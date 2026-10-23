# SPDX-License-Identifier: GPL-3.0-or-later
"""Inline double division, as clang expands a / b.

tramp3d's functors divide three field differences by cell volumes; a
microbenchmark of one (ia64perf bench/mb/apress.c) costs 23 cycles more a
call on an Itanium 9340 than the same code with the divisions made
multiplies, where the model says 14.  These kernels take the division
sequence apart.

fdv_c3: six ldfd (three numerators, three denominators), then the three
interleaved divisions exactly as clang scheduled them in apress.c, each F
op in an MFI of its own; fdv_c3_np: the same without qualifying
predicates; fdv_c1: the first division alone (its loads and its (p8) ops).

The rest are chains, one instruction a group: frcpa_lat (frcpa into its own
divisor), frcpa_fma (frcpa, then a (p8) fma.s1 of its result: the predicate
and the value), fmas1_lat (fma.s1), fnmas1_lat; and issue tests of eight
independent ops in one group: f8_{op}, f8p_{op} with a true (p6)
qualifying predicate.
"""

from emit import Bundle as B, pair

# apress.c's body, from frcpa to the sum before the compare.
CLANG3 = """\
frcpa f11, p8 = f9, f33 ;;
frcpa f32, p9 = f8, f12
fma.d f10 = f15, f14, f10
(p8) fma.s1 f14 = f9, f11, f0
(p8) fnma.s1 f15 = f33, f11, f1 ;;
(p9) fma.s1 f34 = f8, f32, f0
(p9) fnma.s1 f35 = f12, f32, f1
frcpa f36, p10 = f10, f13
(p8) fma.s1 f14 = f15, f14, f14
(p8) fma.s1 f37 = f15, f15, f0
(p8) fma.s1 f15 = f15, f11, f11 ;;
(p9) fma.s1 f34 = f35, f34, f34
(p9) fma.s1 f38 = f35, f35, f0
(p10) fma.s1 f39 = f10, f36, f0
(p10) fnma.s1 f40 = f13, f36, f1
(p8) fma.s1 f14 = f37, f14, f14
(p8) fma.s1 f41 = f37, f37, f0
(p9) fma.s1 f35 = f35, f32, f32
(p8) fma.s1 f15 = f37, f15, f15 ;;
(p9) fma.s1 f34 = f38, f34, f34
(p9) fma.s1 f42 = f38, f38, f0
(p10) fma.s1 f39 = f40, f39, f39
(p10) fma.s1 f43 = f40, f40, f0
(p8) fma.d.s1 f14 = f41, f14, f14
(p10) fma.s1 f37 = f40, f36, f36
(p9) fma.s1 f35 = f38, f35, f35
(p8) fma.s1 f15 = f41, f15, f15 ;;
(p9) fma.d.s1 f34 = f42, f34, f34
(p10) fma.s1 f38 = f43, f39, f39
(p10) fma.s1 f39 = f43, f43, f0
(p8) fnma.s1 f9 = f33, f14, f9
(p9) fma.s1 f33 = f42, f35, f35 ;;
(p9) fnma.s1 f8 = f12, f34, f8
(p10) fma.s1 f12 = f43, f37, f37
(p10) fma.d.s1 f35 = f39, f38, f38
(p8) fma.d f11 = f9, f15, f14 ;;
(p9) fma.d f32 = f8, f33, f34
(p10) fma.s1 f8 = f39, f12, f12
(p10) fnma.s1 f9 = f13, f35, f10 ;;
(p10) fma.d f36 = f9, f8, f35
fma.d f8 = f32, f1, f11 ;;
fma.d f8 = f8, f1, f36 ;;
"""

NUM = ("f9", "f8", "f10")
DEN = ("f33", "f12", "f13")
ONE = 0x3ff0000000000000
THREE = 0x4008000000000000

SETUP = [f"movl r18 = {ONE:#x}", "adds r34 = 64, r33 ;;", "st8 [r34] = r18",
         "adds r35 = 72, r33", f"movl r19 = {THREE:#x} ;;",
         "st8 [r35] = r19 ;;", "mov r18 = 1 ;;", "setf.sig f20 = r18 ;;",
         "fcvt.xf f20 = f20 ;;", "fnorm.d f20 = f20 ;;", "mov f33 = f20",
         "mov f34 = f20", "cmp.eq p6, p7 = r0, r0"]


def ops(text, strip=False, only=None):
    out = []
    for l in text.splitlines():
        l = l.strip()
        if only and not (l.startswith(f"({only})") or l.startswith("frcpa")
                         and f", {only} " in l):
            # Keep the stop of a dropped op on the op before it.
            if l.endswith(";;") and out and not out[-1].endswith(";;"):
                out[-1] += " ;;"
            continue
        if strip and l.startswith("("):
            l = l[l.index(")") + 1:].strip()
        out.append(l)
    if out and not out[-1].endswith(";;"):
        out[-1] += " ;;"
    return out


def loads(regs):
    out = []
    for k in range(0, len(regs), 2):
        pair_ = [f"ldfd {r} = [r34]" if r in NUM else f"ldfd {r} = [r35]"
                 for r in regs[k:k + 2]]
        while len(pair_) < 2:
            pair_.append(None)
        last = k + 2 >= len(regs)
        out.append(B("mmi", pair_[0], pair_[1], "nop.i 0 ;;" if last
                     else None))
    return out


def fbody(lines):
    out = []
    for l in lines:
        stop = l.endswith(";;")
        out.append(B("mfi", None, l.rstrip(" ;"),
                     "nop.i 0 ;;" if stop else None))
    return out


def chain(insns):
    return [B("mfi", None, t, "nop.i 0 ;;") for t in insns]



# apress.c's body as clang bundled it in prof.c's call(), from the fneg
# of the field differences to the compare that guards the update (the
# branch dropped): the FP loads, the products and sums that feed frcpa,
# the three divisions and their sum.  Registers the harness keeps (r14,
# r15) and gp are renamed away, and the index getf reads f0.
CTX = """\
mmf | add r8=r18,r0 | shladd r3=r23,3,r21 | fneg f10=f10;;
mmf | shladd r11=r30,3,r8 | ldfd f13=[r3] | fneg f8=f8
mii | getf.sig r23=f0 | nop.i 0 | nop.i 0
mfi | nop.m 0 | fmpy.d.s0 f9=f11,f10 | nop.i 0
mii | ldfd f10=[r28] | nop.i 0 | nop.i 0
mmi | ldfd f11=[r19];; | nop.m 0 | nop.i 0
mfi | nop.m 0 | fmpy.d.s0 f8=f12,f8 | nop.i 0
mmf | ldfd f12=[r10] | shladd r24=r23,3,r21 | fma.d.s0 f9=f14,f15,f9
mfi | nop.m 0 | fneg f13=f13 | nop.i 0;;
mii | ldfd f14=[r24] | nop.i 0 | nop.i 0
mii | ldfd f15=[r45] | shladd r45=r0,3,r25 | nop.i 0
mfi | nop.m 0 | fma.d.s0 f8=f10,f11,f8 | nop.i 0
mfi | nop.m 0 | fmpy.d.s0 f10=f32,f13 | nop.i 0
mfi | ldfd f13=[r11] | frcpa.s0 f11,p8=f9,f33 | nop.i 0;;
mfi | nop.m 0 | frcpa.s0 f32,p9=f8,f12 | nop.i 0
mfi | nop.m 0 | fma.d.s0 f10=f15,f14,f10 | nop.i 0
mfi | nop.m 0 | (p8) fmpy.s1 f14=f9,f11 | nop.i 0
mfi | nop.m 0 | (p8) fnma.s1 f15=f33,f11,f1 | nop.i 0;;
mfi | nop.m 0 | (p9) fmpy.s1 f34=f8,f32 | nop.i 0
mfi | nop.m 0 | (p9) fnma.s1 f35=f12,f32,f1 | nop.i 0
mfi | nop.m 0 | frcpa.s0 f36,p10=f10,f13 | nop.i 0
mfi | nop.m 0 | (p8) fma.s1 f14=f15,f14,f14 | nop.i 0
mfi | nop.m 0 | (p8) fmpy.s1 f37=f15,f15 | nop.i 0
mfi | nop.m 0 | (p8) fma.s1 f15=f15,f11,f11 | nop.i 0;;
mfi | nop.m 0 | (p9) fma.s1 f34=f35,f34,f34 | nop.i 0
mfi | nop.m 0 | (p9) fmpy.s1 f38=f35,f35 | nop.i 0
mfi | nop.m 0 | (p10) fmpy.s1 f39=f10,f36 | nop.i 0
mfi | nop.m 0 | (p10) fnma.s1 f40=f13,f36,f1 | nop.i 0
mfi | nop.m 0 | (p8) fma.s1 f14=f37,f14,f14 | nop.i 0
mfi | nop.m 0 | (p8) fmpy.s1 f41=f37,f37 | nop.i 0
mfi | nop.m 0 | (p9) fma.s1 f35=f35,f32,f32 | nop.i 0
mfi | nop.m 0 | (p8) fma.s1 f15=f37,f15,f15 | nop.i 0;;
mfi | nop.m 0 | (p9) fma.s1 f34=f38,f34,f34 | nop.i 0
mfi | nop.m 0 | (p9) fmpy.s1 f42=f38,f38 | nop.i 0
mfi | nop.m 0 | (p10) fma.s1 f39=f40,f39,f39 | nop.i 0
mfi | nop.m 0 | (p10) fmpy.s1 f43=f40,f40 | nop.i 0
mfi | nop.m 0 | (p8) fma.d.s1 f14=f41,f14,f14 | nop.i 0
mfi | nop.m 0 | (p10) fma.s1 f37=f40,f36,f36 | nop.i 0
mfi | nop.m 0 | (p9) fma.s1 f35=f38,f35,f35 | nop.i 0
mfi | nop.m 0 | (p8) fma.s1 f15=f41,f15,f15 | nop.i 0;;
mfi | nop.m 0 | (p9) fma.d.s1 f34=f42,f34,f34 | nop.i 0
mfi | nop.m 0 | (p10) fma.s1 f38=f43,f39,f39 | nop.i 0
mfi | nop.m 0 | (p10) fmpy.s1 f39=f43,f43 | nop.i 0
mfi | nop.m 0 | (p8) fnma.s1 f9=f33,f14,f9 | nop.i 0
mfi | nop.m 0 | (p9) fma.s1 f33=f42,f35,f35 | nop.i 0;;
mfi | nop.m 0 | (p9) fnma.s1 f8=f12,f34,f8 | nop.i 0
mfi | nop.m 0 | (p10) fma.s1 f12=f43,f37,f37 | nop.i 0
mfi | nop.m 0 | (p10) fma.d.s1 f35=f39,f38,f38 | nop.i 0
mfi | nop.m 0 | (p8) fma.d.s0 f11=f9,f15,f14 | nop.i 0;;
mfi | nop.m 0 | (p9) fma.d.s0 f32=f8,f33,f34 | nop.i 0
mfi | nop.m 0 | (p10) fma.s1 f8=f39,f12,f12 | nop.i 0
mfi | nop.m 0 | (p10) fnma.s1 f9=f13,f35,f10 | nop.i 0;;
mfi | nop.m 0 | (p10) fma.d.s0 f36=f9,f8,f35 | nop.i 0
mfi | nop.m 0 | fadd.d.s0 f8=f32,f11 | nop.i 0;;
mfi | nop.m 0 | fadd.d.s0 f8=f8,f36 | nop.i 0;;
mfb | nop.m 0 | fcmp.le.s0 p6,p7=f0,f8 | nop.b 0;;
"""

# Each address register of CTX and of the prelude, and the double there.
CTX_ADDR = (("r3", 1.25), ("r28", 0.75), ("r19", 1.5), ("r10", 2.5),
            ("r21", 1.75), ("r25", 0.625), ("r18", 3.25))
# The FP registers CTX reads before writing, each from a bank of its own.
CTX_LIVE = (("f10", 1.125), ("f8", 0.875), ("f11", 1.375), ("f12", 2.25),
            ("f14", 0.5), ("f15", 1.625), ("f32", 2.75), ("f33", 1.0625))


def ctx_body(text):
    out = []
    for l in text.splitlines():
        t, *slots = [x.strip() for x in l.split("|")]
        out.append(B(t, *[None if x.startswith("nop") and not
                          x.endswith(";;") else x for x in slots]))
    return out


def ctx_kernel(name, text):
    import struct
    setup, k = [], 0
    for r, v in CTX_ADDR:
        bits = struct.unpack("<Q", struct.pack("<d", v))[0]
        setup += [f"adds {r} = {256 + 16 * k}, r33", f"movl r40 = {bits:#x} ;;",
                  f"st8 [{r}] = r40 ;;"]
        k += 1
    # r11 is r30 * 8 + r18; r3 and r24 are r23 * 8 + r21, one address.
    # r45, which f15's load reads before the shladd that sets it, is r25.
    setup += ["mov r30 = r0", "mov r23 = r0", "mov r45 = r25"]
    pre = []
    for i, (f, v) in enumerate(CTX_LIVE):
        bits = struct.unpack("<Q", struct.pack("<d", v))[0]
        setup += [f"adds r{50 + i} = {256 + 16 * k}, r33",
                  f"movl r40 = {bits:#x} ;;", f"st8 [r{50 + i}] = r40 ;;"]
        k += 1
    for i in range(0, len(CTX_LIVE), 2):
        pre.append(B("mmi", f"ldfd {CTX_LIVE[i][0]} = [r{50 + i}]",
                     f"ldfd {CTX_LIVE[i + 1][0]} = [r{51 + i}]",
                     "nop.i 0 ;;" if i % 4 == 2 else None))
    return pair(name, "fdiv", pre + ctx_body(text), SETUP + setup, bodies=2)


# The update that follows when the sum is negative, from the same call():
# the strides of P, R and C loaded and multiplied by j and k, the loads of
# R, P and C at (i, j, k), two stores to P.  The structs are {p, sj, sk}
# with post-increments dropped (r73-r75 are their sk fields), r14 (i) is
# r62 and counts up a body, and r15-r17, which the harness keeps, are
# r64, r76 and r77; a prelude bundle pair gives f6 j,
# f7 k and f8 q again, and recomputes the sj addresses.
UPD = """\
mmi | ld8 r77=[r77] | ld8 r76=[r76] | adds r21=8,r72
mlx | ld8 r18=[r71] | movl r24=0x3fd3333333333333;;
mmi | setf.sig f9=r77 | ld8 r19=[r70] | nop.i 0
mmi | setf.sig f10=r76 | ld8 r22=[r73] | nop.i 0;;
mlx | nop.m 0 | movl r77=0xbfe6666666666666
mmi | ld8 r23=[r74] | setf.sig f11=r22 | nop.i 0;;
mmi | setf.sig f12=r23 | ldfd f13=[r64] | nop.i 0
mmi | ld8 r30=[r72];; | ld8 r21=[r21] | nop.i 0
mfi | ld8 r9=[r75] | xmpy.l f9=f9,f6 | nop.i 0
mfi | nop.m 0 | xmpy.l f10=f10,f6 | nop.i 0
mfi | nop.m 0 | xmpy.l f11=f11,f7 | nop.i 0
mfi | nop.m 0 | xmpy.l f12=f12,f7 | nop.i 0
mfi | nop.m 0 | fmpy.d.s0 f8=f8,f13 | nop.i 0;;
mmi | getf.sig r25=f9 | setf.d f9=r24 | nop.i 0
mmi | getf.sig r26=f10 | setf.sig f10=r21 | nop.i 0
mmi | getf.sig r27=f11 | setf.sig f11=r9 | nop.i 0
mmi | getf.sig r20=f12;; | shladd r28=r25,3,r18 | shladd r29=r26,3,r19
mfi | nop.m 0 | fmpy.d.s0 f9=f8,f9 | nop.i 0;;
mmf | shladd r31=r62,3,r28 | shladd r8=r62,3,r29 | xmpy.l f6=f10,f6;;
mii | shladd r76=r27,3,r31 | shladd r10=r20,3,r8;; | nop.i 0
mmf | ldfd f12=[r76] | ldfd.sa f10=[r76] | xmpy.l f7=f11,f7
mmf | ldfd f13=[r10] | getf.sig r11=f6 | fmpy.d.s0 f9=f8,f9;;
mfi | getf.sig r3=f7 | fma.d.s0 f9=f9,f12,f13 | shladd r64=r11,3,r30;;
mii | setf.d f6=r77 | shladd r63=r62,3,r64;; | shladd r63=r3,3,r63
mmi | stfd [r10]=f9 | ldfd.c.clr f10=[r76] | nop.i 0;;
mbb | ldfd f7=[r63] | nop.b 0 | nop.b 0
mfi | nop.m 0 | fmpy.d.s0 f6=f10,f6 | nop.i 0;;
mfi | nop.m 0 | fmpy.d.s0 f6=f6,f7 | nop.i 0;;
mfi | nop.m 0 | fma.d.s0 f6=f6,f8,f9 | nop.i 0;;
mii | stfd [r10]=f6 | nop.i 0 | nop.i 0
"""


def upd_kernel(name, text, apart=0xB640):
    import struct
    # A cut must end its group: the prelude writes f6-f8, which the cut's
    # last loads may write too.
    if not text.rstrip().endswith(";;"):
        text = text.rstrip() + ";;\n"
    neg = struct.unpack("<Q", struct.pack("<d", -0.5))[0]
    setup = [f"movl r40 = {neg:#x} ;;", "setf.d f21 = r40",
             "mov r60 = 3", "mov r61 = 5", "mov r62 = 1"]
    # P, R and C: struct f at 0x2000 + 64n, its array at 0x10000 + apart
    # n: 0xB640 as apress.c's 18^3 doubles are laid out, so no two match
    # in address bits 11:0, which the ALAT compares.
    for n, (r, rk) in enumerate((("r70", "r74"), ("r71", "r73"),
                                 ("r72", "r75"))):
        setup += [f"movl r41 = {0x2000 + 64 * n:#x} ;;",
                  f"add {r} = r41, r33", f"movl r42 = {0x10000 + apart * n:#x} ;;",
                  f"add r42 = r42, r33 ;;", f"st8 [{r}] = r42",
                  f"adds r43 = 8, {r}", f"adds {rk} = 16, {r} ;;",
                  "mov r44 = 18 ;;", "st8 [r43] = r44", "mov r44 = 324 ;;",
                  f"st8 [{rk}] = r44 ;;"]
    setup += ["movl r41 = 0x70000 ;;", "add r64 = r41, r33"]
    pre = [B("mmi", "adds r77 = 8, r71", "adds r76 = 8, r70", "adds r62 = 1, r62"),
           B("mmf", "setf.sig f6 = r60", "setf.sig f7 = r61", "mov f8 = f21 ;;")]
    return pair(name, "fdiv", pre + ctx_body(text), SETUP + setup, bodies=2)

def kernels():
    ks = []
    regs = [r for p in zip(NUM, DEN) for r in p]
    ks += pair("fdv_c3", "fdiv", loads(regs) + fbody(ops(CLANG3)), SETUP,
               bodies=2)
    ks += pair("fdv_c3_np", "fdiv",
               loads(regs) + fbody(ops(CLANG3, strip=True)), SETUP, bodies=2)
    # The same with other work in the M slots beside each F op: an ldfd or
    # ld8 to a register of its own, or a setf.sig, all from/to the scratch.
    for tag, mk in (("ldf", lambda k: f"ldfd f{50 + k % 16} = [r36]"),
                    ("ld8", lambda k: f"ld8 r{40 + k % 16} = [r36]"),
                    ("setf", lambda k: f"setf.sig f{50 + k % 16} = r0")):
        fb = []
        for k, b in enumerate(fbody(ops(CLANG3))):
            t = b.text()
            fop, stop = ops(CLANG3)[k].rstrip(" ;"), ops(CLANG3)[k].endswith(";;")
            fb.append(B("mmf", mk(k), None, fop + (" ;;" if stop else "")))
        ks += pair(f"fdv_c3_{tag}", "fdiv", loads(regs) + fb,
                   SETUP + ["adds r36 = 128, r33"], bodies=2)
    c1 = [l for l in ops(CLANG3, only="p8")]
    ks += pair("fdv_c1", "fdiv", loads(["f9", "f33"]) + fbody(c1), SETUP,
               bodies=2)
    ks += pair("frcpa_lat", "fdiv", chain(["frcpa f33, p8 = f20, f33"]),
               SETUP, meta={"chain": ["frcpa"]})
    ks += pair("frcpa_fma", "fdiv",
               chain(["frcpa f33, p8 = f20, f33",
                      "(p8) fma.s1 f33 = f33, f1, f0"]), SETUP)
    # The producers frcpa reads in apress.c: an fma.d.s0 into the
    # numerator or the divisor, and fmerge.ns into an fma.
    for name, c in (
            ("fmad_frcpa_den", ["frcpa f33, p8 = f20, f33",
                                "fma.d f33 = f33, f1, f0"]),
            ("fmad_frcpa_num", ["frcpa f33, p8 = f33, f20",
                                "fma.d f33 = f33, f1, f0"]),
            ("fmerge_fmad", ["fmerge.ns f33 = f33, f33",
                             "fma.d f33 = f33, f1, f0"]),
            ("fmad_fmas1", ["fma.d f33 = f33, f1, f0",
                            "fma.s1 f33 = f33, f1, f0"]),
            ("fmas1_fmad_s0", ["fma.s1 f33 = f33, f1, f0",
                               "fma.d.s0 f33 = f33, f1, f0"]),
            ("frcpa_fnma_fmad", ["frcpa f35, p8 = f20, f33",
                                 "(p8) fnma.s1 f33 = f33, f35, f1",
                                 "fma.d f33 = f33, f1, f0"])):
        ks += pair(name, "fdiv", chain(c), SETUP)
    ks += pair("fmas1_lat", "fdiv", chain(["fma.s1 f33 = f33, f1, f0"]),
               SETUP, meta={"expect_cyc_per_body": 4.0})
    ks += pair("fnmas1_lat", "fdiv", chain(["fnma.s1 f33 = f34, f33, f1"]),
               SETUP)
    for op in ("fma.s1", "fma.d", "fnma.s1", "frcpa"):
        for p in ("", "(p6) "):
            body = []
            for k in range(8):
                d = f"f{40 + k}"
                t = (f"{p}frcpa {d}, p{8 + k} = f20, f34" if op == "frcpa"
                     else f"{p}{op} {d} = f34, f20, f1")
                body.append(B("mfi", None, t, "nop.i 0 ;;" if k == 7
                              else None))
            ks += pair(f"f8{'p' if p else ''}_{op.replace('.', '')}",
                       "fdiv", body, SETUP)
    ks += ctx_kernel("fdv_ctx", CTX)
    ks += upd_kernel("fdv_upd", UPD)
    # Without each store, without the ALAT pair, and with P's load moved
    # off the address the stores write.
    for tag, a, b in (("nost1", "stfd [r10]=f9", "nop.m 0"),
                      ("nost2", "stfd [r10]=f6", "nop.m 0"),
                      ("noalat", "ldfd.sa f10=[r76]", "ldfd f10=[r76]"),
                      ("ldoff", "ldfd f13=[r10]", "ldfd f13=[r64]")):
        t = UPD.replace(a, b)
        if tag == "noalat":
            t = t.replace("ldfd.c.clr f10=[r76]", "nop.m 0")
        ks += upd_kernel(f"fdv_upd_{tag}", t)
    # The update cut after its first n bundles, to find where hw spends
    # the cycles the model does not.
    for n in (8, 13, 17, 21, 24, 26):
        ks += upd_kernel(f"fdv_updcut{n}",
                         "\n".join(UPD.splitlines()[:n]) + "\n")
    # cut26 (ending at the stfd, the hitting ldfd.c.clr beside it and the
    # ldfd of C) without the check, with a plain reload for it, without
    # the store, with ldfd.c.nc, with the check a group later, and without
    # C's load.
    c26 = "\n".join(UPD.splitlines()[:26]) + "\n"
    chk = "stfd [r10]=f9 | ldfd.c.clr f10=[r76] | nop.i 0;;"
    assert c26.count(chk) == 1
    for tag, t in (
            ("nochk", c26.replace(chk, "stfd [r10]=f9 | nop.m 0 | nop.i 0;;")),
            ("reld", c26.replace("ldfd.c.clr", "ldfd")),
            ("nost", c26.replace("stfd [r10]=f9", "nop.m 0")),
            ("nc", c26.replace("ldfd.c.clr", "ldfd.c.nc")),
            ("late", c26.replace(chk, "stfd [r10]=f9 | nop.m 0 | nop.i 0;;")
             .replace("mbb | ldfd f7=[r63] | nop.b 0 | nop.b 0",
                      "mmi | ldfd f7=[r63] | ldfd.c.clr f10=[r76] | nop.i 0")),
            ("nold", c26.replace("ldfd f7=[r63]", "nop.m 0")),
            # Other addresses for C's load: P's (r10), and r64, which
            # leaves the getf of k * sk out of the address; an ld8 of it.
            ("ldp", c26.replace("ldfd f7=[r63]", "ldfd f7=[r10]")),
            ("ld64", c26.replace("ldfd f7=[r63]", "ldfd f7=[r64]")),
            ("ld8", c26.replace("ldfd f7=[r63]", "ld8 r40=[r63]"))):
        ks += upd_kernel(f"fdv_c26_{tag}", t)
    # P and R a multiple of 4 KB apart: the store to P kills R's entry.
    ks += upd_kernel("fdv_upd_alias", UPD, apart=0x20000)
    return ks
