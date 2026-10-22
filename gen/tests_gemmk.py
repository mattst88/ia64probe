# SPDX-License-Identifier: GPL-3.0-or-later
"""gemm's two inner loops, as clang compiles them.

Polybench gemm runs C[i][j] += alpha * A[i][k] * B[k][j] twice: fused
(kernel_gemm, an fma.d an element) and with the multiply and the add
apart (kernel_gemm_StrictFP, fma.d with f0 and then with f1).  On an
Itanium 9340 timing each phase with ar.itc gives 901K cycles for the
first and 1280K for the second, where the model says 1099K and 1170K.

gk1/gk2: each loop's body as clang emits it, the loop branch a nop.b and
a stop added where the taken branch ended the group.  B is read through
two pointers (even and odd elements), C through two and written through
two more, 32 bytes (gk1) or 48 bytes (gk2) behind its reads.
_nost: the stores made nops.  _nold: the loads made nops.

gkl_r{R}_d{d}: a group of two ldfd d bytes apart, the targets written
again R groups later: a group costs the loads' latency over R.
gkm_{same,split}_d{d}: four ldfd a group, the pair d bytes apart issued
in one cycle or in two, the targets written again in the next group.
gkp_{group}: the group twelve times, each copy loading its own
registers, so no copy waits on another: the cycles per body over 12 are
the cycles a group takes to issue.  f is an ldfd, p an ldfpd, i an ld8,
each from its own 16-byte chunk in banks 0 to 3.
gkh_{prod}_{order}: does a stall hold what issued before it in its cycle?
A group of the producer (getf.sig, or an ld8 two lines on), then a group
of a pointer-chasing ld8 and an add of the producer's result, the load
first (ld) or the add first (add).  If the add's wait holds the load,
the chase adds a cycle a body for both orders.  _use: the chase goes
through an add in the producer's group, so a held load is a cycle late.
gkg_{getf,ldm}_{held,add}: the same with a getf as the instruction that
may be held, its result read in the next group, and the producer a getf
or an ld8 that misses L1D.
"""

import re

from emit import Bundle as B, pair

K1 = [
    ("mfi", "mov r37 = r3", "fma.d f35 = f8, f9, f12", "adds r3 = -2, r3"),
    ("mmf", "ldfd f9 = [r30], 16", "ldfd f12 = [r31], 16",
     "fma.d f36 = f8, f13, f10"),
    ("mmf", "stfd [r10] = f33, 16", "ldfd f10 = [r8], 16",
     "fma.d f33 = f8, f11, f15"),
    ("mii", "ldfd f13 = [r9], 16", "nop.i 0", "nop.i 0"),
    ("mfi", "stfd [r11] = f34, 16", "fma.d f34 = f8, f32, f14", "nop.i 0 ;;"),
    ("mmi", "cmp.ltu p6, p7 = 1, r37", "ldfd f11 = [r30], 16", "nop.i 0"),
    ("mmi", "ldfd f15 = [r31], 16", "stfd [r10] = f35, 16", "nop.i 0"),
    ("mmi", "ldfd f14 = [r8], 16", "ldfd f32 = [r9], 16", "nop.i 0"),
    ("mib", "stfd [r11] = f36, 16", "nop.i 0", "nop.b 0 ;;"),
]

K2 = [
    ("mmf", "ldfd f35 = [r30], 16", "ldfd f36 = [r31], 16",
     "fma.d f40 = f8, f13, f0"),
    ("mfi", "mov r43 = r3", "fma.d f39 = f8, f14, f0", "adds r3 = -3, r3"),
    ("mmi", "ldfd f37 = [r8], 16", "ldfd f38 = [r9], 16", "nop.i 0 ;;"),
    ("mfi", "stfd [r10] = f15, 16", "fma.d f15 = f12, f1, f33", "nop.i 0"),
    ("mfi", "stfd [r11] = f32, 16", "fma.d f32 = f11, f1, f34", "nop.i 0"),
    ("mmi", "ldfd f34 = [r30], 16", "ldfd f33 = [r31], 16", "nop.i 0"),
    ("mmi", "ldfd f11 = [r8], 16", "ldfd f12 = [r9], 16", "nop.i 0 ;;"),
    ("mmf", "stfd [r10] = f15, 16", "stfd [r11] = f32, 16",
     "fma.d f32 = f8, f35, f0"),
    ("mmf", "ldfd f13 = [r30], 16", "ldfd f14 = [r31], 16",
     "fma.d f36 = f8, f36, f0"),
    ("mfi", "nop.m 0", "fma.d f35 = f10, f1, f39", "nop.i 0"),
    ("mfi", "nop.m 0", "fma.d f39 = f9, f1, f40", "nop.i 0"),
    ("mmf", "ldfd f9 = [r8], 16", "ldfd f10 = [r9], 16",
     "fma.d f33 = f8, f33, f0"),
    ("mfi", "nop.m 0", "fma.d f34 = f8, f34, f0", "nop.i 0 ;;"),
    ("mmf", "stfd [r10] = f35, 16", "stfd [r11] = f39, 16",
     "fma.d f15 = f38, f1, f36"),
    ("mfb", "cmp.ltu p6, p7 = 2, r43", "fma.d f32 = f37, f1, f32",
     "nop.b 0 ;;"),
]


def strip(body, kind):
    out = []
    for t, *slots in body:
        s = []
        for x in slots:
            if re.match(r"(ldfd|stfd) ", x) and x.startswith(kind):
                x = "nop.m 0"
            s.append(x)
        out.append(B(t, *s))
    return out


def rename(body, op, base):
    """Each op's target a register of its own, from fbase on."""
    n = base
    out = []
    for t, *slots in body:
        s = []
        for x in slots:
            if x.startswith(op + " "):
                x = re.sub(r"= f\d+|f\d+ =", lambda m: (
                    f"f{n} =" if m.group(0).endswith("=") else m.group(0)),
                    x, count=1)
                n += 1
            s.append(x)
        out.append((t, *s))
    return out


def setup(b_even, b_odd, c_even, c_odd, lag, bo=8, co=8):
    """B at 1 MB, C at 1 MB + 64 KB + 128: the stores to C trail its loads
    by lag bytes."""
    return ["movl r18 = 0x100000", "add r19 = r33, r18",
            "movl r18 = 0x110080", "add r20 = r33, r18",
            f"mov r{b_even} = r19", f"adds r{b_odd} = {bo}, r19",
            "adds r10 = 0, r20", f"adds r11 = {co}, r20",
            f"adds r{c_even} = {lag}, r20", f"adds r{c_odd} = {lag + co}, r20",
            "mov r3 = 1000", "movl r18 = 0x3ff8000000000000",
            "setf.d f8 = r18"] + [f"ldfd f{r} = [r19]" for r in
                                  (9, 10, 11, 12, 13, 14, 15, 32, 33, 34,
                                   35, 36, 37, 38, 39, 40)]


def fpl(R, d, n=16):
    """n groups of two ldfd d bytes apart, 16 bytes on a group, targets
    reused every R groups; d 0 is one load a group."""
    out = []
    for g in range(n):
        inc = 16 if g < n - 1 else -16 * (n - 1)
        a = f"ldfd f{32 + 2 * (g % R)} = [r60], {inc}"
        b = f"ldfd f{33 + 2 * (g % R)} = [r61], {inc}" if d else None
        out += [B("mmi", a, b, "nop.i 0 ;;")]
    return out


def fpo(order, n=16, R=1, use=False):
    """Four ldfd a group, two a cycle, in the order given: a and b one
    chunk, x and y another chunk 64 KB + 160 bytes on."""
    out = []
    for g in range(n):
        inc = 16 if g < n - 1 else -16 * (n - 1)
        t = 32 + 4 * (g % R)
        ld = {c: f"ldfd f{t + j} = [r{60 + j}], {inc}"
              for j, c in enumerate("abxy")}
        m = [ld[c] for c in order]
        # With use, an fma of the previous group's b and a.
        u = (f"fma.d f{64 + g % 4} = f{32 + 4 * ((g - 1) % R) + 1}, f1, "
             f"f{32 + 4 * ((g - 1) % R)}" if use else "nop.f 0")
        out += [B("mmf", m[0], m[1], u),
                B("mmi", m[2], m[3], "nop.i 0 ;;")]
    return out


def fpc(d, gap, n=16):
    """A group of an ldfd, gap bundles of nops, and a second ldfd d bytes
    on: the second issues gap // 2 cycles after... or with gap 0 in the
    same bundle.  Targets written again in the next group."""
    out = []
    for g in range(n):
        inc = 16 if g < n - 1 else -16 * (n - 1)
        a = f"ldfd f32 = [r60], {inc}"
        b = f"ldfd f33 = [r61], {inc}"
        if gap == 0:
            out += [B("mmi", a, b, "nop.i 0 ;;")]
            continue
        out += [B("mmi", a, None, "nop.i 0")]
        out += [B("mfi", None, None, "nop.i 0")] * (gap - 1)
        out += [B("mmi", b, None, "nop.i 0 ;;")]
    return out


def fpm(split, d, n=16):
    """fpl with four loads a group, two a cycle: the pair d apart in one
    cycle, or with split, one in each; the others to banks 2 and 3."""
    out = []
    for g in range(n):
        inc = 16 if g < n - 1 else -16 * (n - 1)
        a, b, x, y = (f"ldfd f{32 + j} = [r{60 + j}], {inc}" for j in range(4))
        if d == 0:
            b = None
        m = [a, x, b, y] if split else [a, b, x, y]
        out += [B("mmi", m[0], m[1], "nop.i 0"),
                B("mmi", m[2], m[3], "nop.i 0 ;;")]
    return out


GKP = ["ff", "fff", "ffff", "ii", "iii", "iiii", "ffii", "iiff", "fi",
       "pp", "pf", "ppff"]


def fpp(group, copies=12):
    """gkp: the group copies times, two memory slots a bundle."""
    out = []
    for c in range(copies):
        ops = []
        for k, t in enumerate(group):
            base = f"r{60 + k}"
            if t == "f":
                ops.append(f"ldfd f{80 + 4 * c + k} = [{base}]")
            elif t == "p":
                ops.append(f"ldfpd f{32 + 8 * (c % 6) + 2 * k}, "
                           f"f{33 + 8 * (c % 6) + 2 * k} = [{base}]")
            else:
                ops.append(f"ld8 r{64 + 4 * c + k} = [{base}]")
        ops += [None] * (len(ops) % 2)
        for i in range(0, len(ops), 2):
            last = "nop.i 0 ;;" if i + 2 >= len(ops) else "nop.i 0"
            out.append(B("mmi", ops[i], ops[i + 1], last))
    return out


def fph(prod, order, n=8, use=False):
    p = ("getf.sig r42 = f50" if prod == "getf" else "ld8 r42 = [r45]")
    ld, add = "ld8 r41 = [r41]", "add r43 = r42, r0"
    if use:
        ld = "ld8 r44 = [r41]"
    out = []
    for _ in range(n):
        out.append(B("mmi", p, None,
                     "adds r41 = 0, r44 ;;" if use else "nop.i 0 ;;"))
        if order == "ld":
            out.append(B("mmi", ld, None, add + " ;;"))
        else:
            out.append(B("mii", None, add, "nop.i 0"))
            out.append(B("mmi", ld, None, "nop.i 0 ;;"))
    return out


def fpg(prod, order, n=8):
    """gkg: fph with a getf as the instruction held and the producer a
    getf or an ld8 that misses L1D (eight lines 4 KB apart)."""
    out = []
    for g in range(n):
        inc = "r46" if g < n - 1 else "r48"
        p = ("getf.sig r42 = f50" if prod == "getf" else
             f"ld8 r42 = [r45], {inc}")
        out.append(B("mmi", p, None, "adds r47 = 0, r44 ;;"))
        held, add = "getf.sig r44 = f51", "add r43 = r42, r0"
        if order == "held":
            out.append(B("mmi", held, None, add + " ;;"))
        else:
            out.append(B("mii", None, add, "nop.i 0"))
            out.append(B("mmi", held, None, "nop.i 0 ;;"))
    return out


def kernels():
    ks = []
    for prod in ("getf", "ldm"):
        for order in ("held", "add"):
            ks += pair(f"gkg_{prod}_{order}", "gemmk", fpg(prod, order),
                       ["movl r18 = 0x100000", "add r45 = r33, r18",
                        "movl r46 = 4096", "movl r48 = -28672",
                        "setf.sig f50 = r18", "setf.sig f51 = r18"], {},
                       bodies=2, iters=500)
    for prod in ("getf", "ld"):
        for order in ("ld", "add"):
            for use in (False, True):
                ks += pair(f"gkh_{prod}_{order}{'_use' if use else ''}",
                           "gemmk", fph(prod, order, use=use),
                           ["movl r18 = 0x100000", "add r41 = r33, r18",
                            "st8 [r41] = r41", "adds r45 = 256, r41",
                            "mov r44 = r41", "setf.sig f50 = r18"], {},
                           bodies=2, iters=500)
    for g in GKP:
        ks += pair(f"gkp_{g}", "gemmk", fpp(g),
                   ["movl r18 = 0x100000", "add r60 = r33, r18",
                    "adds r61 = 16, r60", "adds r62 = 32, r60",
                    "adds r63 = 48, r60"], {"copies": 12}, bodies=2,
                   iters=500)
    for d in (0, 8, 256):
        for gap in (0, 1, 2, 3, 4):
            ks += pair(f"gkc_d{d}_g{gap}", "gemmk", fpc(d, gap),
                       ["movl r18 = 0x100000", "add r60 = r33, r18",
                        f"adds r61 = {d or 64}, r60"],
                       {"d": d, "gap": gap}, bodies=2, iters=250)
    for order, R, use in (("abxy", 1, 0), ("axby", 1, 0), ("axyb", 1, 0),
                          ("xaby", 1, 0), ("axyb", 2, 0), ("axyb", 4, 0),
                          ("axyb", 4, 1), ("abxy", 4, 1)):
        ks += pair(f"gko_{order}" + (f"_r{R}" if R > 1 else "") +
                   ("_use" if use else ""), "gemmk", fpo(order, R=R, use=use),
                   ["movl r18 = 0x100000", "add r60 = r33, r18",
                    "adds r61 = 8, r60", "movl r18 = 0x100a0",
                    "add r62 = r60, r18", "adds r63 = 8, r62"],
                   {"order": order}, bodies=2, iters=250)
    for split in (False, True):
        for d in (0, 8, 16, 256):
            ks += pair(f"gkm_{'split' if split else 'same'}_d{d}", "gemmk",
                       fpm(split, d),
                       ["movl r18 = 0x100000", "add r60 = r33, r18",
                        f"adds r61 = {d}, r60", "movl r18 = 0x10020",
                        "add r62 = r60, r18", "adds r63 = 16, r62"],
                       {"split": split, "d": d}, bodies=2, iters=250)
    for R in (1, 2, 3):
        for d in (0, 8, 16, 24, 256, 264, 4096):
            ks += pair(f"gkl_r{R}_d{d}", "gemmk", fpl(R, d),
                       ["movl r18 = 0x100000", "add r60 = r33, r18",
                        f"adds r61 = {d}, r60" if d < 8192 else
                        "mov r61 = r60"], {"R": R, "d": d}, bodies=2,
                       iters=250)
    for name, body, regs, lag in (("gk1", K1, (30, 9, 31, 8), 32),
                                  ("gk2", K2, (30, 31, 9, 8), 48)):
        for sfx, kind in (("", "x"), ("_nost", "stfd"), ("_nold", "ldfd")):
            ks += pair(f"{name}{sfx}", "gemmk", strip(body, kind),
                       setup(*regs, lag), {}, bodies=8, iters=60)
        for sfx, op, base in (("_fren", "fma.d", 64), ("_lren", "ldfd", 80)):
            ks += pair(f"{name}_nost{sfx}", "gemmk",
                       strip(rename(body, op, base), "stfd"),
                       setup(*regs, lag), {}, bodies=8, iters=60)
            for bo, co in ((24, 8), (8, 24), (24, 24)):
                ks += pair(f"{name}_nost{sfx}_b{bo}_c{co}", "gemmk",
                           strip(rename(body, op, base), "stfd"),
                           setup(*regs, lag, bo, co), {}, bodies=8,
                           iters=60)
    return ks
