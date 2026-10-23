# SPDX-License-Identifier: GPL-3.0-or-later
"""fdtd-2d's ey update, as clang pipelines it.

Polybench fdtd-2d's ey[i][j] -= 0.5 * (hz[i][j] - hz[i-1][j]), two
elements an iteration: six ldfd (ey, hz[i] and hz[i-1], even and odd
elements, 8 bytes apart, each pointer stepping 16) and two stfd to ey, in
three groups.  Packing four of the FP loads into one cycle (fdk_new)
instead of two a cycle (fdk_old) made fdtd2d 9.8% slower on an Itanium
9340, where the model said 2%.

Each body is eight iterations with targets of their own, so no load
waits on another's target and the FP operations read only f8 to f15.
fdk_{order}: the groups of an iteration, by stream: e/o an even/odd
element, E ey, H hz[i], P hz[i-1], s a store to ey.  _nost: no stores.

fpi_{incs}: can all four FP loads of a cycle update their base?  251109
2.2 gives the peak as four loads and two post-increment ALU ops a clock.

fsl_s{sd}_l{ld}_x{x} (m for minus): two stfd sd bytes apart and two ldfd ld bytes apart
in one cycle, the stores x bytes from the loads; _split: the stores a
cycle before the loads.

fsm_s{ns}_{f,i}{nl}: ns stfd and nl ldfd (f) or ld8 (i) a cycle, each
to a stream of its own.

fsg_{x}: a group of stores and FP loads at the offsets FSG gives.
fsgi{n}_{x}: the same, each group followed by n empty ones.

frb_{v}_x{x}: fdtd-2d's ex update with hz[j - 1] taken from the iteration
before, as clang pipelines it at II 2: a group of an stfd, then a group of
two ldfd x bytes apart (ex[j] and hz[j], 81920 bytes apart in polybench's
layout), every pointer stepping 8.  v: base as above; split, one load moved
to the store's group; nost, no store; one, the second load a nop; u{R},
base with an fms in the store's group of the two loads R copies (2R
cycles) back, as the loop subtracts them; a{R}/b{R}, the fms of only the
first/second load R copies back (the other operand f9); a{R}s, a{R} with
the two loads in one bundle instead of two.

frh_{st}_d{d}: fdtd-2d's hz update with ex[j] taken from the iteration
before, as clang pipelines it at II 3: a group of an stfd to hz[j - 7] and
ldfd of ey[i][j] and hz[j], an empty group, and a group of ldfd of ex[j +
1] and ey[i + 1][j], at polybench's offsets (each array 0xa000 bytes after
the last, a row 640 bytes), every pointer stepping 8.  st: 1, the store as
above; 2, in the empty group; 3, in the last group; n, none.  d: bytes
added to the hz load's address.  _c{k...}: an fms of load k (0 ex, 1
ey, 2 ey[i + 1], 3 hz) as many cycles after it as the loop's: 6, 13, 10
and 17.

frq_x{x}_{cb}_g{g}_c{k}: does a bank conflict delay a later load to the
bank?  Copies of eight cycles: at cycle 0, ldfd A and B x bytes apart; at
cycle g, ldfd C, 0x28000 bytes from A (cb s, A's bank) or 0x28040 (d,
another bank); an fms of load k (a, b or c) 6 cycles after it.
frq_{o}_g{g}_c{k}: frq_x81920_d, and at cycle g also ldfd D, 0x3c0a0
bytes from A, after C (o cd) or before it (dc); k may be d.
"""

from emit import Bundle as B

PTR = {"Ee": 60, "Eo": 61, "He": 62, "Pe": 63, "Ho": 64, "Po": 65,
       "se": 66, "so": 67}

# Each bundle's memory slots, a stop after every second bundle; old and
# new as clang emitted them (fdtd2d at 0x2480).
ORDERS = {
    "old": [["He", "Pe"], ["se"], ["so", "Ho"], ["Po"], ["Ee"], ["Eo"]],
    "new": [["He", "Pe"], ["Po", "Ho"], ["so", "se"], ["Eo", "Ee"], [], []],
    "newsplit": [["He", "Pe"], ["Ee", "Eo"], ["so", "se"], ["Po", "Ho"],
                 [], []],
}


def body(order, copies=8, nost=False):
    out = []
    ld = st = fp = 0
    for c in range(copies):
        for i, slots in enumerate(ORDERS[order]):
            m = []
            for s in slots:
                if s.startswith("s") and nost:
                    continue
                if s.startswith("s"):
                    m.append(f"stfd [r{PTR[s]}] = f9, 16")
                else:
                    m.append(f"ldfd f{32 + ld} = [r{PTR[s]}], 16")
                    ld += 1
            f = f"fsub.d f{80 + fp % 40} = f10, f11"
            fp += 1
            stop = " ;;" if i % 2 else ""
            if len(m) == 2:
                out.append(B("mmf", m[0], m[1], f + stop))
            elif len(m) == 1:
                out.append(B("mfi", m[0], f, "nop.i 0" + stop))
            else:
                out.append(B("mfi", None, f, "nop.i 0" + stop))
    return out


def setup():
    """ey at 1 MB, hz[i] at 1 MB + 64 KB + 128, hz[i-1] 640 bytes below;
    the stores six iterations (96 bytes) behind the loads, as in the loop
    (p22 against p16)."""
    return ["movl r18 = 0x100000", "add r60 = r33, r18",
            "adds r61 = 8, r60", "adds r66 = -96, r60", "adds r67 = -88, r60",
            "movl r18 = 0x110080", "add r62 = r33, r18",
            "adds r64 = 8, r62", "adds r63 = -640, r62",
            "adds r65 = -632, r62", "movl r18 = 0x3fe0000000000000",
            "setf.d f8 = r18", "setf.d f9 = r18", "setf.d f10 = r18",
            "setf.d f11 = r18"]


def fpi(incs, copies=12):
    """fpi_{incs}: a group of four ldfd copies times, each to registers of
    its own; incs says which update their base (1) and which do not (0),
    +16 in even copies and -16 in odd ones so the streams stay put."""
    out = []
    for c in range(copies):
        ops = []
        for k, u in enumerate(incs):
            ld = f"ldfd f{32 + 4 * c + k} = [r{60 + k}]"
            if u == "1":
                ld += ", 16" if c % 2 == 0 else ", -16"
            ops.append(ld)
        out += [B("mmi", ops[0], ops[1], "nop.i 0"),
                B("mmi", ops[2], ops[3], "nop.i 0 ;;")]
    return out


def fsl(split, copies=12):
    """fsl: two stfd and two ldfd a group, the stores to r66/r67 and the
    loads from r60/r61, each pointer stepping 16; with split, the stores
    in one cycle and the loads in the next."""
    out = []
    for c in range(copies):
        s = [f"stfd [r{66 + k}] = f9, 16" for k in range(2)]
        ld = [f"ldfd f{32 + 2 * c + k} = [r{60 + k}], 16" for k in range(2)]
        if split:
            out += [B("mmi", s[0], s[1], "nop.i 0 ;;"),
                    B("mmi", ld[0], ld[1], "nop.i 0 ;;")]
        else:
            out += [B("mmi", s[0], s[1], "nop.i 0"),
                    B("mmi", ld[0], ld[1], "nop.i 0 ;;")]
    return out


def fsm(ns, nl, kind="f", copies=12):
    """fsm: ns stfd and nl loads a group, each from a stream of its own
    64 bytes from the last; the loads ldfd, or ld8 with kind i."""
    out = []
    for c in range(copies):
        ops = [f"stfd [r{66 + k}] = f9, 16" for k in range(ns)]
        for k in range(nl):
            ops.append(f"ldfd f{32 + 4 * c + k} = [r{60 + k}], 16" if kind == "f"
                       else f"ld8 r{70 + 4 * (c % 4) + k} = [r{60 + k}], 16")
        ops += [None] * (len(ops) % 2)
        for i in range(0, len(ops), 2):
            last = "nop.i 0 ;;" if i + 2 >= len(ops) else "nop.i 0"
            out.append(B("mmi", ops[i], ops[i + 1], last))
    return out


FSM = [(1, 0), (2, 0), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (0, 2),
       (0, 4)]


# (stores apart, loads apart, store to load): fdtd2d's is (8, 8, -96).
FSL = [(8, 8, -96), (64, 8, -96), (8, 64, -96), (64, 64, -96),
       (8, 8, -4096), (8, 8, 4096 + 256), (8, 8, 65536 + 32), (8, 8, -16),
       (8, 8, -128), (64, 64, 4352), (64, 8, 4352), (8, 64, 4352),
       (64, 64, 4096), (64, 64, -4096), (64, 64, -64), (64, 64, 96),
       (64, 64, -224)]


def fsg(S, L, copies=12, idle=0):
    """fsg: one group a copy of stfd to each offset in S and ldfd from
    each offset in L (stores first), each pointer stepping 16."""
    out = []
    for c in range(copies):
        ops = [f"stfd [r{66 + k}] = f9, 16" for k in range(len(S))]
        ops += [f"ldfd f{32 + 4 * c + k} = [r{60 + k}], 16"
                for k in range(len(L))]
        ops += [None] * (len(ops) % 2)
        for i in range(0, len(ops), 2):
            last = "nop.i 0 ;;" if i + 2 >= len(ops) else "nop.i 0"
            out.append(B("mmi", ops[i], ops[i + 1], last))
        out += [B("mii", None, None, "nop.i 0 ;;")] * idle
    return out


FSG = {"a": ([0, 8], []), "b": ([0, 8], [4352]),
       "c": ([0, 64], [4352, 4360]), "d": ([0], [96, 104]),
       "e": ([0], [96, 160]), "f": ([0, 64], [96, 160]),
       "g": ([0, 8], [4352, 4416]), "h": ([0, 16], [4352, 4416]),
       "i": ([0, 128], [4352, 4416]), "j": ([0, 256], [4352, 4416]),
       "k": ([0, 32], [4352, 4416]), "l": ([0, 16], []),
       "m": ([0, 128], []), "n": ([0, 8], [4352, 4416, 8128]),
       "o": ([0, 8], [8128]), "p": ([0], [-96, -88]),
       "q": ([0], [4352]), "r": ([0], [4352, 4360])}
# fsg_sd{d}: two stores d bytes apart, two loads far from them.
for d in (24, 48, 80, 96, 112, 160, 192, 224, 320, 384, 448, 512, 1024,
          2048, 4096):
    FSG[f"sd{d}"] = ([0, d], [6144, 6208])
# fsg_t{L}_{d}: two stores d bytes apart and loads at L ("u": 4352, 4416;
# "v": 6144, 6208; "w": 6144 alone; "x": none).
for L, lo in (("u", [4352, 4416]), ("v", [6144, 6208]), ("w", [6144])):
    for d in range(8, 72, 8):
        FSG[f"t{L}_{d}"] = ([0, d], lo)


def frb(v, copies=12):
    """frb: copies of an stfd group and a group of two ldfd (see above)."""
    out = []
    for c in range(copies):
        st = None if v == "nost" else "stfd [r66] = f9, 8"
        a = f"ldfd f{32 + 2 * c} = [r60], 8"
        b = None if v == "one" else f"ldfd f{33 + 2 * c} = [r61], 8"
        f = "nop.f 0"
        if v[0] in "uab" and v[1].isdigit():
            p = (c - int(v[1:].rstrip("s"))) % copies
            x, y = f"f{32 + 2 * p}", f"f{33 + 2 * p}"
            x, y = {"u": (x, y), "a": (x, "f9"), "b": ("f9", y)}[v[0]]
            f = f"fms.d f{64 + c} = {x}, f1, {y}"
        if v == "split":
            out += [B("mmi", st, a, "nop.i 0 ;;"), B("mmi", b, None, "nop.i 0 ;;")]
        else:
            out += [B("mmf", st, None, f + " ;;")]
            if v[0] in "ab" and v[1].isdigit() and not v.endswith("s"):
                out += [B("mfi", a, "nop.f 0", "nop.i 0"),
                        B("mfi", b, "nop.f 0", "nop.i 0 ;;")]
            else:
                out += [B("mmi", a, b, "nop.i 0 ;;")]
    return out


def frh(st, cons="", copies=12):
    """frh: copies of fdtd-2d's hz loop's three groups (see above)."""
    out = []

    def use(k, c, back):
        # An fms of load k (0 ex, 1 ey, 2 ey1, 3 hz) of the copy back copies
        # before, or a nop if cons doesn't name it.
        if str(k) not in cons:
            return "nop.f 0"
        return (f"fms.d f{80 + 4 * c + k} = "
                f"f{32 + 4 * ((c - back) % copies) + (2, 0, 3, 1)[k]}, f1, f9")
    for c in range(copies):
        t = 32 + 4 * c
        s = "stfd [r66] = f9, 8"
        out += [B("mmf", s if st == "1" else None, f"ldfd f{t} = [r61], 8",
                  use(2, c, 4)),
                B("mmi", f"ldfd f{t + 1} = [r62], 8", None, "nop.i 0 ;;"),
                B("mfi", s if st == "2" else None, use(1, c, 4),
                  "nop.i 0 ;;"),
                B("mmf", f"ldfd f{t + 2} = [r60], 8",
                  f"ldfd f{t + 3} = [r63], 8", use(0, c, 2)),
                B("mfi", s if st == "3" else None, use(3, c, 5),
                  "nop.i 0 ;;")]
    return out


FRH = [("1", 0, ""), ("2", 0, ""), ("3", 0, ""), ("n", 0, ""),
       ("1", 8, ""), ("1", 16, ""), ("1", 64, ""), ("1", 128, ""),
       ("n", 128, "")]
FRH += [(st, d, c) for st in "1n" for d in (0, 128)
        for c in ("0", "1", "2", "3", "0123")]


def frq(g, k, copies=12, length=8, two=""):
    """frq: copies of loads A and B at cycle 0, C at g (see above)."""
    cyc = [[[], []] for _ in range(copies * length)]
    for c in range(copies):
        t, at = 32 + 4 * c, c * length
        cyc[at][0] += [f"ldfd f{t} = [r60], 8", f"ldfd f{t + 1} = [r61], 8"]
        cd = [f"ldfd f{t + 2} = [r62], 8", f"ldfd f{t + 3} = [r63], 8"]
        cyc[at + g][0] += {"": cd[:1], "cd": cd, "dc": cd[::-1]}[two]
        u = "abcd".index(k)
        cyc[(at + (g if k in "cd" else 0) + 6) % len(cyc)][1] += [
            f"fms.d f{80 + c} = f{t + u}, f1, f9"]
    out = []
    for m, f in cyc:
        m += [None] * (2 - len(m))
        out += [B("mmf", m[0], m[1], (f[0] if f else "nop.f 0") + " ;;")]
    return out


FRB = [0x14000, 0x14008, 0x14010, 0x14020, 0x14080, 8, 16]


FPI = ["0000", "1000", "1100", "0011", "1010", "1110", "1111"]


def kernels():
    from emit import pair
    ks = []
    for order in ORDERS:
        for nost in (False, True):
            ks += pair(f"fdk_{order}" + ("_nost" if nost else ""), "fdtdk",
                       body(order, nost=nost), setup(), {"order": order},
                       bodies=2, iters=30)
    for sd, ld, x in FSL:
        for split in (False, True):
            ks += pair(f"fsl_s{sd}_l{ld}_x{x}".replace("-", "m") +
                       ("_split" if split else ""),
                       "fdtdk", fsl(split),
                       ["movl r18 = 0x100000", "add r60 = r33, r18",
                        f"adds r61 = {ld}, r60", f"movl r18 = {x}",
                        "add r66 = r60, r18", f"adds r67 = {sd}, r66",
                        "movl r18 = 0x3fe0000000000000", "setf.d f9 = r18"],
                       {"sd": sd, "ld": ld, "x": x}, bodies=2, iters=60)
    for ns, nl in FSM:
        for kind in ("f", "i"):
            if kind == "i" and (nl > 2 or nl == 0):
                continue
            ks += pair(f"fsm_s{ns}_{kind}{nl}", "fdtdk", fsm(ns, nl, kind),
                       ["movl r18 = 0x100000", "add r60 = r33, r18",
                        "adds r61 = 64, r60", "adds r62 = 128, r60",
                        "adds r63 = 192, r60", "movl r18 = 4352",
                        "add r66 = r60, r18", "adds r67 = 64, r66",
                        "movl r18 = 0x3fe0000000000000", "setf.d f9 = r18"],
                       {"ns": ns, "nl": nl, "kind": kind}, bodies=2,
                       iters=60)
    for name, (S, L) in FSG.items():
        regs = [f"adds r{60 + k} = {x}, r59" for k, x in enumerate(L)]
        regs += [f"adds r{66 + k} = {x}, r59" for k, x in enumerate(S)]
        ks += pair(f"fsg_{name}", "fdtdk", fsg(S, L),
                   ["movl r18 = 0x100000", "add r59 = r33, r18"] + regs +
                   ["movl r18 = 0x3fe0000000000000", "setf.d f9 = r18"],
                   {"S": S, "L": L}, bodies=2, iters=60)
    for idle in (1, 2):
        for name in "bghjkfo":
            S, L = FSG[name]
            ks += pair(f"fsgi{idle}_{name}", "fdtdk", fsg(S, L, idle=idle),
                       ["movl r18 = 0x100000", "add r59 = r33, r18"] +
                       [f"adds r{60 + k} = {x}, r59" for k, x in enumerate(L)] +
                       [f"adds r{66 + k} = {x}, r59" for k, x in enumerate(S)] +
                       ["movl r18 = 0x3fe0000000000000", "setf.d f9 = r18"],
                       {"S": S, "L": L, "idle": idle}, bodies=2, iters=60)
    for x in FRB:
        for v in ("base", "split", "nost", "one", "u3", "u4", "u5", "u6",
                  "u7", "a3", "a4", "b3", "b4", "a3s", "b3s"):
            if (v == "one" or v[1].isdigit()) and x not in (FRB[0], 16):
                continue
            ks += pair(f"frb_{v}_x{x}", "fdtdk", frb(v),
                       ["movl r18 = 0x100000", "add r60 = r33, r18",
                        f"movl r18 = {x}", "add r61 = r60, r18",
                        "adds r66 = -80, r60",
                        "movl r18 = 0x3fe0000000000000", "setf.d f9 = r18"],
                       {"v": v, "x": x}, bodies=2, iters=60)
    for st, d, cons in FRH:
        ks += pair(f"frh_{st}_d{d}" + (f"_c{cons}" if cons else ""),
                   "fdtdk", frh(st, cons),
                   ["movl r18 = 0x100000", "add r59 = r33, r18",
                    "adds r60 = 0x278, r59", "movl r18 = 0xa270",
                    "add r61 = r59, r18", "adds r63 = 0x280, r61",
                    f"movl r18 = {0x14270 + d}", "add r62 = r59, r18",
                    "movl r18 = 0x14238", "add r66 = r59, r18",
                    "movl r18 = 0x3fe0000000000000", "setf.d f9 = r18"],
                   {"st": st, "d": d, "cons": cons}, bodies=2, iters=60)
    for x in (0x14000, 16):
        for cb, y in (("s", 0x28000), ("d", 0x28040)):
            for g in (1, 2, 3, 4, 5):
                for k in "abc":
                    ks += pair(f"frq_x{x}_{cb}_g{g}_c{k}", "fdtdk", frq(g, k),
                               ["movl r18 = 0x100000", "add r60 = r33, r18",
                                f"movl r18 = {x}", "add r61 = r60, r18",
                                f"movl r18 = {y}", "add r62 = r60, r18",
                                "movl r18 = 0x3fe0000000000000",
                                "setf.d f9 = r18"],
                               {"x": x, "cb": cb, "g": g, "k": k},
                               bodies=2, iters=60)
    for two in ("cd", "dc"):
        for g in (2, 1, 3):
            for k in "cd":
                ks += pair(f"frq_{two}_g{g}_c{k}", "fdtdk",
                           frq(g, k, two=two),
                           ["movl r18 = 0x100000", "add r60 = r33, r18",
                            "movl r18 = 0x14000", "add r61 = r60, r18",
                            "movl r18 = 0x28040", "add r62 = r60, r18",
                            "movl r18 = 0x3c0a0", "add r63 = r60, r18",
                            "movl r18 = 0x3fe0000000000000",
                            "setf.d f9 = r18"],
                           {"two": two, "g": g, "k": k}, bodies=2, iters=60)
    for incs in FPI:
        ks += pair(f"fpi_{incs}", "fdtdk", fpi(incs),
                   ["movl r18 = 0x100000", "add r60 = r33, r18",
                    "adds r61 = 64, r60", "adds r62 = 128, r60",
                    "adds r63 = 192, r60"], {"incs": incs}, bodies=2,
                   iters=500)
    return ks
