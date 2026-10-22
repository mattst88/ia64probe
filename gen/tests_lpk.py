# SPDX-License-Identifier: GPL-3.0-or-later
"""linpack's saxpy, as clang pipelines it in dgefa.

y[i] += a * x[i], two floats an iteration: ldfs of x[i] and x[i+1]
through two pointers 4 bytes apart, of y[i] and y[i+1] likewise, and two
stfs of y 48 bytes (six iterations) behind its loads, each pointer
stepping 8, in three cycles.  Moving the two y loads up beside the x
loads, and the stores down a cycle (lpk_fl4), made linpack 4.9% slower on
an Itanium 9340, where the model said 1% faster.

lpk_{order}_x{d}: x d bytes after y (mod 64 KB); lpk_{order}_y{o}_x{d}:
y o bytes past 16-byte alignment; lpkd_{order}_x{d}: in doubles, as
gemm's saxpy, x d bytes after it; every iteration
of a body writes targets of its own.  _nost: no stores; _nold: no y
loads.
"""

from emit import Bundle as B

PTR = {"xa": 60, "xb": 61, "ya": 62, "yb": 63, "sa": 66, "sb": 67}

# Each cycle's two bundles' memory slots.
ORDERS = {
    "sp": [[["xa", "xb"], ["sa", "sb"]], [["ya"], ["yb"]], [[], []]],
    "fl4": [[["xa", "xb"], ["ya", "yb"]], [["sa", "sb"], []], [[], []]],
    "stfirst": [[["sa", "sb"], ["xa", "xb"]], [["ya"], ["yb"]], [[], []]],
    "yfirst": [[["ya", "yb"], ["sa", "sb"]], [["xa"], ["xb"]], [[], []]],
}


def body(order, copies=8, nost=False, nold=False, wide=False):
    sz, step = ("d", 16) if wide else ("s", 8)
    out = []
    ld = fp = 0
    for c in range(copies):
        for cyc in ORDERS[order]:
            for i, slots in enumerate(cyc):
                m = []
                for s in slots:
                    if s.startswith("s"):
                        if not nost:
                            m.append(f"stf{sz} [r{PTR[s]}] = f9, {step}")
                    elif not (nold and s.startswith("y")):
                        m.append(f"ldf{sz} f{32 + ld} = [r{PTR[s]}], {step}")
                        ld += 1
                m += [None] * (2 - len(m))
                f = f"fnorm.s f{80 + fp % 40} = f10" + (" ;;" if i else "")
                fp += 1
                out.append(B("mmf", m[0], m[1], f))
    return out


def setup(xd, yo=0, w=4):
    return [f"movl r18 = {0x100000 + yo}", "add r62 = r33, r18",
            f"adds r63 = {w}, r62", f"adds r66 = {-12 * w}, r62",
            f"adds r67 = {-11 * w}, r62", f"movl r18 = {0x10000 + xd}",
            "add r60 = r62, r18", f"adds r61 = {w}, r60",
            "movl r18 = 0x3f800000", "setf.s f9 = r18",
            "mov f10 = f1"]


def kernels():
    from emit import pair
    ks = []
    for order in ORDERS:
        for yo in (4, 8, 12):
            for xd in (32, 804, 1608):
                ks += pair(f"lpk_{order}_y{yo}_x{xd}", "lpk",
                           body(order), setup(xd, yo),
                           {"order": order, "xd": xd, "yo": yo},
                           bodies=2, iters=60)
            for sfx, kw in (("_nost", {"nost": True}),
                            ("_nold", {"nold": True})):
                ks += pair(f"lpk_{order}_y{yo}_x32{sfx}", "lpk",
                           body(order, **kw), setup(32, yo),
                           {"order": order, "xd": 32, "yo": yo},
                           bodies=2, iters=60)
        for xd in (32, 64, 128):
            ks += pair(f"lpkd_{order}_x{xd}", "lpk", body(order, wide=True),
                       setup(xd, 0, 8), {"order": order, "xd": xd},
                       bodies=2, iters=60)
        for xd in range(0, 256, 32):
            for sfx, kw in (("", {}), ("_nost", {"nost": True}),
                            ("_nold", {"nold": True})):
                if sfx and xd:
                    continue
                ks += pair(f"lpk_{order}_x{xd}{sfx}", "lpk",
                           body(order, **kw), setup(xd),
                           {"order": order, "xd": xd}, bodies=2, iters=60)
    return ks
