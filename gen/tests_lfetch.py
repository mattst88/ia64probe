# SPDX-License-Identifier: GPL-3.0-or-later
"""What an lfetch costs beside FP loads and stores.

icc 11.1's linpack saxpy issues an lfetch.nt1 an iteration, in the group
of its two ldfs; built with -opt-prefetch=0 it ran 19% faster on an Itanium
9340, where the model said the prefetches cost nothing.

lfp_{pos}_{kind}_{s}{d}: lpk's sp order (clang's saxpy, three cycles an
iteration, see tests_lpk) with one lfetch of stream s (x, the loads; y,
the loaded and stored line) d bytes ahead added to cycle pos (1, the y
loads' cycle; 2, the empty one), stepping 8 like the loads.  kind: nt1,
nta, plain (no hint), excl (.excl.nt1), or none.

lfi_{v}_{kind}_d{d}: icc's shape, one float an iteration in two groups:
ldfs x and ldfs y (and the lfetch), then stfs y 12 iterations back.  v:
base; nost, no store; mf, an fnorm-only group after each copy.

lfd_{v}_{kind}: lfo_1_{kind}_s0 at 0x1c0000, after an ld8 of its 16 KB page
(warm), 16 KB on (far), 64 KB on (far64), N MB on (farNm), an st8 of it (st), 1 or
3 adds (alu1, alu3), an ldfd of it (fp), or nothing (cold).

lfo_{n}_{kind}_s{st}[_i{it}]: only lfetches (or ld8s or ldfds with no use),
n a group, each pointer stepping st, 60 (or it) iterations.
"""

from emit import Bundle as B
from tests_lpk import ORDERS, PTR, setup as lpk_setup

KINDS = {"nt1": "lfetch.nt1", "nta": "lfetch.nta", "plain": "lfetch",
         "excl": "lfetch.excl.nt1"}


def lfp(pos, kind, copies=8):
    out = []
    ld = fp = 0
    for c in range(copies):
        for ci, cyc in enumerate(ORDERS["sp"]):
            for i, slots in enumerate(cyc):
                m = []
                for s in slots:
                    if s.startswith("s"):
                        m.append(f"stfs [r{PTR[s]}] = f9, 8")
                    else:
                        m.append(f"ldfs f{32 + ld} = [r{PTR[s]}], 8")
                        ld += 1
                if ci == pos and i == 0 and kind != "none":
                    m.append(f"{KINDS[kind]} [r68], 8")
                m += [None] * (2 - len(m))
                f = f"fnorm.s f{80 + fp % 40} = f10" + (" ;;" if i else "")
                fp += 1
                out.append(B("mmf", m[0], m[1], f))
    return out


def lfi(v, kind, copies=12):
    out = []
    for c in range(copies):
        lf = None if kind == "none" else f"{KINDS[kind]} [r68], 4"
        out.append(B("mmi", f"ldfs f{32 + 2 * c} = [r60], 4",
                     f"ldfs f{33 + 2 * c} = [r62], 4", "nop.i 0"))
        out.append(B("mfi", lf, "nop.f 0", "nop.i 0 ;;"))
        st = None if v == "nost" else "stfs [r66] = f9, 4"
        out.append(B("mfi", st, "nop.f 0", "nop.i 0 ;;"))
        if v == "mf":
            out.append(B("mfi", None, f"fnorm.s f{80 + c} = f10", "nop.i 0 ;;"))
    return out


def lfo(n, kind, st, copies=16):
    out = []
    for c in range(copies):
        if kind == "ld8":
            m = [f"ld8 r{40 + k} = [r{68 + k}], {st}" for k in range(n)]
        elif kind == "ldfd":
            m = [f"ldfd f{40 + k} = [r{68 + k}], {st}" for k in range(n)]
        else:
            m = [f"{KINDS[kind]} [r{68 + k}], {st}" for k in range(n)]
        m += [None] * (2 - len(m))
        out.append(B("mmi", m[0], m[1], "nop.i 0 ;;"))
    return out


def kernels():
    from emit import pair
    ks = []
    for pos in (1, 2):
        for kind in ("none", "nt1", "nta", "plain", "excl"):
            for s, d in (("x", 0), ("x", 128), ("x", 512), ("x", 2048),
                         ("y", 0), ("y", 512)):
                if kind == "none" and (s, d) != ("x", 0):
                    continue
                if kind in ("nta", "plain", "excl") and d not in (0, 512):
                    continue
                base = "r60" if s == "x" else "r62"
                ks += pair(f"lfp_{pos}_{kind}_{s}{d}", "lfetch",
                           lfp(pos, kind),
                           lpk_setup(32, 4) + [f"adds r68 = {d}, {base}"],
                           {"pos": pos, "kind": kind, "s": s, "d": d},
                           bodies=2, iters=60)
    for v in ("base", "nost", "mf"):
        for kind in ("none", "nt1", "plain"):
            for d in (0, 512):
                if kind == "none" and d:
                    continue
                ks += pair(f"lfi_{v}_{kind}_d{d}", "lfetch", lfi(v, kind),
                           lpk_setup(32, 4, 4) + [f"adds r68 = {d}, r60",
                                                  "adds r66 = -48, r62"],
                           {"v": v, "kind": kind, "d": d},
                           bodies=2, iters=60)
    for n in (1, 2):
        for kind in ("nt1", "plain", "ld8", "ldfd"):
            for st in (0, 8, 64, 128):
                for it in (60, 2, 8):
                    if it != 60 and (st < 64 or n != 1):
                        continue
                    if kind in ("ld8", "ldfd") and st < 64:
                        continue
                    sfx = "" if it == 60 else f"_i{it}"
                    ks += pair(f"lfo_{n}_{kind}_s{st}{sfx}", "lfetch",
                               lfo(n, kind, st),
                               ["movl r18 = 0x100000", "add r68 = r33, r18",
                                "movl r18 = 0x110040", "add r69 = r33, r18"],
                               {"n": n, "kind": kind, "st": st, "it": it},
                               bodies=2, iters=it)
    # lfd_{v}_{kind}: lfetches of one line, after an ld8 of their 16 KB
    # page (warm), of the next page (far), an ldfd of it (fp), or nothing.
    pre = {"cold": [], "warm": ["ld8 r40 = [r68]"],
           "far": ["movl r18 = 0x4000", "add r18 = r68, r18",
                   "ld8 r40 = [r18]"],
           "far64": ["movl r18 = 0x10000", "add r18 = r68, r18",
                     "ld8 r40 = [r18]"],
           "fp": ["ldfd f40 = [r68]"]}
    pre["alu1"] = ["add r40 = r0, r0"]
    pre["alu3"] = ["add r40 = r0, r0", "add r41 = r0, r0", "add r42 = r0, r0"]
    pre["st"] = ["st8 [r68] = r0"]
    for d in (0x100000, 0x1000000, 0x4000000, 0xc000000):
        pre[f"far{d >> 20}m"] = [f"movl r18 = {d:#x}", "add r18 = r68, r18",
                                 "ld8 r40 = [r18]"]
    for v, ins in pre.items():
        for kind in ("nt1", "plain"):
            ks += pair(f"lfd_{v}_{kind}", "lfetch", lfo(1, kind, 0),
                       ["movl r18 = 0x1c0000", "add r68 = r33, r18"] + ins,
                       {"v": v, "kind": kind}, bodies=2, iters=60)
    return ks
