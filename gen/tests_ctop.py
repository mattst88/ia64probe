# SPDX-License-Identifier: GPL-3.0-or-later
"""br.ctop after an F-unit predicate write in its own group.

Each body runs a counted loop of N passes, one group a pass, closed by
br.ctop (or br.cloop as a control).  The group's F-unit instruction either
writes no predicate (fma), or writes one with frcpa or fcmp: a rotating
predicate, which br.ctop renames as the instruction writes it, or a static
one.  "_stop" variants put the write a group before the branch.

The other kernels look for when the loop predictor calls the exit:
ctops_ loops of `groups` groups a pass and n passes, ctope{0,3}_ the same
with ar.ec 0 or 3, ctopn_ with the mov ar.lc `gap` groups ahead of the
loop, and ctopm_ around the 10-cycle edge.
"""

from __future__ import annotations

from emit import Bundle as B, pair

N = 32

F = {"fma": "fma f32 = f6, f6, f0",
     "frcpa_rot": "frcpa f32, p17 = f1, f6",
     "frcpa_static": "frcpa f32, p6 = f1, f6",
     "fcmp_rot": "fcmp.eq p17, p18 = f6, f6",
     "fcmp_static": "fcmp.eq p6, p7 = f6, f6"}


def body(f, br, stop):
    out = [B("mii", None, "mov r62 = ar.lc", f"adds r22 = {N - 1}, r0 ;;"),
           B("mii", None, "mov.i ar.lc = r22", "mov.i ar.ec = 1 ;;"),
           B("mii", None, "mov pr.rot = 0x10000", "nop.i 0 ;;")]
    if stop:
        out += [B("mfi", None, f, "nop.i 0 ;;", label="1"),
                B("mib", None, None, f"{br}.sptk.few 1b ;;")]
    else:
        out += [B("mfb", None, f, f"{br}.sptk.few 1b ;;", label="1")]
    out += [B("mii", None, "mov.i ar.lc = r62", "nop.i 0 ;;")]
    return out


def sweep(br, groups, n, ec=1):
    out = [B("mii", None, "mov r62 = ar.lc", f"adds r22 = {n - 1}, r0 ;;"),
           B("mii", None, "mov.i ar.lc = r22", f"mov.i ar.ec = {ec} ;;"),
           B("mii", None, "mov pr.rot = 0x10000", "nop.i 0 ;;")]
    for g in range(groups - 1):
        out += [B("mfi", None, "fma f32 = f6, f6, f0", "nop.i 0 ;;",
                  label="1" if g == 0 else None)]
    out += [B("mfb", None, "fma f33 = f6, f6, f0", f"{br}.sptk.few 1b ;;",
              label="1" if groups == 1 else None)]
    out += [B("mii", None, "mov.i ar.lc = r62", "nop.i 0 ;;")]
    return out


def near(br, groups, n, ec, gap):
    """ar.lc written gap groups before the loop's first group."""
    out = [B("mii", None, "mov r62 = ar.lc", f"adds r22 = {n - 1}, r0"),
           B("mii", None, f"mov.i ar.ec = {ec}", "nop.i 0 ;;"),
           B("mii", None, "mov pr.rot = 0x10000", "nop.i 0 ;;"),
           B("mii", None, "mov.i ar.lc = r22", "nop.i 0 ;;")]
    out += [B("mii", None, None, "nop.i 0 ;;")] * gap
    for g in range(groups - 1):
        out += [B("mfi", None, "fma f32 = f6, f6, f0", "nop.i 0 ;;",
                  label="1" if g == 0 else None)]
    out += [B("mfb", None, "fma f33 = f6, f6, f0", f"{br}.sptk.few 1b ;;",
              label="1" if groups == 1 else None)]
    out += [B("mii", None, "mov.i ar.lc = r62", "nop.i 0 ;;")]
    return out


def late(br, groups, n, ec, src):
    """ar.lc from a value that arrives late: ready (adds), through
    setf/getf (getf), or through setf/xmpy/getf (xmpy), as fread's
    item-size multiply feeds memcpy's byte loop."""
    out = [B("mii", None, "mov r62 = ar.lc", f"adds r22 = {n}, r0"),
           B("mii", None, f"mov.i ar.ec = {ec}", "nop.i 0 ;;"),
           B("mii", None, "mov pr.rot = 0x10000", "nop.i 0 ;;")]
    if src == "getf":
        out += [B("mmi", "setf.sig f7 = r22", None, "nop.i 0 ;;"),
                B("mmi", "getf.sig r22 = f7", None, "nop.i 0 ;;")]
    elif src == "xmpy":
        out += [B("mii", None, "adds r23 = 1, r0", "nop.i 0 ;;"),
                B("mmi", "setf.sig f7 = r22", "setf.sig f8 = r23",
                  "nop.i 0 ;;"),
                B("mmf", None, None, "xmpy.l f7 = f7, f8 ;;"),
                B("mmi", "getf.sig r22 = f7", None, "nop.i 0 ;;")]
    out += [B("mii", None, "adds r22 = -1, r22", "nop.i 0 ;;"),
            B("mii", None, "mov.i ar.lc = r22", "nop.i 0 ;;")]
    for g in range(groups - 1):
        out += [B("mfi", None, "fma f32 = f6, f6, f0", "nop.i 0 ;;",
                  label="1" if g == 0 else None)]
    out += [B("mfb", None, "fma f33 = f6, f6, f0", f"{br}.sptk.few 1b ;;",
              label="1" if groups == 1 else None)]
    out += [B("mii", None, "mov.i ar.lc = r62", "nop.i 0 ;;")]
    return out


def daxpy(n, ec, four=False):
    out = [B("mii", None, "mov r62 = ar.lc", f"adds r22 = {n - 1}, r0 ;;"),
           B("mii", None, "mov.i ar.lc = r22", f"mov.i ar.ec = {ec} ;;"),
           B("mii", None, "mov pr.rot = 0x10000", "nop.i 0 ;;"),
           B("mfi", None, "fma f32 = f6, f6, f0", "nop.i 0 ;;", label="1")]
    if four:
        out[-1] = B("mfi", None, "fma f32 = f6, f6, f0", None, label="1")
        out += [B("mfi", None, "fma f34 = f6, f6, f0", "nop.i 0 ;;")]
    out += [B("mmf", None, None, "fma f33 = f6, f6, f0"),
           B("mbb", None, "nop.b 0", "br.ctop.sptk.few 1b ;;"),
           B("mii", None, "mov.i ar.lc = r62", "nop.i 0 ;;")]
    return out


def daxpy_call(n, ec, alloc, four=False):
    """daxpy() in a callee, entered by br.call through b6, whose frame is
    the four outputs of the setup's alloc."""
    loop = daxpy(n, ec, four)
    loop[0] = B("mii", "alloc r34 = ar.pfs, 0, 4, 0, 0" if alloc else None,
                "mov r32 = ar.lc" if not alloc else "mov r35 = ar.lc",
                f"adds r22 = {n - 1}, r0 ;;", label="2")
    loop[-1] = B("mii", None,
                 "mov.i ar.lc = r35" if alloc else "mov.i ar.lc = r32",
                 "nop.i 0 ;;")
    pad = [B("mii", None, None, "nop.i 0 ;;")]
    if four:
        # The loop's first bundle starts a pair, as clang aligns it.
        loop = loop[:3] + pad + loop[3:]
    out = ([B("mib", None, None, "br.call.sptk.many b6 = 2f ;;"),
            B("mib", None, None, "br.cond.sptk.few 3f ;;")] + loop +
           [B("mib", None, None, "br.ret.sptk.many b6 ;;"),
            B("mmi", None, None, "nop.i 0 ;;", label="3")])
    return out + pad * (len(out) % 2)


def kernels():
    ks = []
    for br in ("br.ctop", "br.cloop"):
        for groups in (1, 2, 3, 6):
            for n in (8, 32, 128):
                ks += pair(f"ctops_{br[3:]}_g{groups}_n{n}", "ctop",
                           sweep(br, groups, n), ["mov f6 = f1"],
                           meta={"br": br, "groups": groups, "n": n},
                           bodies=2)
    for br in ("br.ctop", "br.cloop"):
        for ec in (0, 3):
            for groups in (1, 2, 3):
                for n in (8, 32):
                    ks += pair(f"ctope{ec}_{br[3:]}_g{groups}_n{n}", "ctop",
                               sweep(br, groups, n, ec), ["mov f6 = f1"],
                               meta={"br": br, "groups": groups, "n": n,
                                     "ec": ec}, bodies=2)
    for name, f in F.items():
        for br in ("br.ctop", "br.cloop"):
            for stop in (False, True):
                k = f"ctop_{name}_{br[3:]}{'_stop' if stop else ''}"
                ks += pair(k, "ctop", body(f, br, stop), ["mov f6 = f1"],
                           meta={"f": name, "br": br, "stop": stop, "n": N},
                           bodies=4)
    for br, ec in (("br.ctop", 1), ("br.cloop", 0)):
        for groups in (2, 3):
            for n in (2, 3, 4, 8, 32):
                for gap in (0, 1, 3):
                    ks += pair(f"ctopn_{br[3:]}_g{groups}_n{n}_gap{gap}",
                               "ctop", near(br, groups, n, ec, gap),
                               ["mov f6 = f1"],
                               meta={"br": br, "groups": groups, "n": n,
                                     "ec": ec, "gap": gap}, bodies=2)
    for br, ec in (("br.ctop", 1), ("br.cloop", 0)):
        for groups in (2, 6, 12):
            for n in (3, 4, 5, 6, 7):
                ks += pair(f"ctopm_{br[3:]}_g{groups}_n{n}", "ctop",
                           near(br, groups, n, ec, 1), ["mov f6 = f1"],
                           meta={"br": br, "groups": groups, "n": n,
                                 "ec": ec}, bodies=2)
    for br, ec in (("br.ctop", 1), ("br.cloop", 0)):
        for groups in (2, 3):
            for n in (4, 16):
                for src in ("adds", "getf", "xmpy"):
                    ks += pair(f"ctopl_{br[3:]}_g{groups}_n{n}_{src}",
                               "ctop", late(br, groups, n, ec, src),
                               ["mov f6 = f1"],
                               meta={"br": br, "groups": groups, "n": n,
                                     "ec": ec, "src": src}, bodies=2)
    # The same loop in a callee, with and without an alloc.
    for alloc in (0, 1):
        for n in range(2, 26):
            ks += pair(f"ctopc_a{alloc}_n{n}", "ctop", daxpy_call(n, 10, alloc),
                       ["alloc r2 = ar.pfs, 2, 60, 4, 0", "mov f6 = f1"],
                       meta={"br": "br.ctop", "groups": 2, "n": n, "ec": 10,
                             "alloc": alloc}, bodies=2)
    # Its four bundles, as clang lays them out, called (with ar.ec 10 or 1)
    # or inline.
    for n in range(2, 33):
        for ec in (10, 1):
            ks += pair(f"ctopc4{'' if ec == 10 else '_e1'}_n{n}", "ctop",
                       daxpy_call(n, ec, 1, True),
                       ["alloc r2 = ar.pfs, 2, 60, 4, 0", "mov f6 = f1"],
                       meta={"br": "br.ctop", "groups": 2, "n": n, "ec": ec},
                       bodies=2)
        loop = daxpy(n, 10, True)
        pad = [B("mii", None, None, "nop.i 0 ;;")]
        ks += pair(f"ctopi4_n{n}", "ctop", loop[:3] + pad + loop[3:] + pad,
                   ["mov f6 = f1"],
                   meta={"br": "br.ctop", "groups": 2, "n": n, "ec": 10},
                   bodies=2)
    # linpack's daxpy as clang pipelines it: three bundles in two groups.
    for ec in (0, 10):
        for n in range(2, 26):
            ks += pair(f"ctopd_e{ec}_n{n}", "ctop", daxpy(n, ec),
                       ["mov f6 = f1"],
                       meta={"br": "br.ctop", "groups": 2, "n": n, "ec": ec},
                       bodies=2)
    # linpack's daxpy: a 2-group kernel with ar.ec 10, short trip counts.
    for ec in (1, 2, 4, 6, 10):
        for n in (2, 4, 6, 8, 12, 16, 32, 64):
            ks += pair(f"ctopk_g2_e{ec}_n{n}", "ctop",
                       sweep("br.ctop", 2, n, ec), ["mov f6 = f1"],
                       meta={"br": "br.ctop", "groups": 2, "n": n, "ec": ec},
                       bodies=2)
    return ks
