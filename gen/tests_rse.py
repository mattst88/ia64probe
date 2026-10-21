# SPDX-License-Identifier: GPL-3.0-or-later
"""The register stack engine: what a call that needs stacked registers costs.

The harness's frame is 96 stacked registers (alloc 2, 94, 0, 0), all of
the Itanium 2's physical ones, so a callee that allocates N locals makes
the RSE spill N of the caller's registers before it runs, and the return
makes it fill them again (251110-003 5.7: lazy mode only, at most two
loads or two stores a cycle).  Each body calls a leaf function placed
after a branch around it: rse_call_nN allocates N, rse_call_n0 measures
the call and return alone.
"""

from emit import Bundle as B, pair


def call_leaf(n, work=()):
    return [B("mib", None, None, "br.call.sptk.many b6 = 2f ;;"),
            B("mib", None, None, "br.cond.sptk.few 3f ;;"),
            B("mmi", f"alloc r2 = ar.pfs, 0, {n}, 0, 0 ;;", None, "nop.i 0",
              label="2"),
            *work,
            B("mib", None, "mov.i ar.pfs = r2", "br.ret.sptk.many b6 ;;"),
            B("mmi", None, None, "nop.i 0 ;;", label="3")]


# A leaf of M locals called from one of N: the harness's frame spills N + M
# registers in two steps, and only the return to it fills them.
def call_two(n, m):
    return [B("mib", None, None, "br.call.sptk.many b6 = 2f ;;"),
            B("mib", None, None, "br.cond.sptk.few 3f ;;"),
            B("mmi", f"alloc r32 = ar.pfs, 0, {n}, 0, 0 ;;", None, "nop.i 0",
              label="2"),
            B("mib", None, None, "br.call.sptk.many b7 = 4f ;;"),
            B("mib", None, "mov.i ar.pfs = r32", "br.ret.sptk.many b6 ;;"),
            B("mmi", f"alloc r32 = ar.pfs, 0, {m}, 0, 0 ;;", None, "nop.i 0",
              label="4"),
            B("mib", None, "mov.i ar.pfs = r32", "br.ret.sptk.many b7 ;;"),
            B("mmi", None, None, "nop.i 0 ;;", label="3")]


# K byte stores to one word ahead of the call: do spills queue behind them?
def stores(k):
    return [B("mmi", "st1 [r36] = r40", None, "nop.i 0 ;;")] * k


def kernels():
    ks = []
    for n in (0, 1, 2, 4, 8, 16, 32, 48, 64, 90):
        ks += pair(f"rse_call_n{n}", "rse", call_leaf(n), bodies=4)
    for n, m in ((8, 8), (16, 16), (40, 40), (8, 1), (1, 8), (48, 1)):
        ks += pair(f"rse_two_n{n}_m{m}", "rse", call_two(n, m), bodies=4)
    for k in (4, 8):
        setup = ["mov r40 = 0", "adds r36 = 72, r33"]
        ks += pair(f"rse_st{k}", "rse", stores(k), setup, bodies=4)
        for n in (0, 8, 32):
            ks += pair(f"rse_st{k}_call_n{n}", "rse", stores(k) + call_leaf(n),
                       setup, bodies=4)
    return ks
