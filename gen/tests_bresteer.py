# SPDX-License-Identifier: GPL-3.0-or-later
"""What follows a mispredicted branch: does a second, well-predicted branch
soon after the resteer cost anything?

Each body branches on bit 0 of the next byte of a random table, as in
tests_bcorr's "one", then runs `gap` empty bundles and the tail:

  br    a branch on bit 7 (never set), dptk, over a bundle
  spnt  the same branch hinted spnt
  none  no branch, the same bundles

All tails take as many bundles either way.
"""

from __future__ import annotations

from emit import Bundle as B, pair
from tests_bcorr import table
from tests_cbranch import setup


def body(tail: str, gap: int, pad: int = 0) -> list:
    out = [B("mmi", "ld1 r20 = [r21], 1", None, "nop.i 0 ;;"),
           B("mmi", "cmp.eq p8, p9 = r21, r19", None, "nop.i 0 ;;"),
           B("mmi", "(p8) mov r21 = r18", None, "tbit.nz p6, p7 = r20, 0 ;;"),
           B("mib", None, None, "(p6) br.cond.dptk.few 1f ;;"),
           B("mib", None, None, "br.sptk.few 2f ;;"),
           B("mii", None, None, "nop.i 0 ;;", label="1"),
           B("mii", None, None, "tbit.nz p10, p11 = r20, 7 ;;", label="2")]
    out += [B("mii", None, None, "nop.i 0 ;;")] * gap
    hint = {"br": "dptk", "spnt": "spnt"}.get(tail)
    if hint:
        out += [B("mib", None, None, f"(p10) br.cond.{hint}.few 3f ;;"),
                B("mib", None, None, "br.sptk.few 4f ;;"),
                B("mii", None, None, "nop.i 0 ;;", label="3")]
    else:
        out += [B("mii", None, None, "nop.i 0 ;;")] * 3
    return (out + [B("mii", None, None, "nop.i 0 ;;", label="4")]
            + [B("mii", None, None, "nop.i 0 ;;")] * pad)


def kernels():
    ks = []
    t = table()
    for tail in ("br", "spnt", "none"):
        for gap in (0, 2, 4, 8):
            ks += pair(f"br_{tail}{gap}", "bresteer", body(tail, gap),
                       setup(t), meta={"tail": tail, "gap": gap})
    for gap in (0, 1):
        for pad in (1, 2, 3):
            ks += pair(f"br_br{gap}p{pad}", "bresteer", body("br", gap, pad),
                       setup(t), meta={"tail": "br", "gap": gap, "pad": pad})
    ks += pair("br_br1", "bresteer", body("br", 1), setup(t),
               meta={"tail": "br", "gap": 1})
    return ks
