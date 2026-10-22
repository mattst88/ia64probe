# SPDX-License-Identifier: GPL-3.0-or-later
"""Counted loops by the alignment of their first bundle.

Each body runs a loop of n passes, each pass `groups` groups of `bpg`
bundles (mfi nops, so two issue together, the last bundle mib with the
br.cloop or br.ctop), with the
loop's first bundle at an even ("a", 32-byte aligned) or odd ("o") bundle.
The front end reads 32-byte bundle pairs, so a pass touches one pair more
when its first bundle is odd or its bundle count is odd, and a loop that
issues two bundles a cycle may be held to a pair a cycle.  Every body has
the same bundles either way: a nop bundle goes before or after the loop.
"""

from __future__ import annotations

from emit import Bundle as B, pair

N = 64


def body(br, ec, groups, bpg, odd):
    pre = [B("mii", None, "mov r62 = ar.lc", f"adds r22 = {N - 1}, r0 ;;"),
           B("mii", None, "mov.i ar.lc = r22", f"mov.i ar.ec = {ec} ;;"),
           B("mii", None, "mov pr.rot = 0x10000", "nop.i 0 ;;")]
    pad = [B("mii", None, None, "nop.i 0 ;;")]
    loop = []
    for g in range(groups):
        for k in range(bpg):
            last = k == bpg - 1
            if last and g == groups - 1:
                loop.append(B("mib", None, None, f"{br}.sptk.few 1b ;;"))
            else:
                loop.append(B("mfi", None, None,
                              "nop.i 0 ;;" if last else "nop.i 0"))
    loop[0].label = "1"
    post = [B("mii", None, "mov.i ar.lc = r62", "nop.i 0 ;;")]
    # The loop starts at bundle len(pre) (+ 1 with the pad first) of a
    # body that starts on a pair boundary.
    first = pad if (len(pre) % 2 == 0) == odd else []
    out = pre + first + loop + post + (pad if not first else [])
    if len(out) % 2:
        out += pad
    return out


def kernels():
    ks = []
    for br, ec in (("br.cloop", 0), ("br.ctop", 1)):
        for groups in (1, 2, 3, 4, 6):
            for bpg in (1, 2, 3):
                for odd in (False, True):
                    name = (f"la_{br[3:]}_g{groups}_b{bpg}_"
                            f"{'o' if odd else 'a'}")
                    ks += pair(name, "lalign",
                               body(br, ec, groups, bpg, odd), [],
                               meta={"br": br, "groups": groups, "bpg": bpg,
                                     "odd": odd, "n": N}, bodies=2)
    return ks
