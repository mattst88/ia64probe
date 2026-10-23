# SPDX-License-Identifier: GPL-3.0-or-later
"""Correlated conditional branches: does the predictor use other branches'
outcomes?

Each body loads the next byte of a random table and branches twice on
bits of it, each branch shaped like tests_cbranch's.  The first branch
tests bit 0, which is random, so it mispredicts about half the time
whatever the predictor.  The second tests:

  one   bit 7, always clear: a branch that never mispredicts, the baseline
  ind   bit 1, random and independent of bit 0
  same  bit 0 again
  not   bit 2, the inverse of bit 0
  lag   bit 3, the previous byte's bit 0, so the outcome of the first
        branch of the body before (four branches back)

A predictor of each branch's own history alone mispredicts the second as
often as the first in all but "one"; one that keeps a history of recent
branches can predict "same", "not" and "lag" from it.
"""

from __future__ import annotations

import random

from emit import Bundle as B, Kernel, pair
from tests_cbranch import setup

KINDS = {"one": 7, "ind": 1, "same": 0, "not": 2, "lag": 3}


def branch(pred: str, then: str) -> list:
    """Branch on pred over a bundle, either way through two more, the last
    (label 2) computing the next predicate with `then`."""
    return [B("mib", None, None, f"({pred}) br.cond.dptk.few 1f ;;"),
            B("mib", None, None, "br.sptk.few 2f ;;"),
            B("mii", None, None, "nop.i 0 ;;", label="1"),
            B("mii", None, None, then, label="2")]


def body(bit: int) -> list:
    return ([B("mmi", "ld1 r20 = [r21], 1", None, "nop.i 0 ;;"),
             B("mmi", "cmp.eq p8, p9 = r21, r19", None, "nop.i 0 ;;"),
             B("mmi", "(p8) mov r21 = r18", None,
               "tbit.nz p6, p7 = r20, 0 ;;")]
            + branch("p6", f"tbit.nz p10, p11 = r20, {bit} ;;")
            + branch("p10", "nop.i 0 ;;"))


def table(n: int = 1024) -> list[int]:
    rng = random.Random(7)
    out, prev = [], 0
    for _ in range(n):
        b0, b1 = rng.randrange(2), rng.randrange(2)
        out.append(b0 | b1 << 1 | (1 - b0) << 2 | prev << 3)
        prev = b0
    return out


def multiway(mbb: bool) -> list:
    """Two branches on bits 0 and 1, never both set, in one MBB bundle (as
    coremark's state machine tests a character for ',' and '.') or in two
    bundles of their own; either way through the same number of bundles."""
    head = [B("mmi", "ld1 r20 = [r21], 1", None, "nop.i 0 ;;"),
            B("mmi", "cmp.eq p8, p9 = r21, r19", None, "nop.i 0 ;;"),
            B("mmi", "(p8) mov r21 = r18", None,
              "tbit.nz p6, p7 = r20, 0"),
            B("mii", None, "tbit.nz p10, p11 = r20, 1", "nop.i 0 ;;")]
    if mbb:
        br = [B("mbb", None, "(p6) br.cond.dpnt.few 1f",
                "(p10) br.cond.dpnt.few 3f ;;"),
              B("mib", None, None, "nop.b 0 ;;")]
    else:
        br = [B("mib", None, None, "(p6) br.cond.dpnt.few 1f ;;"),
              B("mib", None, None, "(p10) br.cond.dpnt.few 3f ;;")]
    return head + br + [
        B("mib", None, None, "br.sptk.few 2f ;;"),
        B("mib", None, None, "br.sptk.few 2f ;;", label="1"),
        B("mib", None, None, "br.sptk.few 2f ;;", label="3"),
        B("mii", None, None, "nop.i 0 ;;", label="2")]


def onehot(pct: int, n: int = 1024) -> list[int]:
    """Each byte 1 or 2 with pct% chance each, else 0."""
    rng = random.Random(pct)
    out = []
    for _ in range(n):
        r = rng.randrange(100)
        out.append(1 if r < pct else 2 if r < 2 * pct else 0)
    return out


def kernels():
    ks = []
    t = table()
    for kind, bit in KINDS.items():
        ks += pair(f"bc_{kind}", "bcorr", body(bit), setup(t),
                   meta={"second": kind})
    for pct in (0, 10, 25):
        for mbb in (True, False):
            ks += pair(f"bm_{'mbb' if mbb else 'split'}{pct}", "bcorr",
                       multiway(mbb), setup(onehot(pct)),
                       meta={"pct": pct, "mbb": int(mbb)})
    # Each of n copies of the body sees every nth outcome, so repeat each
    # outcome n times for each copy to see the sequence as coremark does.
    for mbb in (True, False):
        name = f"bm_{'mbb' if mbb else 'split'}cm"
        ks += [Kernel(name, "bcorr", multiway(mbb), n,
                      setup([o for o in coremark() for _ in range(n)]),
                      {"pct": -1, "mbb": int(mbb)}) for n in (16, 32)]
    return ks


# One period of the outcomes of coremark's core_state_transition bundle
# that tests a character for ',' (b, bit 0) and then '.' (a, bit 1).
CM = ("...b...b.a.aaaa...b...b...b..b..b..b..a..a....b....b.a..b..b..baaa"
      "a.a...b...b...b.a.aaaa...b...b...b..b..b..b..a..a....b....b.a..b.."
      "b..baaaa.a...b...b...b.a.aaaa...b...b...b..b..b..b..a..a....b....b"
      ".a..b")


def coremark() -> list[int]:
    return [{".": 0, "b": 1, "a": 2}[c] for c in CM]
