# SPDX-License-Identifier: GPL-3.0-or-later
"""Instruction fetch: straight-line code and taken branches over footprints
from inside L1I (16 KB) out past L2I.

Each kernel is a pair at one and two copies of a body of n bundles, so the
two members' footprints are n and 2n bundles.  Read each member on its own:
its cycles over iters x bodies x n is the cost of a bundle at that
footprint.  The differential mixes two footprints and means little here.
"""

from __future__ import annotations

import random

from emit import Bundle as B, Kernel

SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def iters_for(n):
    return max(8, (1 << 21) // n)


def seq1(n):
    """One bundle a group: 16 bytes a cycle."""
    return [B("mii", "adds r20 = 1, r0", "adds r21 = 1, r0",
              "adds r22 = 1, r0 ;;") for _ in range(n)]


def seq2(n):
    """Two bundles a group: 32 bytes a cycle, the most fetch can give."""
    out = []
    for _ in range(n // 2):
        out += [B("mmi", "adds r20 = 1, r0", "adds r21 = 1, r0",
                  "adds r22 = 1, r0"),
                B("mmi", "adds r23 = 1, r0", "adds r24 = 1, r0",
                  "adds r25 = 1, r0 ;;")]
    return out


def hops(n, shuffle, seed=1):
    """Lines of four one-group bundles, each line ending in a taken branch
    to the next line of a walk.  The walk starts at the body's first line
    and leaves through its last, which falls through.  shuffle puts the
    lines between in random order; otherwise each branch goes to the line
    right after it."""
    lines = n // 4
    mid = list(range(1, lines - 1))
    if shuffle:
        random.Random(seed).shuffle(mid)
    walk = [0] + mid + [lines - 1]
    nxt = {walk[i]: walk[i + 1] for i in range(len(walk) - 1)}
    out = []
    for p in range(lines):
        lab = str(10000 + p)
        out.append(B("mii", "adds r20 = 1, r0", "adds r21 = 1, r0",
                     "adds r22 = 1, r0 ;;", label=lab))
        out += [B("mii", "adds r20 = 1, r0", "adds r21 = 1, r0",
                  "adds r22 = 1, r0 ;;") for _ in range(2)]
        if p in nxt:
            t = nxt[p]
            ref = f"{10000 + t}{'f' if t > p else 'b'}"
            out.append(B("mib", "adds r20 = 1, r0", "adds r21 = 1, r0",
                         f"br.sptk.few {ref} ;;"))
        else:
            out.append(B("mib", "adds r20 = 1, r0", "adds r21 = 1, r0",
                         "nop.b 0 ;;"))
    return out


def ipair(name, body, n):
    it = iters_for(n)
    meta = {"bundles": n}
    return [Kernel(name, "icache", body, 1, meta=meta, iters=it),
            Kernel(name, "icache", body, 2, meta=meta, iters=it)]


def kernels():
    ks = []
    for n in SIZES:
        ks += ipair(f"ic_seq1_n{n}", seq1(n), n)
        ks += ipair(f"ic_seq2_n{n}", seq2(n), n)
        ks += ipair(f"ic_hopseq_n{n}", hops(n, False), n)
        ks += ipair(f"ic_hoprnd_n{n}", hops(n, True), n)
    return ks
