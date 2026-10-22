# SPDX-License-Identifier: GPL-3.0-or-later
"""Conditional branches: what the predictor learns, and what a miss costs.

Each body loads the next byte of a table of outcomes, branches on it, and
either way runs two more bundles.  The bodies share one pointer, so with N
copies of the body each static branch sees every Nth outcome.  A table of
period p with a single not-taken outcome (the exit of a loop of p - 1
iterations) looks the same to every copy when p is odd, so those are the
patterns ("lp<p>"); "rnd<k>" tables are random with k% not taken, and "t"
and "nt" never change.  _sptk/_dpnt/_spnt change the hint from dptk; cbs_
computes the predicate in the branch's own group.
"""

from __future__ import annotations

import random

from emit import Bundle as B, pair

TABLE = 0xE0000


def body(hint, same=False):
    return [B("mmi", "ld1 r20 = [r21], 1", None, "nop.i 0 ;;"),
            B("mmi", "cmp.eq p8, p9 = r21, r19", None, "nop.i 0 ;;"),
            B("mmi", "(p8) mov r21 = r18", None,
              "nop.i 0 ;;" if same else "cmp.ne p6, p7 = r0, r20 ;;"),
            B("mib", "cmp.ne p6, p7 = r0, r20" if same else None, None,
              f"(p6) br.cond.{hint}.few 1f ;;"),
            B("mib", None, None, "br.sptk.few 2f ;;"),
            B("mii", None, None, "nop.i 0 ;;", label="1"),
            B("mii", None, None, "nop.i 0 ;;", label="2")]


def setup(outcomes):
    out = [f"movl r18 = {TABLE:#x}", "add r18 = r33, r18", "mov r25 = r18"]
    data = bytes(outcomes) + bytes(-len(outcomes) % 8)
    for i in range(0, len(data), 8):
        out += [f"movl r24 = {int.from_bytes(data[i:i + 8], 'little'):#x}",
                "st8 [r25] = r24, 8"]
    out += [f"adds r19 = {len(outcomes)}, r18", "mov r21 = r18"]
    return out


def loop(p, n=512):
    """p - 1 taken outcomes and one not taken, repeated to about n."""
    return ([1] * (p - 1) + [0]) * max(1, n // p)


def rnd(k, n=1024):
    rng = random.Random(k)
    return [0 if rng.randrange(100) < k else 1 for _ in range(n)]


def kernels():
    ks = []
    tables = {"t": [1] * 64, "nt": [0] * 64}
    for p in (3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 33):
        tables[f"lp{p}"] = loop(p)
    for k in (10, 25, 50):
        tables[f"rnd{k}"] = rnd(k)
    for name, t in tables.items():
        ks += pair(f"cb_{name}", "cbranch", body("dptk"), setup(t),
                   meta={"pattern": name, "taken": sum(t), "n": len(t)})
    for hint in ("sptk", "dpnt", "spnt"):
        for name in ("t", "nt", "lp5", "rnd50"):
            ks += pair(f"cb_{name}_{hint}", "cbranch", body(hint),
                       setup(tables[name]),
                       meta={"pattern": name, "hint": hint})
    # The compare in the branch's group, as glibc's fwrite tests the
    # result of its _IO_acquire_lock (cmp4.eq and .spnt branch in one MIB).
    for hint in ("dptk", "sptk", "dpnt", "spnt"):
        for name in ("t", "nt", "lp5"):
            ks += pair(f"cbs_{name}_{hint}", "cbranch", body(hint, True),
                       setup(tables[name]),
                       meta={"pattern": name, "hint": hint, "same": 1})
    return ks
