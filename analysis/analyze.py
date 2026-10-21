#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Differential table: documented vs hardware vs the ia64perf model.

cycles per body = (t(2B) - t(B)) * ratio / (iters * B)

where t is the minimum over repetitions and ratio converts ar.itc ticks to
core cycles (1 for the model, whose log is in cycles already).
"""

from __future__ import annotations

import argparse
import json


def load(path: str) -> dict[str, float]:
    out = {}
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        return out
    for l in lines:
        if not l.startswith("{"):
            continue
        r = json.loads(l)
        d = (r["min"][1] - r["min"][0]) * r["ratio"]
        out[r["name"]] = d / (r["iters"] * r["bodies"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", default="results/hw.jsonl")
    ap.add_argument("--sim", default="results/sim.jsonl")
    ap.add_argument("--meta", default="build/kernels.json")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="relative tolerance for agreement")
    ap.add_argument("--diff", action="store_true",
                    help="only kernels where the model and hardware differ")
    ap.add_argument("prefix", nargs="*")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    hw, sim = load(args.hw), load(args.sim)
    print(f"{'kernel':34} {'claim':>7} {'hw':>8} {'model':>8}  verdict")
    for name, m in meta.items():
        if args.prefix and not any(name.startswith(p) for p in args.prefix):
            continue
        exp = m.get("expect_cyc_per_body")
        h, s = hw.get(name), sim.get(name)

        def fmt(x):
            return f"{x:8.3f}" if x is not None else f"{'-':>8}"

        def agree(a, b):
            return a is not None and b is not None and \
                abs(a - b) <= args.tol * max(abs(b), 0.25)

        v = []
        if exp is not None and h is not None:
            v.append("doc ok" if agree(h, exp) else "DOC DIFFERS")
        if s is not None and h is not None:
            v.append("model ok" if agree(s, h) else "MODEL DIFFERS")
        if args.diff and "MODEL DIFFERS" not in v:
            continue
        e = f"{exp:7.2f}" if exp is not None else f"{'-':>7}"
        print(f"{name:34} {e} {fmt(h)} {fmt(s)}  {', '.join(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
