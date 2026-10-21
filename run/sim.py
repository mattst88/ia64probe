#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run build/probe under qemu-ia64 with the ia64perf cycle model.

The plugin logs the modeled cycle at each ar.itc read; every kernel call
reads it twice, plus any reads its body makes (itc_reads_per_body in the
metadata), and the probe prints which kernel each call was, so the log
pairs up with the calls in order.  Writes one JSON line per kernel pair in
the same shape as the hardware results, with ratio 1 (the log is already in
cycles).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HOME = os.path.expanduser("~")
PERF = os.environ.get("IA64PERF", f"{HOME}/projects/ia64perf")
QEMU = os.environ.get("QEMU", f"{HOME}/projects/qemu-ia64/build/qemu-ia64")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="build/probe")
    ap.add_argument("--meta", default="build/kernels.json")
    ap.add_argument("--out", default="results/sim.jsonl")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--perfargs", default=",pref=1")
    ap.add_argument("prefix", nargs="*")
    args = ap.parse_args()

    tab = args.probe + ".tab"
    with open(tab, "w") as f:
        subprocess.run([sys.executable, f"{PERF}/mktab.py", args.probe],
                       stdout=f, check=True)
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "itc")
        plugin = (f"{PERF}/ia64perf.so,table={tab},itc={log},"
                  f"out={d}/summary{args.perfargs}")
        res = subprocess.run([QEMU, "-plugin", plugin, args.probe, "-s",
                              "-r", str(args.reps)] + args.prefix,
                             capture_output=True, text=True, check=True)
        stamps = [int(x) for x in open(log).read().split()]
    calls = [l.split() for l in res.stdout.splitlines()
             if l.startswith("call ")]
    meta = json.load(open(args.meta))
    spans = []
    for _, name, bodies, iters in calls:
        inner = meta[name].get("itc_reads_per_body", 0)
        spans.append(2 + inner * int(bodies) * int(iters))
    if len(stamps) != sum(spans):
        raise SystemExit(f"{len(stamps)} itc reads for {len(calls)} calls, "
                         f"expected {sum(spans)}")
    best: dict[tuple[str, int], tuple[int, int]] = {}
    order = []
    at = 0
    for (_, name, bodies, iters), n in zip(calls, spans):
        key = (name, int(bodies))
        dt = stamps[at + n - 1] - stamps[at]
        at += n
        if key not in best:
            order.append(key)
            best[key] = (dt, int(iters))
        elif dt < best[key][0]:
            best[key] = (dt, int(iters))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for name, bodies in order:
            if (name, 2 * bodies) not in best:
                continue
            a, iters = best[(name, bodies)]
            b, _ = best[(name, 2 * bodies)]
            f.write(json.dumps({"name": name, "bodies": bodies,
                                "iters": iters, "min": [a, b],
                                "ratio": 1.0}) + "\n")
    print(f"{len(order) // 2} kernels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
