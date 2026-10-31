#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build, copy to the Itanium host, run, collect.

The host has no PMU and an ondemand cpufreq governor, so the probe times
with ar.itc and calibrates against a known add chain (see harness/probe.c).
It is a shared machine: before running, the runner samples /proc/stat and
picks a core whose two SMT threads are both idle, then pins to one of them.
"""

from __future__ import annotations

import argparse
import os
import subprocess

HOST = os.environ.get("IA64HOST", "ia64dev.matoro.tk")
RDIR = "/tmp/ia64probe-mattst88"

PICK = r"""
grep '^cpu[0-9]' /proc/stat > %(d)s/s1; sleep 1
grep '^cpu[0-9]' /proc/stat > %(d)s/s2
paste %(d)s/s1 %(d)s/s2 | awk '{t=0; for (i=2;i<=11;i++) t+=$(i+11)-$i;
  n=substr($1,4); busy[n]=1-($16-$5)/t}
  END {best=-1; smt=length(busy)==16 ? 2 : 1;
    for (c=0;c<length(busy);c+=smt) { b=busy[c]+(smt>1 ? busy[c+1] : 0);
    if (best<0 || b<bb) {best=c; bb=b} } print best}'
"""


def ssh(cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", HOST, cmd],
                          text=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/hw.jsonl")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--reps", type=int, default=31)
    ap.add_argument("--cpu", type=int, default=None)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("prefix", nargs="*")
    args = ap.parse_args()

    if not args.no_build:
        only = " ".join(args.only)
        subprocess.run(["make", "-B", f"ONLY={only}"], check=True)
    ssh(f"mkdir -p {RDIR}", check=True)
    subprocess.run(["scp", "-q", "build/probe", f"{HOST}:{RDIR}/probe"],
                   check=True)
    cpu = args.cpu
    if cpu is None:
        cpu = int(ssh(PICK % {"d": RDIR}, capture_output=True,
                      check=True).stdout)
    print(f"running on cpu {cpu}")
    res = ssh(f"{RDIR}/probe -c {cpu} -r {args.reps} "
              + " ".join(args.prefix), capture_output=True, check=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(res.stdout)
    print(f"{len(res.stdout.splitlines())} kernels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
