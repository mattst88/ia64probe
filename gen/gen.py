#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate build/kernels.{s,inc,json} and build/rings.inc."""

from __future__ import annotations

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emit import emit_file, emit_metadata, emit_registry, emit_rings  # noqa: E402

MODULES = [
    "tests_sanity",
    "tests_latency",
    "tests_dispersal",
    "tests_waw",
    "tests_memory",
    "tests_rse",
    "tests_icache",
    "tests_branch",
    "tests_stld",
    "tests_taken",
    "tests_ldstream",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="build")
    ap.add_argument("--only", action="append", default=None,
                    help="restrict to these test modules")
    args = ap.parse_args()

    modules = args.only or MODULES
    if "tests_sanity" not in modules:
        modules = ["tests_sanity"] + modules
    kernels = []
    rings = []
    for name in modules:
        mod = importlib.import_module(name)
        kernels.extend(mod.kernels())
        rings.extend(getattr(mod, "RINGS", []))

    seen = set()
    for k in kernels:
        if k.symbol in seen:
            raise SystemExit(f"duplicate kernel symbol {k.symbol}")
        seen.add(k.symbol)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "kernels.s"), "w") as f:
        f.write(emit_file(kernels))
    with open(os.path.join(args.outdir, "kernels.inc"), "w") as f:
        f.write(emit_registry(kernels))
    with open(os.path.join(args.outdir, "rings.inc"), "w") as f:
        f.write(emit_rings(rings))
    with open(os.path.join(args.outdir, "kernels.json"), "w") as f:
        f.write(emit_metadata(kernels))

    print(f"generated {len(kernels)} kernels from {len(modules)} module(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
