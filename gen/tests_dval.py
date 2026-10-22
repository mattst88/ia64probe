# SPDX-License-Identifier: GPL-3.0-or-later
"""Double-precision adds by operand value.

The 9340 stalls a single-precision add by its operands' exponents (st_zz_sm
in tests_memory).  These kernels look for the same in double precision:
eight independent ops, one a group, each "op f = f20, f1, f21" (a product
1.5 * 2^ea and an addend 1.25 * 2^eb) or "fadd.d f = f20, f21", for a grid
of addend exponents eb and distances d = ea - eb.

The dc_ kernels instead chain eight accumulations into one register,
f21 += f20, starting each body from 1.25 * 2^eb with f20 = m * 2^(eb + d),
m 1.5 (exact sums, "x") or 1/3 (inexact, "t"), so a value-dependent
latency would show up in the chain.
"""

from __future__ import annotations

import struct

from emit import Bundle as B, pair


def bits(m, e):
    return struct.unpack("<Q", struct.pack("<d", m * 2.0 ** e))[0]


def body(op):
    out = []
    for k in range(8):
        out += [B("mfi", None, f"{op} f{32 + k} = f20, f1, f21"
                  if not op.startswith("fadd") else
                  f"{op} f{32 + k} = f20, f21", "nop.i 0 ;;")]
    return out


def setup(ea, eb):
    return [f"movl r18 = {bits(1.5, ea):#x}", "st8 [r33] = r18 ;;",
            "ldfd f20 = [r33] ;;",
            f"movl r18 = {bits(1.25, eb):#x}", "st8 [r33] = r18 ;;",
            "ldfd f21 = [r33] ;;"]


def chain(op):
    """Eight dependent accumulations, f21 += f20 * 1, one a group."""
    out = []
    for k in range(8):
        src = "f20, f1, f21" if not op.startswith("fadd") else "f20, f21"
        out += [B("mfi", None, f"{op} f21 = {src}", "nop.i 0 ;;")]
    return out


def kernels():
    ks = []
    for op in ("fma.d", "fma.d.s1", "fma", "fma.s1", "fadd.d"):
        for m in (1.5, 1 / 3):
            for eb in (0, 10):
                for d in (-60, -40, -30, -20, -12, -8, -3, 0):
                    name = (f"dc_{op.replace('.', '')}_{'x' if m == 1.5 else 't'}"
                            f"_b{eb}_d{d}".replace("-", "m"))
                    st = [f"movl r18 = {bits(m, eb + d):#x}",
                          "st8 [r33] = r18 ;;", "ldfd f20 = [r33] ;;"]
                    ks += pair(name, "dval",
                               [B("mmi", f"ldfd f21 = [r34]", None,
                                  "nop.i 0 ;;")] + chain(op),
                               st + [f"movl r18 = {bits(1.25, eb):#x}",
                                     "adds r34 = 8, r33 ;;",
                                     "st8 [r34] = r18 ;;"],
                               meta={"op": op, "m": m, "eb": eb, "d": d},
                               bodies=2)
    for op in ("fma.d", "fma.d.s1", "fadd.d", "fma"):
        for eb in (-20, -2, -1, 0, 1, 2, 20):
            for d in (-8, -3, -2, -1, 0, 1, 2, 3, 8):
                name = (f"dv_{op.replace('.', '')}_b{eb}_d{d}"
                        .replace("-", "m"))
                ks += pair(name, "dval", body(op), setup(eb + d, eb),
                           meta={"op": op, "eb": eb, "d": d}, bodies=2)
    return ks
