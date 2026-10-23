# SPDX-License-Identifier: GPL-3.0-or-later
"""Output dependences: does a write wait for an earlier write in flight?

Each body writes one register and reads nothing the loop writes, so with
no wait for an earlier write it would issue a group a cycle.  A body of
one instruction measures X after X; a body of two, X then Y and Y then X.
"""

from emit import Bundle as B, pair
from tests_latency import one

SETUP = ["mov r40 = 0", "mov r41 = 0", "st8 [r33] = r0", "mov r18 = 1",
         "setf.sig f32 = r18", "fcvt.xf f32 = f32", "fnorm.d f32 = f32",
         "mov f33 = f32", "mov b6 = r40"]

LD = ("m", "ld8 r50 = [r33]")
ADD = ("a", "add r50 = r40, r41")
GETF = ("m", "getf.sig r50 = f32")
SETF = ("m", "setf.sig f50 = r40")
LDF = ("m", "ldf8 f50 = [r33]")
FMA = ("f", "fma.d f50 = f32, f32, f33")
POPCNT = ("i", "popcnt r50 = r40")
MOVBR = ("i", "mov r50 = b6")

BODIES = {
    "ld8": [LD], "getf": [GETF], "setf": [SETF], "ldf8": [LDF],
    "fma": [FMA], "popcnt": [POPCNT], "movbr": [MOVBR],
    "ld8_add": [LD, ADD], "getf_add": [GETF, ADD], "ldf8_fma": [LDF, FMA],
    "setf_fma": [SETF, FMA], "popcnt_add": [POPCNT, ADD],
    "movbr_add": [MOVBR, ADD],
    "ld8_ld8": [LD, ("m", "ld8 r50 = [r33]")],
    # An F-unit or load write in flight, then an M-unit write: the
    # update in tramp3d's APressure rewrites each xma.l or ldfd target
    # with a setf a few groups on.
    "fma_setf": [FMA, SETF], "fma_ldf8": [FMA, LDF], "ldf8_setf": [LDF, SETF],
    "xma_setf": [("f", "xma.l f50 = f32, f32, f0"), SETF],
}


def kernels():
    ks = []
    for name, chain in BODIES.items():
        ks += pair(f"waw_{name}", "waw", [one(u, t) for u, t in chain],
                   SETUP, {"chain": [t for _, t in chain]})
    return ks
