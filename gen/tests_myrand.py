# SPDX-License-Identifier: GPL-3.0-or-later
"""health's my_rand, as clang compiles it, inline.

my_rand(idum) xors a mask, divides by 127773 with xma.h, multiplies the
remainder back with shladds, converts the result with setf.sig, fcvt.xf,
fma.d and fma.s; the caller scales it by 2^31 - 1 and truncates it back
into the next seed with fcvt.fx.trunc and getf.sig, so a call depends on
the last.  Called in a loop on an Itanium 9340 it takes 88-91 cycles,
where the model says 73-76.  These kernels replay it without the call,
whole and in parts, to find where.
"""

import copy
import re

from emit import Bundle as B, pair

SETUP = ["mov r60 = 12345",
         "movl r18 = 0x41dfffffffc00000", "setf.d f12 = r18"]

HEAD = [
    B("mlx", None, "movl r44 = 123459876"),
    B("mlx", None, "movl r46 = 4730756183288445817 ;;"),
    B("mlx", "setf.sig f7 = r46", "movl r24 = 2024023771"),
    B("mlx", "xor r45 = r60, r44", "movl r25 = 4467570830353629184 ;;"),
]
# idum ^ mask in r45 to idum in r47.
INT = [
    B("mlx", "setf.sig f6 = r45", "nop.x 0 ;;"),
    B("mfi", None, "xma.h f6 = f6, f7, f0", "nop.i 0"),
    B("mmi", "setf.d f7 = r25 ;;", "getf.sig r46 = f6", "nop.i 0 ;;"),
    B("mii", None, "shr.u r47 = r46, 63", "shr r46 = r46, 15 ;;"),
    B("mmi", "add r46 = r46, r47 ;;", None, "shl r47 = r46, 2"),
    B("mii", None, "shl r18 = r46, 5", "shladd r21 = r46, 2, r46"),
    B("mii", None, "shl r19 = r46, 3 ;;", "sub r47 = r47, r46"),
    B("mmi", "sub r18 = r18, r46", "shladd r22 = r21, 1, r46",
      "sub r19 = r19, r46 ;;"),
    B("mii", "shladd r47 = r18, 4, r47", "shladd r23 = r22, 4, r46",
      "shladd r20 = r19, 2, r46 ;;"),
    B("mii", "shladd r46 = r23, 2, r46", "shl r47 = r47, 8",
      "sub r45 = r45, r20 ;;"),
    B("mii", "sub r45 = r45, r47", "shl r46 = r46, 2 ;;",
      "shladd r47 = r45, 4, r45 ;;"),
    B("mmi", "shladd r47 = r47, 1, r45 ;;", "shladd r47 = r47, 2, r47",
      "nop.i 0 ;;"),
    B("mmi", "shladd r47 = r47, 4, r45 ;;", "shladd r47 = r47, 1, r47",
      "nop.i 0 ;;"),
    B("mmi", "shladd r45 = r47, 1, r45 ;;", "sub r45 = r45, r46",
      "nop.i 0 ;;"),
    B("mii", "xor r47 = r45, r44", "add r44 = r44, r24 ;;",
      "cmp.lt p7, p6 = -1, r45"),
    B("mmi", "add r44 = r47, r44 ;;", "(p6) mov r47 = r44", "nop.i 0 ;;"),
]
# idum in r47 to the next seed in r60.
FP = [
    B("mmi", "setf.sig f6 = r47 ;;", None, None),
    B("mfi", None, "fcvt.xf f6 = f6", "nop.i 0 ;;"),
    B("mfi", None, "fma.d f6 = f6, f1, f0", "nop.i 0 ;;"),
    B("mfi", None, "fma.d f6 = f6, f7, f0", "nop.i 0 ;;"),
    B("mfi", None, "fma.s f8 = f6, f1, f0", "nop.i 0 ;;"),
    B("mfi", None, "fma.d f6 = f8, f1, f0", "nop.i 0 ;;"),
    B("mfi", None, "fma.d f6 = f6, f12, f0", "nop.i 0 ;;"),
    B("mfi", None, "fcvt.fx.trunc f6 = f6", "nop.i 0 ;;"),
    B("mmi", "getf.sig r60 = f6 ;;", None, None),
]


def static(bundles):
    """The bundles with the stacked registers they use moved to static
    ones, for a callee that has no frame of its own."""
    m = {"r44": "r26", "r45": "r27", "r46": "r28", "r47": "r29",
         "r60": "r30"}
    out = []
    for b in bundles:
        c = copy.copy(b)
        c.slots = [re.sub(r"\br(44|45|46|47|60)\b", lambda x: m[x.group(0)],
                          t) if t else t for t in b.slots]
        out.append(c)
    return out


# The same, as a call: the callee ends with the fma.s in the br.ret's group,
# as clang emits it, or a group before it (_sep).
def call(sep=False):
    ret = ([B("mfi", None, "fma.s f8 = f6, f1, f0", "nop.i 0 ;;"),
            B("mib", None, None, "br.ret.sptk.many b6 ;;")] if sep else
           [B("mfb", None, "fma.s f8 = f6, f1, f0",
              "br.ret.sptk.many b6 ;;")])
    return [B("mib", "mov r30 = r60", None, "br.call.sptk.many b6 = 2f ;;"),
            *FP[5:],
            B("mib", None, None, "br.cond.sptk.few 3f ;;"),
            B("mlx", None, "movl r26 = 123459876", label="2"),
            *static(HEAD[1:] + INT + FP[:4]), *ret,
            B("mmi", None, None, "nop.i 0 ;;", label="3")]


def kernels():
    ks = []
    ks += pair("mr_full", "myrand", HEAD + INT + FP, SETUP, {}, bodies=2,
               iters=1000)
    ks += pair("mr_int", "myrand",
               HEAD + INT + [B("mmi", "mov r60 = r47 ;;", None, None)],
               SETUP, {}, bodies=2, iters=1000)
    ks += pair("mr_fp", "myrand",
               HEAD + [B("mmi", "mov r47 = r45", "setf.d f7 = r25",
                         "nop.i 0 ;;")] + FP,
               SETUP, {}, bodies=2, iters=1000)
    ks += pair("mr_call", "myrand", call(), SETUP, {}, bodies=2, iters=1000)
    ks += pair("mr_call_sep", "myrand", call(True), SETUP, {}, bodies=2,
               iters=1000)
    return ks
