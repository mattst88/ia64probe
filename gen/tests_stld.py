# SPDX-License-Identifier: GPL-3.0-or-later
"""A load of a word a store wrote a few cycles before it.

bh's walksub loads its stack arguments right after its caller stores
them, and its caller loads the struct it returns right after the callee
stores it; on hardware those loads cost about 7 cycles more than an L1D
hit.  Each body stores r40, waits d cycles, loads the word back and
makes the next value from it, so the body's cost is d plus the
store-to-load-to-use path.  stld_dD reloads the stored word, stld_oD
another word of its line, stld_xD a word of another line (no conflict);
stld_cD and stld_coD put a call and return in the wait.
"""

from emit import Bundle as B, pair
from tests_rse import call_leaf

SETUP = ["mov r40 = 0", "adds r36 = 64, r33", "adds r37 = 128, r33",
         "ld8 r41 = [r36]", "ld8 r41 = [r37]"]


def wait(d):
    return [B("mmi", None, None, "nop.i 0 ;;")] * d


def body(d, off, call=False, other=False):
    base = "r37" if other else "r36"
    mid = call_leaf(0) if call else wait(d)
    ld = f"adds r38 = {off}, {base} ;;" if off else "mov r38 = r36 ;;"
    return [B("mmi", "st8 [r36] = r40", None, "nop.i 0 ;;"),
            *mid,
            B("mmi", "ld8 r41 = [r38] ;;", None, None),
            B("mmi", None, None, "adds r40 = 1, r41 ;;")], ld


def kernels():
    ks = []
    for d in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24):
        for tag, off, other in (("d", 0, False), ("o", 8, False),
                                ("x", 0, True)):
            b, ld = body(d, off, other=other)
            ks += pair(f"stld_{tag}{d}", "stld", b,
                       SETUP + [ld.rstrip(" ;")], bodies=4)
    for tag, off in (("c", 0), ("co", 8)):
        b, ld = body(0, off, call=True)
        ks += pair(f"stld_{tag}", "stld", b, SETUP + [ld.rstrip(" ;")],
                   bodies=4)
    # A store to another word of the loaded line, then a load chained
    # through its own address (the word holds 0): stld_fD an stf.spill,
    # stld_f8D an stf8, stld_iD an st8, stld_fxD an stf.spill to another
    # line.
    for d in (0, 1, 2, 4, 8, 16):
        for tag, st, other in (("f", "stf.spill [r39] = f0", False),
                               ("f8", "stf8 [r39] = f0", False),
                               ("i", "st8 [r39] = r0", False),
                               ("fx", "stf.spill [r39] = f0", True)):
            setup = ["adds r36 = 64, r33", "adds r37 = 128, r33",
                     "st8 [r36] = r0", "ld8 r41 = [r36]", "ld8 r41 = [r37]",
                     "mov r38 = r36",
                     f"adds r39 = 16, {'r37' if other else 'r36'}"]
            b = [B("mmi", st, None, "nop.i 0 ;;"), *wait(d),
                 B("mmi", "ld8 r41 = [r38] ;;", None, None),
                 B("mmi", None, None, "add r38 = r41, r38 ;;")]
            ks += pair(f"stld_{tag}{d}", "stld", b, setup, bodies=4)
    return ks
