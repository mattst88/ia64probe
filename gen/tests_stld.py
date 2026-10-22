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
    # A 4-byte store, then after d cycles a 4-byte load of the same bytes
    # (s4_dD), of the other half of the doubleword (s4_hD), or of the next
    # doubleword (s4_oD); s44_hD stores the word twice in a row first, as
    # bzip2's partition loop swaps two equal pointers.
    for d in (0, 1, 2, 4, 8, 16):
        for tag, off, twice in (("s4_d", 0, False), ("s4_h", 4, False),
                                ("s4_o", 8, False), ("s44_h", 4, True)):
            ld = f"adds r38 = {off}, r36" if off else "mov r38 = r36"
            sts = [B("mmi", "st4 [r36] = r40", None, "nop.i 0 ;;")]
            if twice:
                sts *= 2
            b = [*sts, *wait(d),
                 B("mmi", "ld4 r41 = [r38] ;;", None, None),
                 B("mmi", None, None, "adds r40 = 1, r41 ;;")]
            ks += pair(f"stld_{tag}{d}", "stld", b, SETUP + [ld], bodies=4)
    # Two stores of a word in a row, then nothing (ss_n), or a load that
    # nothing waits for of the other half (ss_h), the same bytes (ss_d) or
    # another line (ss_x): does a load end a run of stores to those bytes?
    for tag, ld in (("n", None), ("h", "ld4 r41 = [r38]"),
                    ("d", "ld4 r41 = [r36]"), ("x", "ld4 r41 = [r37]")):
        st = B("mmi", "st4 [r36] = r40", None, "nop.i 0 ;;")
        b = [st, st, B("mmi", ld, None, "nop.i 0 ;;"),
             B("mmi", None, None, "nop.i 0 ;;")]
        ks += pair(f"stld_ss_{tag}", "stld", b,
                   SETUP + ["adds r38 = 4, r36"], bodies=4)
    # A store and a load of different sizes in one doubleword, d cycles
    # apart, as zstd's match copy does: stld_sz_S_L_oO stores S bytes at
    # r36 and loads L bytes at r36 + O.
    for d in (0, 1, 2, 3, 4, 6, 8):
        for sz, ld_sz, off in ((8, 1, 6), (8, 1, 0), (8, 4, 4), (1, 8, 0),
                               (1, 1, 0), (1, 1, 1), (4, 1, 2)):
            stop = {1: "st1", 4: "st4", 8: "st8"}[sz]
            ldop = {1: "ld1", 4: "ld4", 8: "ld8"}[ld_sz]
            b = [B("mmi", f"{stop} [r36] = r40", None, "nop.i 0 ;;"),
                 *wait(d),
                 B("mmi", f"{ldop} r41 = [r38] ;;", None, None),
                 B("mmi", None, None, "adds r40 = 1, r41 ;;")]
            ld = f"adds r38 = {off}, r36" if off else "mov r38 = r36"
            ks += pair(f"stld_sz_{sz}_{ld_sz}_o{off}_d{d}", "stld", b,
                       SETUP + [ld], bodies=4)
    # Bursts of k st8s a cycle to one doubleword, then g empty groups, to a
    # line that a load brought into L1D (_l) or that nothing loads (_n):
    # does the 5-cycle wait between such stores depend on L1D holding the
    # line?
    for k, g in ((1, 0), (2, 2), (4, 4), (8, 8), (8, 40), (16, 48)):
        for tag, setup in (("l", ["ld8 r41 = [r39] ;;"]), ("n", [])):
            b = ([B("mmi", "st8 [r39] = r40", None, "nop.i 0 ;;")] * k
                 + wait(g))
            ks += pair(f"stld_sr_b{k}_g{g}_{tag}", "stld", b,
                       SETUP + ["adds r39 = 192, r33"] + setup,
                       meta={"k": k, "g": g, "l1": tag == "l"})
    return ks
