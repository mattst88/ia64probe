# SPDX-License-Identifier: GPL-3.0-or-later
"""A callee's stores to a struct its caller loads right after the return.

sqlite3BtreeParseCellPtr fills a CellInfo on its caller's stack: an st8
first, then st4/st8/st2s to the rest of the 32 bytes in the groups just
before its br.ret, and the caller loads info.nSize (an ld2 of bytes the
callee's st2 wrote) as soon as it is back.  Each body stores r40 like
that and makes the next r40 from the loaded halfword, so it costs the
stores' groups plus the store-to-load-to-use path.

ci_all_gG puts G empty groups between the first st8 and the rest; ci_one
does only the st2 that the ld2 reads; ci_call_gG does ci_all_gG's stores
in a callee.
"""

from emit import Bundle as B, pair

SETUP = ["mov r40 = 0", "adds r36 = 64, r33", "adds r42 = 72, r33",
         "adds r43 = 80, r33", "adds r44 = 84, r33", "adds r45 = 88, r33",
         "adds r46 = 90, r33", "adds r47 = 92, r33", "adds r48 = 94, r33",
         "ld8 r41 = [r36]"]

NOP = B("mmi", None, None, "nop.i 0 ;;")


def stores(g, first=True, rest=True, call=False):
    lab = "2" if call else None
    out = []
    if first:
        out += [B("mmi", "st8 [r36] = r40", None, "nop.i 0 ;;", label=lab)]
        lab = None
    out += [NOP] * g
    if rest:
        out += [B("mmi", "st4 [r43] = r40", "st8 [r42] = r40", "nop.i 0 ;;",
                  label=lab),
                B("mmi", "st4 [r44] = r40", "st2 [r45] = r40", "nop.i 0 ;;"),
                B("mmi", "st2 [r46] = r40", "st2 [r47] = r40", "nop.i 0 ;;")]
    end = "br.ret.sptk.many b6 ;;" if call else None
    out += [B("mib", "st2 [r48] = r40" if rest else None, None, end)]
    return out


USE = [B("mmi", "ld8 r41 = [r36]", "ld2 r49 = [r45]", "nop.i 0 ;;"),
       B("mmi", None, None, "adds r40 = 1, r49 ;;")]


def called(g):
    return [B("mib", None, None, "br.call.sptk.many b6 = 2f ;;"),
            *USE,
            B("mib", None, None, "br.cond.sptk.few 3f ;;"),
            *stores(g, call=True),
            B("mmi", None, None, "nop.i 0 ;;", label="3")]


def kernels():
    ks = []
    for g in (0, 4, 10):
        ks += pair(f"ci_all_g{g}", "cellinfo", stores(g) + USE, SETUP,
                   bodies=4)
        ks += pair(f"ci_call_g{g}", "cellinfo", called(g), SETUP,
                   bodies=4)
    ks += pair("ci_one", "cellinfo",
               [B("mmi", "st2 [r45] = r40", None, "nop.i 0 ;;")] + USE,
               SETUP, bodies=4)
    ks += pair("ci_first", "cellinfo", stores(4, rest=False) + USE, SETUP,
               bodies=4)
    return ks + sweep_kernels() + size_kernels() + wait_kernels() + cover_kernels() + byte_kernels() + group_kernels()


# The first n of the CellInfo stores, one a group, then an ld2 of info.nSize
# (_l), of another line (_x) or no load (_n), after w empty groups.
ST = ["st8 [r36] = r40", "st4 [r43] = r40", "st8 [r42] = r40",
      "st4 [r44] = r40", "st2 [r45] = r40", "st2 [r46] = r40",
      "st2 [r47] = r40", "st2 [r48] = r40"]


def sweep(n, load, w):
    body = [B("mmi", s, None, "nop.i 0 ;;") for s in ST[:n]] + [NOP] * w
    if load == "n":
        return body + [B("mmi", None, None, "adds r40 = 1, r40 ;;")]
    src = "r45" if load == "l" else "r37"
    return body + [B("mmi", f"ld2 r49 = [{src}]", None, "nop.i 0 ;;"),
                   B("mmi", None, None, "adds r40 = 1, r49 ;;")]


def sweep_kernels():
    ks = []
    for n in (1, 2, 3, 4, 5, 6, 8):
        for load in "lxn":
            for w in (0, 8):
                ks += pair(f"cis_{n}{load}_w{w}", "cellinfo", sweep(n, load, w),
                           SETUP + ["adds r37 = 128, r33", "ld8 r41 = [r37]"],
                           bodies=4)
    return ks


# n stores of sz bytes, one a group, packed into consecutive bytes (_p) or
# one to each doubleword (_d), then w empty groups and no load.
def size_kernels():
    ks = []
    for sz, op in ((8, "st8"), (4, "st4"), (2, "st2"), (1, "st1")):
        for lay in "pd":
            for n in (4, 6, 8):
                offs = [i * (8 if lay == "d" else sz) for i in range(n)]
                setup = ["mov r40 = 0"] + [f"adds r{50 + i} = {64 + o}, r33"
                                           for i, o in enumerate(offs)]
                for w in (0, 16):
                    body = ([B("mmi", f"{op} [r{50 + i}] = r40", None,
                               "nop.i 0 ;;") for i in range(n)] +
                            [NOP] * w +
                            [B("mmi", None, None, "adds r40 = 1, r40 ;;")])
                    ks += pair(f"ciz_{op}{lay}{n}_w{w}", "cellinfo", body,
                               setup, bodies=4)
    return ks


# The eight CellInfo stores, then a load chained to the next r40 and 40
# empty groups, so the load's own wait shows: ld2 of each st2's bytes
# (+24..+30), an ld8 of their doubleword, an ld4 of the st4s' (+16), an
# ld8 of the first st8's (+0).
LOADS = {"h24": "ld2 r49 = [r45]", "h26": "ld2 r49 = [r46]",
         "h30": "ld2 r49 = [r48]", "d24": "ld8 r49 = [r51]",
         "w16": "ld4 r49 = [r43]", "d0": "ld8 r49 = [r36]"}


def wait_kernels():
    ks = []
    for tag, ld in LOADS.items():
        for n in (8, 5):
            body = ([B("mmi", s, None, "nop.i 0 ;;") for s in ST[:n]] +
                    [B("mmi", ld, None, "nop.i 0 ;;"),
                     B("mmi", None, None, "adds r40 = 1, r49 ;;")] +
                    [NOP] * 40)
            ks += pair(f"ciw_{n}_{tag}", "cellinfo", body,
                       SETUP + ["adds r51 = 88, r33"], bodies=2)
    return ks


# Whether the load's bytes were all written by the queued 1- and 2-byte
# stores to its doubleword decides how long it waits: each variant drops
# the CellInfo st2s in `drop` and loads with `ld`.
COVER = {
    "h24": ((), "ld2 r49 = [r45]"),           # its own st2's bytes
    "h24x": ((4,), "ld2 r49 = [r45]"),         # bytes no queued store wrote
    "b25": ((), "ld1 r49 = [r52]"),            # half of a queued st2's
    "w24": ((), "ld4 r49 = [r45]"),            # two queued st2s'
    "w24x": ((5,), "ld4 r49 = [r45]"),         # one queued st2's and others
    "w28x": ((4, 5), "ld4 r49 = [r47]"),       # other bytes of its word
}


def cover_kernels():
    ks = []
    for tag, (drop, ld) in COVER.items():
        body = ([B("mmi", s, None, "nop.i 0 ;;")
                 for i, s in enumerate(ST) if i not in drop] +
                [B("mmi", ld, None, "nop.i 0 ;;"),
                 B("mmi", None, None, "adds r40 = 1, r49 ;;")] + [NOP] * 40)
        ks += pair(f"ciq_{tag}", "cellinfo", body,
                   SETUP + ["adds r52 = 89, r33"], bodies=2)
    return ks


# st1s where ciq has st2s: n st1s to bytes 24.. of the line, one a group,
# then an ld1 of the first (_w) or of the byte after the last (_o), with
# no other stores; and n st2s the same way.
def byte_kernels():
    ks = []
    setup = ["mov r40 = 0"] + [f"adds r{50 + i} = {88 + i}, r33"
                               for i in range(9)] + [
        f"adds r{60 + i} = {88 + 2 * i}, r33" for i in range(5)]
    for n in (2, 4, 8):
        for tag, ld in (("w", "ld1 r49 = [r50]"), ("o", f"ld1 r49 = [r{50 + n}]")):
            body = ([B("mmi", f"st1 [r{50 + i}] = r40", None, "nop.i 0 ;;")
                     for i in range(n)] +
                    [B("mmi", ld, None, "nop.i 0 ;;"),
                     B("mmi", None, None, "adds r40 = 1, r49 ;;")] + [NOP] * 40)
            ks += pair(f"ciq1_{n}{tag}", "cellinfo", body, setup, bodies=2)
    for n in (2, 4):
        for tag, ld in (("w", "ld2 r49 = [r60]"),
                        ("o", f"ld2 r49 = [r{60 + n}]")):
            body = ([B("mmi", f"st2 [r{60 + i}] = r40", None, "nop.i 0 ;;")
                     for i in range(n)] +
                    [B("mmi", ld, None, "nop.i 0 ;;"),
                     B("mmi", None, None, "adds r40 = 1, r49 ;;")] + [NOP] * 40)
            ks += pair(f"ciq2_{n}{tag}", "cellinfo", body, setup, bodies=2)
    return ks


# A load in the same group as n-1 earlier 1-byte stores to its doubleword
# and one in it: the store first in the group (_s), then an ld1 of the
# byte after its own (_o) or of an unwritten byte of another doubleword
# of the line (_x); stores alone (_n).
def group_kernels():
    ks = []
    setup = ["mov r40 = 0"] + [f"adds r{50 + i} = {88 + i}, r33"
                               for i in range(9)] + ["adds r59 = 64, r33"]
    for n in (1, 2, 4, 8):
        for tag in "oxn":
            ld = {"o": f"ld1 r49 = [r{50 + n}]", "x": "ld1 r49 = [r59]",
                  "n": None}[tag]
            body = ([B("mmi", f"st1 [r{50 + i}] = r40", None, "nop.i 0 ;;")
                     for i in range(n - 1)] +
                    [B("mmi", f"st1 [r{50 + n - 1}] = r40", ld, "nop.i 0 ;;"),
                     B("mmi", None, None,
                       "adds r40 = 1, r49 ;;" if ld else "adds r40 = 1, r40 ;;")]
                    + [NOP] * 40)
            ks += pair(f"cig_{n}{tag}", "cellinfo", body, setup, bodies=2)
    return ks
