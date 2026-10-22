# SPDX-License-Identifier: GPL-3.0-or-later
"""The memory hierarchy: load latency by footprint, and store throughput.

chase_*: ld8 r34 = [r34] around a ring of pointers covering a region of the
given size, one pointer per 128-byte line in a random order (the harness
builds the rings; see harness/scratch.c).  The cycles per body are the load
latency to an address consumer, N + 1 in 308065-001 Table 2-8, so N is one
less: 1 for L1D and 5 for L2D, as the table says; the 9340 measures 16 to
19 for L3 (the table's 14-15 is Montecito's) and 310 to 330 for memory.
The cache sizes are those the 9340 reports: L1D 16 KB, L2D 256 KB, L3 5 MB.
Rings beyond 128 KB visit more 64 KB pages than the data TLB holds, so the
large ones include TLB misses, as random accesses in real code do.

fchase_*: the same with ldf8 and getf.sig, so N is replaced by an FP load's
M + 1 (FP loads skip L1D) plus getf's 6 to an address.

st_*: stores, and loads and stores together: store throughput (same, seq,
gap, alt), store-to-load forwarding (stld, 251110-003 Table 6-5), loads
and stores in one group (mtf, pair), the L1D fill a store cancels (bi,
251110-003 6.6.3.2), and ALAT entries a store kills (alat, alat1, alatd:
the ALAT compares the bytes in address bits 11:0 only).  Then 1- and 2-byte
stores to one word, which leave the store queue 5 or 6 cycles apart (seqst,
st1st8, st8st1, zst, zsq, zsqw), and how many queue before the pipeline
waits (strun, stq, stqw: about 18); glibc's byte copy loop (copy1,
copyrun), a load in a store's group with the same address bits 11:2
(alias), and zstd's match copies (zcopy, zmatch, zline, zlg, zbyte, qld,
ldst), whose loads of lines that stores missed L2 in wait for them.
"""

from emit import Bundle as B, pair
from tests_latency import one

KB, MB = 1 << 10, 1 << 20
RING_BASE = 9 * MB
STREAM = 1 * MB
SIZES = [8 * KB, 16 * KB, 32 * KB, 64 * KB, 128 * KB, 192 * KB, 256 * KB,
         384 * KB, 512 * KB, 1 * MB, 2 * MB, 4 * MB, 6 * MB, 8 * MB,
         16 * MB, 64 * MB]
SEQ = [256 * KB, 2 * MB, 64 * MB]


def _layout():
    rings, at = {}, RING_BASE
    for seq, sizes in ((False, SIZES), (True, SEQ)):
        for size in sizes:
            at = (at + 64 * KB - 1) & ~(64 * KB - 1)
            rings[(size, seq)] = (at, size, 128, seq)
            at += size
    return rings


LAYOUT = _layout()
RINGS = list(LAYOUT.values())


def name_of(size):
    return f"{size // MB}m" if size >= MB else f"{size // KB}k"


# Each ring's cursor: where its last call stopped, so that every call
# continues around the ring instead of revisiting the same first lines --
# a call walks 8000-16000 of them, which a large ring's start would keep
# warm in L3 from call to call.  Zero (the scratch is zeroed) means start.
CURSORS = 0xF0000


def ring_setup(size, seq=False):
    off = LAYOUT[(size, seq)][0]
    cur = CURSORS + 8 * RINGS.index(LAYOUT[(size, seq)])
    return [f"movl r18 = {off:#x}", f"movl r19 = {cur:#x}",
            "add r19 = r33, r19", "ld8 r34 = [r19]",
            "cmp.eq p6, p0 = 0, r34", "(p6) add r34 = r33, r18",
            "mov r40 = 0"]


RING_TEARDOWN = ["st8 [r19] = r34"]


def level(size):
    """The claimed load latency N of the level a ring of this size lives in
    (308065-001 Table 2-8 note 3, lower bounds)."""
    if size <= 16 * KB:
        return 1
    if size <= 256 * KB:
        return 5
    if size <= 4 * MB:
        return 14
    return None


SSETUP = ["mov r40 = 0", "adds r42 = 64, r33", f"movl r18 = {STREAM:#x}",
          "add r34 = r33, r18", "adds r35 = 8, r34",
          "st8 [r33] = r33"]


def st(addr, inc=None, data="r40"):
    return f"st8 [{addr}] = {data}" + (f", {inc}" if inc else "")


# An empty issue group.
GAP = B("mmi", None, None, "nop.i 0 ;;")

# body, and setup beyond SSETUP.  r36 is a second fixed address.
STORES = {
    "same": ([B("mmi", st("r33"), None, "nop.i 0 ;;")], []),
    "same_x2": ([B("mmi", st("r33"), st("r42"), "nop.i 0 ;;")], []),
    "seq8": ([B("mmi", st("r34", 8), None, "nop.i 0 ;;")], []),
    "seq128": ([B("mmi", st("r34", 128), None, "nop.i 0 ;;")], []),
    "seq16_x2": ([B("mmi", st("r34", 16), st("r35", 16), "nop.i 0 ;;")],
                 []),
    "seq8_ld": ([B("mmi", st("r34", 8), "ld8 r50 = [r42]", "nop.i 0 ;;")],
                []),
    # A load of what the previous group stored, and the store of the
    # loaded value: store -> load forwarding.
    "st_ld_same": ([one("m", "st8 [r33] = r34"), one("m", "ld8 r34 = [r33]")],
                   []),
    "st_ld_other": ([one("m", "st8 [r42] = r34"),
                     one("m", "ld8 r34 = [r33]")], []),
}
# The same address stored every cycle by other kinds of store, and the
# next cycle's first store over what the cycle before's second wrote
# (st_zz_pair_d8_*_l0 costs 1 a group with stfs): which stores wait.
for name, op in (("st4", "st4 [r33] = r40"), ("stfs", "stfs [r33] = f1"),
                 ("stfd", "stfd [r33] = f1"), ("stf8", "stf8 [r33] = f1")):
    STORES[f"same_{name}"] = ([B("mmi", op, None, "nop.i 0 ;;")], [])
for name, op in (("st8", "st8 [r{}] = r40, 8"), ("st4", "st4 [r{}] = r40, 8"),
                 ("stfs", "stfs [r{}] = f1, 8"), ("stfd", "stfd [r{}] = f1, 8")):
    STORES[f"chain_{name}"] = ([B("mmi", op.format(34), op.format(35),
                                  "nop.i 0 ;;")], [])
    STORES[f"chainrev_{name}"] = ([B("mmi", op.format(35), op.format(34),
                                     "nop.i 0 ;;")], [])
STORES["same_slot1"] = ([B("mmi", None, st("r33"), "nop.i 0 ;;")], [])
STORES["same_swap"] = ([B("mmi", st("r33"), None, "nop.i 0 ;;"),
                        B("mmi", None, st("r33"), "nop.i 0 ;;")], [])
STORES["same_swap_o"] = ([B("mmi", st("r33"), "nop.m 0", "nop.i 0 ;;"),
                          B("mmi", "ld8 r50 = [r42]", st("r33"),
                            "nop.i 0 ;;")], [])
# k stores in a row, one a cycle, to each doubleword of a stream: is only a
# third store to an address held up?  And a store to the address the store
# k back wrote, in a stream: alt's pattern without a fixed address.
for k in (2, 3, 4, 5, 6, 8, 12):
    STORES[f"rep{k}"] = ([one("m", st("r34"))] * (k - 1)
                         + [one("m", st("r34", 8))], [])
STORES["same_r34"] = ([one("m", st("r34"))], [])
STORES["rep24"] = ([one("m", st("r34"))] * 23 + [one("m", st("r34", 8))], [])
STORES["rep4_r33"] = ([one("m", st("r33"))] * 3 + [one("m", st("r42"))], [])
for k in (8, 24):
    STORES[f"fix{k}"] = ([one("m", st("r34"))] * k, [])
for k, g in ((4, 4), (8, 8), (8, 24), (8, 40), (16, 16), (16, 48), (24, 0),
             (24, 24), (24, 40)):
    STORES[f"burst{k}_g{g}"] = ([one("m", st("r34"))] * k + [GAP] * g, [])
STORES["burst23_other"] = ([one("m", st("r34"))] * 23
                           + [one("m", st("r42"))], [])
STORES["rep48"] = ([one("m", st("r34"))] * 47 + [one("m", st("r34", 8))], [])
STORES["rep24_back"] = ([one("m", st("r34"))] * 23
                        + [one("m", st("r34", 8))]
                        + [one("m", st("r34"))] * 23
                        + [one("m", st("r34", -8))], [])
STORES["rep2_g1"] = ([one("m", st("r34")), GAP, one("m", st("r34", 8))], [])
# One store to a fixed address every k cycles: how long a store to an
# address keeps the next one to it waiting.
for k in range(1, 9):
    STORES[f"gap{k}"] = ([one("m", st("r33"))] + [GAP] * (k - 1), [])
# Stores alternating between two fixed addresses `off` bytes apart, one a
# cycle: what granule the wait is on.
for off in (8, 16, 32, 64, 128, 256, 4096):
    STORES[f"alt{off}"] = ([one("m", st("r33")), one("m", st("r36"))],
                           [f"movl r18 = {off}", "add r36 = r33, r18"])
# A store, k-1 empty groups, and a load from the address (or 8 bytes on, or
# from the other half of the word for st4) feeding the next store's data:
# when a load may read what a store wrote.
for k in range(1, 7):
    STORES[f"stld_gap{k}"] = ([one("m", "st8 [r33] = r34")] + [GAP] * (k - 1)
                              + [one("m", "ld8 r34 = [r33]")], [])
STORES["stld_next"] = ([one("m", "st8 [r33] = r34"),
                        one("m", "ld8 r34 = [r36]")],
                       ["adds r36 = 8, r33"])
STORES["stld_line"] = ([one("m", "st8 [r33] = r34"),
                        one("m", "ld8 r34 = [r36]")],
                       ["adds r36 = 64, r33"])
STORES["st4ld8"] = ([one("m", "st4 [r33] = r34"),
                     one("m", "ld8 r34 = [r33]")], [])

# Sub-word stores and loads: a store of SW bytes at r33+SOFF, k-1 empty
# groups, and a load of LW bytes at r33+LOFF feeding the next store's data.
# bzip2's move-to-front shift as clang and gcc schedule it: load byte k and
# store the byte loaded before over it, walking a 2^BITS-byte window.  r27
# is the pointer, r25 the byte to store.
MTF_SETUP = ["mov r27 = r33", "mov r25 = 0"]
for bits in (8, 12):
    STORES[f"mtf_clang_{bits}"] = ([
        B("mmi", "ld1 r28 = [r27]", "st1 [r27] = r25", "adds r29 = 1, r27 ;;"),
        B("mii", "nop.m 0", "mov r25 = r28",
          f"dep r27 = r29, r33, 0, {bits} ;;")], MTF_SETUP)
    STORES[f"mtf_gcc_{bits}"] = ([
        B("mii", "ld1 r28 = [r27]", "mov r26 = r27", "adds r29 = 1, r27 ;;"),
        B("mii", "nop.m 0", "mov r24 = r25",
          f"dep r27 = r29, r33, 0, {bits}"),
        B("mmi", "nop.m 0", "nop.m 0", "mov r25 = r28 ;;"),
        B("mmi", "st1 [r26] = r24", "nop.m 0", "nop.i 0 ;;")], MTF_SETUP)
    # The store a group after the load ("next"), the store of the byte
    # before ("prev"), and icc's: the store of the byte before, a group
    # before the next load.
    STORES[f"mtf_next_{bits}"] = ([
        B("mmi", "ld1 r28 = [r27]", "nop.m 0", "adds r29 = 1, r27 ;;"),
        B("mii", "st1 [r27] = r25", "mov r25 = r28",
          f"dep r27 = r29, r33, 0, {bits} ;;")], MTF_SETUP)
    STORES[f"mtf_prev_{bits}"] = ([
        B("mmi", "ld1 r28 = [r27]", "nop.m 0", "adds r29 = 1, r27 ;;"),
        B("mii", "st1 [r26] = r25", "mov r25 = r28", "mov r26 = r27"),
        B("mmi", "nop.m 0", "nop.m 0", f"dep r27 = r29, r33, 0, {bits} ;;")],
        MTF_SETUP + ["mov r26 = r33"])
    STORES[f"mtf_icc_{bits}"] = ([
        B("mii", "st1 [r26] = r24", "mov r26 = r27", "adds r29 = 1, r27 ;;"),
        B("mii", "ld1 r28 = [r27]", "mov r24 = r25",
          f"dep r27 = r29, r33, 0, {bits} ;;"),
        B("mmi", "nop.m 0", "nop.m 0", "mov r25 = r28 ;;")],
        MTF_SETUP + ["mov r26 = r33", "mov r24 = 0"])
# The load, K empty groups, and the store of the same byte ("d") or the
# byte before ("p"): how far apart a load and a store to its byte must be.
for k in range(0, 6):
    for name, addr, setup in (("d", "r27", []), ("p", "r26", ["mov r26 = r33"])):
        STORES[f"mtf{name}_k{k}"] = (
            [B("mmi", "ld1 r28 = [r27]", "nop.m 0", "adds r29 = 1, r27 ;;")]
            + [GAP] * k
            + [B("mii", f"st1 [{addr}] = r25", "mov r25 = r28",
                 "mov r26 = r27"),
               B("mmi", "nop.m 0", "nop.m 0", "dep r27 = r29, r33, 0, 12 ;;")],
            MTF_SETUP + setup)
# A read-modify-write walk (a[i] += 1) of SZ-byte elements: the store of
# the incremented element ("d"), or of the element before ("p").
for sz in (1, 4, 8):
    for name, addr in (("d", "r27"), ("p", "r26")):
        STORES[f"rmw{name}_{sz}"] = ([
            B("mmi", f"ld{sz} r28 = [r27]", "nop.m 0",
              f"adds r29 = {sz}, r27 ;;"),
            B("mmi", "nop.m 0", "nop.m 0", "adds r25 = 1, r28 ;;"),
            B("mii", f"st{sz} [{addr}] = r25", "mov r26 = r27",
              "dep r27 = r29, r33, 0, 12 ;;")], MTF_SETUP + ["mov r26 = r33"])
# What separates mtfd (12 cycles) from rmwd (4.75): the store's data and
# where its register is written.
LD1 = B("mmi", "ld1 r28 = [r27]", "nop.m 0", "adds r29 = 1, r27 ;;")
STDEP = "dep r27 = r29, r33, 0, 12 ;;"
MTFX = {
    "war_out": [LD1, GAP,
                B("mii", "st1 [r27] = r25", "mov r26 = r27", STDEP),
                B("mmi", "nop.m 0", "nop.m 0", "mov r25 = r28 ;;")],
    "cur": [LD1, B("mmi", "nop.m 0", "nop.m 0", "mov r25 = r28 ;;"),
            B("mii", "st1 [r27] = r25", "mov r26 = r27", STDEP)],
    "staged": [LD1, B("mii", "nop.m 0", "mov r24 = r25", "mov r25 = r28 ;;"),
               B("mii", "st1 [r27] = r24", "mov r26 = r27", STDEP)],
    "const": [LD1, GAP, B("mii", "st1 [r27] = r0", "mov r26 = r27", STDEP)],
    "const7": [LD1, GAP, B("mii", "st1 [r27] = r23", "mov r26 = r27", STDEP)],
    "noload": [B("mmi", "nop.m 0", "nop.m 0", "adds r29 = 1, r27 ;;"), GAP,
               B("mii", "st1 [r27] = r25", "mov r26 = r27", STDEP)],
}
MTFX["stg2"] = [
    B("mii", "ld1 r28 = [r27]", "mov r24 = r25", "adds r29 = 1, r27 ;;"),
    B("mii", "st1 [r27] = r24", "mov r25 = r28", STDEP)]
MTFX["stg2b"] = [
    B("mii", "ld1 r28 = [r27]", "mov r26 = r27", "mov r24 = r25"),
    B("mmi", "nop.m 0", "nop.m 0", "dep r27 = r29, r33, 0, 12 ;;"),
    B("mii", "st1 [r26] = r24", "mov r25 = r28", "adds r29 = 1, r27 ;;")]
MTFX["stg3"] = [
    B("mii", "ld1 r28 = [r27]", "mov r24 = r25", "adds r29 = 1, r27 ;;"),
    GAP, B("mii", "st1 [r27] = r24", "mov r25 = r28", STDEP)]
MTFX["same2"] = [
    B("mmi", "ld1 r28 = [r27]", "st1 [r27] = r24", "adds r29 = 1, r27 ;;"),
    B("mii", "mov r24 = r25", "mov r25 = r28", STDEP)]
MTFX["cmp"] = [LD1, B("mmi", "nop.m 0", "nop.m 0",
                      "cmp.eq p6, p7 = r23, r28 ;;"),
               B("mii", "st1 [r27] = r25", "mov r25 = r28", STDEP)]
MTFX["cmp_same"] = [LD1, B("mii", "st1 [r27] = r25", "mov r25 = r28",
                           "cmp.eq p6, p7 = r23, r28"),
                    B("mmi", "nop.m 0", "nop.m 0", STDEP)]
for name, body in MTFX.items():
    STORES[f"mtfx_{name}"] = (body, MTF_SETUP + ["mov r24 = 0", "mov r23 = 7",
                                                 "adds r29 = 1, r33"])
# Variants of the clang shape: the pointer stepping STRIDE bytes, GAPS empty
# groups after, and the store SOFF bytes from the load.
for stride, gaps, soff in ((4, 0, 0), (8, 0, 0), (64, 0, 0), (1, 1, 0),
                           (1, 2, 0), (8, 1, 0), (1, 0, 128), (1, 0, 8),
                           (8, 0, 128), (0, 0, 0)):
    STORES[f"mtfv_s{stride}_g{gaps}_o{soff}"] = ([
        B("mmi", "ld1 r28 = [r27]", "st1 [r26] = r25",
          f"adds r29 = {stride}, r27 ;;"),
        B("mii", "nop.m 0", "mov r25 = r28", "dep r27 = r29, r33, 0, 12 ;;"),
        B("mmi", "nop.m 0", "nop.m 0", f"adds r26 = {soff}, r27 ;;")]
        + [GAP] * gaps, MTF_SETUP + [f"adds r26 = {soff}, r33"])
# One load and one store in an issue group, a fixed address apart: which
# pairs cost what (mtfv: about 10 cycles when they are the same byte).
# NAME: (first, second, setup); r27 = r33, r26 = r33 + SOFF.
PAIRS = {
    "ld1st1": ("ld1 r28 = [r27]", "st1 [r26] = r25"),
    "st1ld1": ("st1 [r26] = r25", "ld1 r28 = [r27]"),
    "ld8st8": ("ld8 r28 = [r27]", "st8 [r26] = r25"),
    "ld8st1": ("ld8 r28 = [r27]", "st1 [r26] = r25"),
    "ld1st8": ("ld1 r28 = [r27]", "st8 [r26] = r25"),
    "ld4st4": ("ld4 r28 = [r27]", "st4 [r26] = r25"),
}
for name, (a, b) in PAIRS.items():
    for soff in ((0, 1, 4, 8, 64, 4096) if name == "ld1st1" else (0, 8)):
        for use in (True, False):
            STORES[f"pair_{name}_o{soff}" + ("" if use else "_nouse")] = ([
                B("mmi", a, b, "nop.i 0 ;;"),
                B("mii", "nop.m 0", "mov r25 = r28" if use else "nop.i 0",
                  "nop.i 0 ;;")],
                ["mov r27 = r33", f"movl r18 = {soff}", "add r26 = r33, r18",
                 "mov r25 = 0"])
# ld8 then st8 in a group, the store SOFF bytes on: which address bits
# decide the cost.
for soff in (16, 24, 32, 40, 48, 56, 72, 128, 192, 256, 512, 1024, 2048,
             4160, 8192, 16384, 65536, 1 << 20):
    STORES[f"pair_ld8st8_o{soff}"] = ([
        B("mmi", "ld8 r28 = [r27]", "st8 [r26] = r25", "nop.i 0 ;;"),
        B("mii", "nop.m 0", "mov r25 = r28", "nop.i 0 ;;")],
        ["mov r27 = r33", f"movl r18 = {soff}", "add r26 = r33, r18",
         "mov r25 = 0"])
STORES["pair_ld8_only"] = ([B("mmi", "ld8 r28 = [r27]", "nop.m 0",
                              "nop.i 0 ;;"),
                            B("mii", "nop.m 0", "mov r25 = r28",
                              "nop.i 0 ;;")], ["mov r27 = r33"])
STORES["pair_st8_only"] = ([B("mmi", "nop.m 0", "st8 [r26] = r25",
                              "nop.i 0 ;;"),
                            B("mii", "nop.m 0", "nop.i 0", "nop.i 0 ;;")],
                           ["mov r26 = r33", "mov r25 = 0"])
# The same shapes with the line first brought into L1D (a load and a use),
# or flushed from every cache (fc): does the cost depend on where the line
# starts?
WARM = ["ld8 r30 = [r33] ;;", "add r30 = r30, r0 ;;"]
COLD = ["fc r33 ;;", "sync.i ;;", "srlz.d ;;"]
for state, pre in (("warm", WARM), ("cold", COLD)):
    STORES[f"bi_pair_ld8st8_o0_{state}"] = ([
        B("mmi", "ld8 r28 = [r27]", "st8 [r26] = r25", "nop.i 0 ;;"),
        B("mii", "nop.m 0", "mov r25 = r28", "nop.i 0 ;;")],
        ["mov r27 = r33", "mov r26 = r33", "mov r25 = 0"] + pre)
    STORES[f"bi_pair_ld8st8_o16_{state}"] = ([
        B("mmi", "ld8 r28 = [r27]", "st8 [r26] = r25", "nop.i 0 ;;"),
        B("mii", "nop.m 0", "mov r25 = r28", "nop.i 0 ;;")],
        ["mov r27 = r33", "adds r26 = 16, r33", "mov r25 = 0"] + pre)
    STORES[f"bi_stld_next_{state}"] = ([
        one("m", "st8 [r33] = r34"), one("m", "ld8 r34 = [r36]")],
        ["adds r36 = 8, r33"] + pre)
    STORES[f"bi_st1ld1_0_8_{state}"] = ([
        one("m", "st1 [r33] = r34"), one("m", "ld1 r34 = [r36]")],
        ["adds r36 = 8, r33"] + pre)
    STORES[f"bi_st1ld1_0_64_{state}"] = ([
        one("m", "st1 [r33] = r34"), one("m", "ld1 r34 = [r36]")],
        ["adds r36 = 64, r33"] + pre)
    STORES[f"bi_st8ld8_0_64_{state}"] = ([
        one("m", "st8 [r33] = r34"), one("m", "ld8 r34 = [r36]")],
        ["adds r36 = 64, r33"] + pre)
# bi_stld_next_cold and st_pair_st1ld1 with the load D bytes past the store:
# in its 16-byte L2 bank (bits 7:4), another bank of its line, or its bank
# in another line.  Which match makes a cold load wait?
for d in (8, 16, 32, 64, 128, 256, 4096):
    STORES[f"stld_d{d}_cold"] = ([
        one("m", "st8 [r33] = r34"), one("m", "ld8 r34 = [r36]")],
        ["movl r18 = %d" % d, "add r36 = r33, r18"] + COLD)
    STORES[f"stld_same_d{d}_cold"] = ([
        B("mmi", "st8 [r33] = r34", "ld8 r34 = [r36]", "nop.i 0 ;;")],
        ["movl r18 = %d" % d, "add r36 = r33, r18"] + COLD)

# A store then, K-1 groups on, a load of the next word of the line, with
# the line cold (flushed) or warm: how far apart they must be for the load's
# fill to survive the next store.
for k in (1, 2, 3, 4, 6, 8, 12):
    for state, pre in (("warm", WARM), ("cold", COLD)):
        STORES[f"bi_stld8_k{k}_{state}"] = (
            [one("m", "st8 [r33] = r34")] + [GAP] * (k - 1)
            + [one("m", "ld8 r34 = [r36]")], ["adds r36 = 8, r33"] + pre)
# A load, K-1 groups, then a store to the next word: the order the manual
# describes (6.6.3.2), without a use of the load in between.
for k in (1, 2, 4, 6, 8):
    STORES[f"bi_ldst8_k{k}_cold"] = (
        [one("m", "ld8 r34 = [r36]")] + [GAP] * (k - 1)
        + [one("m", "st8 [r33] = r40")], ["adds r36 = 8, r33", "mov r40 = 0"]
        + COLD)

# bi_ldst8_k1_cold without the fc: a load, then a store a group later to the
# word above or below it, in r33's line or L lines on.  Does the store's
# direction or the line's L1D set decide whether the line stays cold?
for line in (0, 1, 2, 3, 8):
    for name, ld, sto in (("up", 0, 8), ("down", 8, 0)):
        STORES[f"ldst_order_l{line}_{name}"] = (
            [one("m", "ld8 r45 = [r36]"), one("m", "st8 [r37] = r40")],
            [f"adds r36 = {64 * line + ld}, r33",
             f"adds r37 = {64 * line + sto}, r33"])

# sha_stream's byte reverse as clang schedules it: ld1 each byte b of a
# doubleword through its own post-incremented pointer (r80+b), st1 it to
# its swap partner in the same 32-bit half (r88+b) a group or a body later.
# The middle group stores two pairs and then loads the other two.  Through
# a stream of new doublewords; through one 64-byte block, eight doublewords
# a body; and through that block after eight st8s rewrite it, as
# sha_update's memcpy does before each byte_reverse.
def brev_element(loads=True, stores=True):
    L = [f"r{80 + b}" for b in range(8)]
    S = [f"r{88 + b}" for b in range(8)]
    D = [f"r{60 + b}" for b in range(8)]
    ld = lambda b: f"ld1 {D[b]} = [{L[b]}], 8" if loads else None
    sto = lambda b: f"st1 [{S[b]}] = {D[b]}, 8" if stores else None
    return [B("mmi", sto(2), sto(1), "nop.i 0"),
            B("mmi", ld(7), ld(4), "nop.i 0"),
            B("mmi", ld(6), ld(5), "nop.i 0 ;;"),
            B("mmi", sto(4), sto(7), "nop.i 0"),
            B("mmi", ld(3), ld(0), "nop.i 0"),
            B("mmi", sto(6), sto(5), "nop.i 0"),
            B("mmi", ld(2), ld(1), "nop.i 0 ;;"),
            B("mmi", sto(3), sto(0), "nop.i 0 ;;")]


BSWAP = {0: 3, 1: 2, 2: 1, 3: 0, 4: 7, 5: 6, 6: 5, 7: 4}
BREV_SETUP = ([f"adds r{80 + b} = {64 + b}, r34" for b in range(8)]
              + [f"adds r{88 + b} = {64 + BSWAP[b] - (8 if b in (1, 2) else 0)}"
                 ", r34" for b in range(8)])
BREV_RESET = [B("mii", f"adds r{80 + i} = -64, r{80 + i}",
                f"adds r{81 + i} = -64, r{81 + i}",
                f"adds r{82 + i} = -64, r{82 + i}" + (" ;;" if i % 6 == 3
                                                       else ""))
              for i in range(0, 15, 3)] + [
    B("mii", "adds r95 = -64, r95", None, "nop.i 0 ;;")]
BREV_COPY = [B("mmi", f"st8 [r{96 + i}] = r40", f"st8 [r{97 + i}] = r40",
               "nop.i 0 ;;") for i in range(0, 8, 2)]
STORES["brev_stream"] = (brev_element(), BREV_SETUP)
# Its loads alone, its stores alone, and both with the loads 32 KB away.
STORES["brev_stream_ld"] = (brev_element(stores=False), BREV_SETUP)
STORES["brev_stream_st"] = (brev_element(loads=False), BREV_SETUP)
STORES["brev_stream_far"] = (brev_element(), [
    "movl r18 = 0x8000", "add r19 = r34, r18"] + [
    l.replace(", r34", ", r19") for l in BREV_SETUP[:8]] + BREV_SETUP[8:])
# Both, with the loads K doublewords ahead of the stores: in the doubleword
# after, another 16-byte bank, or the next line.
for k in (1, 2, 4, 8, 16):
    STORES[f"brev_stream_a{k}"] = (brev_element(), [
        f"adds r{80 + b} = {64 + 8 * k + b}, r34" for b in range(8)]
        + BREV_SETUP[8:])
# One doubleword's eight st1s two a group, and one a group, in brev's order
# and ascending; and eight st1s to eight doublewords' byte 0.
BREV_ORDER = [2, 1, 4, 7, 6, 5, 3, 0]
for name, order in (("brev", BREV_ORDER), ("up", list(range(8)))):
    STORES[f"w8_{name}_two"] = ([
        B("mmi", f"st1 [r{88 + order[i]}] = r40, 8",
          f"st1 [r{88 + order[i + 1]}] = r40, 8", "nop.i 0 ;;")
        for i in range(0, 8, 2)], BREV_SETUP)
    STORES[f"w8_{name}_one"] = ([one("m", f"st1 [r{88 + b}] = r40, 8")
                                 for b in order], BREV_SETUP)
STORES["w8_spread"] = ([one("m", f"st1 [r{88 + b}] = r40, 64")
                        for b in range(8)],
                       [f"adds r{88 + b} = {8 * b}, r34" for b in range(8)])
# sha's byte reverse over 32-bit words, ld1 and st1 through one pointer
# per access advancing 4: each group lists (op, byte, lead), a load of
# byte b of the word lead words ahead or a store to byte b of this word of
# the byte 3 - b loaded earlier.  "icc" is icc 11.1's kernel, a load and a
# store a group, two of the loads a word ahead; "p2_aK" stores a word two
# bytes a group and loads the word K ahead.  sha_info->data is 4 mod 8,
# so each doubleword holds two words; "_o0" starts at 0 mod 8.
BREVW_SHAPES = {
    "icc": [[("st", 0, 0), ("ld", 1, 0)], [("ld", 0, 1), ("st", 3, 0)],
            [("ld", 2, 0), ("st", 2, 0)], [("ld", 3, 1), ("st", 1, 0)]],
    "icc_nodep": [[("st", 0, 0), ("ld", 1, 1)], [("ld", 0, 1), ("st", 3, 0)],
                  [("ld", 2, 1), ("st", 2, 0)], [("ld", 3, 1), ("st", 1, 0)]],
}
for k in (1, 2, 4):
    BREVW_SHAPES[f"p2_a{k}"] = [
        [("st", 0, 0), ("st", 1, 0), ("ld", 0, k), ("ld", 1, k)],
        [("st", 2, 0), ("st", 3, 0), ("ld", 2, k), ("ld", 3, k)]]
    BREVW_SHAPES[f"p4_a{k}"] = [
        [("st", 0, 0), ("st", 1, 0), ("ld", 0, k), ("ld", 1, k)],
        [("st", 2, 0), ("st", 3, 0), ("ld", 2, k), ("ld", 3, k)],
        [], []]


def brevw(shape, words=4):
    ops = [op for g in shape for op in g]
    ptr = {i: f"r{80 + i}" for i in range(len(ops))}
    val = lambda b, w: f"r{56 + 4 * (w % 4) + b}"
    body = []
    for w in range(words):
        i = 0
        for g in shape:
            ins = []
            for op, b, lead in g:
                if op == "ld":
                    ins.append(f"ld1 {val(b, w + lead)} = [{ptr[i]}], 4")
                else:
                    ins.append(f"st1 [{ptr[i]}] = {val(3 - b, w)}, 4")
                i += 1
            ins += [None] * (4 - len(ins))
            body += [B("mmi", ins[0], ins[1], "nop.i 0"),
                     B("mmi", ins[2], ins[3], "nop.i 0 ;;")]
    return body


def brevw_setup(shape, off):
    ops = [op for g in shape for op in g]
    return [f"adds r{80 + i} = {64 + off + b + 4 * (lead if op == 'ld' else 0)}"
            ", r34" for i, (op, b, lead) in enumerate(ops)]


for name, shape in BREVW_SHAPES.items():
    for off, sfx in ((4, ""), (0, "_o0")):
        STORES[f"brevw_{name}{sfx}"] = (brevw(shape),
                                       brevw_setup(shape, off))

# Eight doublewords a body: too long for STORES' eight bodies.
BLOCKS = {"brev_block": (brev_element() * 8 + BREV_RESET, BREV_SETUP)}
BLOCKS["brev_block_copy"] = (BREV_COPY + brev_element() * 8 + BREV_RESET,
                             BREV_SETUP + [f"adds r{96 + i} = {64 + 8 * i}, r34"
                                           for i in range(8)])


# The same through one 64-byte block, sixteen words a body: alone, after
# eight st8s rewrite it, and after those and sixteen ld4s read it, as
# sha_update's memcpy and sha_transform do around each byte_reverse.
def brevw_reset(n):
    rs = [f"adds r{80 + i} = -64, r{80 + i}" for i in range(n)]
    rs += [None] * (-len(rs) % 3)
    return [B("mii", rs[i], rs[i + 1], rs[i + 2] or "nop.i 0")
            for i in range(0, len(rs), 3)] + [B("mii", None, None,
                                                  "nop.i 0 ;;")]


BREVW_READ = [B("mmi", f"ld4 r{104 + (i % 4)} = [r{96 + (i // 2) % 8}]",
                None, "nop.i 0 ;;") for i in range(16)]
for name, shape in BREVW_SHAPES.items():
    n = sum(len(g) for g in shape)
    for pre, sfx in (([], ""), (BREV_COPY, "_copy"),
                     (BREV_COPY + BREVW_READ, "_copyrd")):
        BLOCKS[f"brevwb_{name}{sfx}"] = (
            pre + brevw(shape, 16) + brevw_reset(n),
            brevw_setup(shape, 4) + [f"adds r{96 + i} = {64 + 8 * i}, r34"
                                     for i in range(8)])


# clang's 32-byte wildcopy loop from zstd's ZSTD_decompressSequences, its
# bundles as compiled, branches made nops: two 16-byte halves of ld1/st1,
# dst1 r50, src2 r51 (src1 is r51 - 16), dst2 r52, each advancing 32.
ZCOPY32 = [
    B('mmf', 'adds r53 = -16, r51', 'mov r8 = r50', 'nop.f 0'),
    B('mii', 'adds r3 = 11, r50', 'adds r9 = 14, r50', 'adds r10 = 13, r50 ;;'),
    B('mii', 'sub r18 = r50, r53', 'adds r19 = 7, r53', 'adds r23 = 8, r53'),
    B('mii', 'adds r24 = 9, r53', 'adds r20 = 6, r53', 'adds r25 = 10, r53'),
    B('mmi', 'adds r26 = 11, r53 ;;', 'adds r30 = 15, r18', 'adds r27 = 12, r53'),
    B('mfi', 'adds r21 = 5, r53', 'nop.f 0', 'adds r28 = 13, r53'),
    B('mii', 'adds r29 = 14, r53', 'adds r22 = 4, r53', 'adds r18 = 1, r53 ;;'),
    B('mib', 'cmp.ltu p7, p6 = 30, r30', 'nop.i 0', 'nop.b 0'),
    B('mmi', 'mov r30 = r53', 'ld1 r26 = [r26]', 'adds r11 = 12, r50 ;;'),
    B('mmi', 'ld1 r31 = [r30], 15', 'ld1 r29 = [r29]', 'nop.i 0'),
    B('mii', 'st1 [r3] = r26', 'adds r3 = 3, r53', 'adds r53 = 2, r53 ;;'),
    B('mmi', 'st1 [r8] = r31, 15', 'ld1 r30 = [r30]', 'adds r26 = 10, r50'),
    B('mmi', 'ld1 r28 = [r28]', 'st1 [r9] = r29', 'adds r29 = 4, r50 ;;'),
    B('mmi', 'adds r9 = 7, r50', 'st1 [r8] = r30', 'adds r8 = 9, r50'),
    B('mmi', 'ld1 r25 = [r25]', 'ld1 r24 = [r24]', 'adds r30 = 8, r50'),
    B('mmi', 'ld1 r27 = [r27]', 'st1 [r10] = r28', 'adds r10 = 6, r50 ;;'),
    B('mmi', 'st1 [r11] = r27', 'ld1 r23 = [r23]', 'nop.i 0'),
    B('mmi', 'st1 [r26] = r25', 'ld1 r19 = [r19]', 'adds r25 = 5, r50 ;;'),
    B('mmi', 'st1 [r8] = r24', 'st1 [r30] = r23', 'adds r30 = 3, r50'),
    B('mmi', 'ld1 r20 = [r20]', 'ld1 r24 = [r21]', 'adds r21 = 2, r50'),
    B('mmf', 'ld1 r11 = [r22]', 'ld1 r22 = [r3]', 'nop.f 0'),
    B('mmi', 'adds r3 = 1, r50', 'st1 [r9] = r19', 'nop.i 0 ;;'),
    B('mmi', 'st1 [r10] = r20', 'ld1 r53 = [r53]', 'nop.i 0'),
    B('mmi', 'ld1 r18 = [r18]', 'st1 [r25] = r24', 'nop.i 0'),
    B('mmi', 'st1 [r29] = r11', 'st1 [r30] = r22', 'nop.i 0 ;;'),
    B('mmi', 'st1 [r21] = r53', 'st1 [r3] = r18', 'nop.i 0'),
    B('mii', 'sub r53 = r52, r51', 'adds r22 = 8, r51', 'adds r23 = 9, r51'),
    B('mii', 'adds r19 = 6, r51', 'adds r24 = 10, r51', 'adds r25 = 11, r51 ;;'),
    B('mmi', 'adds r18 = 7, r51', 'adds r29 = 15, r53', 'adds r26 = 12, r51'),
    B('mmi', 'adds r20 = 5, r51', 'adds r27 = 13, r51', 'adds r28 = 14, r51'),
    B('mmi', 'adds r21 = 4, r51', 'adds r53 = 1, r51', 'mov r31 = r52'),
    B('mmi', 'adds r8 = 14, r52', 'adds r11 = 11, r52', 'adds r9 = 13, r52 ;;'),
    B('mib', 'cmp.ltu p7, p6 = 30, r29', 'nop.i 0', 'nop.b 0'),
    B('mmi', 'mov r29 = r51', 'ld1 r28 = [r28]', 'adds r10 = 12, r52'),
    B('mii', 'adds r3 = 10, r52', 'adds r50 = 32, r50 ;;', 'nop.i 0'),
    B('mmi', 'ld1 r30 = [r29], 15', 'ld1 r25 = [r25]', 'nop.i 0'),
    B('mii', 'st1 [r8] = r28', 'adds r28 = 3, r51', 'adds r8 = 6, r52 ;;'),
    B('mmi', 'st1 [r31] = r30, 15', 'ld1 r29 = [r29]', 'adds r30 = 9, r52'),
    B('mii', 'st1 [r11] = r25', 'adds r25 = 8, r52', 'adds r11 = 2, r51'),
    B('mmi', 'ld1 r27 = [r27]', 'ld1 r26 = [r26]', 'nop.i 0 ;;'),
    B('mii', 'st1 [r31] = r29', 'adds r29 = 7, r52', 'adds r51 = 32, r51'),
    B('mmi', 'ld1 r24 = [r24]', 'ld1 r23 = [r23]', 'nop.i 0'),
    B('mmi', 'st1 [r9] = r27', 'st1 [r10] = r26', 'adds r10 = 4, r52'),
    B('mmi', 'ld1 r22 = [r22] ;;', 'st1 [r3] = r24', 'adds r24 = 5, r52'),
    B('mmi', 'st1 [r30] = r23', 'ld1 r18 = [r18]', 'adds r23 = 1, r52'),
    B('mmi', 'st1 [r25] = r22', 'ld1 r19 = [r19]', 'nop.i 0'),
    B('mii', 'ld1 r31 = [r20]', 'adds r20 = 3, r52', 'adds r22 = 2, r52'),
    B('mmi', 'adds r52 = 32, r52', 'ld1 r27 = [r21]', 'nop.i 0'),
    B('mmi', 'ld1 r21 = [r28] ;;', 'st1 [r29] = r18', 'nop.i 0'),
    B('mmi', 'st1 [r8] = r19', 'ld1 r28 = [r11]', 'nop.i 0'),
    B('mmi', 'ld1 r53 = [r53]', 'st1 [r24] = r31', 'nop.i 0'),
    B('mmi', 'st1 [r10] = r27', 'st1 [r20] = r21', 'nop.i 0 ;;'),
    B('mmb', 'st1 [r22] = r28', 'st1 [r23] = r53', 'nop.b 0')]
for off in (16, 32, 64, 4096, 524288):
    BLOCKS[f"zcopy32_o{off}"] = (ZCOPY32, [
        "adds r51 = 16, r34", f"movl r19 = {off}", "add r50 = r34, r19",
        "add r52 = r34, r19 ;;", "adds r52 = 16, r52"])


# The L1 DTLB: a chain of loads, each address waiting on the load before,
# over N lines STEP bytes apart, each in its own L1D set.  With more pages
# than the L1 DTLB's 32 entries, each load misses it and so L1D.
def dtlb_chain(n, step=4096):
    body = []
    for k in range(n):
        nxt = 60 + (k + 1) % n
        body += [B("mmi", f"ld8 r45 = [r{60 + k}]", None, "nop.i 0 ;;"),
                 B("mii", None, f"add r{nxt} = r{nxt}, r45", "nop.i 0 ;;")]
    setup = []
    for k in range(n):
        setup += [f"movl r19 = {k * step + (k % 64) * 64}",
                  f"add r{60 + k} = r34, r19 ;;"]
    return body, setup


# The same as zlgm with C dependent L1D loads, 2 cycles each, between the
# stores and the loads of their line: how long the line takes to come from
# memory.  4224 bytes apart, 17 MB a call.
for c in (25, 50, 100, 150, 200, 240):
    BLOCKS[f"zlgmc_c{c}"] = ([
        one("m", "st8 [r34] = r40, 8"), one("m", "st8 [r34] = r40, 8"),
        B("mii", None, "adds r37 = -2, r34", "adds r38 = -16, r34 ;;")] +
        [one("m", "ld8 r52 = [r52]")] * c + [
        one("m", "ld1 r44 = [r37]"), one("m", "ld1 r45 = [r38]"),
        B("mmi", "st1 [r34] = r45", "ld1 r44 = [r37]",
          "adds r39 = 14, r34 ;;"),
        B("mmi", "st1 [r39] = r44", None, "adds r34 = 4208, r34 ;;")],
        ["mov r52 = r33"])

for n in (8, 16, 24, 28, 30, 31, 32, 33, 34, 36, 40, 48, 64):
    BLOCKS[f"dtlb_chain_p{n}"] = dtlb_chain(n)
for n in (16, 31, 32, 33, 34, 48, 64):
    for step in (16384, 65536):
        BLOCKS[f"dtlb_chain_s{step // 1024}k_p{n}"] = dtlb_chain(n, step)

# glibc's memcpy tail, its byte loop as built (st1 and the next ld1 in one
# group, br.cloop) for N bytes from r60 to r61, the harness's ar.lc kept in
# r62.  "stream" moves both on by N each body; "warm" copies the same bytes.
def mcpy(n, step):
    return [B("mii", None, "mov r62 = ar.lc", f"adds r22 = {n - 1}, r0 ;;"),
            B("mii", "ld1 r23 = [r60], 1", "mov.i ar.lc = r22",
              "cmp.eq p0, p6 = 0, r22 ;;"),
            B("mmi", "st1 [r61] = r23, 1", "(p6) ld1 r23 = [r60], 1",
              "nop.i 0", label="1"),
            B("mib", "cmp.lt p6, p0 = 1, r22", "adds r22 = -1, r22",
              "br.cloop.dptk.few 1b ;;"),
            B("mii", f"adds r60 = {step - n}, r60",
              f"adds r61 = {step - n}, r61", "mov.i ar.lc = r62 ;;")]


for n in (1, 4, 8, 16, 32, 64):
    for kind, step in (("stream", n), ("warm", 0)):
        BLOCKS[f"mcpy_n{n}_{kind}"] = (mcpy(n, step), [
            "mov r60 = r34", "movl r19 = 0x100000",
            "add r61 = r34, r19"])

# MiBench aes's CBC step in decfile, clang's bundles as built, branches made
# nops: xor the 16 bytes decrypt wrote at r43 into the chaining block at
# r37 + 0..15 (r37, r38, r35, r47-r58, r60 hold its byte addresses), a
# byte at a time.  "dx" first stores r43's 16 bytes in decrypt's order.
CBCX = [
    B('mmi', 'adds r23 = 4, r43', 'adds r24 = 5, r43', 'adds r27 = 6, r43'),
    B('mmi', 'adds r28 = 7, r43', 'adds r10 = 9, r43', 'mov r70 = r43 ;;'),
    B('mmi', 'ld1 r23 = [r23]', 'ld1 r25 = [r48]', 'adds r20 = 3, r43'),
    B('mmi', 'ld1 r24 = [r24]', 'ld1 r26 = [r49]', 'adds r8 = 8, r43'),
    B('mmi', 'adds r72 = 1, r43', 'adds r19 = 2, r43', 'adds r73 = 11, r43 ;;'),
    B('mmi', 'xor r18 = r25, r23', 'adds r23 = 10, r43', 'mov r64 = r43'),
    B('mmi', 'ld1 r27 = [r27]', 'ld1 r29 = [r50]', 'xor r21 = r26, r24 ;;'),
    B('mmi', 'st1 [r48] = r18', 'ld1 r30 = [r28]', 'adds r24 = 13, r43'),
    B('mmi', 'ld1 r31 = [r51]', 'st1 [r49] = r21', 'mov r65 = r42'),
    B('mmi', 'ld1 r3 = [r10]', 'ld1 r18 = [r53]', 'xor r11 = r29, r27'),
    B('mmi', 'mov r66 = r40', 'mov r67 = r74', 'mov r75 = r46 ;;'),
    B('mmi', 'xor r26 = r31, r30', 'adds r31 = 12, r43', 'adds r30 = 14, r43'),
    B('mmi', 'ld1 r71 = [r70], 15', 'ld1 r20 = [r20]', 'xor r29 = r18, r3 ;;'),
    B('mmi', 'st1 [r51] = r26', 'ld1 r22 = [r47]', 'nop.i 0'),
    B('mmi', 'ld1 r10 = [r37]', 'st1 [r50] = r11', 'nop.i 0'),
    B('mmi', 'st1 [r53] = r29', 'ld1 r9 = [r8]', 'nop.i 0'),
    B('mmi', 'ld1 r21 = [r52] ;;', 'xor r20 = r22, r20', 'xor r71 = r10, r71'),
    B('mmi', 'ld1 r72 = [r72]', 'ld1 r3 = [r38]', 'xor r26 = r21, r9 ;;'),
    B('mmi', 'st1 [r47] = r20', 'ld1 r19 = [r19]', 'nop.i 0'),
    B('mii', 'ld1 r22 = [r35]', 'xor r72 = r3, r72', 'nop.i 0'),
    B('mmi', 'st1 [r37] = r71', 'st1 [r52] = r26', 'nop.i 0'),
    B('mmi', 'ld1 r27 = [r73]', 'ld1 r8 = [r31]', 'nop.i 0 ;;'),
    B('mmi', 'st1 [r38] = r72', 'ld1 r31 = [r23]', 'xor r19 = r22, r19'),
    B('mmi', 'ld1 r73 = [r54] ;;', 'st1 [r35] = r19', 'nop.i 0'),
    B('mmi', 'ld1 r20 = [r55]', 'ld1 r18 = [r56]', 'xor r23 = r73, r31'),
    B('mmi', 'ld1 r71 = [r24]', 'ld1 r21 = [r57]', 'nop.i 0 ;;'),
    B('mmi', 'xor r73 = r18, r8', 'st1 [r54] = r23', 'xor r72 = r20, r27'),
    B('mmi', 'ld1 r28 = [r30]', 'ld1 r19 = [r58]', 'xor r18 = r21, r71 ;;'),
    B('mmi', 'st1 [r55] = r72', 'st1 [r56] = r73', 'nop.i 0'),
    B('mmi', 'ld1 r70 = [r70]', 'ld1 r71 = [r60]', 'xor r72 = r19, r28 ;;'),
    B('mmi', 'st1 [r57] = r18', 'st1 [r58] = r72', 'xor r70 = r71, r70 ;;'),
    B('mib', 'st1 [r60] = r70', 'nop.i 0', 'nop.b 0')]
CBC_A = {0: 37, 1: 38, 2: 35, 3: 47, 4: 48, 5: 49, 6: 50, 7: 51, 8: 52,
         9: 53, 10: 54, 11: 55, 12: 56, 13: 57, 14: 58, 15: 60}
CBC_SETUP = ["adds r43 = 1024, r33", "adds r37 = 1280, r33"] + [
    f"adds r{r} = {k}, r37" for k, r in CBC_A.items() if k]
DECRYPT_ORDER = [1, 5, 7, 9, 6, 8, 3, 0, 2, 12, 13, 4, 10, 11, 14, 15]
BLOCKS["cbc_x"] = (CBCX, CBC_SETUP)
BLOCKS["cbc_dx"] = ([B("mii", None, f"adds r{100 + i} = {k}, r43",
                       "nop.i 0 ;;" if i % 4 == 3 else "nop.i 0")
                     for i, k in enumerate(DECRYPT_ORDER)] +
                    [one("m", f"st1 [r{100 + i}] = r40") for i in range(16)] +
                    CBCX, CBC_SETUP)

# Advanced loads and checks: ld8.a, a store SOFF bytes on (0 kills the
# ALAT entry), then ld8.c.clr, its value feeding the next store; "plain"
# uses ordinary loads.
for soff in (0, 8, 72, 128, 4096, 4100, 4104, 8192, 16384, 32768, 65536,
             1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 24):
    for kind, la, lc in (("alat", "ld8.a", "ld8.c.clr"), ("plain", "ld8", "ld8")):
        STORES[f"alat_{kind}_o{soff}"] = ([
            one("m", f"{la} r34 = [r33]"), one("m", "st8 [r36] = r34"),
            one("m", f"{lc} r34 = [r33]")],
            [f"movl r18 = {soff}", "add r36 = r33, r18"] + WARM)
# One store a cycle to consecutive addresses, SIZE bytes each and STRIDE
# apart, or one every GAP+1 cycles: does a store wait for one to the same
# word a cycle earlier when they write different bytes of it?
for size in (1, 2, 4, 8):
    for stride in sorted({size, 8}):
        for gap in (0, 1):
            STORES[f"seqst{size}_s{stride}_g{gap}"] = (
                [one("m", f"st{size} [r34] = r40, {stride}")] + [GAP] * gap,
                [])
# How long a sub-word store keeps a later one to its word waiting, and
# stores that write the same bytes again (stride 0).
for size, stride in ((1, 1), (1, 0), (2, 0), (4, 0), (8, 0)):
    for gap in (2, 3, 4, 5, 6):
        if stride == 0 and gap > 2:
            continue
        STORES[f"seqst{size}_s{stride}_g{gap}"] = (
            [one("m", f"st{size} [r34] = r40, {stride}")] + [GAP] * gap, [])
for size in (1, 2, 4):
    STORES[f"seqst{size}_s0_g0"] = (
        [one("m", f"st{size} [r34] = r40")], [])
# A byte store and then a doubleword store to its word, and the reverse.
STORES["st1st8"] = ([one("m", "st1 [r34] = r40"), one("m", "st8 [r34] = r40, 8")], [])
STORES["st8st1"] = ([one("m", "st8 [r34] = r40"), one("m", "st1 [r34] = r40, 8")], [])
# glibc's byte copy loop: each group stores the byte the previous one
# loaded and loads the next.
STORES["copy1"] = ([B("mmi", "st1 [r34] = r40, 1", "ld1 r40 = [r27], 1",
                      "nop.i 0 ;;")],
                   ["movl r18 = 0x80000", "add r27 = r33, r18"])
# Which half of the copy costs: the byte loads alone, the loads with
# doubleword stores, the copy from a source 64 bytes behind the
# destination (zstd's overlapping match copy), and the loads not feeding the
# stores.
STORES["copy1_ldonly"] = ([one("m", "ld1 r40 = [r27], 1")],
                          ["movl r18 = 0x80000", "add r27 = r33, r18"])
STORES["copy1_st8"] = ([B("mmi", "st8 [r34] = r40, 8", "ld1 r40 = [r27], 1",
                          "nop.i 0 ;;")],
                       ["movl r18 = 0x80000", "add r27 = r33, r18"])
STORES["copy1_behind64"] = ([B("mmi", "st1 [r34] = r40, 1",
                               "ld1 r40 = [r27], 1", "nop.i 0 ;;")],
                            ["adds r27 = -64, r34"])
STORES["copy1_nodep"] = ([B("mmi", "st1 [r34] = r41, 1",
                            "ld1 r40 = [r27], 1", "nop.i 0 ;;")],
                         ["movl r18 = 0x80000", "add r27 = r33, r18",
                          "mov r41 = 0"])
# The far copy with the source the given distance from the destination: is
# the cost an L2 bank conflict (the same address bits 7:4)?
for off in (8, 16, 32, 64, 128, 256, 4096, 0x80000 + 16, 0x80000 + 64,
            0x80000 + 128):
    STORES[f"copy1_d{off}"] = ([B("mmi", "st1 [r34] = r40, 1",
                                  "ld1 r40 = [r27], 1", "nop.i 0 ;;")],
                               [f"movl r18 = {off}", "add r27 = r34, r18"])
# A store and a load 4 KB on (the same address bits 11:0), in one group or
# the load a group later, both pointers stepping SIZE: the false conflict.
for size in (1, 8):
    for gap in (0, 1, 2):
        STORES[f"alias{size}_g{gap}"] = (
            [B("mmi", f"st{size} [r34] = r41, {size}",
               None if gap else f"ld{size} r40 = [r27], {size}",
               "nop.i 0 ;;")] + [GAP] * (gap - 1 if gap else 0)
            + ([one("m", f"ld{size} r40 = [r27], {size}")] if gap else []),
            ["movl r18 = 4096", "add r27 = r34, r18", "mov r41 = 0"])
        STORES[f"alias{size}_g{gap}_d4160"] = (
            STORES[f"alias{size}_g{gap}"][0],
            ["movl r18 = 4160", "add r27 = r34, r18", "mov r41 = 0"])
# Short copies, as zstd makes them (about 16 bytes a call): RUN copy groups,
# then GAPS empty groups standing for the call around them.  Do the first
# stores of a run go at one a cycle?
for run in (2, 4, 8, 16, 32):
    for gaps in (4, 16):
        STORES[f"copyrun{run}_g{gaps}"] = (
            [B("mmi", "st1 [r34] = r40, 1", "ld1 r40 = [r27], 1",
               "nop.i 0 ;;")] * run + [GAP] * gaps,
            ["movl r18 = 0x80000", "add r27 = r33, r18"])
        STORES[f"strun{run}_g{gaps}"] = (
            [one("m", "st1 [r34] = r40, 1")] * run + [GAP] * gaps, [])
# How many conflicting sub-word stores the pipeline queues before it
# stalls: RUN st1s to the bytes of one word, then 48 groups the queue can
# drain under, the first of them stepping to the next word.
for run in range(2, 9):
    STORES[f"stq{run}"] = (
        [one("m", "st1 [r34] = r40, 1")] * run
        + [B("mmi", None, None, f"adds r34 = {8 - run}, r34 ;;")]
        + [GAP] * 47, [])
# The strun4_g16 drain with loads in the groups after the stores: from a
# warm line elsewhere (r33), from the stored line's next word, and an
# integer op standing for the rest.  Does a load wait for the queue?
for run in (4, 8):
    for kind, text in (("ldfar", "ld8 r41 = [r33]"),
                       ("ldline", "ld8 r41 = [r35]"),
                       ("ld1far", "ld1 r41 = [r33]"),
                       ("add", None)):
        g = (one("m", text) if text else
             B("mmi", None, None, "add r41 = r40, r40 ;;"))
        STORES[f"strun{run}_{kind}"] = (
            [one("m", "st1 [r34] = r40, 1")] * run + [g] * 16,
            ["adds r35 = 8, r34"] + WARM)
# aes's in-place byte loop (decfile): load a byte, then store it back a
# group after the next, so the next load reads the word the store is
# queued for.  _far loads from a stream 64 bytes on instead; _dep makes the
# store's data the load's.
for name, ld, setup in (("same", "ld1 r40 = [r34], 1", []),
                        ("far", "ld1 r40 = [r27], 1", ["adds r27 = 64, r34"])):
    for dep in ("", "_dep"):
        STORES[f"inplace_{name}{dep}"] = (
            [one("m", ld), B("mmi", None, None, "adds r41 = -1, r34 ;;"
                             if name == "same" else "nop.i 0 ;;"),
             one("m", (f"st1 [r41] = {'r40' if dep else 'r42'}"
                       if name == "same" else
                       f"st1 [r34] = {'r40' if dep else 'r42'}, 1"))],
            setup + ["mov r42 = 0"])
# The same in runs of RUN bytes, then 16 empty groups, as decfile's
# 16-byte blocks between calls.
for run in (1, 2, 4, 8, 16):
    STORES[f"inplace_run{run}"] = (
        [one("m", "ld1 r40 = [r34], 1"),
         B("mmi", None, None, "adds r41 = -1, r34 ;;"),
         one("m", "st1 [r41] = r40")] * run + [GAP] * 16,
        ["mov r42 = 0"])
# The same over one warm block, back to its start every body.
for run in (1, 2, 4, 8, 16):
    for dep in ("", "_nodep"):
        STORES[f"inplace_warm{run}{dep}"] = (
            [one("m", "ld1 r40 = [r34], 1"),
             B("mmi", None, None, "adds r41 = -1, r34 ;;"),
             one("m", f"st1 [r41] = {'r42' if dep else 'r40'}")] * run
            + [B("mmi", None, None, f"adds r34 = -{run}, r34 ;;")]
            + [GAP] * 15,
            ["mov r42 = 0"])
# Without the store at all.
STORES["alat_nostore"] = ([one("m", "ld8.a r34 = [r33]"), GAP,
                           one("m", "ld8.c.clr r34 = [r33]")], WARM)
STORES["alat_nostore_nc"] = ([one("m", "ld8.a r34 = [r33]"), GAP,
                              one("m", "ld8.c.nc r34 = [r33]")], WARM)
# clang's unrolled match copy in zstd: ld1.sa of byte 14 hoisted above the
# copy, the first byte's load and store, and the check in the store's group
# ("same") or the next ("next"), the store OFF bytes on from the source; "plain"
# loads byte 14 again with a plain ld1, "nost" leaves the store out.
for off in (16, 64, 4096 + 16):
    for kind in ("same", "next", "plain", "nost"):
        la, lc = ("ld1", "ld1") if kind == "plain" else ("ld1.sa", "ld1.c.clr")
        stb = None if kind == "nost" else "st1 [r38] = r45"
        tail = ([B("mmi", stb, None, "nop.i 0 ;;"), one("m", f"{lc} r44 = [r37]")]
                if kind == "next" else
                [B("mmi", stb, f"{lc} r44 = [r37]", "nop.i 0 ;;")])
        STORES[f"zcopy_{kind}_o{off}"] = (
            [one("m", f"{la} r44 = [r37]"), one("m", "ld1 r45 = [r33]")]
            + tail + [one("m", "st1 [r39] = r44")],
            ["adds r37 = 14, r33", f"movl r18 = {off}", "add r38 = r33, r18",
             "adds r39 = 256, r33"] + WARM)
# ld1.sa of byte 14 of a line, a st1 to byte K of it, then ld1.c.clr: which
# byte stores kill the entry (the ld8 probes above store doublewords).
for k in (0, 7, 8, 13, 14, 15, 16, 17, 22, 24, 31, 32, 40, 48, 63, 64):
    STORES[f"alat1_k{k}"] = ([
        one("m", "ld1.sa r44 = [r37]"), one("m", "st1 [r38] = r45"),
        one("m", "ld1.c.clr r44 = [r37]"), one("m", "st1 [r39] = r44")],
        ["adds r37 = 14, r33", f"adds r38 = {k}, r33",
         "adds r39 = 256, r33", "mov r45 = 0"] + WARM)
# A check GAP empty groups after its advanced load (0: the next group),
# with a store to another line in the check's group or without.
for sz in (1, 8):
    for gap in (0, 1, 2, 3, 4, 6):
        for stk in ("", "_st"):
            stb = "st8 [r39] = r45" if stk else None
            STORES[f"alatd{sz}_g{gap}{stk}"] = ([
                one("m", f"ld{sz}.sa r44 = [r37]")] + [GAP] * gap + [
                B("mmi", stb, f"ld{sz}.c.clr r44 = [r37]", "nop.i 0 ;;"),
                one("m", "st8 [r38] = r44")],
                ["mov r37 = r33", "adds r38 = 256, r33",
                 "adds r39 = 512, r33", "mov r45 = 0"] + WARM)
# How many entries the ALAT holds: N ld8.a to N registers and addresses,
# then a ld8.c.clr of each in the same order, which misses where a later
# load replaced its entry and so delays the add of its value in the next
# group (cap_n); and a window of K live entries, each group adding one and
# checking the one added K-1 groups before (win_k): can a check free its
# entry for an advanced load in its own group?  cap_n_gG puts G empty groups
# between the loads and the checks, in case an entry is written some cycles
# after its load issues.
ALAT_ADDRS = [f"adds r{84 + k} = {8 * k}, r33" for k in range(40)]
for n, g in ([(n, 0) for n in (16, 24, 28, 30, 31, 32, 33, 34, 35, 36, 37,
                                38, 40)] +
             [(n, 8) for n in (32, 33, 34, 35, 36)]):
    BLOCKS[f"alatcap_n{n}" + (f"_g{g}" if g else "")] = (
        [one("m", f"ld8.a r{44 + k} = [r{84 + k}]") for k in range(n)] +
        [GAP] * g +
        [B("mmi", f"ld8.c.clr r{44 + k} = [r{84 + k}]", None,
           f"add r40 = r40, r{44 + k - 1} ;;" if k else "nop.i 0 ;;")
         for k in range(n)] +
        [B("mmi", None, None, f"add r40 = r40, r{44 + n - 1} ;;")],
        ALAT_ADDRS + WARM)
    if g:
        body, setup = BLOCKS[f"alatcap_n{n}_g{g}"]
        BLOCKS[f"alatcap_n{n}_g{g}_inv"] = (
            body + [B("mmi", "invala", None, "nop.i 0 ;;")], setup)
for k in (8, 16, 28, 30, 31, 32, 33, 34):
    BLOCKS[f"alatwin_k{k}"] = (
        [B("mmi", f"ld8.a r{44 + j} = [r{84 + j}]",
           f"ld8.c.clr r{44 + (j + 1) % k} = [r{84 + (j + 1) % k}]",
           f"add r40 = r40, r{44 + (j - 1) % k} ;;") for j in range(k)]
        * (2 if k < 16 else 1),
        ALAT_ADDRS + WARM)
# The same with the check in M0 ahead of the load in M1: whether the order
# within a group decides if the load has the entry the check frees.
for k in (32, 33, 34):
    BLOCKS[f"alatwin_k{k}_cf"] = (
        [B("mmi", f"ld8.c.clr r{44 + (j + 1) % k} = [r{84 + (j + 1) % k}]",
           f"ld8.a r{44 + j} = [r{84 + j}]",
           f"add r40 = r40, r{44 + (j - 1) % k} ;;") for j in range(k)],
        ALAT_ADDRS + WARM)
# Which entry a 33rd advanced load replaces: 32 ld8.a fill the empty ALAT
# (after invala) in order, a 33rd issues on port M0 or M1, and a check of
# the first (old) or the 32nd (new) misses where it went.  The manual has
# pointers for each port that advance with every replacement and only
# approximate FIFO: an old check that misses every time means the oldest
# entry, one that misses one time in 32 means a pointer.
for port in (0, 1):
    for which, r in [("old", 0), ("new", 31), ("x", 32)] + [
            (f"e{e}", e) for e in range(1, 31)]:
        extra = f"ld8.a r76 = [r116]"
        BLOCKS[f"alatvic_m{port}_{which}"] = (
            [one("m", f"ld8.a r{44 + k} = [r{84 + k}]") for k in range(32)] +
            [B("mmi", None if port else extra, extra if port else None,
               "nop.i 0 ;;")] + [GAP] * 8 +
            [B("mmi", f"ld8.c.clr r{44 + r} = [r{84 + r}]", None,
               "nop.i 0 ;;"),
             B("mmi", "invala", None, f"add r40 = r40, r{44 + r} ;;")],
            ALAT_ADDRS + WARM)
# The same with a store that kills the first entry: what a miss costs.
BLOCKS["alatvic_kill"] = (
    [one("m", f"ld8.a r{44 + k} = [r{84 + k}]") for k in range(32)] +
    [one("m", "st8 [r84] = r0")] + [GAP] * 8 +
    [B("mmi", "ld8.c.clr r44 = [r84]", None, "nop.i 0 ;;"),
     B("mmi", "invala", None, "add r40 = r40, r44 ;;")],
    ALAT_ADDRS + WARM)
# Whether two registers D apart share an ALAT tag: ld8.a of each, then a
# ld8.c.clr of each, the first of which would clear an entry both match.
for d in (1, 16, 31, 32, 33, 39):
    BLOCKS[f"alatalias_d{d}"] = (
        [one("m", "ld8.a r44 = [r84]"), one("m", f"ld8.a r{44 + d} = [r85]")] +
        [GAP] * 8 +
        [one("m", "ld8.c.clr r44 = [r84]"),
         one("m", f"ld8.c.clr r{44 + d} = [r85]"),
         B("mmi", None, None, f"add r40 = r40, r{44 + d} ;;")],
        ALAT_ADDRS + WARM)
# N ld8.a after invala, a gap, then checks of the entries in ORDER: the
# first only, the last only, all oldest first, all newest first.
def alat_checks(regs):
    return ([B("mmi", f"ld8.c.clr r{44 + r} = [r{84 + r}]", None,
               "nop.i 0 ;;") for r in regs] +
            [B("mmi", "invala", None, f"add r40 = r40, r{44 + regs[-1]} ;;")])


for n in (33, 34, 35, 36, 40):
    for order, regs in [("first", [0]), ("last", [n - 1]),
                        ("fwd", list(range(n))),
                        ("rev", list(range(n - 1, -1, -1))),
                        ("pair", [0, n - 1]), ("rpair", [n - 1, 0])] + (
                        [(f"head{h}", list(range(h)))
                         for h in (2, 4, 8, 16, 24, 28, 30, 31, 32)] +
                        [(f"tail{h}", list(range(n - h, n)))
                         for h in (2, 4, 8, 16, 24, 28, 30, 31, 32)]
                        if n in (33, 36) else []):
        BLOCKS[f"alatord_n{n}_{order}"] = (
            [one("m", f"ld8.a r{44 + k} = [r{84 + k}]") for k in range(n)] +
            [GAP] * 8 + alat_checks(regs),
            ALAT_ADDRS + WARM)
# n33: a check that clears entry A, then a check of entry J.
for a in (0, 1, 5, 20):
    for j in range(33):
        if j != a:
            BLOCKS[f"alatpair_a{a}_j{j}"] = (
                [one("m", f"ld8.a r{44 + k} = [r{84 + k}]")
                 for k in range(33)] + [GAP] * 8 + alat_checks([a, j]),
                ALAT_ADDRS + WARM)
# A chk.a.clr of an ld8.a, with a store between that kills its entry
# (fail) or to another line (pass): what branching to recovery code (a
# reload and a branch back) costs.  The check's group follows the store's
# by two cycles, so the ALAT compares 20 address bits.
for kind, sto in (("fail", 0), ("pass", 256)):
    BLOCKS[f"chka_{kind}"] = ([
        one("m", "ld8.a r44 = [r33]"), one("m", "st8 [r36] = r45"), GAP,
        one("m", "chk.a.clr r44, 1f"),
        B("mmi", None, None, "add r40 = r40, r44 ;;", label="2"),
        B("mib", None, None, "br.cond.sptk.few 3f ;;"),
        B("mmi", "ld8 r44 = [r33]", None, "nop.i 0 ;;", label="1"),
        B("mib", None, None, "br.cond.sptk.few 2b ;;"),
        B("mmi", None, None, "nop.i 0 ;;", label="3")],
        [f"adds r36 = {sto}, r33", "mov r45 = 0"] + WARM)
# The same copy in a stream: 16 bytes written with two st8, then bytes 0 and
# 14 copied from OFF bytes behind the write pointer, as zstd's decoder copies
# a match from output it wrote shortly before; "plain" without the advanced
# load.
for off in (16, 32, 64, 128, 256, 1024, 8192):
    for kind in ("alat", "plain"):
        la, lc = ("ld1.sa", "ld1.c.clr") if kind == "alat" else ("ld1", "ld1")
        STORES[f"zmatch_{kind}_o{off}"] = ([
            one("m", "st8 [r34] = r40, 8"), one("m", "st8 [r34] = r40, 8"),
            B("mii", None, f"adds r37 = {14 - off}, r34",
              f"adds r38 = {-off}, r34 ;;"),
            one("m", f"{la} r44 = [r37]"), one("m", "ld1 r45 = [r38]"),
            B("mmi", "st1 [r34] = r45", f"{lc} r44 = [r37]",
              "adds r39 = 14, r34 ;;"),
            B("mmi", "st1 [r39] = r44", None, "adds r34 = 16, r34 ;;")], [])
# The same writing a new 128-byte L2 line each body, 1-2 MB a call, so every
# line the stores reach misses L2.
for off in (16, 32, 64, 128, 256, 1024):
    STORES[f"zline_o{off}"] = ([
        one("m", "st8 [r34] = r40, 8"), one("m", "st8 [r34] = r40, 8"),
        B("mii", None, f"adds r37 = {14 - off}, r34",
          f"adds r38 = {-off}, r34 ;;"),
        one("m", "ld1 r44 = [r37]"), one("m", "ld1 r45 = [r38]"),
        B("mmi", "st1 [r34] = r45", "ld1 r44 = [r37]",
          "adds r39 = 14, r34 ;;"),
        B("mmi", "st1 [r39] = r44", None, "adds r34 = 112, r34 ;;")], [])
# With GAP empty groups between the stores and the loads: how long after
# the store missed L2 a load of its line, or of the line before, still waits.
for off in (16, 128):
    for gap in (4, 8, 16, 24, 32, 40, 56):
        STORES[f"zlg_o{off}_g{gap}"] = ([
            one("m", "st8 [r34] = r40, 8"), one("m", "st8 [r34] = r40, 8"),
            B("mii", None, f"adds r37 = {14 - off}, r34",
              f"adds r38 = {-off}, r34 ;;")] + [GAP] * gap + [
            one("m", "ld1 r44 = [r37]"), one("m", "ld1 r45 = [r38]"),
            B("mmi", "st1 [r34] = r45", "ld1 r44 = [r37]",
              "adds r39 = 14, r34 ;;"),
            B("mmi", "st1 [r39] = r44", None, "adds r34 = 112, r34 ;;")], [])
# zline_o16 and zlg_o16 with the stores 2176 bytes apart, 35 MB a call,
# so every line they reach comes from memory, as when malloc writes a chunk
# header into the top of a heap that has just grown.
for gap in (0, 8, 24, 56):
    STORES[f"zlgm_g{gap}"] = ([
        one("m", "st8 [r34] = r40, 8"), one("m", "st8 [r34] = r40, 8"),
        B("mii", None, "adds r37 = -2, r34", "adds r38 = -16, r34 ;;")] +
        [GAP] * gap + [
        one("m", "ld1 r44 = [r37]"), one("m", "ld1 r45 = [r38]"),
        B("mmi", "st1 [r34] = r45", "ld1 r44 = [r37]",
          "adds r39 = 14, r34 ;;"),
        B("mmi", "st1 [r39] = r44", None, "adds r34 = 2160, r34 ;;")], [])


# The same with the write pointer back where it started each body, so every
# line stays in L1D and L2.
for off in (16, 32, 64, 256):
    STORES[f"zmatch_warm_o{off}"] = ([
        one("m", "st8 [r34] = r40, 8"), one("m", "st8 [r34] = r40, 8"),
        B("mii", None, f"adds r37 = {14 - off}, r34",
          f"adds r38 = {-off}, r34 ;;"),
        one("m", "ld1 r44 = [r37]"), one("m", "ld1 r45 = [r38]"),
        B("mmi", "st1 [r34] = r45", "ld1 r44 = [r37]",
          "adds r39 = 14, r34 ;;"),
        B("mmi", "st1 [r39] = r44", None, "adds r34 = -16, r34 ;;")], [])

# zmatch_warm_o16 with one of its memory operations made a nop (zmw_xN,
# N counting from the first st8), to find what its 12 cycles wait on.
ZMW = [("st8 [r34] = r40, 8", "adds r34 = 8, r34"),
       ("st8 [r34] = r40, 8", "adds r34 = 8, r34"),
       ("ld1 r44 = [r37]", "mov r44 = r0"), ("ld1 r45 = [r38]", "mov r45 = r0"),
       ("st1 [r34] = r45", None), ("ld1 r44 = [r37]", "mov r44 = r0"),
       ("st1 [r39] = r44", None)]
for x in range(-1, len(ZMW)):
    o = [m if i != x else r for i, (m, r) in enumerate(ZMW)]
    STORES[f"zmw_x{x + 1}" if x >= 0 else "zmw_full"] = ([
        one("m", o[0]), one("m", o[1]),
        B("mii", None, "adds r37 = -2, r34", "adds r38 = -16, r34 ;;"),
        one("m", o[2]), one("m", o[3]),
        B("mmi", o[4], o[5], "adds r39 = 14, r34 ;;"),
        B("mmi", o[6], None, "adds r34 = -16, r34 ;;")], [])

# zmw_full with the second st1 moved to r34+16+A (zmv_aA), and with the
# load whose byte the first st1 stores reading r34+16+L (zmv_lL), to find
# which address relations its 3 cycles need.  On an Itanium 9340 the best
# run takes 12 cycles with the second st1 in the first st1's 16 bytes and
# not 4-byte aligned (or on its byte), else 9; 10 with the load from the
# second st8's word, 9 from the next 128-byte line; and one empty group
# anywhere takes the extra cycles away.  Many runs instead settle at 21 to
# 25 cycles a body for all of them.
def zmv(a=14, l=-16, g45=0, g56=0):
    return [one("m", "st8 [r34] = r40, 8"), one("m", "st8 [r34] = r40, 8"),
            B("mii", None, "adds r37 = -2, r34", f"adds r38 = {l}, r34 ;;"),
            one("m", "ld1 r44 = [r37]"), one("m", "ld1 r45 = [r38]")] + \
        [GAP] * g45 + [
            B("mmi", "st1 [r34] = r45", "ld1 r44 = [r37]",
              f"adds r39 = {a}, r34 ;;")] + [GAP] * g56 + [
            B("mmi", "st1 [r39] = r44", None, "adds r34 = -16, r34 ;;")]
# ... with G empty groups before the first st1 (zmv_pG) or after it
# (zmv_qG), and the same with the load from the next 128-byte line.
for g in (1, 2, 3, 4, 6):
    STORES[f"zmv_p{g}"] = (zmv(g45=g), [])
    STORES[f"zmv_q{g}"] = (zmv(g56=g), [])
    STORES[f"zmv_p{g}_l112"] = (zmv(l=112, g45=g), [])
    STORES[f"zmv_q{g}_l112"] = (zmv(l=112, g56=g), [])
for a in list(range(32)) + [48, 64, 112]:
    STORES[f"zmv_a{a}"] = (zmv(a=a), [])
for l in list(range(-16, 16)) + [16, 32, 48, 64, 112]:
    STORES[f"zmv_l{l}".replace("-", "m")] = (zmv(l=l), [])

# zstd's 16-byte match copy as clang schedules it in ZSTD_decompressSequences:
# byte 14 read ahead by ld1.sa, then sixteen ld1/st1 pairs two to a group, the
# check of byte 14 in the group after the first store. The source is OFF bytes
# behind the destination; "stream" writes on, "warm" rewrites the same 16.
def zbyte(off, adv, step):
    la, lc = ("ld1.sa", "ld1.c.clr") if adv else ("ld1", "ld1")
    addrs = [B("mii", None, f"adds r{64 + i} = {i - off}, r34",
               f"adds r{65 + i} = {i + 1 - off}, r34" + (" ;;" if i % 4 == 2
                                                          else ""))
             for i in range(0, 16, 2)]
    addrs += [B("mii", None, f"adds r{80 + i} = {i}, r34",
                f"adds r{81 + i} = {i + 1}, r34" + (" ;;" if i % 4 == 2
                                                    else ""))
              for i in range(0, 16, 2)]
    ld = lambda i: f"ld1 r{44 + i} = [r{64 + i}]"
    sb = lambda i: f"st1 [r{80 + i}] = r{44 + i}"
    body = addrs + [
        one("m", f"{la} r58 = [r78]"),
        one("m", ld(0)),
        B("mmi", ld(15), sb(0), "nop.i 0 ;;"),
        B("mmi", f"{lc} r58 = [r78]", ld(13), "nop.i 0 ;;"),
        B("mmi", f"st1 [r95] = r59", ld(11), "nop.i 0 ;;"),
        B("mmi", "st1 [r94] = r58", ld(10), "nop.i 0 ;;"),
        B("mmi", ld(8), sb(13), "nop.i 0 ;;"),
        B("mmi", ld(12), ld(9), "nop.i 0 ;;"),
        B("mmi", sb(11), ld(7), "nop.i 0 ;;"),
        B("mmi", sb(10), ld(6), "nop.i 0 ;;"),
        B("mmi", sb(8), ld(5), "nop.i 0 ;;"),
        B("mmi", sb(12), ld(4), "nop.i 0 ;;"),
        B("mmi", sb(9), ld(3), "nop.i 0 ;;"),
        B("mmi", sb(7), ld(2), "nop.i 0 ;;"),
        B("mmi", sb(6), ld(1), "nop.i 0 ;;"),
        B("mmi", sb(5), sb(4), "nop.i 0 ;;"),
        B("mmi", sb(3), sb(2), "nop.i 0 ;;"),
        B("mmi", sb(1), None, f"adds r34 = {step}, r34 ;;")]
    return body


ZBYTE = {}
for off in (16, 32, 64, 128, 1024):
    for adv in (True, False):
        for kind, step in (("stream", 16), ("warm", 0)):
            ZBYTE[f"zbyte_{'alat' if adv else 'plain'}_{kind}_o{off}"] = (
                zbyte(off, adv, step))
# Only zbyte's stores, one a group, in its byte order, ascending or
# descending; "pair" keeps zbyte's groups, two stores to one.
ZORDER = {"z": [0, 15, 14, 13, 11, 10, 8, 12, 9, 7, 6, 5, 4, 3, 2, 1],
          "up": list(range(16)), "down": list(range(15, -1, -1)),
          "alt": [b for i in range(8) for b in (i, 8 + i)],
          "alt2": [b for i in range(0, 8, 2) for b in (i, i + 1, 8 + i, 9 + i)],
          "zc": [11, 0, 14, 15, 13, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
for oname, order in ZORDER.items():
    for kind, step in (("stream", 16), ("warm", 0)):
        addrs = [B("mii", None, f"adds r{80 + i} = {i}, r34",
                   f"adds r{81 + i} = {i + 1}, r34" + (" ;;" if i % 4 == 2
                                                       else ""))
                 for i in range(0, 16, 2)]
        ZBYTE[f"zst_{oname}_{kind}"] = addrs + [
            one("m", f"st1 [r{80 + i}] = r40") for i in order] + [
            B("mii", None, None, f"adds r34 = {step}, r34 ;;")]
# Byte stores in a given order to offsets from r34, one a group, each
# address in its own register; "stream" moves on by STEP each body.
def zsx(offs, step):
    n = len(offs)
    addrs = [B("mii", None, f"adds r{80 + i} = {o}, r34",
               "nop.i 0 ;;" if i % 4 == 3 else "nop.i 0")
             for i, o in enumerate(offs)]
    return {"stream": addrs + [one("m", f"st1 [r{80 + i}] = r40")
                               for i in range(n)] + [
                B("mii", None, None, f"adds r34 = {step}, r34 ;;")],
            "warm": addrs + [one("m", f"st1 [r{80 + i}] = r40")
                             for i in range(n)]}


ZC = ZORDER["zc"]
for oname, offs, step in (
        ("alt3", [b for i in range(8) for b in (i, 8 + i, 16 + i)], 24),
        ("altbank", [b for i in range(8) for b in (i, 16 + i)], 8),
        ("altline", [b for i in range(8) for b in (i, 64 + i)], 8),
        ("alt4", [b for h in (0, 4) for b in (h, h + 1, h + 2, h + 3,
                                              8 + h, 9 + h, 10 + h, 11 + h)],
         16),
        ("zc2", ZC + [16 + b for b in ZC], 32)):
    for kind, body in zsx(offs, step).items():
        ZBYTE[f"zsx_{oname}_{kind}"] = body
# zst_up_stream's 16 byte stores through one post-incremented pointer, with
# zst's eight groups of adds in front or without them.
for pre in ("", "_adds"):
    addrs = [B("mii", None, f"adds r{80 + i} = {i}, r34",
               f"adds r{81 + i} = {i + 1}, r34" + (" ;;" if i % 4 == 2
                                                   else ""))
             for i in range(0, 16, 2)] if pre else []
    ZBYTE[f"zsq_post{pre}"] = addrs + [
        one("m", "st1 [r34] = r40, 1") for i in range(16)]
# A load of a doubleword stored behind B others in the queue: a fixed one
# whose line a load keeps in L1D, or the last of a stream of new lines.
for b in (0, 4, 8, 16):
    ZBYTE[f"qld_fixed_b{b}"] = [one("m", "ld8 r45 = [r42]")] + [
        one("m", "st8 [r34] = r40, 8")] * b + [
        one("m", "st8 [r36] = r40"), one("m", "ld8 r44 = [r36]"),
        B("mii", None, "add r46 = r44, r0", "nop.i 0 ;;")]
    ZBYTE[f"qld_stream_b{b}"] = [one("m", "st8 [r34] = r40, 8")] * (b + 1) + [
        B("mii", None, "adds r37 = -8, r34", "nop.i 0 ;;"),
        one("m", "ld8 r44 = [r37]"),
        B("mii", None, "add r46 = r44, r0", "nop.i 0 ;;")]
# The same 16 bytes rewritten each body: through a post-incremented pointer
# with or without zst's adds in front, or through address registers set up
# once.
ADDRS = [B("mii", None, f"adds r{80 + i} = {i}, r34",
           f"adds r{81 + i} = {i + 1}, r34" + (" ;;" if i % 4 == 2 else ""))
         for i in range(0, 16, 2)]
for pre in ("", "_adds"):
    ZBYTE[f"zsqw_post{pre}"] = (ADDRS if pre else []) + [
        one("m", "st1 [r34] = r40, 1") for i in range(16)] + [
        B("mii", None, None, "adds r34 = -16, r34 ;;")]
ZBYTE["zsqw_regs"] = [one("m", f"st1 [r{80 + i}] = r40") for i in range(16)]
ZBYTE["zsqw_regs_adds"] = [B("mii", None, "nop.i 0",
                            "nop.i 0" + (" ;;" if i % 4 == 2 else ""))
                          for i in range(0, 16, 2)] + ZBYTE["zsqw_regs"]
# qld_fixed_b0's pieces: the load before the store gone, to another line, or
# a group further away; and the store's load gone or its result unused.
LDST = {"noa": [None, "st8 [r36] = r40", "ld8 r44 = [r36]", True],
        "far": ["ld8 r45 = [r39]", "st8 [r36] = r40", "ld8 r44 = [r36]", True],
        "gap": ["ld8 r45 = [r42]", None, "st8 [r36] = r40", "ld8 r44 = [r36]",
                True],
        "nold": ["ld8 r45 = [r42]", "st8 [r36] = r40", None, False],
        "nouse": ["ld8 r45 = [r42]", "st8 [r36] = r40", "ld8 r44 = [r36]",
                  False],
        "same": ["ld8 r45 = [r42]", "st8 [r36] = r40", "ld8 r44 = [r36]",
                 True]}
for name, parts in LDST.items():
    body = [one("m", p) if p else GAP for p in parts[:-1]]
    if parts[-1]:
        body.append(B("mii", None, "add r46 = r44, r0", "nop.i 0 ;;"))
    ZBYTE[f"ldst_{name}"] = body


# linpack's saxpy loop as clang pipelines it: each iteration loads x[i],
# x[i+1], y[i] and y[i+1] with four ldfs through four pointers, and stores
# the y[i] and y[i+1] of 7 iterations back with two stfs, in one group of
# MMF MMF MMF MFB.  The loop rotates its registers; here a body is 8
# iterations, each writing its own registers, whose values nothing waits
# for (the fmas read f100-f103, the stores store f100).  x at XOFF and y
# at 128 KB + YOFF in the stream, both in L2 after the first call.  "mem"
# has only the loads and stores, "ld" only the loads, "st" only the stores.
def saxpy_setup(xoff, yoff):
    y = 128 * 1024 + yoff
    return [f"movl r18 = {STREAM + xoff:#x}", "add r60 = r33, r18",
            "adds r61 = 4, r60", f"movl r18 = {STREAM + y:#x}",
            "add r62 = r33, r18", "adds r63 = 4, r62",
            "adds r58 = -56, r62", "adds r59 = -52, r62"] + [
                f"mov f{r} = f1" for r in range(100, 104)]


def saxpy_group(k, loads=True, stores=True, fmas=True):
    def f(t):
        return t if fmas else None
    ld = loads or None
    st = stores or None
    r = 32 + 8 * k
    return [B("mmf", st and "stfs [r58] = f100, 8",
              st and "stfs [r59] = f100, 8",
              f(f"fma.s f{r} = f100, f1, f101")),
            B("mmf", ld and f"ldfs f{r + 1} = [r60], 8",
              ld and f"ldfs f{r + 2} = [r61], 8",
              f(f"fma.s f{r + 3} = f102, f103, f0")),
            B("mmf", ld and f"ldfs f{r + 4} = [r62], 8",
              ld and f"ldfs f{r + 5} = [r63], 8",
              f(f"fma.s f{r + 6} = f102, f101, f0")),
            B("mfb", None, f(f"fma.s f{r + 7} = f103, f1, f100"),
              "nop.b 0 ;;")]


SAXPY = {}
for xo, yo in ((0, 0), (4, 4), (0, 4), (804, 1608)):
    for kind, kw in (("full", {}), ("mem", {"fmas": False}),
                     ("ld", {"stores": False, "fmas": False}),
                     ("st", {"loads": False, "fmas": False}),
                     ("fma", {"loads": False, "stores": False})):
        SAXPY[f"saxpy_{kind}_x{xo}_y{yo}"] = (
            [b for k in range(8) for b in saxpy_group(k, **kw)],
            saxpy_setup(xo, yo))

# gemm's inner loop as clang pipelines it: saxpy in doubles, y += a * x
# over a row, each iteration loading x[j], x[j+1], y[j] and y[j+1] with
# four ldfds through four pointers and storing y 7 iterations (112 bytes)
# back with two stfds.  gemm's rows are 560 bytes, so x's L2 bank against
# y's takes every value: x at XOFF, y at 128 KB + YOFF.
def saxpyd_setup(xoff, yoff):
    y = 128 * 1024 + yoff
    return [f"movl r18 = {STREAM + xoff:#x}", "add r60 = r33, r18",
            "adds r61 = 8, r60", f"movl r18 = {STREAM + y:#x}",
            "add r62 = r33, r18", "adds r63 = 8, r62",
            "adds r58 = -112, r62", "adds r59 = -104, r62"] + [
                f"mov f{r} = f1" for r in range(100, 104)]


def saxpyd_group(k):
    r = 32 + 8 * k
    return [B("mmf", "stfd [r58] = f100, 16", "stfd [r59] = f100, 16", None),
            B("mmf", f"ldfd f{r + 1} = [r60], 16",
              f"ldfd f{r + 2} = [r61], 16", None),
            B("mmf", f"ldfd f{r + 4} = [r62], 16",
              f"ldfd f{r + 5} = [r63], 16", None),
            B("mfb", None, None, "nop.b 0 ;;")]


for xo, yo in [(x, 0) for x in range(0, 256, 16)] + [(8, 0), (0, 8), (8, 8)]:
    SAXPY[f"saxpyd_x{xo}_y{yo}"] = (
        [b for k in range(8) for b in saxpyd_group(k)], saxpyd_setup(xo, yo))


def kernels():
    ks = []
    for size in SIZES:
        n = level(size)
        meta = {"footprint": size}
        if n is not None:
            meta["expect_cyc_per_body"] = float(n + 1)
        ks += pair(f"chase_{name_of(size)}", "memory",
                   [one("m", "ld8 r34 = [r34]")], ring_setup(size), meta,
                   teardown=RING_TEARDOWN)
    # An L2-resident chase behind K st1s to one word, and those st1s
    # alone: does a load that misses L1D wait for the stores queued ahead?
    for k in (1, 2, 4, 8):
        sts = [one("m", "st1 [r36] = r40")] * k
        ks += pair(f"qchase_k{k}", "memory",
                   sts + [one("m", "ld8 r34 = [r34]")],
                   ring_setup(128 * KB) + ["adds r36 = 72, r33"],
                   teardown=RING_TEARDOWN)
        ks += pair(f"qchase_k{k}_st", "memory", sts,
                   ring_setup(128 * KB) + ["adds r36 = 72, r33"],
                   teardown=RING_TEARDOWN)
    # The chase writing the second word of the line it is about to load
    # (chase_dirty_*), so the store misses first, or of the line it just
    # loaded (chase_dcur_*); either way every line L3 evicts is dirty.
    for size in (2 * MB, 6 * MB, 16 * MB, 64 * MB):
        ks += pair(f"chase_dirty_{name_of(size)}", "memory",
                   [one("m", "ld8 r34 = [r34]"),
                    B("mii", None, "adds r41 = 8, r34", "nop.i 0 ;;"),
                    one("m", "st8 [r41] = r40")], ring_setup(size),
                   {"footprint": size}, teardown=RING_TEARDOWN)
        ks += pair(f"chase_dcur_{name_of(size)}", "memory",
                   [B("mii", None, "adds r41 = 8, r34", "nop.i 0 ;;"),
                    one("m", "ld8 r34 = [r34]"),
                    one("m", "st8 [r41] = r40")], ring_setup(size),
                   {"footprint": size}, teardown=RING_TEARDOWN)
    # mst's HashLookup: the key's ld4 misses, the compare waiting for it
    # shares a group with a ld8.sa of the next pointer from the same line
    # (hl_spec), and a ld8.c.clr checks it in the next group; or a plain
    # ld8 there (hl_early), or in the check's place (hl_late).
    for size in (16 * KB, 128 * KB, 2 * MB, 64 * MB):
        for kind, early, late in (
                ("spec", "ld8.sa r37 = [r34]", "ld8.c.clr r37 = [r34]"),
                ("early", "ld8 r37 = [r34]", None),
                ("late", None, "ld8 r37 = [r34]")):
            ks += pair(f"hl_{kind}_{name_of(size)}", "memory",
                       [one("m", "ld4 r36 = [r35]"),
                        B("mmi", early, "cmp4.eq p6, p0 = r36, r40",
                          "nop.i 0 ;;"),
                        one("m", late) if late else GAP,
                        B("mii", None, "mov r34 = r37",
                          "adds r35 = 16, r37 ;;")],
                       ring_setup(size) + ["adds r35 = 16, r34"],
                       {"footprint": size}, teardown=RING_TEARDOWN)
    # A second load of the line the key's ld4 missed, K empty groups after
    # it: when does a secondary miss get its data?
    for size in (128 * KB, 2 * MB, 64 * MB):
        for k in (0, 1, 2, 4, 8, 12, 16, 24):
            first = (B("mmi", "ld4 r36 = [r35]", "ld8 r37 = [r34]",
                       "nop.i 0 ;;") if k == 0 else
                     one("m", "ld4 r36 = [r35]"))
            ks += pair(f"hlsec_k{k}_{name_of(size)}", "memory",
                       [first] + [GAP] * (k - 1) +
                       ([one("m", "ld8 r37 = [r34]")] if k else []) +
                       [B("mmi", None, "cmp4.eq p6, p0 = r36, r40",
                          "nop.i 0 ;;"),
                        B("mii", None, "mov r34 = r37",
                          "adds r35 = 16, r37 ;;")],
                       ring_setup(size) + ["adds r35 = 16, r34"],
                       {"footprint": size}, teardown=RING_TEARDOWN)
    # The same with the key in the line's other 64-byte half.
    for size in (128 * KB, 2 * MB, 64 * MB):
        for k in (1, 4, 12):
            ks += pair(f"hlsec_h{k}_{name_of(size)}", "memory",
                       [one("m", "ld4 r36 = [r35]")] + [GAP] * (k - 1) +
                       [one("m", "ld8 r37 = [r34]"),
                        B("mmi", None, "cmp4.eq p6, p0 = r36, r40",
                          "nop.i 0 ;;"),
                        B("mii", None, "mov r34 = r37",
                          "adds r35 = 80, r37 ;;")],
                       ring_setup(size) + ["adds r35 = 80, r34"],
                       {"footprint": size}, teardown=RING_TEARDOWN)
    # bzip2's cftab pass: a ld8 the chase goes through and a second load
    # of the same word (or the next) in its group, in either slot: does
    # the second hold back the first's data?
    for size in (16 * KB, 128 * KB, 2 * MB):
        for kind, second, off, swap in (
                ("none", None, 0, False),
                ("same", "ld8 r36 = [r35]", 0, False),
                ("sa", "ld8.sa r36 = [r35]", 0, False),
                ("off8", "ld8 r36 = [r35]", 8, False),
                ("same_sw", "ld8 r36 = [r35]", 0, True),
                ("sa_sw", "ld8.sa r36 = [r35]", 0, True)):
            loads = ([second, "ld8 r37 = [r34]"] if swap else
                     ["ld8 r37 = [r34]", second])
            ks += pair(f"ldpair_{kind}_{name_of(size)}", "memory",
                       [B("mmi", loads[0], loads[1], "nop.i 0 ;;"),
                        B("mii", None, "mov r34 = r37",
                          f"adds r35 = {off}, r37 ;;")],
                       ring_setup(size) + [f"adds r35 = {off}, r34"],
                       {"footprint": size}, teardown=RING_TEARDOWN)
    for size in SEQ:
        ks += pair(f"chase_seq_{name_of(size)}", "memory",
                   [one("m", "ld8 r34 = [r34]")], ring_setup(size, True),
                   {"footprint": size}, teardown=RING_TEARDOWN)
    for size in (8 * KB, 128 * KB, 2 * MB, 64 * MB):
        m = level(size)
        meta = {"footprint": size}
        if m is not None:
            meta["expect_cyc_per_body"] = float(max(m, 5) + 1 + 6)
        ks += pair(f"fchase_{name_of(size)}", "memory",
                   [one("m", "ldf8 f32 = [r34]"),
                    one("m", "getf.sig r34 = f32")], ring_setup(size), meta,
                   teardown=RING_TEARDOWN)
    for name, (body, setup) in STORES.items():
        ks += pair(f"st_{name}", "memory", body, SSETUP + setup)
    for name, (body, setup) in BLOCKS.items():
        ks += pair(f"st_{name}", "memory", body, SSETUP + setup, bodies=2)
    # The ld8/st8 pair of one address, and mtfv's walk, as the store setup
    # leaves them, with 8 empty groups after its st8 [r33], and without
    # it: does that store, a cycle or two before the first load of r33,
    # decide whether the line starts in L1D?
    for name in ("pair_ld8st8_o0", "mtfv_s4_g0_o0"):
        body, setup = STORES[name]
        ks += pair(f"st_start_{name}_gap", "memory", body,
                   SSETUP + ["nop.i 0 ;;"] * 8 + setup)
        ks += pair(f"st_start_{name}_nost", "memory", body,
                   [l for l in SSETUP if not l.startswith("st8")] + setup)
    for name, body in ZBYTE.items():
        ks += pair(f"st_{name}", "memory", body,
                   SSETUP + ["adds r36 = 72, r33", "adds r39 = 256, r33"] +
                   [f"adds r{80 + i} = {i}, r34" for i in range(16)],
                   bodies=4)
    # Deeper than stq: RUN st1s over RUN/8 words, then enough groups to drain.
    for run in (8, 12, 16, 20, 24, 32):
        ks += pair(f"st_stqw{run}", "memory",
                   [one("m", "st1 [r34] = r40, 1")] * run
                   + [B("mmi", None, None, f"adds r34 = {-run % 8}, r34 ;;")]
                   + [GAP] * (127 - run), SSETUP, bodies=4)
    for name, (body, setup) in SAXPY.items():
        ks += pair(f"st_{name}", "memory", body, SSETUP + setup, bodies=2,
                   iters=250)
    return ks


def saxpy_mix(k, nld, nst, sname):
    r = 32 + 8 * k
    lds = [f"ldfs f{r + 1} = [r60], 8", f"ldfs f{r + 2} = [r61], 8",
           f"ldfs f{r + 4} = [r62], 8", f"ldfs f{r + 5} = [r63], 8"][:nld]
    sts = [f"stfs [r{sname}] = f100, 8", f"stfs [r{sname + 1}] = f100, 8"]
    sts = sts[:nst]
    m = (sts + [None] * (2 - nst)) + (lds + [None] * (4 - nld))
    return [B("mmf", m[0], m[1], None), B("mmf", m[2], m[3], None),
            B("mmf", m[4], m[5], None), B("mfb", None, None, "nop.b 0 ;;")]


# r58/r59 store 56 bytes behind the y loads, r56/r57 64 KB on from them.
for nld in (0, 1, 2, 3, 4):
    for nst in (1, 2):
        for sname, base in (("near", 58), ("far", 56)):
            SAXPY[f"zz_mix_l{nld}_s{nst}_{sname}"] = (
                [b for k in range(8) for b in saxpy_mix(k, nld, nst, base)],
                saxpy_setup(0, 0) + ["movl r18 = 0x10000", "add r56 = r62, r18",
                                     "adds r57 = 4, r56"])


def dense(k, nld, nst, sbase, fp=True):
    """nld loads (through r60-r63) and nst stores in one MMI MMI group."""
    r = 32 + 8 * k
    if fp:
        lds = [f"ldfs f{r + j} = [r{60 + j}], 8" for j in range(nld)]
        sts = [f"stfs [r{sbase + j}] = f100, 8" for j in range(nst)]
    else:
        lds = [f"ld4 r{64 + 4 * (k % 4) + j} = [r{60 + j}], 8"
               for j in range(nld)]
        sts = [f"st4 [r{sbase + j}] = r40, 8" for j in range(nst)]
    m = sts + lds + [None] * (4 - nld - nst)
    return [B("mmi", m[0], m[1], "nop.i 0"), B("mmi", m[2], m[3], "nop.i 0 ;;")]


for fp in (True, False):
    for nld, nst in ((1, 0), (2, 0), (3, 0), (4, 0), (0, 1), (0, 2), (1, 1),
                     (2, 1), (2, 2), (3, 1), (1, 2)):
        for sname, base in (("near", 58), ("far", 56)):
            if nst == 0 and sname == "far":
                continue
            for yo in (0, 16, 804):
                SAXPY[f"zz_dense_{'f' if fp else 'i'}_l{nld}_s{nst}_{sname}_y{yo}"] = (
                    [b for k in range(8) for b in dense(k, nld, nst, base, fp)],
                    saxpy_setup(0, yo) + ["movl r18 = 0x10000",
                                          "add r56 = r62, r18",
                                          "adds r57 = 4, r56"])


def l2lat(R, nld, nst, sbase, n=24):
    """n groups of nst stores and nld FP loads, load targets reused every R
    groups: past the issue rate, a group costs the load latency over R."""
    out = []
    for g in range(n):
        # The last group steps the pointers back, so the body covers the
        # same 8n bytes each time.
        inc = 8 if g < n - 1 else -8 * (n - 1)
        lds = [f"ldfs f{32 + 4 * (g % R) + j} = [r{60 + j}], {inc}"
               for j in range(nld)]
        sts = [f"stfs [r{sbase + j}] = f100, {inc}" for j in range(nst)]
        m = sts + lds + [None] * (4 - nld - nst)
        out += [B("mmi", m[0], m[1], "nop.i 0"),
                B("mmi", m[2], m[3], "nop.i 0 ;;")]
    return out


for R in (2, 3, 4, 6, 8, 12, 24):
    for nld in (1, 2):
        for nst, sname, base in ((0, "none", 58), (1, "near", 58),
                                 (2, "near", 58), (1, "far", 56),
                                 (2, "far", 56)):
            SAXPY[f"zz_lat_r{R}_l{nld}_s{nst}{sname}"] = (
                l2lat(R, nld, nst, base),
                saxpy_setup(0, 804) + ["movl r18 = 0x10000",
                                       "add r56 = r62, r18",
                                       "adds r57 = 4, r56"])


def l2pair(R, nld, nst, n=24):
    """l2lat with the stores through r58 and r57, which setup places apart."""
    out = []
    for g in range(n):
        inc = 8 if g < n - 1 else -8 * (n - 1)
        lds = [f"ldfs f{32 + 4 * (g % R) + j} = [r{60 + j}], {inc}"
               for j in range(nld)]
        sts = [f"stfs [r{(58, 57)[j]}] = f100, {inc}" for j in range(nst)]
        m = sts + lds + [None] * (4 - nld - nst)
        out += [B("mmi", m[0], m[1], "nop.i 0"),
                B("mmi", m[2], m[3], "nop.i 0 ;;")]
    return out


for d in (4, 8, 16, 128, 256, 512):
    for R in (6, 24):
        for nld in (0, 1, 2):
            SAXPY[f"zz_pair_d{d}_r{R}_l{nld}"] = (
                l2pair(R, nld, 2),
                saxpy_setup(0, 804) + [f"adds r57 = {d}, r58"])


def _fma_variant(k, op, n, src):
    r = 32 + 8 * k
    fs = [f"{op} f{r + i} = {src}" for i in range(n)] + [None] * 4
    return [B("mmf", None, None, fs[0]), B("mmf", None, None, fs[1]),
            B("mmf", None, None, fs[2]), B("mfb", None, fs[3], "nop.b 0 ;;")]


def _zz_spaced(op, src, per, gap):
    """per ops in a group, then gap empty groups; 8 such per body."""
    out = []
    for k in range(8):
        out += _fma_variant(k, op, per, src)
        out += [B("mfi", None, None, "nop.i 0 ;;")] * gap
    return out


_ZV = ["movl r18 = 0x3e99999a", "st4 [r33] = r18 ;;", "ldfs f9 = [r33] ;;",
       "movl r18 = 0x3f333333", "st4 [r33] = r18 ;;", "ldfs f10 = [r33] ;;",
       "mov f11 = f0"]
for op in ("fadd.s", "fadd.d", "fadd.s.s1", "fadd.s.s3", "fsub.s"):
    for per, gap in ((1, 0), (2, 0), (1, 1), (1, 3), (1, 7)):
        SAXPY[f"zz_sp_{op.replace('.', '')}_{per}_{gap}"] = (_zz_spaced(op, "f100, f101", per,
                                                         gap),
                                              saxpy_setup(0, 0))
SAXPY["zz_sp_fadds_inexact"] = (_zz_spaced("fadd.s", "f9, f10", 1, 0),
                                saxpy_setup(0, 0) + _ZV)
SAXPY["zz_sp_fmas_addzero"] = (_zz_spaced("fma.s", "f9, f10, f11", 1, 0),
                               saxpy_setup(0, 0) + _ZV)
SAXPY["zz_sp_fmas_inexact"] = (_zz_spaced("fma.s", "f9, f10, f9", 1, 0),
                               saxpy_setup(0, 0) + _ZV)


_ZLD = ["movl r18 = 0x3f800000", "st4 [r33] = r18 ;;", "ldfs f6 = [r33]",
        "ldfs f7 = [r33]", "ldfs f8 = [r33] ;;"]
for name, op, src, setup in (
        ("fmas", "fma.s", "f100, f1, f101", []),
        ("fmas_ld", "fma.s", "f6, f7, f8", _ZLD),
        ("fmas1", "fma.s.s1", "f100, f1, f101", []),
        ("fmas2", "fma.s.s2", "f100, f1, f101", []),
        ("fma", "fma", "f100, f1, f101", []),
        ("fmad_ld", "fma.d", "f6, f7, f8", _ZLD),
        ("fmpys", "fmpy.s", "f100, f101", []),
        ("fadds", "fadd.s", "f100, f101", []),
        ("fmas_f1", "fma.s", "f1, f1, f0", [])):
    SAXPY[f"zz_{name}"] = (
        [b for k in range(8) for b in _fma_variant(k, op, 1, src)],
        saxpy_setup(0, 0) + setup)


# What avoids the single-precision add stall: fnorm.s (an fma.s with f0
# for the addend) of a double, fadd.d then fnorm.s, the parallel fpma, and
# which operand of fadd.s must be zero for it to be fast.
_ZQ = _ZV + ["fadd.d f12 = f9, f10",
             "movl r18 = 0x3f8000003f800000", "st8 [r33] = r18 ;;",
             "ldf8 f13 = [r33]", "movl r18 = 0x3e99999a3f333333 ;;",
             "st8 [r33] = r18 ;;", "ldf8 f14 = [r33] ;;"]
for name, op, src, per in (
        ("fnorms", "fnorm.s", "f9", 1),
        ("fnorms_d", "fnorm.s", "f12", 1),
        ("fnorms_d2", "fnorm.s", "f12", 2),
        ("faddd", "fadd.d", "f9, f10", 1),
        ("fmpys", "fmpy.s", "f9, f10", 1),
        ("fmss", "fms.s", "f9, f10, f9", 1),
        ("fnmas", "fnma.s", "f9, f10, f9", 1),
        ("fadds_x0", "fadd.s", "f9, f11", 1),
        ("fadds_0x", "fadd.s", "f11, f9", 1),
        ("fmad", "fma.d", "f9, f10, f9", 1),
        ("fpma1", "fpma", "f13, f13, f13", 1),
        ("fpma", "fpma", "f14, f14, f14", 1),
        ("fpma2", "fpma", "f14, f14, f14", 2),
        ("fpmpy", "fpmpy", "f14, f14", 1)):
    SAXPY[f"zz_sq_{name}"] = (_zz_spaced(op, src, per, 0),
                              saxpy_setup(0, 0) + _ZQ)


def _sq_pair(k):
    """fadd.d and an fnorm.s of the fadd.d two groups before."""
    r = 32 + 8 * k
    return [B("mmf", None, None, f"fadd.d f{r} = f9, f10"),
            B("mmf", None, None, f"fnorm.s f{r + 1} = f{32 + 8 * ((k - 2) % 8)}"),
            B("mmf", None, None, None), B("mfb", None, None, "nop.b 0 ;;")]


SAXPY["zz_sq_faddd_fnorms"] = ([b for k in range(8) for b in _sq_pair(k)],
                               saxpy_setup(0, 0) + _ZQ +
                               [f"mov f{32 + 8 * k} = f9" for k in range(8)])


# Is the single-precision add stall value dependent?
def _zv_setup(a, b):
    return [f"movl r18 = {a}", "st4 [r33] = r18 ;;", "ldfs f20 = [r33] ;;",
            f"movl r18 = {b}", "st4 [r33] = r18 ;;", "ldfs f21 = [r33] ;;"]


for name, a, b in (
        ("1p1", "0x3f800000", "0x3f800000"),
        ("1p05", "0x3f800000", "0x3f000000"),
        ("1m05", "0x3f800000", "0xbf000000"),
        ("15p15", "0x3fc00000", "0x3fc00000"),
        ("1ptiny", "0x3f800000", "0x30800000"),
        ("3p5", "0x40400000", "0x40a00000"),
        ("125p05", "0x3fa00000", "0x3f000000"),
        ("03p02", "0x3e99999a", "0x3e4ccccd"),
        ("07p01", "0x3f333333", "0x3dcccccd"),
        ("1p3", "0x3f800000", "0x40400000"),
        ("m1p3", "0xbf800000", "0x40400000"),
        ("3m1", "0x40400000", "0xbf800000"),
        ("1pm1", "0x3f800000", "0xbf800000"),
        ("2p1", "0x40000000", "0x3f800000"),
        ("big", "0x60ad78ec", "0x5fd02ab5")):
    for op in ("fadd.s", "fadd.d"):
        SAXPY[f"zz_sv_{op.replace('.', '')}_{name}"] = (
            _zz_spaced(op, "f20, f21", 1, 0), saxpy_setup(0, 0) + _zv_setup(a, b))


# fma.s a*b+c: an exact (x) or inexact (i) product, c 2^-d its size
# (name d+3), added (p) or subtracted (m).
def _zg_setup(a, b, c):
    return _zv_setup(a, b) + [f"movl r18 = {c}", "st4 [r33] = r18 ;;",
                              "ldfs f22 = [r33] ;;"]


for name, a, b, c in (
        ("x_d0p", "0x3fc00000", "0x3f800000", "0x41200000"),
        ("x_d0m", "0x3fc00000", "0x3f800000", "0xc1200000"),
        ("x_d1p", "0x3fc00000", "0x3f800000", "0x40a00000"),
        ("x_d1m", "0x3fc00000", "0x3f800000", "0xc0a00000"),
        ("x_d2p", "0x3fc00000", "0x3f800000", "0x40200000"),
        ("x_d2m", "0x3fc00000", "0x3f800000", "0xc0200000"),
        ("x_d3p", "0x3fc00000", "0x3f800000", "0x3fa00000"),
        ("x_d3m", "0x3fc00000", "0x3f800000", "0xbfa00000"),
        ("x_d4p", "0x3fc00000", "0x3f800000", "0x3f200000"),
        ("x_d4m", "0x3fc00000", "0x3f800000", "0xbf200000"),
        ("x_d5p", "0x3fc00000", "0x3f800000", "0x3ea00000"),
        ("x_d5m", "0x3fc00000", "0x3f800000", "0xbea00000"),
        ("x_d6p", "0x3fc00000", "0x3f800000", "0x3e200000"),
        ("x_d6m", "0x3fc00000", "0x3f800000", "0xbe200000"),
        ("i_d0p", "0x3f8ccccd", "0x3fa66666", "0x41200000"),
        ("i_d0m", "0x3f8ccccd", "0x3fa66666", "0xc1200000"),
        ("i_d1p", "0x3f8ccccd", "0x3fa66666", "0x40a00000"),
        ("i_d1m", "0x3f8ccccd", "0x3fa66666", "0xc0a00000"),
        ("i_d2p", "0x3f8ccccd", "0x3fa66666", "0x40200000"),
        ("i_d2m", "0x3f8ccccd", "0x3fa66666", "0xc0200000"),
        ("i_d3p", "0x3f8ccccd", "0x3fa66666", "0x3fa00000"),
        ("i_d3m", "0x3f8ccccd", "0x3fa66666", "0xbfa00000"),
        ("i_d4p", "0x3f8ccccd", "0x3fa66666", "0x3f200000"),
        ("i_d4m", "0x3f8ccccd", "0x3fa66666", "0xbf200000"),
        ("i_d5p", "0x3f8ccccd", "0x3fa66666", "0x3ea00000"),
        ("i_d5m", "0x3f8ccccd", "0x3fa66666", "0xbea00000"),
        ("i_d6p", "0x3f8ccccd", "0x3fa66666", "0x3e200000"),
        ("i_d6m", "0x3f8ccccd", "0x3fa66666", "0xbe200000")):
    SAXPY[f"zz_sg_{name}"] = (_zz_spaced("fma.s", "f20, f21, f22", 1, 0),
                               saxpy_setup(0, 0) + _zg_setup(a, b, c))


# fadd.s at an exponent gap of -1 and +-2, with long significands.
for name, a, b in (
        ("m1_03_07", "0x3e99999a", "0x3f333333"),
        ("m1_12_28", "0x3f99999a", "0x40333333"),
        ("m1_11_23", "0x3f8ccccd", "0x40133333"),
        ("m1_19_29", "0x3ff33333", "0x4039999a"),
        ("m1_15_35", "0x3fc00000", "0x40600000"),
        ("m1_125_225", "0x3fa00000", "0x40100000"),
        ("m1_11_25", "0x3f8ccccd", "0x40200000"),
        ("m1_15_23", "0x3fc00000", "0x40133333"),
        ("m2_11_47", "0x3f8ccccd", "0x40966666"),
        ("m2_15_45", "0x3fc00000", "0x40900000"),
        ("p2_47_11", "0x40966666", "0x3f8ccccd"),
        ("m1_m11_23", "0xbf8ccccd", "0x40133333"),
        ("m1_11_m23", "0x3f8ccccd", "0xc0133333"),
        ("m1_1p_2", "0x3f800001", "0x40000000"),
        ("m1_1_2p", "0x3f800000", "0x40000001")):
    SAXPY[f"zz_sw_{name}"] = (_zz_spaced("fadd.s", "f20, f21", 1, 0),
                               saxpy_setup(0, 0) + _zv_setup(a, b))


# fadd.s 1.25*2^e + 1.5*2^(e-d) (named e+8, d+3) over exponents.
for name, a, b in (
        ("e0_d1", "0x3ba00000", "0x3cc00000"),
        ("e0_d2", "0x3ba00000", "0x3c400000"),
        ("e0_d5", "0x3ba00000", "0x3ac00000"),
        ("e1_d1", "0x3c200000", "0x3d400000"),
        ("e1_d2", "0x3c200000", "0x3cc00000"),
        ("e1_d5", "0x3c200000", "0x3b400000"),
        ("e2_d1", "0x3ca00000", "0x3dc00000"),
        ("e2_d2", "0x3ca00000", "0x3d400000"),
        ("e2_d5", "0x3ca00000", "0x3bc00000"),
        ("e3_d1", "0x3d200000", "0x3e400000"),
        ("e3_d2", "0x3d200000", "0x3dc00000"),
        ("e3_d5", "0x3d200000", "0x3c400000"),
        ("e4_d1", "0x3da00000", "0x3ec00000"),
        ("e4_d2", "0x3da00000", "0x3e400000"),
        ("e4_d5", "0x3da00000", "0x3cc00000"),
        ("e5_d1", "0x3e200000", "0x3f400000"),
        ("e5_d2", "0x3e200000", "0x3ec00000"),
        ("e5_d5", "0x3e200000", "0x3d400000"),
        ("e6_d1", "0x3ea00000", "0x3fc00000"),
        ("e6_d2", "0x3ea00000", "0x3f400000"),
        ("e6_d5", "0x3ea00000", "0x3dc00000"),
        ("e7_d1", "0x3f200000", "0x40400000"),
        ("e7_d2", "0x3f200000", "0x3fc00000"),
        ("e7_d5", "0x3f200000", "0x3e400000"),
        ("e8_d1", "0x3fa00000", "0x40c00000"),
        ("e8_d2", "0x3fa00000", "0x40400000"),
        ("e8_d5", "0x3fa00000", "0x3ec00000"),
        ("e9_d1", "0x40200000", "0x41400000"),
        ("e9_d2", "0x40200000", "0x40c00000"),
        ("e9_d5", "0x40200000", "0x3f400000"),
        ("e10_d1", "0x40a00000", "0x41c00000"),
        ("e10_d2", "0x40a00000", "0x41400000"),
        ("e10_d5", "0x40a00000", "0x3fc00000"),
        ("e11_d1", "0x41200000", "0x42400000"),
        ("e11_d2", "0x41200000", "0x41c00000"),
        ("e11_d5", "0x41200000", "0x40400000"),
        ("e12_d1", "0x41a00000", "0x42c00000"),
        ("e12_d2", "0x41a00000", "0x42400000"),
        ("e12_d5", "0x41a00000", "0x40c00000"),
        ("e13_d1", "0x42200000", "0x43400000"),
        ("e13_d2", "0x42200000", "0x42c00000"),
        ("e13_d5", "0x42200000", "0x41400000"),
        ("e14_d1", "0x42a00000", "0x43c00000"),
        ("e14_d2", "0x42a00000", "0x43400000"),
        ("e14_d5", "0x42a00000", "0x41c00000"),
        ("e15_d1", "0x43200000", "0x44400000"),
        ("e15_d2", "0x43200000", "0x43c00000"),
        ("e15_d5", "0x43200000", "0x42400000"),
        ("e16_d1", "0x43a00000", "0x44c00000"),
        ("e16_d2", "0x43a00000", "0x44400000"),
        ("e16_d5", "0x43a00000", "0x42c00000")):
    SAXPY[f"zz_se_{name}"] = (_zz_spaced("fadd.s", "f20, f21", 1, 0),
                               saxpy_setup(0, 0) + _zv_setup(a, b))


# fadd.s 1.25*2^e + 1.5*2^(e-d) (named e+50, d+4): the stall's region.
# On the 9340 a single-precision add (fadd.s, fma.s, fms.s, fnma.s in a
# status field without wre) stalls the pipeline 3 cycles, once per issue
# group, when its addend b is nonzero and below 2 in magnitude and the
# product a has an exponent from 2 below b's to 1 above it:
# exp(b) <= 0 and -2 <= exp(a) - exp(b) <= 1, for e from -40 to 40.
for name, a, b in (
        ("e40_d0", "0x3aa00000", "0x3cc00000"),
        ("e40_d1", "0x3aa00000", "0x3c400000"),
        ("e40_d2", "0x3aa00000", "0x3bc00000"),
        ("e40_d3", "0x3aa00000", "0x3b400000"),
        ("e40_d4", "0x3aa00000", "0x3ac00000"),
        ("e40_d5", "0x3aa00000", "0x3a400000"),
        ("e40_d6", "0x3aa00000", "0x39c00000"),
        ("e40_d7", "0x3aa00000", "0x39400000"),
        ("e40_d8", "0x3aa00000", "0x38c00000"),
        ("e41_d0", "0x3b200000", "0x3d400000"),
        ("e41_d1", "0x3b200000", "0x3cc00000"),
        ("e41_d2", "0x3b200000", "0x3c400000"),
        ("e41_d3", "0x3b200000", "0x3bc00000"),
        ("e41_d4", "0x3b200000", "0x3b400000"),
        ("e41_d5", "0x3b200000", "0x3ac00000"),
        ("e41_d6", "0x3b200000", "0x3a400000"),
        ("e41_d7", "0x3b200000", "0x39c00000"),
        ("e41_d8", "0x3b200000", "0x39400000"),
        ("e42_d0", "0x3ba00000", "0x3dc00000"),
        ("e42_d1", "0x3ba00000", "0x3d400000"),
        ("e42_d2", "0x3ba00000", "0x3cc00000"),
        ("e42_d3", "0x3ba00000", "0x3c400000"),
        ("e42_d4", "0x3ba00000", "0x3bc00000"),
        ("e42_d5", "0x3ba00000", "0x3b400000"),
        ("e42_d6", "0x3ba00000", "0x3ac00000"),
        ("e42_d7", "0x3ba00000", "0x3a400000"),
        ("e42_d8", "0x3ba00000", "0x39c00000"),
        ("e43_d0", "0x3c200000", "0x3e400000"),
        ("e43_d1", "0x3c200000", "0x3dc00000"),
        ("e43_d2", "0x3c200000", "0x3d400000"),
        ("e43_d3", "0x3c200000", "0x3cc00000"),
        ("e43_d4", "0x3c200000", "0x3c400000"),
        ("e43_d5", "0x3c200000", "0x3bc00000"),
        ("e43_d6", "0x3c200000", "0x3b400000"),
        ("e43_d7", "0x3c200000", "0x3ac00000"),
        ("e43_d8", "0x3c200000", "0x3a400000"),
        ("e44_d0", "0x3ca00000", "0x3ec00000"),
        ("e44_d1", "0x3ca00000", "0x3e400000"),
        ("e44_d2", "0x3ca00000", "0x3dc00000"),
        ("e44_d3", "0x3ca00000", "0x3d400000"),
        ("e44_d4", "0x3ca00000", "0x3cc00000"),
        ("e44_d5", "0x3ca00000", "0x3c400000"),
        ("e44_d6", "0x3ca00000", "0x3bc00000"),
        ("e44_d7", "0x3ca00000", "0x3b400000"),
        ("e44_d8", "0x3ca00000", "0x3ac00000"),
        ("e45_d0", "0x3d200000", "0x3f400000"),
        ("e45_d1", "0x3d200000", "0x3ec00000"),
        ("e45_d2", "0x3d200000", "0x3e400000"),
        ("e45_d3", "0x3d200000", "0x3dc00000"),
        ("e45_d4", "0x3d200000", "0x3d400000"),
        ("e45_d5", "0x3d200000", "0x3cc00000"),
        ("e45_d6", "0x3d200000", "0x3c400000"),
        ("e45_d7", "0x3d200000", "0x3bc00000"),
        ("e45_d8", "0x3d200000", "0x3b400000"),
        ("e46_d0", "0x3da00000", "0x3fc00000"),
        ("e46_d1", "0x3da00000", "0x3f400000"),
        ("e46_d2", "0x3da00000", "0x3ec00000"),
        ("e46_d3", "0x3da00000", "0x3e400000"),
        ("e46_d4", "0x3da00000", "0x3dc00000"),
        ("e46_d5", "0x3da00000", "0x3d400000"),
        ("e46_d6", "0x3da00000", "0x3cc00000"),
        ("e46_d7", "0x3da00000", "0x3c400000"),
        ("e46_d8", "0x3da00000", "0x3bc00000"),
        ("e47_d0", "0x3e200000", "0x40400000"),
        ("e47_d1", "0x3e200000", "0x3fc00000"),
        ("e47_d2", "0x3e200000", "0x3f400000"),
        ("e47_d3", "0x3e200000", "0x3ec00000"),
        ("e47_d4", "0x3e200000", "0x3e400000"),
        ("e47_d5", "0x3e200000", "0x3dc00000"),
        ("e47_d6", "0x3e200000", "0x3d400000"),
        ("e47_d7", "0x3e200000", "0x3cc00000"),
        ("e47_d8", "0x3e200000", "0x3c400000"),
        ("e48_d0", "0x3ea00000", "0x40c00000"),
        ("e48_d1", "0x3ea00000", "0x40400000"),
        ("e48_d2", "0x3ea00000", "0x3fc00000"),
        ("e48_d3", "0x3ea00000", "0x3f400000"),
        ("e48_d4", "0x3ea00000", "0x3ec00000"),
        ("e48_d5", "0x3ea00000", "0x3e400000"),
        ("e48_d6", "0x3ea00000", "0x3dc00000"),
        ("e48_d7", "0x3ea00000", "0x3d400000"),
        ("e48_d8", "0x3ea00000", "0x3cc00000"),
        ("e49_d0", "0x3f200000", "0x41400000"),
        ("e49_d1", "0x3f200000", "0x40c00000"),
        ("e49_d2", "0x3f200000", "0x40400000"),
        ("e49_d3", "0x3f200000", "0x3fc00000"),
        ("e49_d4", "0x3f200000", "0x3f400000"),
        ("e49_d5", "0x3f200000", "0x3ec00000"),
        ("e49_d6", "0x3f200000", "0x3e400000"),
        ("e49_d7", "0x3f200000", "0x3dc00000"),
        ("e49_d8", "0x3f200000", "0x3d400000"),
        ("e50_d0", "0x3fa00000", "0x41c00000"),
        ("e50_d1", "0x3fa00000", "0x41400000"),
        ("e50_d2", "0x3fa00000", "0x40c00000"),
        ("e50_d3", "0x3fa00000", "0x40400000"),
        ("e50_d4", "0x3fa00000", "0x3fc00000"),
        ("e50_d5", "0x3fa00000", "0x3f400000"),
        ("e50_d6", "0x3fa00000", "0x3ec00000"),
        ("e50_d7", "0x3fa00000", "0x3e400000"),
        ("e50_d8", "0x3fa00000", "0x3dc00000"),
        ("e51_d0", "0x40200000", "0x42400000"),
        ("e51_d1", "0x40200000", "0x41c00000"),
        ("e51_d2", "0x40200000", "0x41400000"),
        ("e51_d3", "0x40200000", "0x40c00000"),
        ("e51_d4", "0x40200000", "0x40400000"),
        ("e51_d5", "0x40200000", "0x3fc00000"),
        ("e51_d6", "0x40200000", "0x3f400000"),
        ("e51_d7", "0x40200000", "0x3ec00000"),
        ("e51_d8", "0x40200000", "0x3e400000"),
        ("e52_d0", "0x40a00000", "0x42c00000"),
        ("e52_d1", "0x40a00000", "0x42400000"),
        ("e52_d2", "0x40a00000", "0x41c00000"),
        ("e52_d3", "0x40a00000", "0x41400000"),
        ("e52_d4", "0x40a00000", "0x40c00000"),
        ("e52_d5", "0x40a00000", "0x40400000"),
        ("e52_d6", "0x40a00000", "0x3fc00000"),
        ("e52_d7", "0x40a00000", "0x3f400000"),
        ("e52_d8", "0x40a00000", "0x3ec00000"),
        ("e53_d0", "0x41200000", "0x43400000"),
        ("e53_d1", "0x41200000", "0x42c00000"),
        ("e53_d2", "0x41200000", "0x42400000"),
        ("e53_d3", "0x41200000", "0x41c00000"),
        ("e53_d4", "0x41200000", "0x41400000"),
        ("e53_d5", "0x41200000", "0x40c00000"),
        ("e53_d6", "0x41200000", "0x40400000"),
        ("e53_d7", "0x41200000", "0x3fc00000"),
        ("e53_d8", "0x41200000", "0x3f400000"),
        ("e54_d0", "0x41a00000", "0x43c00000"),
        ("e54_d1", "0x41a00000", "0x43400000"),
        ("e54_d2", "0x41a00000", "0x42c00000"),
        ("e54_d3", "0x41a00000", "0x42400000"),
        ("e54_d4", "0x41a00000", "0x41c00000"),
        ("e54_d5", "0x41a00000", "0x41400000"),
        ("e54_d6", "0x41a00000", "0x40c00000"),
        ("e54_d7", "0x41a00000", "0x40400000"),
        ("e54_d8", "0x41a00000", "0x3fc00000"),
        ("e55_d0", "0x42200000", "0x44400000"),
        ("e55_d1", "0x42200000", "0x43c00000"),
        ("e55_d2", "0x42200000", "0x43400000"),
        ("e55_d3", "0x42200000", "0x42c00000"),
        ("e55_d4", "0x42200000", "0x42400000"),
        ("e55_d5", "0x42200000", "0x41c00000"),
        ("e55_d6", "0x42200000", "0x41400000"),
        ("e55_d7", "0x42200000", "0x40c00000"),
        ("e55_d8", "0x42200000", "0x40400000"),
        ("e56_d0", "0x42a00000", "0x44c00000"),
        ("e56_d1", "0x42a00000", "0x44400000"),
        ("e56_d2", "0x42a00000", "0x43c00000"),
        ("e56_d3", "0x42a00000", "0x43400000"),
        ("e56_d4", "0x42a00000", "0x42c00000"),
        ("e56_d5", "0x42a00000", "0x42400000"),
        ("e56_d6", "0x42a00000", "0x41c00000"),
        ("e56_d7", "0x42a00000", "0x41400000"),
        ("e56_d8", "0x42a00000", "0x40c00000"),
        ("e57_d0", "0x43200000", "0x45400000"),
        ("e57_d1", "0x43200000", "0x44c00000"),
        ("e57_d2", "0x43200000", "0x44400000"),
        ("e57_d3", "0x43200000", "0x43c00000"),
        ("e57_d4", "0x43200000", "0x43400000"),
        ("e57_d5", "0x43200000", "0x42c00000"),
        ("e57_d6", "0x43200000", "0x42400000"),
        ("e57_d7", "0x43200000", "0x41c00000"),
        ("e57_d8", "0x43200000", "0x41400000"),
        ("e58_d0", "0x43a00000", "0x45c00000"),
        ("e58_d1", "0x43a00000", "0x45400000"),
        ("e58_d2", "0x43a00000", "0x44c00000"),
        ("e58_d3", "0x43a00000", "0x44400000"),
        ("e58_d4", "0x43a00000", "0x43c00000"),
        ("e58_d5", "0x43a00000", "0x43400000"),
        ("e58_d6", "0x43a00000", "0x42c00000"),
        ("e58_d7", "0x43a00000", "0x42400000"),
        ("e58_d8", "0x43a00000", "0x41c00000"),
        ("e59_d0", "0x44200000", "0x46400000"),
        ("e59_d1", "0x44200000", "0x45c00000"),
        ("e59_d2", "0x44200000", "0x45400000"),
        ("e59_d3", "0x44200000", "0x44c00000"),
        ("e59_d4", "0x44200000", "0x44400000"),
        ("e59_d5", "0x44200000", "0x43c00000"),
        ("e59_d6", "0x44200000", "0x43400000"),
        ("e59_d7", "0x44200000", "0x42c00000"),
        ("e59_d8", "0x44200000", "0x42400000"),
        ("e60_d0", "0x44a00000", "0x46c00000"),
        ("e60_d1", "0x44a00000", "0x46400000"),
        ("e60_d2", "0x44a00000", "0x45c00000"),
        ("e60_d3", "0x44a00000", "0x45400000"),
        ("e60_d4", "0x44a00000", "0x44c00000"),
        ("e60_d5", "0x44a00000", "0x44400000"),
        ("e60_d6", "0x44a00000", "0x43c00000"),
        ("e60_d7", "0x44a00000", "0x43400000"),
        ("e60_d8", "0x44a00000", "0x42c00000"),
        ("e10_d0", "0x2ba00000", "0x2dc00000"),
        ("e10_d1", "0x2ba00000", "0x2d400000"),
        ("e10_d2", "0x2ba00000", "0x2cc00000"),
        ("e10_d3", "0x2ba00000", "0x2c400000"),
        ("e10_d4", "0x2ba00000", "0x2bc00000"),
        ("e10_d5", "0x2ba00000", "0x2b400000"),
        ("e10_d6", "0x2ba00000", "0x2ac00000"),
        ("e10_d7", "0x2ba00000", "0x2a400000"),
        ("e10_d8", "0x2ba00000", "0x29c00000"),
        ("e30_d0", "0x35a00000", "0x37c00000"),
        ("e30_d1", "0x35a00000", "0x37400000"),
        ("e30_d2", "0x35a00000", "0x36c00000"),
        ("e30_d3", "0x35a00000", "0x36400000"),
        ("e30_d4", "0x35a00000", "0x35c00000"),
        ("e30_d5", "0x35a00000", "0x35400000"),
        ("e30_d6", "0x35a00000", "0x34c00000"),
        ("e30_d7", "0x35a00000", "0x34400000"),
        ("e30_d8", "0x35a00000", "0x33c00000"),
        ("e70_d0", "0x49a00000", "0x4bc00000"),
        ("e70_d1", "0x49a00000", "0x4b400000"),
        ("e70_d2", "0x49a00000", "0x4ac00000"),
        ("e70_d3", "0x49a00000", "0x4a400000"),
        ("e70_d4", "0x49a00000", "0x49c00000"),
        ("e70_d5", "0x49a00000", "0x49400000"),
        ("e70_d6", "0x49a00000", "0x48c00000"),
        ("e70_d7", "0x49a00000", "0x48400000"),
        ("e70_d8", "0x49a00000", "0x47c00000"),
        ("e90_d0", "0x53a00000", "0x55c00000"),
        ("e90_d1", "0x53a00000", "0x55400000"),
        ("e90_d2", "0x53a00000", "0x54c00000"),
        ("e90_d3", "0x53a00000", "0x54400000"),
        ("e90_d4", "0x53a00000", "0x53c00000"),
        ("e90_d5", "0x53a00000", "0x53400000"),
        ("e90_d6", "0x53a00000", "0x52c00000"),
        ("e90_d7", "0x53a00000", "0x52400000"),
        ("e90_d8", "0x53a00000", "0x51c00000")):
    SAXPY[f"zz_sm_{name}"] = (_zz_spaced("fadd.s", "f20, f21", 1, 0),
                               saxpy_setup(0, 0) + _zv_setup(a, b))
