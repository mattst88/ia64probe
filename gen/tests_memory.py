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
251110-003 6.6.3.2), and ALAT entries a store kills (alat: the ALAT
compares address bits 11:3 only).  Then 1- and 2-byte stores to one word,
which leave the store queue 5 cycles apart (seqst, st1st8, st8st1), and how
many queue before the pipeline waits (strun, stq, stqw: about 18); glibc's
byte copy loop (copy1, copyrun), and a load in a store's group with the
same address bits 11:2 (alias).
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
# Without the store at all.
STORES["alat_nostore"] = ([one("m", "ld8.a r34 = [r33]"), GAP,
                           one("m", "ld8.c.clr r34 = [r33]")], WARM)
STORES["alat_nostore_nc"] = ([one("m", "ld8.a r34 = [r33]"), GAP,
                              one("m", "ld8.c.nc r34 = [r33]")], WARM)

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
    # Deeper than stq: RUN st1s over RUN/8 words, then enough groups to drain.
    for run in (8, 12, 16, 20, 24, 32):
        ks += pair(f"st_stqw{run}", "memory",
                   [one("m", "st1 [r34] = r40, 1")] * run
                   + [B("mmi", None, None, f"adds r34 = {-run % 8}, r34 ;;")]
                   + [GAP] * (127 - run), SSETUP, bodies=4)
    return ks
