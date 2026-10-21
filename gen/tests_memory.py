# SPDX-License-Identifier: GPL-3.0-or-later
"""The memory hierarchy: load latency by footprint, and store throughput.

chase_*: ld8 r34 = [r34] around a ring of pointers covering a region of the
given size, one pointer per 128-byte line in a random order (the harness
builds the rings; see harness/scratch.c).  The cycles per body are the load
latency to an address consumer, N + 1 in 308065-001 Table 2-8, so N is one
less: 1 for L1D, 5 for L2D, 14-15 for L3 and about 180-225 for memory.  The
cache sizes are those the 9340 reports: L1D 16 KB, L2D 256 KB, L3 5 MB.
Rings beyond 128 KB visit more 64 KB pages than the data TLB holds, so the
large ones include TLB misses, as random accesses in real code do.

fchase_*: the same with ldf8 and getf.sig, so N is replaced by an FP load's
M + 1 (FP loads skip L1D) plus getf's 6 to an address.

st_*: store throughput, measured only.
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
    return ks
