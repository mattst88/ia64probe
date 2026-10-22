# SPDX-License-Identifier: GPL-3.0-or-later
"""zstd's hash-table insert, as clang compiles it.

ZSTD_compressBlock_doubleFast, after a match, hashes two positions and
stores their indices into a hash table: an unaligned ld8 done as two
aligned ld8 and shifts, setf.sig, xma.l by a prime, getf.sig, shr.u,
shladd, st4.  On an Itanium 9340 the getf.sig of the second hash is
where the samples land: about 34 cycles an execution, where the model
says 4.  These kernels replay that code over 32 KB of pseudo-random data,
storing into a table of 2^(64-S) words (zh_t{S}).

Hardware: 41 cycles a body at every table size, 4 MB included, and 41
with the data 2 MB long; 43 with an ld4 of each slot before its st4, a
group after the slot's address.  So the code alone does not stall at the
getf.sig: the samples there come from its context in zstd.
"""

from emit import Bundle as B, pair

DATA = 0x80000
TABLE = 2 << 20


def setup(shift, data=DATA, words=4096):
    return [f"movl r18 = {data:#x}", "add r38 = r33, r18",
            "mov r20 = r38", f"movl r21 = {words}",
            "movl r24 = 0x9e3779b97f4a7c15",
            "1:", "shl r25 = r24, 13", "xor r24 = r24, r25",
            "shr.u r25 = r24, 7", "xor r24 = r24, r25",
            "shl r25 = r24, 17", "xor r24 = r24, r25",
            "st8 [r20] = r24, 8", "adds r21 = -1, r21",
            "cmp.ne p6, p0 = 0, r21", "(p6) br.cond.sptk 1b",
            "mov r39 = 0", f"movl r37 = {words * 8 - 17:#x}",
            f"movl r18 = {TABLE:#x}", "add r22 = r33, r18",
            f"mov r30 = {shift}", "mov r26 = 1234",
            "movl r18 = 0xcf1bbcdcb7a56463", "setf.sig f6 = r18",
            "movl r18 = 0x9e3779b185ebca87", "setf.sig f8 = r18"]


BODY = [
    B("mmi", "add r45 = r38, r39", "adds r39 = 13, r39", "nop.i 0 ;;"),
    B("mii", "adds r50 = 8, r45", "and r39 = r39, r37",
      "dep.z r52 = r45, 3, 3 ;;"),
    B("mmi", "and r50 = -8, r50", "and r53 = -8, r45", "nop.i 0 ;;"),
    # From here on as at 0x40000000000b5850 in the zstd built by clang.
    B("mmi", "ld8 r50 = [r50]", None, "nop.i 0 ;;"),
    B("mmi", "adds r46 = 7, r45", "and r48 = -8, r45",
      "dep.z r45 = r45, 3, 3 ;;"),
    B("mmi", "and r46 = -8, r46", "ld8 r48 = [r48]", "sub r51 = 64, r45 ;;"),
    B("mii", "ld8 r46 = [r46]", "shr.u r45 = r48, r45",
      "sub r48 = 64, r52 ;;"),
    B("mii", None, "shl r46 = r46, r51", "nop.i 0"),
    B("mii", "ld8 r51 = [r53]", "shl r48 = r50, r48 ;;",
      "shr.u r50 = r51, r52"),
    B("mmi", "or r45 = r45, r46 ;;", "setf.sig f9 = r45",
      "or r46 = r50, r48"),
    B("mii", None, None, "nop.i 0 ;;"),
    B("mbb", "setf.sig f10 = r46", "nop.b 0", "nop.b 0"),
    B("mfi", None, "xma.l f11 = f9, f6, f0", "nop.i 0"),
    B("mfi", None, "xma.l f9 = f9, f8, f0", "nop.i 0 ;;"),
    B("mfi", "getf.sig r45 = f11", "xma.l f10 = f10, f6, f0", "nop.i 0 ;;"),
    B("mbb", "getf.sig r46 = f10", "nop.b 0", "nop.b 0"),
    B("mii", None, "shr.u r45 = r45, r30 ;;", "shr.u r46 = r46, r30"),
    B("mmi", "shladd r45 = r45, 2, r22 ;;", "shladd r46 = r46, 2, r22",
      "nop.i 0"),
    B("mii", "st4 [r45] = r26", None, "nop.i 0 ;;"),
    B("mii", "st4 [r46] = r48", None, "nop.i 0 ;;"),
]


def kernels():
    ks = []
    for s in (64, 56, 52, 49, 47, 44):
        ks += pair(f"zh_t{s}", "zhash", BODY, setup(s), {"shift": s},
                   bodies=4, iters=500)
    # Without the stores.
    nost = BODY[:-2] + [B("mii", None, None, "nop.i 0 ;;")]
    ks += pair("zh_nost", "zhash", nost, setup(47), {}, bodies=4, iters=500)
    # 2 MB of data at 4 MB, so the unaligned loads miss L1D and L2.
    ks += pair("zh_d2m", "zhash", BODY, setup(47, 4 << 20, 1 << 18), {},
               bodies=4, iters=500)
    # Each st4 after an ld4 of its slot, as the main loop does.
    # The ld4 of r46's slot can't share the group that computes r46.
    ldst = BODY[:-3] + [
        B("mmi", "shladd r45 = r45, 2, r22 ;;", "shladd r46 = r46, 2, r22",
          "nop.i 0 ;;"),
        B("mii", "ld4 r47 = [r45]", None, "nop.i 0"),
        B("mii", "ld4 r49 = [r46]", None, "nop.i 0 ;;"),
        B("mii", "st4 [r45] = r26", None, "nop.i 0 ;;"),
        B("mii", "st4 [r46] = r48", None, "nop.i 0 ;;")]
    for s in (52, 47, 44):
        ks += pair(f"zh_ld_t{s}", "zhash", ldst, setup(s), {}, bodies=4,
                   iters=500)
    return ks
