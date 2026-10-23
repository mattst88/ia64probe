# SPDX-License-Identifier: GPL-3.0-or-later
"""coremark's core_list_find loop: a list walk whose next-pointer load
clang hoists above the loop's compare-and-branch as ld8.sa, checked by a
ld8.c.clr after it.

Each body is one pass of the loop as clang lays it out: node->info (at
+8), info->idx (at +2), compare and a never-taken exit, then node->next,
and a taken branch on to the next body.  The list is 32 nodes 32 bytes
apart, all in L1D.  The next-pointer load is:

  sa     ld8.sa beside the idx load, ld8.c.clr after the exit (clang)
  plain  ld8 after the exit, where the source has it
  early  ld8 beside the idx load, no check
"""

from __future__ import annotations

from emit import Bundle as B, pair

TABLE = 0xE0000
NODES = 32


def setup() -> list[str]:
    out = []
    for i in range(NODES):
        node = TABLE + 32 * i
        out += [f"movl r24 = {TABLE + 32 * ((i + 1) % NODES):#x}",
                f"movl r25 = {node:#x}", "add r24 = r33, r24",
                "add r25 = r33, r25 ;;", "st8 [r25] = r24, 8",
                "adds r24 = 16, r25 ;;", "st8 [r25] = r24, 8 ;;",
                "st8 [r25] = r0"]
    return out + [f"movl r20 = {TABLE:#x}", "mov r27 = 0xffff",
                  "add r20 = r33, r20"]


def body(kind: str) -> list:
    early = {"sa": "ld8.sa r22 = [r20]", "early": "ld8 r22 = [r20]"}
    late = {"sa": "ld8.c.clr r22 = [r20] ;;", "plain": "ld8 r22 = [r20] ;;"}
    return [B("mmi", "adds r22 = 8, r20 ;;", "ld8 r22 = [r22]", "nop.i 0 ;;"),
            B("mmi", "adds r22 = 2, r22", None, "nop.i 0 ;;"),
            B("mmi", "ld2 r23 = [r22]", early.get(kind), "nop.i 0 ;;"),
            B("mib", "cmp.eq p6, p7 = r23, r27", None,
              "(p6) br.cond.dpnt.few 9f"),
            B("mmi", late.get(kind), "mov r20 = r22", "cmp.eq p9, p8 = 0, r22"),
            B("mib", None, None, "(p8) br.cond.dptk.few 9f ;;"),
            B("mii", None, None, "nop.i 0 ;;", label="9")]


def loop_setup(n: int) -> list[str]:
    """n nodes, the last pointing to NULL, r26 the head."""
    out = []
    for i in range(n):
        node = TABLE + 32 * i
        nxt = TABLE + 32 * (i + 1) if i + 1 < n else 0
        out += [f"movl r24 = {nxt:#x}", f"movl r25 = {node:#x}",
                "add r24 = r33, r24" if nxt else "mov r24 = r0",
                "add r25 = r33, r25 ;;", "st8 [r25] = r24, 8",
                "adds r24 = 16, r25 ;;", "st8 [r25] = r24, 8 ;;",
                "st8 [r25] = r0"]
    return out + [f"movl r26 = {TABLE:#x}", "mov r27 = 0xffff ;;",
                  "add r26 = r33, r26 ;;", "mov r20 = r26"]


def loop_body(pre: int) -> list:
    """The whole loop as clang lays it out, its first bundle pre bundles
    into a 32-byte-aligned 8-bundle body, then back to the list's head."""
    lp = [B("mmi", "adds r22 = 8, r20 ;;", "ld8 r22 = [r22]", "nop.i 0 ;;",
            label="7"),
          B("mlx", "adds r22 = 2, r22", "nop.x 0 ;;"),
          B("mmi", "ld2 r23 = [r22]", "ld8.sa r22 = [r20]", "nop.i 0 ;;"),
          B("mib", "cmp.eq p6, p7 = r23, r27", None,
            "(p6) br.cond.dpnt.few 9f"),
          B("mmi", "ld8.c.clr r22 = [r20] ;;", "mov r20 = r22",
            "cmp.eq p9, p8 = 0, r22"),
          B("mib", None, None, "(p8) br.cond.dptk.few 7b ;;")]
    return ([B("mii", None, None, "nop.i 0 ;;")] * pre + lp
            + [B("mii", "mov r20 = r26", None, "nop.i 0 ;;", label="9")]
            + [B("mii", None, None, "nop.i 0 ;;")] * (1 - pre % 2))


def reverse_body(pre: int, rev: bool) -> list:
    """loop_body's find, then core_list_reverse as clang lays it out (or
    as many bundles of nops), so each find walks a list just rewritten."""
    find = loop_body(pre)[:-1 - (1 - pre % 2)]
    if rev:
        tail = [B("mii", "mov r28 = r26", "mov r29 = r0", "nop.i 0 ;;"),
                B("mmi", "mov r30 = r28", "ld8 r28 = [r28]", "nop.i 0 ;;",
                  label="8"),
                B("mmi", "cmp.eq p7, p6 = 0, r28 ;;", "st8 [r30] = r29",
                  "mov r29 = r30"),
                B("mib", None, None, "(p6) br.cond.dptk.few 8b ;;"),
                B("mii", "mov r26 = r30", None, "nop.i 0 ;;")]
    else:
        tail = [B("mii", None, None, "nop.i 0 ;;")] * 5
    return (find + [B("mii", None, None, "nop.i 0 ;;", label="9")] + tail
            + [B("mii", "mov r20 = r26", None, "nop.i 0 ;;")]
            + [B("mii", None, None, "nop.i 0 ;;")] * (pre % 2))


# coremark's list as core_list_find walks it: each node's address and its
# info's, as offsets in its data segment.
CM_NODES = [0x330, 0x340, 0x360, 0x190, 0x210, 0x290, 0x310, 0x220, 0x2a0,
            0x1b0, 0x230, 0x2b0, 0x1c0, 0x240, 0x2c0, 0x1d0, 0x250, 0x2d0,
            0x1e0, 0x260, 0x2e0, 0x1f0, 0x270, 0x2f0, 0x200, 0x350, 0x320]
CM_INFO = [0x3e8, 0x3ec, 0x3f4, 0x380, 0x3a0, 0x3c0, 0x3e0, 0x3a4, 0x3c4,
           0x388, 0x3a8, 0x3c8, 0x38c, 0x3ac, 0x3cc, 0x390, 0x3b0, 0x3d0,
           0x394, 0x3b4, 0x3d4, 0x398, 0x3b8, 0x3d8, 0x39c, 0x3f0, 0x3e4]


def cm_setup(nodes, info) -> list[str]:
    out = []
    for i, (n, f) in enumerate(zip(nodes, info)):
        nxt = TABLE + nodes[i + 1] if i + 1 < len(nodes) else 0
        out += [f"movl r24 = {nxt:#x}", f"movl r25 = {TABLE + n:#x}",
                "add r24 = r33, r24" if nxt else "mov r24 = r0",
                "add r25 = r33, r25 ;;", "st8 [r25] = r24, 8",
                f"movl r24 = {TABLE + f:#x} ;;", "add r24 = r33, r24 ;;",
                "st8 [r25] = r24"]
    return out + [f"movl r26 = {TABLE + nodes[0]:#x}", "mov r27 = 0xffff ;;",
                  "add r26 = r33, r26 ;;", "mov r20 = r26"]


def kernels():
    ks = []
    ks += pair("lfc_find", "lfind", loop_body(0), cm_setup(CM_NODES, CM_INFO),
               meta={"nodes": len(CM_NODES)})
    ks += pair("lfc_rev", "lfind", reverse_body(0, True),
               cm_setup(CM_NODES, CM_INFO), meta={"nodes": len(CM_NODES)})
    seq = [0x100 + 16 * i for i in range(len(CM_NODES))]
    ks += pair("lfc_seqnode", "lfind", loop_body(0), cm_setup(seq, CM_INFO),
               meta={"nodes": len(CM_NODES)})
    for rev in (True, False):
        ks += pair(f"lfr_{int(rev)}", "lfind", reverse_body(0, rev),
                   loop_setup(28), meta={"nodes": 28, "rev": int(rev)})
    for pre in (0, 1):
        for n in (4, 28):
            ks += pair(f"lfl_{n}_{pre}", "lfind", loop_body(pre),
                       loop_setup(n), meta={"nodes": n, "pre": pre})
    for kind in ("sa", "plain", "early"):
        ks += pair(f"lf_{kind}", "lfind", body(kind), setup(),
                   meta={"kind": kind})
    return ks
