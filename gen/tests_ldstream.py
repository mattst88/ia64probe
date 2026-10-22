# SPDX-License-Identifier: GPL-3.0-or-later
"""Streams of integer loads that miss L1D and hit L2.

st_zz_dense_i_l{1..4}_s0 (one to four ld4 streams, a group a cycle, each
load's target written again four groups later) cost 13, 15, 23 and 25
cycles a body of 8 groups on an Itanium 9340, where the model says 9, 9,
16 and 16.  Each line of a stream misses L1D once.  Is that the miss
itself, or the next write to the load's target waiting for it?

ldst_n{N}_r{R}: N ld8 streams through r60-r63, a group of loads a cycle,
stride 8 (one miss in 8 loads), each target written again R groups later.
_s64 strides a line per load, so every load misses.  _w steps the
pointers back at the end of each body, so its 64 bytes stay in L1D: the
cost of the body without misses.

ldsx_k{K}: a line a body, missed in group 0 and loaded again in group K.
8 cycles a body at K 0-3 and 7, but 12, 11 and 10 at K 4, 5 and 6: a load
of a line whose fill is arriving waits until it is written.  ldsx_s{S}:
one stream of ld8 at stride S, 1 to 8 lines a body: 11, 13.3, 17 and 16.
"""

from emit import Bundle as B, pair

KB = 1 << 10
STREAM = 1 << 20


def setup(n):
    out = []
    for j in range(n):
        out += [f"movl r18 = {STREAM + j * 256 * KB + j * 64:#x}",
                f"add r{60 + j} = r33, r18"]
    return out


def body(n, R, stride, warm):
    out = []
    for g in range(8):
        inc = stride if not warm or g < 7 else -stride * 7
        lds = [f"ld8 r{64 + 4 * (g % R) + j} = [r{60 + j}], {inc}"
               for j in range(n)] + [None] * (4 - n)
        out += [B("mmi", lds[0], lds[1], "nop.i 0"),
                B("mmi", lds[2], lds[3], "nop.i 0 ;;")]
    return out


def kernels():
    ks = []
    for n in (1, 2, 4):
        for R in (1, 2, 4, 8):
            ks += pair(f"ldst_n{n}_r{R}", "ldstream", body(n, R, 8, False),
                       setup(n), {"n": n, "R": R}, bodies=2, iters=250)
            ks += pair(f"ldst_n{n}_r{R}_w", "ldstream", body(n, R, 8, True),
                       setup(n), {"n": n, "R": R}, bodies=2, iters=250)
    for R in (1, 2, 4, 8):
        ks += pair(f"ldst_n1_r{R}_s64", "ldstream", body(1, R, 64, False),
                   setup(1), {"n": 1, "R": R}, bodies=2, iters=60)
    # One line a body: a load of it in group 0, and with k > 0 a second
    # load of its next word in group k.  Without a target written again.
    for k in range(8):
        out = []
        for g in range(8):
            a = "ld8 r64 = [r60], 64" if g == 0 else None
            b = "ld8 r65 = [r61], 64" if k and g == k else None
            out += [B("mmi", a, b, "nop.i 0"),
                    B("mmi", None, None, "nop.i 0 ;;")]
        if not k:
            out[0] = B("mmi", "ld8 r64 = [r60], 64", None, "nop.i 0")
        ks += pair(f"ldsx_k{k}", "ldstream", out,
                   ["movl r18 = 0x100000", "add r60 = r33, r18",
                    "adds r61 = 8, r60"], {"k": k}, bodies=2, iters=250)
    # One stream, a load a group, R = 8, at strides 8 to 64: 1 to 8 lines
    # a body.
    for st in (8, 16, 32, 64):
        ks += pair(f"ldsx_s{st}", "ldstream", body(1, 8, st, False),
                   setup(1), {"stride": st}, bodies=2,
                   iters=250 if st < 64 else 60)
    return ks
