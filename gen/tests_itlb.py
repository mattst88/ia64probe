# SPDX-License-Identifier: GPL-3.0-or-later
"""The L1 ITLB, apart from the caches.

ic_itlb_s{stride}_p{n}: n taken branches stride bytes apart, then back to
the first.  A stride of 4096 + 64 or 16384 + 64 puts each branch in its own
L1I set, so n up to 64 lines stay in L1I and only the ITLB can miss.  Each
branch is in its own 4 KB page, and with the larger stride its own 16 KB
one.  Madison's L1 ITLB is documented as 32 entries of 4 KB (251110-003
6.1); a jump in the cost per branch past p32 at one stride but not the
other says how large its entries are.
"""

from __future__ import annotations

from emit import Bundle as B, Kernel


def body(n, stride):
    out = []
    pad = stride // 16 - 1
    for p in range(n):
        lab = str(20000 + p)
        if p + 1 < n:
            br = f"br.sptk.few {20000 + p + 1}f ;;"
        else:
            br = "nop.b 0 ;;"
        out.append(B("mib", "nop.m 0", "nop.i 0", br, label=lab))
        if p + 1 < n:
            out += [B("mii", "nop.m 0", "nop.i 0", "nop.i 0 ;;")
                    for _ in range(pad)]
    return out


def kernels():
    ks = []
    for stride in (4160, 16448):
        for n in (8, 16, 24, 31, 32, 33, 34, 40, 48, 64):
            name = f"ic_itlb_s{stride}_p{n}"
            b = body(n, stride)
            meta = {"n": n, "stride": stride}
            ks += [Kernel(name, "itlb", b, 1, meta=meta, iters=4096),
                   Kernel(name, "itlb", b, 2, meta=meta, iters=4096)]
    # The L2 ITLB: one chain of n 16 KB pages, each branch still in its own
    # L1I set modulo 64 (L1I holds 256 lines), so past the L1 ITLB only the
    # second level and the VHPT walk change the cost per branch.
    for n in (64, 96, 112, 120, 124, 128, 132, 136, 144, 160, 192, 224, 255):
        name = f"ic_itlb2_s16448_p{n}"
        b = body(n, 16448)
        meta = {"n": n, "stride": 16448}
        ks += [Kernel(name, "itlb", b, 1, meta=meta, iters=1024),
               Kernel(name, "itlb", b, 2, meta=meta, iters=1024)]
    return ks
