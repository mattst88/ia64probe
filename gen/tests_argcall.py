# SPDX-License-Identifier: GPL-3.0-or-later
"""A load's value crossing a call or a return in a stacked register.

The caller's output registers are the callee's r32 on, so a callee that
reads an argument a load is still writing waits for it, as health's
my_rand waits on its caller's load of village->seed.  Each body follows a
ring of pointers through a call: argcall_SIZE loads the next pointer into
out0 and the callee moves its in0 to the cursor; retcall_SIZE has the
callee load it into its r32 and the caller move its out0 to the cursor;
statcall_SIZE loads it into the cursor itself, around an empty call.
Each should take the ring's load latency plus the call's.

The setup reallocates the harness's frame as 2 inputs, 60 locals and 4
outputs (r94-r97), so a callee with no alloc of its own has those four.
"""

import re

from emit import Bundle as B, pair
from tests_memory import KB, MB, name_of, ring_setup, RING_TEARDOWN


def body(caller, callee, after=None):
    return [B("mib", caller, None, "br.call.sptk.many b6 = 2f ;;"),
            B("mib", after, None, "br.cond.sptk.few 3f ;;"),
            B("mib", callee, None, "br.ret.sptk.many b6 ;;", label="2"),
            B("mmi", None, None, "nop.i 0 ;;", label="3")]


VARIANTS = {
    "argcall": body("ld8 r94 = [r20]", "mov r20 = r32"),
    "retcall": body(None, "ld8 r32 = [r20]", "mov r20 = r94"),
    "statcall": body("ld8 r20 = [r20]", None),
}


# A call and return alone: cr_gN puts N groups in the callee before its
# br.ret, and cr_aN N groups in the caller after the call returns to it.
def callret(g, a=0):
    nop = B("mmi", None, None, "nop.i 0 ;;")
    return ([B("mib", None, None, "br.call.sptk.many b6 = 2f ;;")] +
            [nop] * a +
            [B("mib", None, None, "br.cond.sptk.few 3f ;;")] +
            [B("mmi", None, None, "nop.i 0 ;;", label="2")] * min(g, 1) +
            [nop] * max(g - 1, 0) +
            [B("mib", None, None, "br.ret.sptk.many b6 ;;",
               label=None if g else "2"),
             B("mmi", None, None, "nop.i 0 ;;", label="3")])


# Two IP-relative taken branches in a row: the first jumps over f1 bundles
# to the second, which jumps over f2 to a nop; f1 = 2 makes the second
# branch the odd bundle of its pair.  The body is padded to an even length
# so every copy has the same parity.
def taken2(f1, f2):
    fill = B("mmi", None, None, "nop.i 0 ;;")
    out = ([B("mib", None, None, "br.sptk.few 1f ;;")] + [fill] * f1 +
           [B("mib", None, None, "br.sptk.few 2f ;;", label="1")] +
           [fill] * f2 + [B("mmi", None, None, "nop.i 0 ;;", label="2")])
    return out + [fill] * (len(out) % 2)


# A return to an odd bundle (_o) or an even one, which branches (_t) or
# does not; the caller pads so that the return lands where it should.
def retpar(odd, taken):
    fill = B("mmi", None, None, "nop.i 0 ;;")
    out = [B("mib", None, None, "br.call.sptk.many b6 = 2f ;;")]
    if not odd:
        out = [fill] + out
    tgt = (B("mib", None, None, "br.sptk.few 3f ;;") if taken else
           B("mmi", None, None, "nop.i 0 ;;"))
    out += [tgt]
    if not taken:
        out += [B("mib", None, None, "br.sptk.few 3f ;;")]
    out += [fill, B("mib", None, None, "br.ret.sptk.many b6 ;;", label="2"),
            B("mmi", None, None, "nop.i 0 ;;", label="3")]
    return out + [fill] * (len(out) % 2)


# fread's entry: setf.sig of size and count in successive groups,
# xmpy.l, getf.sig, and a compare that branches on the product.  fm_full
# is the code as glibc has it, fm_nob without the branch, fm_nomul with
# the multiply replaced by nops and the compare on count, fm_inl the full
# sequence without the call around it.
def fmul(mul=True, br=True, call=True, args=False, pad=0, ind=False):
    # Inline, the arguments are the harness frame's outputs.
    a = ({} if call else {"r32": "r94", "r33": "r95", "r34": "r96"})
    def r(x):
        return x and re.sub(r"r3[234]\b", lambda g: a.get(g[0], g[0]), x)
    m = (lambda x: r(x)) if mul else (lambda x: None)
    seq = [B("mmb", r("alloc r35 = ar.pfs, 3, 1, 0, 0") if call else None,
             "mov r21 = r12", "nop.b 0", label="2" if call else None),
           B("mmi", "adds r22 = -16, r12", m("setf.sig f6 = r33"),
             "mov r23 = b0 ;;"),
           B("mmi", m("setf.sig f7 = r34"), None, r("adds r24 = 136, r32")),
           B("mmi", "mov r25 = r1", None, "adds r26 = -1856, r13 ;;"),
           B("mmf", None, None, "xmpy.l f6 = f6, f7 ;;" if mul
             else "nop.f 0 ;;"),
           B("mmi", None, m("getf.sig r20 = f6"), "nop.i 0 ;;"),
           B("mib", "cmp.eq p6, p7 = 0, r20" if mul
             else r("cmp.eq p6, p7 = 0, r34"), None,
             "(p6) br.cond.dpnt.few 3f ;;" if br else "nop.b 0 ;;")]
    tail = [B("mmi", None, None, "nop.i 0 ;;")] * pad
    if not call:
        return seq + [B("mmi", None, None, "nop.i 0 ;;", label="3")] + tail
    return ([B("mmi", "mov r95 = 1", "mov r96 = 16", "nop.i 0 ;;")] * args +
            ([B("mlx", None, "movl r27 = 2f ;;"),
              B("mmi", None, None, "mov b7 = r27 ;;"),
              B("mib", None, None, "br.call.sptk.many b6 = b7 ;;")] if ind
             else [B("mib", None, None, "br.call.sptk.many b6 = 2f ;;")]) +
            [B("mib", None, None, "br.cond.sptk.few 3f ;;")] + seq +
            [B("mib", None, "mov.i ar.pfs = r35", "br.ret.sptk.many b6 ;;"),
             B("mmi", None, None, "nop.i 0 ;;", label="3")] + tail)


def kernels():
    ks = []
    fsetup = ["alloc r2 = ar.pfs, 2, 60, 4, 0", "mov r95 = 1",
              "mov r96 = 16"]
    for name, kw in (("full", {}), ("nob", {"br": False}),
                     ("nomul", {"mul": False}), ("inl", {"call": False}),
                     ("inlnomul", {"call": False, "mul": False}),
                     ("args", {"args": True}),
                     ("argsnomul", {"args": True, "mul": False}),
                     ("ind", {"ind": True, "args": True}),
                     ("indnomul", {"ind": True, "args": True,
                                   "mul": False}),
                     ("pad", {"pad": 64}),
                     ("padnomul", {"pad": 64, "mul": False}),
                     ("inlpad", {"call": False, "pad": 64}),
                     ("inlpadnomul", {"call": False, "pad": 64,
                                      "mul": False})):
        ks += pair(f"fm_{name}", "argcall", fmul(**kw), fsetup, bodies=4)
    for f1 in (1, 2):
        for f2 in (1, 2):
            ks += pair(f"tt2_f{f1}{f2}", "argcall", taken2(f1, f2), bodies=4)
    for odd in (0, 1):
        for taken in (0, 1):
            ks += pair(f"retpar_{'o' if odd else 'e'}{'t' if taken else 'n'}",
                       "argcall", retpar(odd, taken), bodies=4)
    for g in range(5):
        for a in range(3):
            ks += pair(f"cr_g{g}_a{a}", "argcall", callret(g, a), bodies=4)
    for size in (16 * KB, 128 * KB, 2 * MB, 64 * MB):
        setup = (["alloc r2 = ar.pfs, 2, 60, 4, 0"] + ring_setup(size) +
                 ["mov r20 = r34"])
        teardown = ["mov r34 = r20"] + RING_TEARDOWN
        for v, b in VARIANTS.items():
            ks += pair(f"{v}_{name_of(size)}", "argcall", b, setup,
                       {"size": size}, bodies=4, teardown=teardown)
    return ks
