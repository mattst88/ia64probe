# SPDX-License-Identifier: GPL-3.0-or-later
"""What waits behind a store that misses L2.

mst's HashInsert stores into freshly allocated lines and into a hot line
of globals; the hashfunc called next loads HashRange from that hot line
and moves it to an FR with setf.sig.  On an Itanium 9340 the samples land
on that setf.sig and on the FP ops after it, about 90 cycles a call, where
the model says 1; and with the call in MakeGraph's loop, on the getf.sig
and setf.sig there.

sm_{x}_s{S}: a body of 8 groups; group 0 stores to a stream at stride S
(S 0 stores to one hot word), group k does x, and group 7 uses x's
result:

  none  nothing
  ld    ld8 of a hot word of another line
  ldsl  ld8 of a word of a hot line that group 0 also stores to
  setf  setf.sig
  getf  getf.sig
"""

from emit import Bundle as B, pair

STREAM = 1 << 20
SETUP = ["mov r40 = 1", "adds r36 = 64, r33", "adds r37 = 128, r33",
         "adds r38 = 136, r33", "ld8 r41 = [r36]", "ld8 r41 = [r37]",
         "setf.sig f32 = r40", f"movl r18 = {STREAM:#x}", "add r60 = r33, r18"]

X = {
    "none": (None, None),
    "ld": ("ld8 r70 = [r36]", "add r41 = r70, r0"),
    "ldsl": ("ld8 r70 = [r38]", "add r41 = r70, r0"),
    "setf": ("setf.sig f40 = r40", None),
    "getf": ("getf.sig r70 = f32", "add r41 = r70, r0"),
}


def setup(stride):
    return SETUP + [f"movl r61 = {stride}"]


def body(x, stride, k):
    op, use = X[x]
    out = []
    for g in range(8):
        m0 = m1 = None
        if g == 0:
            m0 = "st8 [r60] = r40"
            if x == "ldsl":
                m1 = "st8 [r37] = r40"
        if g == k:
            m1 = op
        i = use if g == 7 else "add r60 = r60, r61" if g == 0 else None
        out.append(B("mmi", m0, m1, (i or "nop.i 0") + " ;;"))
    return out


def kernels():
    ks = []
    for stride, iters in ((0, 250), (64, 250), (4160, 200), (65600, 12)):
        for x in X:
            for k in ((1,) if x == "none" else (1, 3, 6)):
                ks += pair(f"sm_{x}_k{k}_s{stride}", "stmiss",
                           body(x, stride, k), setup(stride),
                           {"x": x, "k": k, "stride": stride},
                           bodies=4, iters=iters)
    return ks
