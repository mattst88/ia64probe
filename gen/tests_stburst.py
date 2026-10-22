# SPDX-License-Identifier: GPL-3.0-or-later
"""What waits behind a burst of stores.

ZSTD_compressBlock_doubleFast copies a match's literals a byte at a time,
ld1 and st1, and then hashes two positions: ld8, setf.sig, xma.l,
getf.sig.  On an Itanium 9340 the samples land on the first xma.l, about
43 cycles an execution, where the model says 3; the same hash code alone
(zh_*) does not stall.

sb_{x}_{w}{N}: a body that does N stores of width w (st1 to consecutive
bytes, st8 to consecutive words) two to a group, then x:

  i  an integer add chain, 4 groups
  f  setf.sig; xma.l; getf.sig; add, a group each
  l  ld8 of a hot word of another line; add
"""

from emit import Bundle as B, pair

SETUP = ["mov r40 = 1", "movl r18 = 4096", "add r36 = r33, r18",
         "adds r38 = 64, r33", "ld8 r41 = [r38]", "setf.sig f32 = r40"]

TAIL = {
    "i": [B("mii", None, "add r41 = r41, r40 ;;", "add r41 = r41, r40 ;;"),
          B("mii", None, "add r41 = r41, r40 ;;", "add r41 = r41, r40 ;;")],
    "f": [B("mmi", "setf.sig f40 = r40", None, "nop.i 0 ;;"),
          B("mfi", None, "xma.l f41 = f40, f32, f0", "nop.i 0 ;;"),
          B("mmi", "getf.sig r70 = f41", None, "nop.i 0 ;;"),
          B("mii", None, "add r41 = r41, r70", "nop.i 0 ;;")],
    "l": [B("mmi", "ld8 r70 = [r38]", None, "nop.i 0 ;;"),
          B("mii", None, "add r41 = r41, r70", "nop.i 0 ;;")],
}


def body(x, w, n):
    st, inc = ("st1", 2) if w == "b" else ("st8", 16)
    out = [B("mii", None, "mov r60 = r36", f"adds r62 = {inc // 2}, r36 ;;")]
    for _ in range(n // 2):
        out.append(B("mmi", f"{st} [r60] = r40, {inc}",
                     f"{st} [r62] = r40, {inc}", "nop.i 0 ;;"))
    return out + TAIL[x]


DRAIN = {
    "i": "add r41 = r41, r40",
    "s": "setf.sig f40 = r40",
    "x": "xma.l f41 = f32, f32, f0",
    "g": "getf.sig r70 = f32",
    "l": "ld8 r70 = [r38]",
}


def drain(x, n, at=1, groups=48):
    """n st1 to consecutive bytes, then 48 groups of integer adds with
    DRAIN[x] in group `at`: does x wait for the stores to leave?"""
    out = body("i", "b", n)[:-2]
    for g in range(groups):
        op = DRAIN[x] if g == at else None
        if x in "sxag" and g == at and x != "g":
            out.append(B("mfi", None, op, "add r42 = r42, r40 ;;")
                       if x in "xa" else
                       B("mmi", op, None, "add r42 = r42, r40 ;;"))
        else:
            out.append(B("mmi", op, None, "add r42 = r42, r40 ;;"))
    return out


def kernels():
    ks = []
    # 16 st1 to consecutive bytes: through one pointer a group each, or
    # through two pointers that alternate, a group each or both in one.
    seq = {
        "one": [B("mmi", "st1 [r60] = r40, 1", None, "nop.i 0 ;;")] * 16,
        "two": [B("mmi", f"st1 [r{60 + 2 * (i % 2)}] = r40, 2", None,
                  "nop.i 0 ;;") for i in range(16)],
        "pair": [B("mmi", "st1 [r60] = r40, 2", "st1 [r62] = r40, 2",
                   "nop.i 0 ;;")] * 8,
    }
    for name, sts in seq.items():
        ks += pair(f"sp_{name}", "stburst",
                   [B("mii", None, "mov r60 = r36", "adds r62 = 1, r36 ;;")]
                   + sts, SETUP, {"x": name}, bodies=4, iters=100)
    for n in (8, 16):
        for x in DRAIN:
            ks += pair(f"sd_{x}_b{n}", "stburst", drain(x, n), SETUP,
                       {"x": x, "n": n}, bodies=4, iters=100)
    for w in ("b", "q"):
        for n in (0, 4, 8, 16, 32):
            if n == 0 and w == "q":
                continue
            for x in TAIL:
                ks += pair(f"sb_{x}_{w}{n}", "stburst", body(x, w, n), SETUP,
                           {"x": x, "w": w, "n": n}, bodies=4, iters=200)
    return ks
