# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectral-norm's II=6 rotating kernel (clang, eval_AtA_times_u's second
loop), whole and with one instruction at a time made a nop.

The kernel's integer chains live in rotating registers the harness owns, so
they are rewritten over static ones with the same dependences: an add per
pass to a running index, an extract, a sign extension and a setf.sig of
their sum.  All of f32-f127 start at 1.0, and the vector the
ldfd walks holds 1.0, so the values stay as tame as spectral-norm's.
"""

from __future__ import annotations

from emit import Bundle as B, pair

N = 128

# (template, slot 0, slot 1, slot 2); ";;" ends a group.
KERNEL = [
    ("mfi", None, "(p27) fma.s1 f32 = f37, f35, f35",
     "(p16) extr.u r24 = r20, 1, 31"),
    ("mfi", "(p16) add r20 = r20, r21", "(p29) fma.d f73 = f39, f41, f64",
     "(p16) adds r21 = 2, r21 ;;"),
    ("mfi", None, "(p26) fma.s1 f36 = f43, f43, f0",
     "(p16) add r24 = r22, r24"),
    ("mfi", None, "(p26) fma.s1 f34 = f43, f70, f70",
     "(p16) adds r22 = 1, r22 ;;"),
    ("mfi", None, "(p28) fnma.s1 f38 = f49, f63, f1", None),
    ("mfi", None, "(p28) fma.s1 f40 = f67, f33, f33",
     "(p16) sxt4 r24 = r24 ;;"),
    ("mfi", None, "(p17) fcvt.xf f44 = f51", None),
    ("mfi", "(p16) setf.sig f50 = r24", "(p25) fnma.s1 f42 = f46, f69, f1",
     "nop.i 0 ;;"),
    ("mfi", None, "(p23) fma.d f58 = f73, f61, f59", None),
    ("mfi", "(p22) ldfd f60 = [r23], 8", "(p27) fma.d.s1 f62 = f66, f32, f32",
     "nop.i 0 ;;"),
    ("mfi", None, "(p26) fma.s1 f65 = f36, f36, f0", None),
    ("mfb", None, "(p18) frcpa f68, p24 = f1, f45", "br.ctop.sptk.few 1b ;;"),
]

NOP = {"m": "nop.m 0", "f": "nop.f 0", "i": "nop.i 0"}


def body(drop=None):
    out = [B("mii", "mov r23 = r33", "mov r62 = ar.lc",
             f"adds r25 = {N - 1}, r0 ;;"),
           B("mii", "mov r20 = 1", "mov r21 = 3", "mov r22 = 5 ;;"),
           B("mii", None, "mov.i ar.lc = r25", "mov.i ar.ec = 1 ;;"),
           B("mii", None, "mov pr.rot = 0x10000", "nop.i 0 ;;")]
    for i, (t, *slots) in enumerate(KERNEL):
        slots = list(slots)
        for j, s in enumerate(slots):
            if drop == (i, j):
                slots[j] = NOP[t[j]] + (" ;;" if s.endswith(";;") else "")
        out.append(B(t, *slots, label="1" if i == 0 else None))
    out += [B("mii", None, "mov.i ar.lc = r62", "nop.i 0 ;;")]
    return out


def setup():
    one = ["movl r18 = 0x3ff0000000000000", "mov r19 = r33"]
    for _ in range(N):
        one.append("st8 [r19] = r18, 8")
    return ([f"mov f{r} = f1" for r in range(32, 128)] + one +
            ["mov r24 = 7"])


def kernels():
    ks = pair("spk_all", "spk", body(), setup(), meta={"n": N}, bodies=2)
    for i, (t, *slots) in enumerate(KERNEL):
        for j, s in enumerate(slots):
            if s is None or s.startswith("nop") or "br.ctop" in s:
                continue
            op = s.split(")")[1].split()[0] if ")" in s else s.split()[0]
            op = op.replace(".", "_")
            ks += pair(f"spk_d{i}_{j}_{op}", "spk", body((i, j)), setup(),
                       meta={"n": N, "drop": s}, bodies=2)
    return ks
