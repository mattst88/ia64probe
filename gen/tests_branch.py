# SPDX-License-Identifier: GPL-3.0-or-later
"""Indirect branches: an interpreter's dispatch.

Each body loads the next entry of a table of P targets, makes it an
address with mov ip, moves it to b6, waits `dist` groups, and branches
through b6 to one of K target blocks, each of which branches to the
body's end.  The table cycles; its entries are a pseudo-random sequence
of the K targets ("rnd"), or each target in turn ("seq").  Without a
penalty a body takes 11 + dist groups.

On a 9340 a body costs 6 cycles more whenever dist (in bundles) is 9 or
less, and nothing from 10 on, whatever K, the order or the hint; the w*
bodies, with two-bundle groups, show that bundles count, not cycles.  The ibv and ibs probes
show the penalty is for a target that differs from the last one taken
through the same register, by any branch: two call sites through b6,
each always to its own target, cost as much as one alternating.  Naming
the branch does not help either: mov.sptk, mov.sptk.imp and mov.dptk to
b6 with the branch as tag cost what a plain mov does, and a brp.sptk
(.imp) of b6 in its own group only adds that group.
"""

from __future__ import annotations

import random

from emit import Bundle as B, pair

TABLE = 0xE0000
P = 16
NOP = B("mii", None, None, "nop.i 0 ;;")


NOP2 = [B("mii", None, None, "nop.i 0"), B("mii", None, None, "nop.i 0 ;;")]


def body(k, dist, hint, direct=False, wide=False, same=False, call=False,
         many=False, movhint=None, brp=None):
    pre = [B("mmi", "ld8 r20 = [r21]", None, "adds r22 = 8, r22 ;;"),
           B("mmi", None, None, "and r22 = r19, r22 ;;"),
           B("mmi", None, None, "add r21 = r18, r22 ;;"),
           B("mii", None, "mov r23 = ip", "nop.i 0 ;;"),
           B("mmi", None, None, "add r20 = r20, r23 ;;")]
    if not same:
        mov = (f"mov.{movhint} b6 = r20, 20002f" if movhint
               else "mov b6 = r20")
        pre.append(B("mii", None, mov, "nop.i 0 ;;"))
    if brp:
        pre.append(B("mib", None, None, f"brp.{brp} b6, 20002f ;;"))
    pre += (NOP2 if wide else [NOP]) * dist
    wh = "many" if many else "few"
    br = (f"br.call.{hint}.{wh} b7 = b6 ;;" if call
          else f"br.{hint}.{wh} b6 ;;")
    pre.append(B("mib", None, "mov b6 = r20" if same else None,
                 "br.sptk.few 20001f ;;" if direct else br,
                 label="20002" if movhint or brp else None))
    if call:
        pre.append(B("mib", None, None, "br.sptk.few 20000f ;;"))
    ip_at = 3
    t0 = len(pre)
    tgt = []
    out = "br.ret.sptk.many b7 ;;" if call else "br.sptk.few 20000f ;;"
    for i in range(k):
        tgt += [B("mii", None, None, "nop.i 0 ;;",
                  label="20001" if i == 0 else None)] + [NOP] * 2 + [B("mib", None, None, out)]
    end = [B("mii", None, None, "nop.i 0 ;;", label="20000")]
    return pre + tgt + end, (t0 - ip_at) * 16


def table(k, order, base):
    if order == "seq":
        seq = [i % k for i in range(P)]
    else:
        rng = random.Random(k)
        seq = [rng.randrange(k) for _ in range(P)]
    return [base + 64 * t for t in seq]


def setup(offs):
    out = [f"movl r18 = {TABLE:#x}", "add r18 = r33, r18", "mov r22 = r0",
           f"mov r19 = {8 * P - 8}", "mov r25 = r18"]
    for o in offs:
        out += [f"movl r24 = {o}", "st8 [r25] = r24, 8"]
    out.append("mov r21 = r18")
    return out


# Two routines after the loop that return through b7; r24 and r25 hold
# their addresses.
VSETUP = ["movl r24 = 77771f", "movl r25 = 77772f"]
VTEARDOWN = ["br.sptk.few 77773f", "77771:", "br.ret.sptk.many b7",
             "77772:", "br.ret.sptk.many b7", "77773:"]


def vbody(dist, alt):
    out = [B("mib", None, "mov b6 = r24", "nop.b 0 ;;")]
    out += [NOP] * dist
    out.append(B("mib", None, None, "br.call.sptk.many b7 = b6 ;;"))
    if alt:
        out += [B("mii", "mov r26 = r24", "mov r24 = r25", "nop.i 0 ;;"),
                B("mii", None, "mov r25 = r26", "nop.i 0 ;;")]
    else:
        out += [NOP, NOP]
    return out


def sbody(second):
    out = []
    for r in ("r24", second):
        out += [B("mib", None, f"mov b6 = {r}", "nop.b 0 ;;"),
                B("mib", None, None, "br.call.sptk.many b7 = b6 ;;"),
                NOP, NOP]
    return out


def kernels():
    ks = []
    for k, order in ((1, "seq"), (2, "seq"), (2, "rnd"), (4, "seq"),
                     (4, "rnd"), (8, "rnd")):
        for dist in (0, 3, 8):
            for hint in ("sptk", "dptk"):
                b, base = body(k, dist, hint)
                ks += pair(f"ib_k{k}_{order}_d{dist}_{hint}", "branch", b,
                           setup(table(k, order, base)), bodies=4)
    # How far ahead of the branch b6 must be written; and the same with an
    # IP-relative branch to the first target, which costs no bubble.
    for k, order in ((1, "seq"), (4, "rnd")):
        for dist in (12, 16, 24, 32):
            b, base = body(k, dist, "sptk")
            ks += pair(f"ib_k{k}_{order}_d{dist}_sptk", "branch", b,
                       setup(table(k, order, base)), bodies=4)
    for dist in (9, 10, 11, 13, 14):
        b, base = body(4, dist, "sptk")
        ks += pair(f"ib_k4_rnd_d{dist}_sptk", "branch", b,
                   setup(table(4, "rnd", base)), bodies=4)
    # dist groups of two bundles: is the distance in groups or bundles?
    for dist in (4, 6, 8, 10, 12):
        b, base = body(4, dist, "sptk", wide=True)
        ks += pair(f"ib_k4_rnd_w{dist}_sptk", "branch", b,
                   setup(table(4, "rnd", base)), bodies=4)
    # glibc's fwrite calls _IO_file_xsputn through the vtable: mov b6 in
    # the branch's own group ("same"), a br.call returning through b7
    # ("call"), and the .many hint.
    for k, order in ((1, "seq"), (4, "rnd")):
        for name, kw in (("same", dict(same=True)),
                         ("many", dict(many=True)),
                         ("call", dict(call=True)),
                         ("callmany", dict(call=True, many=True)),
                         ("callsame", dict(call=True, many=True,
                                           same=True))):
            b, base = body(k, 0, "sptk", **kw)
            ks += pair(f"ib_{name}_k{k}_{order}_d0", "branch", b,
                       setup(table(k, order, base)), bodies=4)
        b, base = body(k, 12, "sptk", call=True, many=True)
        ks += pair(f"ib_callmany_k{k}_{order}_d12", "branch", b,
                   setup(table(k, order, base)), bodies=4)
    # A call through a b6 written just before it, to one routine outside
    # the bodies ("same"), or to one of two in turn ("alt"): is the
    # penalty for a late write, or for a target that changed?
    for alt in (False, True):
        for dist in (0, 3, 12):
            ks += pair(f"ibv_{'alt' if alt else 'same'}_d{dist}", "branch",
                       vbody(dist, alt), VSETUP, bodies=4,
                       teardown=VTEARDOWN)
    # Two call sites through b6, each with its own fixed target ("xy"),
    # or both to the same one ("xx"): is "same target" judged per
    # register or per branch?
    for name, second in (("xx", "r24"), ("xy", "r25")):
        ks += pair(f"ibs_{name}_d0", "branch", sbody(second), VSETUP,
                   bodies=4, teardown=VTEARDOWN)
    # The hinted move to b6, and brp, which name the branch: does either
    # tell the front end the new target in time?
    for name, kw in (("movsptk", dict(movhint="sptk")),
                     ("movimp", dict(movhint="sptk.imp")),
                     ("movdptk", dict(movhint="dptk")),
                     ("brp", dict(brp="sptk")),
                     ("brpimp", dict(brp="sptk.imp"))):
        for dist in (0, 3, 6):
            b, base = body(4, dist, "sptk", **kw)
            ks += pair(f"ib_{name}_k4_rnd_d{dist}", "branch", b,
                       setup(table(4, "rnd", base)), bodies=4)
    for dist in (0, 8):
        b, base = body(1, dist, "sptk", direct=True)
        ks += pair(f"ib_direct_d{dist}", "branch", b,
                   setup(table(1, "seq", base)), bodies=4)
    return ks
