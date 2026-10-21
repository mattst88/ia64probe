# SPDX-License-Identifier: GPL-3.0-or-later
"""Sanity kernels: the gate every other measurement depends on.

There is no PMU on the machine this was written for, so the cycle counts are
ar.itc ticks scaled by the core-to-ITC clock ratio, and that ratio is itself
measured: `cal_add` is a chain of dependent adds, two per bundle with a stop
after each, so it takes exactly one cycle per add if the documented 1-cycle
ALU-to-ALU latency holds.  The harness runs it before every kernel.  The
other sanity kernels then check that the ratio makes sense independently of
the add chain:

    sanity_fma_chain    dependent fma.d: 4 cycles each (Fmac -> Fmac).
    sanity_nop_pair     two bundles of nops ending in one stop: one cycle,
                        since two bundles issue per cycle.
    sanity_add6         six independent adds in an MII MII pair: one cycle,
                        since A-type instructions in the third and fourth I
                        slots go to free M ports (251110-003 section 3.3.2).
    sanity_add_chain    the calibration chain itself, as an ordinary kernel:
                        must come out at 1.0 per add, or the ratio drifted.
"""

from emit import Bundle as B, pair

ISETUP = [f"mov r{r} = {r}" for r in range(34, 50)]
FSETUP = ["mov r18 = 1", "setf.sig f32 = r18", "fcvt.xf f32 = f32",
          "fnorm.d f32 = f32", "mov f33 = f0", "mov f34 = f0"]

ADD_CHAIN = [B("mmi", "add r34 = r34, r35 ;;", "add r34 = r34, r35",
               "nop.i 0 ;;")]


def kernels():
    ks = []
    ks += pair("cal_add", "cal", ADD_CHAIN * 4, ISETUP,
               {"expect_cyc_per_body": 8.0}, bodies=32, iters=2000)
    ks += pair("sanity_add_chain", "sanity", ADD_CHAIN, ISETUP,
               {"expect_cyc_per_body": 2.0})
    ks += pair("sanity_fma_chain", "sanity",
               [B("mfi", None, "fma.d f33 = f33, f32, f34", "nop.i 0 ;;")],
               FSETUP, {"expect_cyc_per_body": 4.0})
    ks += pair("sanity_nop_pair", "sanity",
               [B("mmi"), B("mmi", None, None, "nop.i 0 ;;")],
               ISETUP, {"expect_cyc_per_body": 1.0})
    ks += pair("sanity_add6", "sanity",
               [B("mii", "add r34 = r40, r41", "add r35 = r40, r41",
                  "add r36 = r40, r41"),
                B("mii", "add r37 = r40, r41", "add r38 = r40, r41",
                  "add r39 = r40, r41 ;;")],
               ISETUP, {"expect_cyc_per_body": 1.0})
    return ks
