# SPDX-License-Identifier: GPL-3.0-or-later
"""Dispersal: which bundles issue together, after 251110-003 section 3.3
and 308065-001 section 2.3.1.

Every kernel is a run of independent instructions ending in a stop, so the
cycles per body are the issue groups it splits into: 1 when the hardware
issues it all at once, more where it runs out of a port.  Nothing in a
body reads a register the body writes.

The claims are the number of cycles the documented port counts allow:
two bundles a cycle; four M ports, of which loads use two and stores two
(M0/M1 and M2/M3); two I ports, with A-type instructions also able to use
the M ports; two F ports; three B ports.  Where the documents disagree
with each other or are silent, the kernel has no claim.
"""

from emit import Bundle as B, pair

SETUP = ["mov r40 = 0", "mov r41 = 0", "st8 [r33] = r0",
         "adds r42 = 64, r33", "adds r43 = 128, r33", "adds r44 = 192, r33",
         "adds r45 = 256, r33", "mov r18 = 1", "setf.sig f32 = r18",
         "fcvt.xf f32 = f32", "fnorm.d f32 = f32", "mov f33 = f32",
         "mov b6 = r40"]


def add(d):
    return f"add r{d} = r40, r41"


def extr(d):
    return f"extr.u r{d} = r40, 3, 20"


def ld(d, a):
    return f"ld8 r{d} = [r{a}]"


def st(a):
    return f"st8 [r{a}] = r40"


def fma(d):
    return f"fma.d f{d} = f32, f32, f33"


# name: (bundles, claimed cycles or None)
GROUPS = {
    # Two bundles a cycle.
    "nop3": ([B("mii"), B("mii"), B("mii", None, None, "nop.i 0 ;;")], 2),
    "nop4": ([B("mii"), B("mii"), B("mii"),
              B("mii", None, None, "nop.i 0 ;;")], 2),
    # A-type in I slots beyond two I ports move to M ports.
    "mii_add6": ([B("mii", add(50), add(51), add(52)),
                  B("mii", add(53), add(54), add(55) + " ;;")], 1),
    "mmi_add6": ([B("mmi", add(50), add(51), add(52)),
                  B("mmi", add(53), add(54), add(55) + " ;;")], 1),
    "mii_add_extr": ([B("mii", add(50), add(51), add(52)),
                      B("mii", add(53), extr(54), add(55) + " ;;")], 1),
    # I-type only on I ports: four need two cycles.
    "mii_extr4": ([B("mii", None, extr(50), extr(51)),
                   B("mii", None, extr(52), extr(53) + " ;;")], 2),
    "mii_extr2": ([B("mii", None, extr(50), extr(51) + " ;;")], 1),
    "mii_extr_nop": ([B("mii", None, "nop.i 0", extr(51) + " ;;")], None),
    "mii_popcnt2": ([B("mii", None, "popcnt r50 = r40",
                       "popcnt r51 = r40 ;;")], None),
    "mii_shrp2": ([B("mii", None, "shrp r50 = r40, r41, 3",
                     "shrp r51 = r40, r41, 3 ;;")], None),
    "mii_pmpy2": ([B("mii", None, "pmpy2.r r50 = r40, r41",
                     "pmpy2.r r51 = r40, r41 ;;")], None),
    "mii_shlvar2": ([B("mii", None, "shl r50 = r40, r41",
                       "shl r51 = r40, r41 ;;")], None),
    "mii_mux2": ([B("mii", None, "mux1 r50 = r40, @rev",
                    "mux1 r51 = r40, @rev ;;")], None),
    "mii_tbit2": ([B("mii", None, "tbit.z p6, p7 = r40, 3",
                     "tbit.z p8, p9 = r40, 3 ;;")], None),
    "mii_dep2": ([B("mii", None, "dep r50 = r40, r41, 3, 7",
                    "dep r51 = r40, r41, 3, 7 ;;")], None),
    "mii_tnat2": ([B("mii", None, "tnat.z p6, p7 = r40",
                     "tnat.z p8, p9 = r40 ;;")], None),
    "mii_pmpyshr2": ([B("mii", None, "pmpyshr2 r50 = r40, r41, 0",
                        "pmpyshr2 r51 = r40, r41, 0 ;;")], None),
    "mii_psad2": ([B("mii", None, "psad1 r50 = r40, r41",
                     "psad1 r51 = r40, r41 ;;")], None),
    "mii_pshl2": ([B("mii", None, "pshl2 r50 = r40, 3",
                     "pshl2 r51 = r40, 3 ;;")], None),
    "mii_mix2": ([B("mii", None, "mix1.l r50 = r40, r41",
                    "mix1.l r51 = r40, r41 ;;")], None),
    "mii_pmpy_extr": ([B("mii", None, "pmpy2.r r50 = r40, r41",
                         extr(51) + " ;;")], None),
    "mii_extr_pmpy": ([B("mii", None, extr(51),
                         "pmpy2.r r50 = r40, r41 ;;")], None),
    "mii_nop_movbr": ([B("mii", None, "nop.i 0", "mov r50 = b6 ;;")], None),
    "mii_add_movbr": ([B("mii", None, add(51), "mov r50 = b6 ;;")], None),
    "mii_movtobr2": ([B("mii", None, "mov b6 = r40", "mov b7 = r40 ;;")],
                     None),
    # nop.i takes an I port as any I-type instruction does.
    "mii_aan_x2": ([B("mii", add(50), add(51), "nop.i 0"),
                    B("mii", add(52), add(53), "nop.i 0 ;;")], None),
    "mii_ann_x2": ([B("mii", add(50), "nop.i 0", "nop.i 0"),
                    B("mii", add(52), "nop.i 0", "nop.i 0 ;;")], None),
    "mii_naa_x2": ([B("mii", add(50), "nop.i 0", add(51)),
                    B("mii", add(52), "nop.i 0", add(53) + " ;;")], None),
    "mii_nnx_mmi": ([B("mii", add(50), "nop.i 0", "nop.i 0"),
                     B("mmi", add(52), add(53), extr(54) + " ;;")], None),
    "mfi_nop3": ([B("mfi"), B("mfi"), B("mfi", None, None, "nop.i 0 ;;")],
                 None),
    "mmf_nop3": ([B("mmf"), B("mmf"), B("mmf", None, None, "nop.f 0 ;;")],
                 None),
    "mmb_nop3": ([B("mmb"), B("mmb"), B("mmb", None, None, "nop.b 0 ;;")],
                 None),
    "mii_zxt2": ([B("mii", None, "zxt4 r50 = r40", "zxt4 r51 = r40 ;;")], 1),
    "mii_movfrombr2": ([B("mii", None, "mov r50 = b6", "mov r51 = b6 ;;")],
                       None),
    "mii_movfrompr2": ([B("mii", None, "mov r50 = pr", "mov r51 = pr ;;")],
                       None),
    "mii_extr_add_x2": ([B("mii", None, extr(50), add(51)),
                         B("mii", None, extr(52), add(53) + " ;;")], None),
    # Loads on two ports, stores on two.
    "ld2": ([B("mmi", ld(50, 33), ld(51, 42), "nop.i 0 ;;")], 1),
    "ld4": ([B("mmi", ld(50, 33), ld(51, 42)),
             B("mmi", ld(52, 43), ld(53, 44), "nop.i 0 ;;")], 2),
    "st2": ([B("mmi", st(33), st(42), "nop.i 0 ;;")], 1),
    "st4": ([B("mmi", st(33), st(42)),
             B("mmi", st(43), st(44), "nop.i 0 ;;")], 2),
    "ld2st2": ([B("mmi", ld(50, 33), ld(51, 42)),
                B("mmi", st(43), st(44), "nop.i 0 ;;")], 1),
    "ld2_add4": ([B("mmi", ld(50, 33), ld(51, 42), add(52)),
                  B("mmi", add(53), add(54), add(55) + " ;;")], 1),
    "ld2st2_add2": ([B("mmi", ld(50, 33), ld(51, 42), add(52)),
                     B("mmi", st(43), st(44), add(55) + " ;;")], 1),
    "getf2": ([B("mmi", "getf.sig r50 = f32", "getf.sig r51 = f32",
                 "nop.i 0 ;;")], None),
    "setf2": ([B("mmi", "setf.sig f50 = r40", "setf.sig f51 = r40",
                 "nop.i 0 ;;")], None),
    "ldf2": ([B("mmi", "ldf8 f50 = [r33]", "ldf8 f51 = [r42]",
                "nop.i 0 ;;")], 1),
    "ldfp2": ([B("mmi", "ldfp8 f50, f51 = [r33]", "ldfp8 f52, f53 = [r43]",
                 "nop.i 0 ;;")], 1),
    "ld_getf": ([B("mmi", ld(50, 33), "getf.sig r51 = f32",
                   "nop.i 0 ;;")], None),
    "st_getf": ([B("mmi", st(33), "getf.sig r51 = f32", "nop.i 0 ;;")], None),
    "st2_getf": ([B("mmi", st(33), st(42)),
                  B("mmi", "getf.sig r51 = f32", None, "nop.i 0 ;;")], None),
    "lfetch2": ([B("mmi", "lfetch [r33]", "lfetch [r42]", "nop.i 0 ;;")],
                None),
    "lfetch_ld2": ([B("mmi", ld(50, 33), ld(51, 42)),
                    B("mmi", "lfetch [r43]", None, "nop.i 0 ;;")], None),
    # F ports.
    "fma2": ([B("mfi", None, fma(50)), B("mfi", None, fma(51), "nop.i 0 ;;")],
             1),
    "mmf_fma2": ([B("mmf", add(50), add(51), fma(50)),
                  B("mmf", add(52), add(53), fma(51) + " ;;")], 1),
    "fma4": ([B("mfi", None, fma(50)), B("mfi", None, fma(51)),
              B("mfi", None, fma(52)), B("mfi", None, fma(53), "nop.i 0 ;;")],
             2),
    "mlx2": ([B("mlx", None, "movl r50 = 0x123456789"),
              B("mlx", None, "movl r51 = 0x123456789 ;;")], 1),
    "mlx_mii": ([B("mlx", add(50), "movl r51 = 0x123456789"),
                 B("mii", add(52), add(53), add(54) + " ;;")], None),
    # Split issue at a B slot that is not a nop or brp.
    "mib_nopb2": ([B("mib"), B("mib", None, None, "nop.b 0 ;;")], 1),
    "mfb_nopb2": ([B("mfb"), B("mfb", None, None, "nop.b 0 ;;")], 1),
    "mmb_nopb2": ([B("mmb"), B("mmb", None, None, "nop.b 0 ;;")], 1),
    "mbb_nop2": ([B("mbb"), B("mbb", None, None, "nop.b 0 ;;")], None),
    "bbb_nop2": ([B("bbb"), B("bbb", None, None, "nop.b 0 ;;")], None),
    "mib_mib_add": ([B("mib", add(50), add(51)),
                     B("mib", add(52), add(53), "nop.b 0 ;;")], 1),
    # A group that starts after a stop in the middle of a bundle, with
    # nothing of its own there: gemm's kernel, whose second group is the
    # nop after the stop and two whole bundles. The rest of the split
    # bundle takes a place in the second group's cycle, so it takes three
    # cycles; with the stop at the end of the bundle it takes two.
    "split_nop": ([B("mmf", add(50), add(51), fma(50)),
                   B("mii", add(52), "nop.i 0 ;;", None),
                   B("mmi", add(53), add(54), None),
                   B("mfi", add(55), fma(51), "nop.i 0 ;;")], 3),
    "split_mnop": ([B("mmf", add(50), add(51), fma(50)),
                    B("mmi", add(52) + " ;;", None, None),
                    B("mmi", add(53), add(54), None),
                    B("mfi", add(55), fma(51), "nop.i 0 ;;")], 3),
    "split_end": ([B("mmf", add(50), add(51), fma(50)),
                   B("mii", add(52), None, "nop.i 0 ;;"),
                   B("mmi", add(53), add(54), None),
                   B("mfi", add(55), fma(51), "nop.i 0 ;;")], 2),
    "split_add": ([B("mmf", add(50), add(51), fma(50)),
                   B("mii", add(52), "nop.i 0 ;;", add(56)),
                   B("mmi", add(53), add(54), None),
                   B("mfi", add(55), fma(51), "nop.i 0 ;;")], 3),
}


def kernels():
    ks = []
    for name, (body, claim) in GROUPS.items():
        meta = {}
        if claim is not None:
            meta["expect_cyc_per_body"] = float(claim)
        ks += pair(f"disp_{name}", "dispersal", body, SETUP, meta)
    return ks
