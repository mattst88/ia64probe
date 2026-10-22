# SPDX-License-Identifier: GPL-3.0-or-later
"""Producer-to-consumer latency, after 308065-001 Table 2-8.

Each kernel is a dependence cycle: every instruction reads what the one
before it wrote, and the first reads what the last wrote, with a stop after
each so every instruction is its own issue group.  The cycles per body are
then the sum of the latencies around the cycle, one edge per instruction.
A cycle of one instruction measures X -> X; a cycle of two measures
X -> Y plus Y -> X, and pairing Y with an operation whose latencies are
known separates the two.

The values are chosen so the chains are stable: r40 is 0 and r41 is 0, so
adds, shifts and deposits keep r34 the same, and the load chains read a
self-pointer that st8 [r33] = r33 left in the scratch area.

Claims are 308065-001 Table 2-8 ("Execution with Bypass Latency Summary").
Where the table gives no number the claim is left out and the kernel is a
measurement only.
"""

from emit import Bundle as B, pair


def one(unit: str, text: str) -> B:
    """`text` alone in a bundle, in a slot of `unit`, then a stop."""
    if unit in "ma":
        return B("mmi", text, None, "nop.i 0 ;;")
    if unit == "i":
        return B("mii", None, text, "nop.i 0 ;;")
    if unit == "f":
        return B("mfi", None, text, "nop.i 0 ;;")
    raise ValueError(unit)


ISETUP = ["mov r34 = 0x1234", "mov r40 = 0", "mov r41 = 0",
          "cmp.eq p6, p7 = r0, r0"]
LSETUP = ISETUP + ["st8 [r33] = r33", "mov r34 = r33"]
# r34 points at a zero, so a load from it adds nothing to r34.
ZSETUP = ISETUP + ["adds r34 = 64, r33", "st8 [r34] = r0"]
FSETUP = ISETUP + ["mov r18 = 1", "setf.sig f32 = r18", "fcvt.xf f32 = f32",
                   "fnorm.d f32 = f32", "mov f33 = f32", "mov f34 = f0"]

ADD = ("a", "add r34 = r34, r40")
SHL = ("i", "shl r34 = r34, r41")

# name: (setup, [(unit, insn), ...], claimed cycles per body or None)
CHAINS = {
    # The adder class: Adder -> ALU is 1.
    "add": (ISETUP, [ADD], 1),
    "shladd": (ISETUP, [("a", "shladd r34 = r40, 2, r34")], 1),
    "addp4": (ISETUP, [("a", "addp4 r34 = r34, r40")], 1),
    "extr_u": (ISETUP, [("i", "extr.u r34 = r34, 0, 40")], 1),
    "extr": (ISETUP, [("i", "extr r34 = r34, 0, 40")], 1),
    "dep_z": (ISETUP, [("i", "dep.z r34 = r34, 0, 40")], 1),
    "dep": (ISETUP, [("i", "dep r34 = r40, r34, 50, 4")], 1),
    "shl_imm": (ISETUP, [("i", "shl r34 = r34, 0")], 1),
    "shrp": (ISETUP, [("i", "shrp r34 = r34, r34, 8")], 1),
    "zxt4": (ISETUP, [("i", "zxt4 r34 = r34")], 1),
    "sxt4": (ISETUP, [("i", "sxt4 r34 = r34")], 1),
    "czx1": (ISETUP, [("i", "czx1.l r34 = r34")], 1),
    # Multimedia: MM -> MM is 2, Adder -> MM 3, MM -> ALU 3.
    "shl_var": (ISETUP, [SHL], 2),
    "shr_var": (ISETUP, [("i", "shr.u r34 = r34, r41")], 2),
    "popcnt": (ISETUP, [("i", "popcnt r34 = r34")], 2),
    "mux1": (ISETUP, [("i", "mux1 r34 = r34, @rev")], 2),
    "mux2": (ISETUP, [("i", "mux2 r34 = r34, 0x1b")], 2),
    "pmpy2": (ISETUP, [("i", "pmpy2.r r34 = r34, r34")], 2),
    "pmpyshr2": (ISETUP, [("i", "pmpyshr2.u r34 = r34, r34, 0")], 2),
    "pshr2": (ISETUP, [("i", "pshr2.u r34 = r34, 1")], 2),
    "mix1": (ISETUP, [("i", "mix1.l r34 = r34, r34")], 2),
    "pmin1": (ISETUP, [("i", "pmin1.u r34 = r34, r34")], 2),
    "padd2": (ISETUP, [("a", "padd2 r34 = r34, r40")], 2),
    "pcmp1": (ISETUP, [("a", "pcmp1.eq r34 = r34, r40")], 2),
    "pshladd2": (ISETUP, [("a", "pshladd2 r34 = r34, 1, r40")], 2),
    "add_shl": (ISETUP, [ADD, SHL], 6),
    "popcnt_add": (ISETUP, [("i", "popcnt r34 = r34"), ADD], 6),
    "padd2_add": (ISETUP, [("a", "padd2 r34 = r34, r40"), ADD], 6),
    # Integer loads hitting L1D: N = 1, load address N + 1, MM N + 2.
    "ld8": (LSETUP, [("m", "ld8 r34 = [r34]")], 2),
    "ld8_add": (LSETUP, [("m", "ld8 r34 = [r34]"), ADD], 2),
    "ld4_add": (ZSETUP, [("m", "ld4 r35 = [r34]"),
                         ("a", "add r34 = r34, r35")], 2),
    "ld2_add": (ZSETUP, [("m", "ld2 r35 = [r34]"),
                         ("a", "add r34 = r34, r35")], 2),
    "ld1_add": (ZSETUP, [("m", "ld1 r35 = [r34]"),
                         ("a", "add r34 = r34, r35")], 2),
    # A write to the target of a load still in flight waits for it.
    "ld8_waw": (LSETUP, [("m", "ld8 r34 = [r34]"), ("m", "ld8 r34 = [r33]")],
                None),
    "ld8_shl": (LSETUP, [("m", "ld8 r34 = [r34]"), SHL], None),
    "ld8_setf": (LSETUP, [("m", "ld8 r34 = [r34]"),
                          ("m", "setf.sig f32 = r34"),
                          ("m", "getf.sig r34 = f32")], 13),
    # FP loads bypass L1D: M = 5 from L2D, plus 1 for the format.
    "ldf8_getf": (LSETUP, [("m", "ldf8 f32 = [r34]"),
                           ("m", "getf.sig r34 = f32")], 12),
    "setf_getf": (ISETUP, [("m", "setf.sig f32 = r34"),
                           ("m", "getf.sig r34 = f32")], 11),
    "setf_getf_d": (FSETUP, [("m", "getf.d r34 = f32"),
                             ("m", "setf.d f32 = r34")], 11),
    "getf_add_setf": (ISETUP, [("m", "setf.sig f32 = r34"),
                               ("m", "getf.sig r34 = f32"), ADD], 12),
    # FP: Fmac and Fmisc are 4 to either.
    "fma": (FSETUP, [("f", "fma.d f33 = f33, f32, f34")], 4),
    "fmerge": (FSETUP, [("f", "fmerge.s f33 = f33, f33")], 4),
    "fma_fmerge": (FSETUP, [("f", "fma.d f33 = f33, f32, f34"),
                            ("f", "fmerge.s f33 = f33, f33")], 8),
    "xma": (FSETUP, [("f", "xma.l f33 = f33, f32, f34")], 4),
    "fnorm": (FSETUP, [("f", "fnorm.d f33 = f33")], 4),
    "fpma": (FSETUP, [("f", "fpma f33 = f33, f32, f34")], 4),
    "fselect": (FSETUP, [("f", "fselect f33 = f33, f32, f33")], 4),
    # Predicates: integer compare -> qualifying predicate 1, fcmp 2.
    "cmp_qp": (ISETUP, [("a", "(p6) cmp.eq p6, p7 = r0, r0")], 1),
    "tbit_qp": (ISETUP, [("i", "(p6) tbit.z p6, p7 = r0, 0")], 1),
    "fcmp_qp": (FSETUP, [("f", "(p6) fcmp.eq p6, p7 = f32, f32")], 2),
    "cmp_add_qp": (ISETUP, [("a", "(p6) cmp.eq p6, p7 = r34, r34"),
                            ("a", "(p6) add r34 = r34, r40")], 2),
    # An instruction whose qualifying predicate is false (p7): does it wait
    # for its sources, and does a later reader wait for its write?
    "fma_off": (FSETUP, [("f", "(p7) fma.d f33 = f33, f32, f34")], None),
    "fma_off_src": (FSETUP, [("f", "fma.d f33 = f33, f32, f34"),
                             ("f", "(p7) fma.d f35 = f33, f32, f34")], None),
    "fma_off_dst": (FSETUP, [("f", "fma.d f33 = f33, f32, f34"),
                             ("f", "(p7) fma.d f33 = f35, f32, f34")], None),
    "add_off_ld": (LSETUP, [("m", "ld8 r34 = [r34]"),
                            ("a", "(p7) add r35 = r34, r40")], None),
    "ld_off_add": (LSETUP, [("m", "ld8 r34 = [r34]"),
                            ("a", "(p7) add r34 = r35, r40")], None),
    # Moves to and from the special registers: measured only.
    "mov_ec": (ISETUP, [("i", "mov.i ar.ec = r34"),
                        ("i", "mov.i r34 = ar.ec")], None),
    "mov_ccv": (ISETUP, [("m", "mov.m ar.ccv = r34"),
                         ("m", "mov.m r34 = ar.ccv")], None),
    "mov_br": (ISETUP, [("i", "mov b6 = r34"), ("i", "mov r34 = b6")], None),
    "mov_br_add": (ISETUP, [("i", "mov r34 = b6"), ADD,
                            ("i", "mov b6 = r34")], None),
    "mov_pr": (ISETUP, [("i", "mov pr = r34, 0x40"), ("i", "mov r34 = pr")],
               None),
    "mov_fpsr": (ISETUP, [("m", "mov.m r34 = ar.fpsr"),
                          ("m", "mov.m ar.fpsr = r34")], None),
}

# Halves of a move through a special register.  A cycle through the register
# cannot separate the write from the read, so these break it: in `from` the
# read does not depend on anything in the loop and only the add waits for
# it, and in `to` the write reads r34, which nothing writes.  Each body is
# then the one latency plus the cycle after the second instruction's group.
SPECIAL = {
    "ec": ("i", "ar.ec"),
    "ccv": ("m", "ar.ccv"),
    "unat": ("m", "ar.unat"),
    "pfs": ("i", "ar.pfs"),
    "br": ("i", "b6"),
}


def special(reg: str, unit: str, name: str):
    mov = {"i": "mov.i", "m": "mov.m"}[unit] if reg.startswith("ar.") \
        else "mov"
    frm = [(unit, f"{mov} r35 = {reg}"), ("a", "add r34 = r34, r35")]
    to = [(unit, f"{mov} {reg} = r34"), (unit, f"{mov} r35 = {reg}")]
    return {f"from_{name}": (ISETUP, frm, None),
            f"to_{name}": (ISETUP, to, None)}


for _n, (_u, _r) in SPECIAL.items():
    CHAINS.update(special(_r, _u, _n))
CHAINS["from_pr"] = (ISETUP, [("i", "mov r35 = pr"),
                              ("a", "add r34 = r34, r35")], None)
CHAINS["to_pr"] = (ISETUP, [("i", "mov pr = r34, 0x40"),
                            ("i", "mov r35 = pr")], None)
CHAINS["from_itc"] = (ISETUP, [("m", "mov.m r35 = ar.itc"),
                               ("a", "add r34 = r34, r35")], None)


def kernels():
    ks = []
    for name, (setup, chain, claim) in CHAINS.items():
        meta = {"chain": [t for _, t in chain]}
        itc = sum("ar.itc" in t for _, t in chain)
        if itc:
            meta["itc_reads_per_body"] = itc
        if claim is not None:
            meta["expect_cyc_per_body"] = float(claim)
        ks += pair(f"lat_{name}", "latency", [one(u, t) for u, t in chain],
                   setup, meta)
    return ks
