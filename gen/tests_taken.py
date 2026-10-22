# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct taken branches: how fast the front end follows them.

Each body starts with a bundle whose branch is taken over `s` bundles that
never run, to n - 1 bundles of nops that fall through to the next body's
branch.  So a body runs n bundles and one taken branch, and its target is
s + 1 bundles after the branch.  Without stops in the nop bundles the back
end could take two bundles a cycle; with them (_st) one.  _c makes the
branch conditional on a true predicate.
"""

from __future__ import annotations

from emit import Bundle as B, pair

SETUP = ["cmp.eq p6, p7 = r0, r0"]


def body(n, s, stops, cond):
    if n < 2:
        raise ValueError("the target needs a bundle of its own")
    br = "(p6) br.cond.dptk.few 1f ;;" if cond else "br.sptk.few 1f ;;"
    out = [B("mib", None, None, br)]
    out += [B("mii", None, None, "nop.i 0 ;;")] * s
    for i in range(n - 1):
        last = "nop.i 0 ;;" if stops else "nop.i 0"
        out.append(B("mii", None, None, last, label="1" if i == 0 else None))
    return out


def kernels():
    ks = []
    for n in (2, 3, 4, 5, 6, 8):
        for s in (0, 1, 2, 3):
            for stops in (False, True):
                name = f"tk_n{n}_s{s}" + ("_st" if stops else "")
                ks += pair(name, "taken", body(n, s, stops, False),
                           setup=SETUP, meta={"n": n, "s": s,
                                              "stops": stops})
    for n in (2, 3, 4):
        for s in (0, 1):
            name = f"tk_c_n{n}_s{s}_st"
            ks += pair(name, "taken", body(n, s, True, True), setup=SETUP,
                       meta={"n": n, "s": s, "stops": True, "cond": True})
    # Back-to-back: a taken branch whose target bundle's branch is taken
    # too, k of them in a row, then one nop bundle.  _cmp computes each
    # branch's predicate in its own bundle.
    for k in (2, 3):
        for s in (0, 1, 2):
            for cmp in (False, True):
                out = []
                for j in range(k):
                    lab = None if j == 0 else str(j)
                    if cmp:
                        out.append(B("mib", "cmp.eq p8, p9 = r0, r0", None,
                                     f"(p8) br.cond.dptk.few {j + 1}f ;;",
                                     label=lab))
                    else:
                        out.append(B("mib", None, None,
                                     f"br.sptk.few {j + 1}f ;;", label=lab))
                    out += [B("mii", None, None, "nop.i 0 ;;")] * s
                out.append(B("mii", None, None, "nop.i 0 ;;", label=str(k)))
                name = f"tk_b{k}_s{s}" + ("_cmp" if cmp else "")
                ks += pair(name, "taken", out, setup=SETUP,
                           meta={"k": k, "s": s, "cmp": cmp})
    # qs's partition loop: a group of an MIB whose branch is not taken and
    # an MBB whose first branch is, their predicates computed by cmps in
    # the same group (_c) or preset; at the target, a second taken branch
    # (_t) or a fall-through.  lead nop bundles go before the pair and the
    # body is padded to an even length, so lead moves the pair's alignment.
    for lead in (1, 2):
        for a in (False, True):
            for c in (False, True):
                for t in (False, True):
                    out = [B("mii", None, None, "nop.i 0 ;;")] * lead
                    if a:
                        out.append(B("mib",
                                     "cmp.ne p6, p7 = r0, r0" if c else None,
                                     None, "(p6) br.cond.dptk.few 2f"))
                    out.append(B("mbb",
                                 "cmp.eq p8, p9 = r0, r0" if c else None,
                                 "(p8) br.cond.dptk.few 1f",
                                 "br.sptk.few 2f ;;"))
                    out.append(B("mii", None, None, "nop.i 0 ;;", label="2"))
                    if t:
                        out.append(B("mib", None, None, "br.sptk.few 3f ;;",
                                     label="1"))
                        out.append(B("mii", None, None, "nop.i 0 ;;"))
                        out.append(B("mii", None, None, "nop.i 0 ;;",
                                     label="3"))
                    else:
                        out.append(B("mii", None, None, "nop.i 0 ;;",
                                     label="1"))
                    if len(out) % 2:
                        out.append(B("mii", None, None, "nop.i 0 ;;"))
                    name = (f"tk_q_l{lead}" + ("_a" if a else "") +
                            ("_c" if c else "") + ("_t" if t else ""))
                    ks += pair(name, "taken", out,
                               setup=SETUP + ["cmp.ne p6, p7 = r0, r0",
                                              "cmp.eq p8, p9 = r0, r0"],
                               meta={"lead": lead, "a": a, "c": c, "t": t})
    # Which bundle pairs with branches issue in one cycle: two bundles in a
    # group whose branches are never taken, then a bundle of their target.
    # 2 cycles a body if the pair issues together, 3 if it splits.
    nt = "(p6) br.cond.dptk.few 1f"
    mii = B("mii")
    pairs = {
        "mii_bbb": [mii, B("bbb", nt, nt, nt + " ;;")],
        "mii_mbb": [mii, B("mbb", None, nt, nt + " ;;")],
        "mii_mib": [mii, B("mib", None, None, nt + " ;;")],
        "mii_mmb": [mii, B("mmb", None, None, nt + " ;;")],
        "mmi_bbb": [B("mmi"), B("bbb", nt, nt, nt + " ;;")],
        "mfi_bbb": [B("mfi"), B("bbb", nt, nt, nt + " ;;")],
        "mii_bbb1": [mii, B("bbb", nt, None, "nop.b 0 ;;")],
        "mii_bbbn": [mii, B("bbb", None, None, "nop.b 0 ;;")],
        "mibn_mii": [B("mib"), B("mii", None, None, "nop.i 0 ;;")],
        "mibn_bbb": [B("mib"), B("bbb", nt, nt, nt + " ;;")],
        "mib_mii": [B("mib", None, None, nt), B("mii", None, None,
                                                 "nop.i 0 ;;")],
        "mmb_mii": [B("mmb", None, None, nt), B("mii", None, None,
                                                 "nop.i 0 ;;")],
        "mfb_mii": [B("mfb", None, None, nt), B("mii", None, None,
                                                 "nop.i 0 ;;")],
        "mbb_mii": [B("mbb", None, nt, nt), B("mii", None, None,
                                               "nop.i 0 ;;")],
        "mbbn_mii": [B("mbb"), B("mii", None, None, "nop.i 0 ;;")],
        "mmbn_mii": [B("mmb"), B("mii", None, None, "nop.i 0 ;;")],
        "mfbn_mii": [B("mfb"), B("mii", None, None, "nop.i 0 ;;")],
        "mii_mii": [mii, B("mii", None, None, "nop.i 0 ;;")],
    }
    for name, two in pairs.items():
        out = two + [B("mii", None, None, "nop.i 0 ;;", label="1"),
                     B("mii", None, None, "nop.i 0 ;;")]
        ks += pair("tk_d_" + name, "taken", out,
                   setup=SETUP + ["cmp.ne p6, p7 = r0, r0"])
    return ks
