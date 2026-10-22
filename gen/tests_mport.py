# SPDX-License-Identifier: GPL-3.0-or-later
"""setf and getf on the M ports.

251110-003 section 3.3 puts getf in the store subtype (M2/M3) and also
says it issues only on M2; the issue logic does not reorder instructions
of one subtype, so a store before a getf in a group takes M2 and splits
the group.  It does not say where setf goes.  tests_dispersal's getf2 and
setf2 write the same registers every body, so they time the write-after-
write wait, not the ports.

mp_{name}: the group {name} twelve times, each copy writing its own
registers, so no copy waits on another: the cycles per body over 12 are
the cycles a group takes to issue.
"""

from emit import Bundle as B, pair

SETUP = ["mov r40 = 1", "adds r41 = 64, r33", "adds r42 = 128, r33",
         "setf.sig f32 = r40"]
COPIES = 12


def G(c):
    """The slot texts of copy c."""
    r = 50 + 6 * c
    f = 50 + 6 * c
    return {
        "s": lambda k: f"setf.sig f{f + k} = r40",
        "g": lambda k: f"getf.sig r{r + k} = f32",
        "a": lambda k: f"add r{r + 2 + k} = r40, r40",
        "l": lambda k: f"ld8 r{r + 4 + k} = [r41]",
        "lf": lambda k: f"ldfd f{f + 4 + k} = [r41]",
        "st": lambda k: "st8 [r42] = r40",
    }


# Each group: a list of bundles, each (template, [slot kinds]).  A kind is
# a key of G with an index for its destination, or None for a nop.
GROUPS = {
    "s": [("mmi", [("s", 0), None])],
    "g": [("mmi", [("g", 0), None])],
    "ss": [("mmi", [("s", 0), ("s", 1)])],
    "gg": [("mmi", [("g", 0), ("g", 1)])],
    "sg": [("mmi", [("s", 0), ("g", 0)])],
    "gs": [("mmi", [("g", 0), ("s", 0)])],
    "sss": [("mmi", [("s", 0), ("s", 1)]), ("mmi", [("s", 2), None])],
    "ssg": [("mmi", [("s", 0), ("s", 1)]), ("mmi", [("g", 0), None])],
    "gss": [("mmi", [("g", 0), ("s", 0)]), ("mmi", [("s", 1), None])],
    "as": [("mmi", [("a", 0), ("s", 0)])],
    "ag": [("mmi", [("a", 0), ("g", 0)])],
    "aas": [("mmi", [("a", 0), ("a", 1)]), ("mmi", [("s", 0), None])],
    "aag": [("mmi", [("a", 0), ("a", 1)]), ("mmi", [("g", 0), None])],
    "aasg": [("mmi", [("a", 0), ("a", 1)]), ("mmi", [("s", 0), ("g", 0)])],
    "aags": [("mmi", [("a", 0), ("a", 1)]), ("mmi", [("g", 0), ("s", 0)])],
    "sts": [("mmi", [("st", 0), ("s", 0)])],
    "sst": [("mmi", [("s", 0), ("st", 0)])],
    "stg": [("mmi", [("st", 0), ("g", 0)])],
    "gst": [("mmi", [("g", 0), ("st", 0)])],
    "ls": [("mmi", [("l", 0), ("s", 0)])],
    "lfs": [("mmi", [("lf", 0), ("s", 1)])],
    "slf": [("mmi", [("s", 1), ("lf", 0)])],
    "lfsg": [("mmi", [("lf", 0), ("s", 1)]), ("mmi", [("g", 0), None])],
    "sglf": [("mmi", [("s", 1), ("g", 0)]), ("mmi", [("lf", 0), None])],
}


def body(spec):
    out = []
    for c in range(COPIES):
        g = G(c)
        for i, (tmpl, kinds) in enumerate(spec):
            slots = [None if k is None else g[k[0]](k[1]) for k in kinds]
            slots += [None] * (2 - len(slots))
            last = "nop.i 0 ;;" if i == len(spec) - 1 else "nop.i 0"
            out.append(B(tmpl, slots[0], slots[1], last))
    return out


def kernels():
    ks = []
    for name, spec in GROUPS.items():
        ks += pair(f"mp_{name}", "mport", body(spec), SETUP, {"copies": COPIES},
                   bodies=2, iters=500)
    return ks
