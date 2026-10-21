# SPDX-License-Identifier: GPL-3.0-or-later
"""Assembly emitter for ia64probe kernels.

Every kernel is one IA-64 function with the signature

    unsigned long kern(unsigned long iters, void *scratch);

returning the ar.itc ticks its loop took.  in0 (r32) is the iteration count
and in1 (r33) the scratch pointer.  The loop runs on ar.lc with br.cloop, so
it costs no general register and no predicate.

The central concern here is *placement*.  On Itanium the processor issues
what the code says: the template, the stops, and the slot order decide the
issue groups and the ports, and the hardware never reorders.  So bodies are
written as explicit bundles -- template, three slots, stops -- and assembled
with gas in explicit mode (-x), which also rejects a body whose group has a
dependency violation instead of quietly inserting a stop.  Setup and
teardown code is not measured and is assembled in .auto mode.

Registers the harness keeps for itself, and which a body must leave alone:

    r14  saved ar.itc at loop entry
    r15  saved pr
    r16  saved ar.pfs
    r17  saved ar.lc
    r32  in0, r33 in1 (scratch pointer; a body may read it)
    r1 (gp), r12 (sp), r13 (tp)

Everything else that is not callee-saved is free: r2-r3, r8-r11, r18-r31,
the stacked locals r34-r127, f6-f15 and f32-f127, all of p1-p63 (pr is
saved), and b6-b7.
"""

from __future__ import annotations

import json
import re

TEMPLATES = {"mii", "mmi", "mfi", "mmf", "mib", "mbb", "bbb", "mmb", "mfb",
             "mlx"}

# Where a template may stop inside the bundle: MI;I stops after slot 1 and
# M;MI after slot 0.  Every template may stop at its end.
MID_STOP = {"mii": 1, "mmi": 0}

NOP = {"m": "nop.m 0", "i": "nop.i 0", "f": "nop.f 0", "b": "nop.b 0"}


class Bundle:
    """One bundle: a template and three slot texts.

    A slot text ending in ';;' puts a stop after it.  Slots given as None
    are filled with the template's nop.  For mlx, slot 1 is the whole
    movl/brl and slot 2 must be None.
    """

    def __init__(self, tmpl: str, *slots) -> None:
        tmpl = tmpl.lower()
        if tmpl not in TEMPLATES:
            raise ValueError(f"unknown template {tmpl}")
        slots = list(slots) + [None] * (3 - len(slots))
        if len(slots) != 3:
            raise ValueError(f"{tmpl}: {len(slots)} slots")
        self.tmpl = tmpl
        self.slots = []
        for i, s in enumerate(slots):
            if tmpl == "mlx" and i == 2:
                if s is not None:
                    raise ValueError("mlx: slot 2 belongs to the movl")
                continue
            unit = "i" if (tmpl == "mlx" and i == 1) else tmpl[i]
            if s is None:
                s = NOP[unit]
            stop = s.rstrip().endswith(";;")
            last = 1 if tmpl == "mlx" else 2
            if stop and i != last and MID_STOP.get(tmpl) != i:
                raise ValueError(f"{tmpl}: no stop after slot {i}: {s}")
            self.slots.append(s)

    def stops(self) -> int:
        return sum(1 for s in self.slots if s.rstrip().endswith(";;"))

    def insns(self) -> int:
        """Slots that are not nops."""
        return sum(1 for s in self.slots if not re.match(r"^\s*nop", s))

    def text(self) -> list[str]:
        out = [f"\t{{ .{self.tmpl}"]
        out += [f"\t  {s}" for s in self.slots]
        out.append("\t}")
        return out


class Kernel:
    def __init__(self, name: str, group: str, body: list, bodies: int,
                 setup: list[str] | None = None, meta: dict | None = None,
                 iters: int = 1000, teardown: list[str] | None = None) -> None:
        self.name = name
        self.group = group
        self.body = body
        self.bodies = bodies
        self.setup = setup or []
        self.teardown = teardown or []
        self.meta = meta or {}
        self.iters = iters

    @property
    def symbol(self) -> str:
        return f"k_{self.name}_x{self.bodies}"

    def emit(self) -> list[str]:
        sym = self.symbol
        out = [
            "\t.text",
            "\t.align 32",
            f"\t.global {sym}",
            f"\t.proc {sym}",
            f"{sym}:",
            "\t.auto",
            "\talloc r16 = ar.pfs, 2, 94, 0, 0",
            "\tmov r17 = ar.lc",
            "\tmov r15 = pr",
            "\tadds r14 = -1, r32",
            "\tmov ar.lc = r14",
        ]
        out += ["\t" + s for s in self.setup]
        # Read the timer last, after every setup instruction has been
        # issued; the reads cancel in the differential.  In .auto mode gas
        # places the stops itself.
        out += [
            "\tmov r14 = ar.itc",
            "\t.explicit",
            "\t.align 32",
            f".L{sym}_loop:",
        ]
        for _ in range(self.bodies):
            for b in self.body:
                out += b.text()
        out += [
            "\t{ .mib",
            "\t  nop.m 0",
            "\t  nop.i 0",
            f"\t  br.cloop.sptk.few .L{sym}_loop ;;",
            "\t}",
            "\t.auto",
            "\tmov r8 = ar.itc",
            "\tsub r8 = r8, r14",
        ]
        out += ["\t" + s for s in self.teardown]
        out += [
            "\tmov ar.lc = r17",
            "\tmov pr = r15, -1",
            "\tmov ar.pfs = r16",
            "\tbr.ret.sptk.many b0",
            f"\t.endp {sym}",
            "",
        ]
        return out


def pair(name: str, group: str, body: list, setup=None, meta=None,
         bodies: int = 8, iters: int = 1000, teardown=None) -> list[Kernel]:
    """The kernel at `bodies` and at twice that many copies of `body`."""
    if 2 * bodies * len(body) > 1024:
        raise ValueError(f"{name}: {2 * bodies * len(body)} bundles "
                         "would not fit the 16 KB L1I")
    return [Kernel(name, group, body, bodies, setup, meta, iters, teardown),
            Kernel(name, group, body, 2 * bodies, setup, meta, iters,
                   teardown)]


def emit_file(kernels: list[Kernel]) -> str:
    out = ["// Generated by gen/gen.py; do not edit.", "\t.pred.safe_across_calls p1-p5,p16-p63"]
    for k in kernels:
        out += k.emit()
    return "\n".join(out) + "\n"


def emit_registry(kernels: list[Kernel]) -> str:
    out = ["/* Generated by gen/gen.py; do not edit. */"]
    for k in kernels:
        out.append(f"unsigned long {k.symbol}(unsigned long, void *);")
    out.append("static const struct kernel kernels[] = {")
    for k in kernels:
        out.append(f'\t{{ "{k.name}", "{k.group}", {k.bodies}, {k.iters}, '
                   f'{k.symbol} }},')
    out.append("};")
    return "\n".join(out) + "\n"


def emit_metadata(kernels: list[Kernel]) -> str:
    seen = {}
    for k in kernels:
        if k.name in seen:
            continue
        seen[k.name] = {
            "group": k.group,
            "bundles_per_body": len(k.body),
            "insns_per_body": sum(b.insns() for b in k.body),
            "stops_per_body": sum(b.stops() for b in k.body),
            "body": [b.text() for b in k.body],
            **k.meta,
        }
    return json.dumps(seen, indent=1) + "\n"


def emit_rings(rings) -> str:
    """The pointer-chase rings the harness lays out in the scratch area:
    (offset, size, stride, sequential) each."""
    out = ["/* Generated by gen/gen.py; do not edit. */",
           "static const struct ring rings[] = {"]
    for off, size, stride, seq in rings:
        out.append(f"\t{{ {off:#x}, {size:#x}, {stride}, {int(seq)} }},")
    out.append("\t{ 0, 0, 0, 0 },")
    out.append("};")
    return "\n".join(out) + "\n"
