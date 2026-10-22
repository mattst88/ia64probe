// SPDX-License-Identifier: GPL-3.0-or-later
/* ia64probe: the scratch area kernels read and write through in1. */
#ifndef IA64PROBE_SCRATCH_H
#define IA64PROBE_SCRATCH_H

#define SCRATCH_BYTES (256UL << 20)
/* Where the pointer-chase rings start; no kernel may write past here. */
#define SCRATCH_RINGS (9UL << 20)
/* And where the store streams after them start. */
#define SCRATCH_FAR (192UL << 20)

unsigned char *scratch_init(void);

#endif /* IA64PROBE_SCRATCH_H */
