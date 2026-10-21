// SPDX-License-Identifier: GPL-3.0-or-later
/* ia64probe: measurement primitives.
 *
 * There is no PMU to read on the machine this targets (perf reports every
 * hardware event as unsupported), so time is ar.itc, the interval time
 * counter, read in user space.  It runs at a fixed rate unrelated to the
 * core clock, which the ondemand governor moves between 0.8 and 1.6 GHz, so
 * every kernel is bracketed by a calibration kernel whose cycle count is
 * known and the ratio of the two converts ticks to core cycles.
 */
#ifndef IA64PROBE_PROBE_H
#define IA64PROBE_PROBE_H

#include <stdint.h>

struct kernel {
	const char *name;
	const char *group;
	unsigned bodies;	/* copies of the body inside the loop */
	unsigned iters;		/* loop trips per call */
	unsigned long (*fn)(unsigned long iters, void *scratch);
};

static inline uint64_t itc(void)
{
	uint64_t t;

	__asm__ __volatile__("mov %0 = ar.itc" : "=r"(t) :: "memory");
	return t;
}

#endif /* IA64PROBE_PROBE_H */
