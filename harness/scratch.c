// SPDX-License-Identifier: GPL-3.0-or-later
/* ia64probe: the scratch area.
 *
 * 256 MB, aligned to 1 MB, zeroed, and laid out by the suites:
 *
 *   0 - 1 MB       words kernels read and write at small offsets from in1
 *   1 - 9 MB       store streams
 *   9 - 178 MB     pointer-chase rings (gen/tests_memory.py RINGS)
 *   192 - 256 MB   store streams too long for 1 - 9 MB (FAR)
 *
 * A ring is a cycle of pointers, one per `stride` bytes of its region, in
 * a random order (or in address order when sequential), so that a chain of
 * ld8 r = [r] visits every line of the region once per lap.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "scratch.h"

struct ring {
	unsigned long off, size, stride, seq;
};

#include "rings.inc"

#define SCRATCH_ALIGN (1UL << 20)

static unsigned long
xorshift(unsigned long *s)
{
	*s ^= *s << 13;
	*s ^= *s >> 7;
	*s ^= *s << 17;
	return *s;
}

static void
ring_build(unsigned char *p, const struct ring *r)
{
	unsigned long n = r->size / r->stride;
	unsigned long *order = malloc(n * sizeof(*order));
	unsigned long seed = 0x9e3779b97f4a7c15UL ^ r->size;

	if (!order) {
		perror("malloc");
		exit(1);
	}
	for (unsigned long i = 0; i < n; i++)
		order[i] = i;
	/* Sattolo's shuffle: one cycle through every element. */
	if (!r->seq) {
		for (unsigned long i = n - 1; i > 0; i--) {
			unsigned long j = xorshift(&seed) % i;
			unsigned long t = order[i];

			order[i] = order[j];
			order[j] = t;
		}
	}
	for (unsigned long i = 0; i < n; i++) {
		unsigned char *from = p + r->off + order[i] * r->stride;
		unsigned char *to = p + r->off + order[(i + 1) % n] * r->stride;

		*(unsigned char **)from = to;
	}
	free(order);
}

unsigned char *
scratch_init(void)
{
	unsigned char *p = mmap(NULL, SCRATCH_BYTES + SCRATCH_ALIGN,
				PROT_READ | PROT_WRITE,
				MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

	if (p == MAP_FAILED) {
		perror("mmap");
		exit(1);
	}
	p = (unsigned char *)(((unsigned long)p + SCRATCH_ALIGN - 1) &
			      ~(SCRATCH_ALIGN - 1));
	/* Touch every page so no kernel takes a first-touch fault. */
	for (unsigned long i = 0; i < SCRATCH_BYTES; i += 4096)
		p[i] = 0;
	for (const struct ring *r = rings; r->size; r++) {
		if (r->off + r->size > SCRATCH_BYTES) {
			fprintf(stderr, "ring at %#lx overflows scratch\n",
				r->off);
			exit(1);
		}
		ring_build(p, r);
	}
	return p;
}
