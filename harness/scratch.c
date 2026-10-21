// SPDX-License-Identifier: GPL-3.0-or-later
/* ia64probe: the scratch area.
 *
 * 64 MB, aligned to 1 MB, zeroed.  Suites that need data laid out in it
 * (pointer-chase rings, branch patterns) describe their regions here so the
 * offsets live in one place.
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#include "scratch.h"

#define SCRATCH_BYTES (64UL << 20)
#define SCRATCH_ALIGN (1UL << 20)

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
	return p;
}
