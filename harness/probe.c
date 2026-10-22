// SPDX-License-Identifier: GPL-3.0-or-later
/* ia64probe: run generated kernels and report ar.itc ticks as JSON lines.
 *
 * One kernel is one loop with B copies of a body, and the generator emits
 * every measurement twice, at B and 2B copies; analysis subtracts the two so
 * that the loop branch, the timer reads and pipeline fill all cancel.
 *
 * On hardware, each repetition runs the calibration pair (cal_add at B and
 * 2B), then the kernel pair, and the minimum over repetitions is kept for
 * each of the four: an interrupt can only add ticks.  The calibration pair
 * is a chain of dependent adds of known length, so its differential gives
 * core cycles per ITC tick at the clock the kernel ran at.  The median is
 * reported too, so a minimum that does not reproduce is visible, and so is
 * each repetition's 2B - B: a kernel whose line state is bimodal can take
 * the two minimums from different modes.
 *
 * With --sim the program runs each kernel --reps times and prints which, in
 * order; the ia64perf plugin's itc= log has the modeled cycle at each ar.itc
 * read, two per call, and run/sim.py pairs them up.
 *
 * With -v it also checks, after each pair, that the pair left the rings
 * as they were, and names any that did not: a kernel that walks off the
 * end of the store streams breaks every later kernel chasing a ring.
 */
#define _GNU_SOURCE
#include <getopt.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "probe.h"
#include "kernels.inc"
#include "scratch.h"

#define NKERNELS (sizeof kernels / sizeof kernels[0])

static int
cmp_u64(const void *a, const void *b)
{
	uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;

	return x < y ? -1 : x > y;
}

static uint64_t
median(uint64_t *v, unsigned n)
{
	qsort(v, n, sizeof *v, cmp_u64);
	return v[n / 2];
}

/* Report, and restore so as to name only the first, each ring word that
 * kernel K changed. */
static void
check(const struct kernel *k, unsigned char *scratch, uint64_t *ringw,
      unsigned npages, unsigned page)
{
	if (!ringw)
		return;
	for (unsigned p = 0; p < npages; p++) {
		uint64_t *w = (uint64_t *)(scratch + SCRATCH_RINGS + p * page);

		if (*w == ringw[p])
			continue;
		fprintf(stderr, "clobber %s %u at %#lx\n", k->name, k->bodies,
			SCRATCH_RINGS + (unsigned long)p * page);
		*w = ringw[p];
	}
}

static const struct kernel *
find(const char *name, unsigned bodies)
{
	for (unsigned i = 0; i < NKERNELS; i++)
		if (!strcmp(kernels[i].name, name) &&
		    kernels[i].bodies == bodies)
			return &kernels[i];
	return NULL;
}

static volatile unsigned long warm_sink;

/* Keep the core busy long enough for ondemand to raise the clock. */
static void
warm(double seconds)
{
	struct timespec a, b;


	clock_gettime(CLOCK_MONOTONIC, &a);
	do {
		for (int i = 0; i < 100000; i++)
			warm_sink += i;
		clock_gettime(CLOCK_MONOTONIC, &b);
	} while ((b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9 <
		 seconds);
}

static int
selected(const char *name, char **pfx, int npfx)
{
	if (npfx == 0)
		return 1;
	for (int i = 0; i < npfx; i++)
		if (!strncmp(name, pfx[i], strlen(pfx[i])))
			return 1;
	return 0;
}

int
main(int argc, char **argv)
{
	unsigned reps = 31;
	int sim = 0, verify = 0, cpu = -1, c;
	double scale = 1.0;

	while ((c = getopt(argc, argv, "r:svc:i:")) != -1) {
		switch (c) {
		case 'r':
			reps = strtoul(optarg, NULL, 0);
			break;
		case 's':
			sim = 1;
			break;
		case 'v':
			verify = 1;
			break;
		case 'c':
			cpu = atoi(optarg);
			break;
		case 'i':
			scale = strtod(optarg, NULL);
			break;
		default:
			fprintf(stderr, "usage: %s [-s] [-v] [-r reps] [-c cpu] "
				"[-i iter-scale] [prefix...]\n", argv[0]);
			return 2;
		}
	}
	if (cpu >= 0) {
		cpu_set_t set;

		CPU_ZERO(&set);
		CPU_SET(cpu, &set);
		if (sched_setaffinity(0, sizeof set, &set)) {
			perror("sched_setaffinity");
			return 1;
		}
	}
	unsigned char *scratch = scratch_init();
	const struct kernel *calb = find("cal_add", 32);
	const struct kernel *cal2b = find("cal_add", 64);
	uint64_t *v = calloc(4 * reps, sizeof *v);
	long long *d = calloc(reps, sizeof *d);
	/* The first word of every page of the rings, which is a pointer of
	 * any ring with a stride that divides the page. */
	enum { PAGE = 4096, NPAGES = (SCRATCH_FAR - SCRATCH_RINGS) / PAGE };
	uint64_t *ringw = verify ? calloc(NPAGES, sizeof *ringw) : NULL;

	for (unsigned p = 0; verify && p < NPAGES; p++)
		ringw[p] = *(uint64_t *)(scratch + SCRATCH_RINGS + p * PAGE);

	if (!sim)
		warm(0.5);
	for (unsigned i = 0; i + 1 < NKERNELS; i += 2) {
		const struct kernel *kb = &kernels[i], *k2b = &kernels[i + 1];
		unsigned long iters = kb->iters * scale;

		if (iters < 2)
			iters = 2;
		if (!selected(kb->name, argv + optind, argc - optind))
			continue;
		if (sim) {
			for (unsigned r = 0; r < reps; r++) {
				kb->fn(iters, scratch);
				check(kb, scratch, ringw, NPAGES, PAGE);
				printf("call %s %u %lu\n", kb->name, kb->bodies,
				       iters);
				k2b->fn(iters, scratch);
				check(k2b, scratch, ringw, NPAGES, PAGE);
				printf("call %s %u %lu\n", k2b->name,
				       k2b->bodies, iters);
			}
			/* As the timed runs below do: the write between
			 * kernels is part of what each one starts from. */
			fflush(stdout);
			continue;
		}
		uint64_t mn[4] = { ~0ULL, ~0ULL, ~0ULL, ~0ULL };
		for (unsigned r = 0; r < reps; r++) {
			uint64_t t[4];

			t[0] = calb->fn(calb->iters, scratch);
			t[1] = cal2b->fn(cal2b->iters, scratch);
			t[2] = kb->fn(iters, scratch);
			t[3] = k2b->fn(iters, scratch);
			for (int j = 0; j < 4; j++) {
				v[j * reps + r] = t[j];
				if (t[j] < mn[j])
					mn[j] = t[j];
			}
		}
		uint64_t md[4];
		/* median() sorts, so take the pairs' differences first. */
		for (unsigned r = 0; r < reps; r++)
			d[r] = (long long)(v[3 * reps + r] - v[2 * reps + r]);
		for (int j = 0; j < 4; j++)
			md[j] = median(v + j * reps, reps);
		/* cal_add runs 8 dependent adds per body. */
		double calcyc = 8.0 * calb->iters * calb->bodies;
		printf("{\"name\": \"%s\", \"bodies\": %u, \"iters\": %lu, "
		       "\"min\": [%llu, %llu], \"med\": [%llu, %llu], "
		       "\"cal_min\": [%llu, %llu], \"cal_med\": [%llu, %llu], "
		       "\"ratio\": %.6f, \"d\": [",
		       kb->name, kb->bodies, iters,
		       (unsigned long long)mn[2], (unsigned long long)mn[3],
		       (unsigned long long)md[2], (unsigned long long)md[3],
		       (unsigned long long)mn[0], (unsigned long long)mn[1],
		       (unsigned long long)md[0], (unsigned long long)md[1],
		       calcyc / (double)(mn[1] - mn[0]));
		for (unsigned r = 0; r < reps; r++)
			printf("%s%lld", r ? ", " : "", d[r]);
		printf("]}\n");
		fflush(stdout);
	}
	return 0;
}
