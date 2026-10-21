# SPDX-License-Identifier: GPL-3.0-or-later
CC = ia64-unknown-linux-gnu-gcc
AS = ia64-unknown-linux-gnu-as
CFLAGS = -O2 -Wall -Iharness -Ibuild
ONLY ?=

all: build/probe

build/kernels.s build/kernels.inc build/kernels.json: gen/*.py
	python3 gen/gen.py --outdir build $(foreach m,$(ONLY),--only $(m))

build/kernels.o: build/kernels.s
	$(AS) -x $< -o $@

build/probe: harness/probe.c harness/scratch.c harness/*.h build/kernels.inc build/kernels.o
	$(CC) $(CFLAGS) -static -o $@ harness/probe.c harness/scratch.c build/kernels.o

clean:
	rm -rf build

.PHONY: all clean
