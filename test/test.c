#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <libgen.h>
#include <getopt.h>

#include "zctap_lib.h"

struct opt {
	const char *iface;
	bool poll;
	bool nonblock;
	bool zc;
	bool rcv;

	/* iou parameters */
	struct {
		int entries;
	} iou;
};

/* defaults */
struct opt opt = {
	.iface = "eth0",
};

static void usage(const char *prog)
{
	error(1, 0, "Usage: %s [options]", prog);
}

#define OPTSTR "i:"

static void
parse_cmdline(int argc, char **argv)
{
	int c;

	while ((c = getopt(argc, argv, OPTSTR)) != -1) {
		switch (c) {
		case 'i':
			opt.iface = optarg;
			break;
		default:
			usage(basename(argv[0]));
		}
	}
}

static void
test_normal(size_t sz, int count)
{
	struct zctap_mem *mem = NULL;
	void *ptr[count];
	int idx[count];
	int i;

	CHK_ERR(zctap_open_memarea(&mem));

	for (i = 0; i < count; i++) {
		ptr[i] = zctap_alloc_memory(sz, MEMTYPE_HOST);
		CHECK(ptr[i]);

		idx[i] = zctap_register_region(mem, ptr[i], sz, MEMTYPE_HOST);
		CHECK(idx[i] >= 0);
	}

	CHK_ERR(zctap_close_memarea(&mem));

	for (i = 0; i < count; i++)
		zctap_free_memory(ptr[i], sz, MEMTYPE_HOST);

	for (i = 0; i < count; i++)
		CHECK_MSG(idx[i] == i+1, "idx[%d] == %d", i, idx[i]);
}

static void
test_one(size_t sz)
{
	test_normal(sz, 1);
}

static void
test_overlap(size_t sz)
{
	struct zctap_mem *mem = NULL;
	void *ptr;
	int idx;

	ptr = zctap_alloc_memory(sz, MEMTYPE_HOST);
	CHECK(ptr);

	CHK_ERR(zctap_open_memarea(&mem));

	idx = zctap_register_region(mem, ptr, sz, MEMTYPE_HOST);
	CHECK(idx >= 0);

	idx = zctap_register_region(mem, ptr, sz, MEMTYPE_HOST);
	CHECK(idx == -EEXIST);

	CHK_ERR(zctap_close_memarea(&mem));

	zctap_free_memory(ptr, sz, MEMTYPE_HOST);
}

static void
test_duplicate(size_t sz)
{
	struct zctap_mem *mem1 = NULL, *mem2 = NULL;
	void *ptr;
	int idx;

	ptr = zctap_alloc_memory(sz, MEMTYPE_HOST);
	CHECK(ptr);

	CHK_ERR(zctap_open_memarea(&mem1));
	CHK_ERR(zctap_open_memarea(&mem2));

	idx = zctap_register_region(mem1, ptr, sz, MEMTYPE_HOST);
	CHECK(idx >= 0);

	idx = zctap_register_region(mem2, ptr, sz, MEMTYPE_HOST);
	CHECK(idx == -EEXIST);

	CHK_ERR(zctap_close_memarea(&mem1));
	CHK_ERR(zctap_close_memarea(&mem2));

	zctap_free_memory(ptr, sz, MEMTYPE_HOST);
}

int
main(int argc, char **argv)
{
	parse_cmdline(argc, argv);

	/* test single regions of different sizes */
	test_one(1024);
	test_one(1024 * 1024);
	test_one(1024 * 1024 * 1024);

	/* multiple regions in same memarea */
	test_normal(1024 * 16, 8);

	/* overlapping regions in same area are disallowed */
	test_overlap(1024 * 16);

	/* duplicate areas in different memareas are disallowed */
	test_duplicate(1024 * 16);

	return 0;
}
