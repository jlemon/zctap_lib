#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <libgen.h>
#include <getopt.h>

#include "zctap_lib.h"

struct {
	const char *ifname;
	int memtype;
} opt = {
	.ifname		= "eth0",
	.memtype	= MEMTYPE_HOST,
};

static void
usage(const char *prog)
{
	error(1, 0, "Usage: %s [options]", prog);
}

#define OPTSTR "i:m"

static void
parse_cmdline(int argc, char **argv)
{
	int c;

	while ((c = getopt(argc, argv, OPTSTR)) != -1) {
		switch (c) {
		case 'i':
			opt.ifname = optarg;
			break;
		case 'm':
			opt.memtype = MEMTYPE_CUDA;
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
		ptr[i] = zctap_alloc_memory(sz, opt.memtype);
		CHECK(ptr[i]);

		idx[i] = zctap_add_memarea(mem, ptr[i], sz, opt.memtype);
		CHECK(idx[i] > 0);
	}

	zctap_close_memarea(&mem);

	for (i = 0; i < count; i++)
		zctap_free_memory(ptr[i], sz, opt.memtype);

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

	ptr = zctap_alloc_memory(sz, opt.memtype);
	CHECK(ptr);

	CHK_ERR(zctap_open_memarea(&mem));

	idx = zctap_add_memarea(mem, ptr, sz, opt.memtype);
	CHECK(idx > 0);

	idx = zctap_add_memarea(mem, ptr, sz, opt.memtype);
	CHECK(idx == -EEXIST);

	zctap_close_memarea(&mem);

	zctap_free_memory(ptr, sz, opt.memtype);
}

static void
test_duplicate(size_t sz)
{
	struct zctap_mem *mem1 = NULL, *mem2 = NULL;
	void *ptr;
	int idx;

	ptr = zctap_alloc_memory(sz, opt.memtype);
	CHECK(ptr);

	CHK_ERR(zctap_open_memarea(&mem1));
	CHK_ERR(zctap_open_memarea(&mem2));

	idx = zctap_add_memarea(mem1, ptr, sz, opt.memtype);
	CHECK(idx > 0);

	idx = zctap_add_memarea(mem2, ptr, sz, opt.memtype);
	CHECK(idx == -EEXIST);

	zctap_close_memarea(&mem1);
	zctap_close_memarea(&mem2);

	zctap_free_memory(ptr, sz, opt.memtype);
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
	test_normal(1024 * 64, 8);

	/* overlapping regions in same area are disallowed */
	test_overlap(1024 * 16);

	/* duplicate areas in different memareas are disallowed */
	test_duplicate(1024 * 16);

	return 0;
}
