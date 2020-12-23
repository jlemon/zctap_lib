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
test_one(const char *ifname)
{
	struct zctap_ctx *ctx = NULL;

	CHK_ERR(zctap_open_ctx(&ctx, ifname));
	zctap_close_ctx(&ctx);
}

static void
test_two(const char *ifname)
{
	struct zctap_ctx *ctx1 = NULL, *ctx2 = NULL;

	CHK_ERR(zctap_open_ctx(&ctx1, ifname));
	CHK_ERR(zctap_open_ctx(&ctx2, ifname));
	zctap_close_ctx(&ctx1);
	zctap_close_ctx(&ctx2);
}

static void
test_mem(const char *ifname, size_t sz)
{
	struct zctap_ctx *ctx = NULL;
	void *ptr;

	ptr = zctap_alloc_memory(sz, opt.memtype);
	CHECK(ptr);

	CHK_ERR(zctap_open_ctx(&ctx, ifname));
	CHK_ERR(zctap_register_memory(ctx, ptr, sz, opt.memtype));

	zctap_close_ctx(&ctx);
	zctap_free_memory(ptr, sz, opt.memtype);
}

static void
test_mem2(const char *ifname, size_t sz)
{
	struct zctap_ctx *ctx = NULL;
	void *ptr, *ptr2;

	ptr = zctap_alloc_memory(sz, opt.memtype);
	CHECK(ptr);
	ptr2 = zctap_alloc_memory(sz, opt.memtype);
	CHECK(ptr2);

	CHK_ERR(zctap_open_ctx(&ctx, ifname));
	CHK_ERR(zctap_register_memory(ctx, ptr, sz, opt.memtype));
	CHK_ERR(zctap_register_memory(ctx, ptr2, sz, opt.memtype));

	zctap_close_ctx(&ctx);
	zctap_free_memory(ptr, sz, opt.memtype);
	zctap_free_memory(ptr2, sz, opt.memtype);
}


static void
test_sharing(const char *ifname, size_t sz)
{
	struct zctap_ctx *ctx1 = NULL, *ctx2 = NULL;
	struct zctap_mem *mem = NULL;
	void *ptr;
	int idx;

	CHK_ERR(zctap_open_memarea(&mem));
	ptr = zctap_alloc_memory(sz, opt.memtype);
	CHECK(ptr);
	idx = zctap_add_memarea(mem, ptr, sz, opt.memtype);
	CHECK(idx > 0);

	CHK_ERR(zctap_open_ctx(&ctx1, ifname));
	CHK_ERR(zctap_attach_region(ctx1, mem, idx));

	CHK_ERR(zctap_open_ctx(&ctx2, ifname));
	CHK_ERR(zctap_attach_region(ctx2, mem, idx));

	zctap_close_ctx(&ctx1);

	CHK_ERR(zctap_open_ctx(&ctx1, ifname));
	CHK_ERR(zctap_attach_region(ctx1, mem, idx));

	zctap_close_ctx(&ctx2);
	zctap_close_ctx(&ctx1);
	zctap_close_memarea(&mem);
	zctap_free_memory(ptr, sz, opt.memtype);
}

static void
test_ordering(const char *ifname, size_t sz)
{
	struct zctap_ctx *ctx = NULL;
	struct zctap_mem *mem = NULL;
	void *ptr, *ptr2;
	int idx, err;

	CHK_ERR(zctap_open_memarea(&mem));
	ptr = zctap_alloc_memory(sz, opt.memtype);
	CHECK(ptr);
	idx = zctap_add_memarea(mem, ptr, sz, opt.memtype);
	CHECK(idx > 0);

	CHK_ERR(zctap_open_ctx(&ctx, ifname));
	CHK_ERR(zctap_attach_region(ctx, mem, idx));

	/* can't add memory region more than once */
	err = zctap_attach_region(ctx, mem, idx);
	CHECK(err == -EEXIST);

	/* adding same memory region to internal memarea is not allowed */
	err = zctap_register_memory(ctx, ptr, sz, opt.memtype);
	CHECK(err == -EEXIST);

	/* close memarea while in use */
	zctap_close_memarea(&mem);

	ptr2 = zctap_alloc_memory(sz, opt.memtype);
	CHECK(ptr2);
	CHK_ERR(zctap_register_memory(ctx, ptr2, sz, opt.memtype));

	/* free memory while in use */
	zctap_free_memory(ptr2, sz, opt.memtype);

	zctap_close_ctx(&ctx);
	zctap_free_memory(ptr, sz, opt.memtype);
}

int
main(int argc, char **argv)
{
	parse_cmdline(argc, argv);

	test_one(opt.ifname);
	test_two(opt.ifname);
	test_mem(opt.ifname, 1024 * 64);
	test_mem2(opt.ifname, 1024 * 64);
	test_sharing(opt.ifname, 1024 * 64);
	test_ordering(opt.ifname, 1024 * 64);

	return 0;
}
