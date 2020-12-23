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
close_memarea(struct zctap_mem **mem, void *ptr, size_t sz)
{
	zctap_close_memarea(mem);
	zctap_free_memory(ptr, sz, opt.memtype);
}

static int
open_memarea(struct zctap_mem **mem, void **ptr, size_t sz)
{
	int idx;

	*ptr = zctap_alloc_memory(sz, opt.memtype);
	CHECK(*ptr);

	CHK_ERR(zctap_open_memarea(mem));

	idx = zctap_add_memarea(*mem, *ptr, sz, opt.memtype);
	CHECK(idx > 0);

	return idx;
}

static void
test_memarea(size_t sz)
{
	struct zctap_mem *mem = NULL;
	void *ptr;

	open_memarea(&mem, &ptr, sz);
	close_memarea(&mem, ptr, sz);
}

static void
test_ctx_nop(const char *ifname, size_t sz)
{
	struct zctap_mem *mem = NULL;
	struct zctap_ctx *ctx = NULL;
	void *ptr;

	open_memarea(&mem, &ptr, sz);

	CHK_ERR(zctap_open_ctx(&ctx, ifname));
	zctap_close_ctx(&ctx);

	close_memarea(&mem, ptr, sz);
}

static void
test_dmamap1(const char *ifname, size_t sz)
{
	struct zctap_mem *mem = NULL;
	struct zctap_ctx *ctx = NULL;
	void *ptr;
	int idx;

	idx = open_memarea(&mem, &ptr, sz);
	CHK_ERR(zctap_open_ctx(&ctx, ifname));

	CHK_ERR(zctap_attach_region(ctx, mem, idx));

	zctap_close_ctx(&ctx);
	close_memarea(&mem, ptr, sz);
}

static void
test_dmamap2(const char *ifname, size_t sz)
{
	struct zctap_mem *mem = NULL;
	struct zctap_ctx *ctx = NULL;
	void *ptr;
	int idx;

	idx = open_memarea(&mem, &ptr, sz);
	CHK_ERR(zctap_open_ctx(&ctx, ifname));

	CHK_ERR(zctap_attach_region(ctx, mem, idx));

	zctap_close_memarea(&mem);
	zctap_close_ctx(&ctx);
	zctap_free_memory(ptr, sz, opt.memtype);
}

static void
test_dmamap3(const char *ifname, size_t sz)
{
	struct zctap_mem *mem = NULL;
	struct zctap_ctx *ctx = NULL;
	void *ptr;
	int idx;

	CHK_ERR(zctap_open_ctx(&ctx, ifname));
	idx = open_memarea(&mem, &ptr, sz);

	CHK_ERR(zctap_attach_region(ctx, mem, idx));

	zctap_close_memarea(&mem);
	zctap_close_ctx(&ctx);
	zctap_free_memory(ptr, sz, opt.memtype);
}

static void
test_dmamap4(const char *ifname, size_t sz)
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

int
main(int argc, char **argv)
{
	parse_cmdline(argc, argv);

	test_memarea(1024 * 64);
	test_ctx_nop(opt.ifname, 1024 * 64);
	test_dmamap1(opt.ifname, 1024 * 64);
	test_dmamap2(opt.ifname, 1024 * 64);
	test_dmamap3(opt.ifname, 1024 * 64);
	test_dmamap4(opt.ifname, 1024 * 64);

	return 0;
}
