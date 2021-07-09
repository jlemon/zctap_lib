#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <libgen.h>
#include <getopt.h>
#include <net/if.h>

#include "zctap_lib.h"

struct {
	const char *ifname;
	int memtype;
	bool debug;
} opt = {
	.ifname		= "eth0",
	.memtype	= MEMTYPE_HOST,
};

static void
usage(const char *prog)
{
	error(1, 0, "Usage: %s [options]", prog);
}

#define OPTSTR "di:m"

static void
parse_cmdline(int argc, char **argv)
{
	int c;

	while ((c = getopt(argc, argv, OPTSTR)) != -1) {
		switch (c) {
		case 'd':
			opt.debug = true;
			break;
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
test_region(size_t sz)
{
	void *ptr;
	int r;

	CHK_FOR(ptr = util_alloc_memory(sz, opt.memtype));
	CHK_ERR(r = util_create_region(ptr, sz, opt.memtype));

	CHK_SYS(close(r));
	util_free_memory(ptr, sz, opt.memtype);
}

static void
test_mem(const char *ifname, size_t sz)
{
	struct zctap_ctx *ctx = NULL;
	void *ptr;
	int r;

	CHK_FOR(ptr = util_alloc_memory(sz, opt.memtype));
	CHK_ERR(r = util_create_region(ptr, sz, opt.memtype));
	CHK_ERR(zctap_open_ctx(&ctx, ifname));

	CHK_ERR(zctap_attach_region(ctx, r));

	zctap_close_ctx(&ctx);
	CHK_SYS(close(r));
	util_free_memory(ptr, sz, opt.memtype);
}

static void
test_mem2(const char *ifname, size_t sz)
{
	struct zctap_ctx *ctx = NULL;
	void *ptr1, *ptr2;

	CHK_FOR(ptr1 = util_alloc_memory(sz, opt.memtype));
	CHK_FOR(ptr2 = util_alloc_memory(sz, opt.memtype));

	CHK_ERR(zctap_open_ctx(&ctx, ifname));
	CHK_ERR(util_register_memory(ctx, ptr1, sz, opt.memtype));
	CHK_ERR(util_register_memory(ctx, ptr2, sz, opt.memtype));

	zctap_close_ctx(&ctx);
	util_free_memory(ptr1, sz, opt.memtype);
	util_free_memory(ptr2, sz, opt.memtype);
}

static void
test_sharing(const char *ifname, size_t sz)
{
	struct zctap_ctx *ctx1 = NULL, *ctx2 = NULL;
	void *ptr;
	int r;

	CHK_FOR(ptr = util_alloc_memory(sz, opt.memtype));
	CHK_ERR(r = util_create_region(ptr, sz, opt.memtype));

	CHK_ERR(zctap_open_ctx(&ctx1, ifname));
	CHK_ERR(zctap_attach_region(ctx1, r));

	CHK_ERR(zctap_open_ctx(&ctx2, ifname));
	CHK_ERR(zctap_attach_region(ctx2, r));

	zctap_close_ctx(&ctx1);

	CHK_ERR(zctap_open_ctx(&ctx1, ifname));
	CHK_ERR(zctap_attach_region(ctx1, r));

	zctap_close_ctx(&ctx2);
	zctap_close_ctx(&ctx1);
	CHK_SYS(close(r));
	util_free_memory(ptr, sz, opt.memtype);
}

static void
test_ordering(const char *ifname, size_t sz)
{
	struct zctap_ctx *ctx = NULL;
	void *ptr, *ptr2;
	int r, r2;
	int err;

	CHK_FOR(ptr = util_alloc_memory(sz, opt.memtype));
	CHK_ERR(r = util_create_region(ptr, sz, opt.memtype));

	CHK_ERR(zctap_open_ctx(&ctx, ifname));
	CHK_ERR(zctap_attach_region(ctx, r));

	/* can't add memory region more than once */
	err = zctap_attach_region(ctx, r);
	CHK_FOR(err == -EEXIST);

	/* creating region over existing one is not allowed */
	r2 = util_create_region(ptr, sz, opt.memtype);
	CHK_FOR(r2 == -EEXIST);

	/* close region while in use */
	CHK_SYS(close(r));

	CHK_FOR(ptr2 = util_alloc_memory(sz, opt.memtype));
	CHK_ERR(util_register_memory(ctx, ptr2, sz, opt.memtype));

	/* free memory while in use */
	util_free_memory(ptr2, sz, opt.memtype);

	zctap_close_ctx(&ctx);
	util_free_memory(ptr, sz, opt.memtype);
}

#define T(fcn, ...) do {						\
	if (opt.debug) {						\
		fprintf(stderr, "Calling %s...\n", #fcn);		\
		sleep(2);						\
	}								\
	fcn(__VA_ARGS__);						\
} while (0)

int
main(int argc, char **argv)
{
	parse_cmdline(argc, argv);

	T(test_one, opt.ifname);
	T(test_two, opt.ifname);
	T(test_region, 1024 * 64);
	T(test_mem, opt.ifname, 1024 * 64);
	T(test_mem2, opt.ifname, 1024 * 64);
	T(test_sharing, opt.ifname, 1024 * 64);
	T(test_ordering, opt.ifname, 1024 * 64);

	return 0;
}
