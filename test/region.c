#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <libgen.h>
#include <getopt.h>
#include <string.h>

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
	void *ptr[count];
	int idx[count];
	int i;

	for (i = 0; i < count; i++) {
		CHK_FOR(ptr[i] = util_alloc_memory(sz, opt.memtype));
		CHK_ERR(idx[i] = util_create_region(ptr[i], sz, opt.memtype));
	}

	for (i = 0; i < count; i++)
		util_free_memory(ptr[i], sz, opt.memtype);

	for (i = 0; i < count; i++)
		CHK_SYS(close(idx[i]));
}

static void
test_one(size_t sz)
{
	test_normal(sz, 1);
}

static void
test_overlap(size_t sz)
{
	void *ptr;
	int idx[2];
	size_t half = sz / 2;

	CHK_FOR(ptr = util_alloc_memory(sz, opt.memtype));

	CHK_ERR(idx[0] = util_create_region(ptr, half, opt.memtype));

	idx[1] = util_create_region(ptr, half, opt.memtype);
	CHK_FOR(idx[1] == -EEXIST);

	idx[1] = util_create_region(ptr + (half/2), (half/2), opt.memtype);
	CHK_FOR(idx[1] == -EEXIST);

	idx[1] = util_create_region(ptr + (half/2), half, opt.memtype);
	CHK_FOR(idx[1] == -EEXIST);

	CHK_ERR(idx[1] = util_create_region(ptr + half, half, opt.memtype));

	CHK_SYS(close(idx[0]));
	CHK_SYS(close(idx[1]));

	util_free_memory(ptr, sz, opt.memtype);
}

static void
test_sanity(void)
{
	struct {
		struct zctap_host_param real;
		int trash;
	} p;
	int rc;

	memset(&p, 0, sizeof(p));
	p.trash = 0xdead;

	/* parameter oversized */
	rc = zctap(ZCTAP_CREATE_HOST_REGION, &p, sizeof(p));
	CHK_FOR(rc == -1 && errno == E2BIG);

	/* zero region length */
	p.trash = 0;
	rc = zctap(ZCTAP_CREATE_HOST_REGION, &p, sizeof(p));
	CHK_FOR(rc == -1 && errno == EINVAL);

	/* too short, so invalid length */
	rc = zctap(ZCTAP_CREATE_HOST_REGION, &p, sizeof(void *));
	CHK_FOR(rc == -1 && errno == EINVAL);

	/* length is too short */
	p.real.iov.iov_len = 1024;
	rc = zctap(ZCTAP_CREATE_HOST_REGION, &p, sizeof(p));
	CHK_FOR(rc == -1 && errno == EINVAL);

	/* length is not page aligned */
	p.real.iov.iov_len = 4096 + 1024;
	rc = zctap(ZCTAP_CREATE_HOST_REGION, &p, sizeof(p));
	CHK_FOR(rc == -1 && errno == EINVAL);

	/* buffer is not page aligned */
	p.real.iov.iov_base = (void *)0xdeadbeef;
	p.real.iov.iov_len = 4096;
	rc = zctap(ZCTAP_CREATE_HOST_REGION, &p, sizeof(p));
	CHK_FOR(rc == -1 && errno == EINVAL);
}

int
main(int argc, char **argv)
{
	parse_cmdline(argc, argv);

	/* test assertion/failure cases. */
	test_sanity();

	/* test single regions of different sizes */
	test_one(4096);
	test_one(1024 * 1024);
	test_one(1024 * 1024 * 1024);

	/* multiple regions in same memarea */
	test_normal(1024 * 64, 8);

	/* overlapping regions in same area are disallowed */
	test_overlap(1024 * 16);

	return 0;
}
