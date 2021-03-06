#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include "zctap_lib.h"

struct {
	const char *ifname;
	size_t sz;
	int nentries;
	int memtype;
} opt = {
	.ifname		= "eth0",
	.sz		= 1024 * 64,
	.nentries	= 1024,
	.memtype	= MEMTYPE_HOST,
};

static void
usage(const char *prog)
{
	error(1, 0, "Usage: %s [options]", prog);
}

#define OPTSTR "i:s:m"

static void
parse_cmdline(int argc, char **argv)
{
	int c;

	while ((c = getopt(argc, argv, OPTSTR)) != -1) {
		switch (c) {
		case 'i':
			opt.ifname = optarg;
			break;
		case 's':
			opt.sz = atoi(optarg);
			break;
		case 'm':
			opt.memtype = MEMTYPE_CUDA;
			break;
		default:
			usage(basename(argv[0]));
		}
	}
}

static struct zctap_ctx *
setup_ctx(int count, void *ptr[])
{
	struct zctap_ctx *ctx = NULL;
	int i;

	CHK_ERR(zctap_open_ctx(&ctx, opt.ifname));

	for (i = 0; i < count; i++) {
		CHK_FOR(ptr[i] = util_alloc_memory(opt.sz, opt.memtype));
		CHK_ERR(util_register_memory(ctx, ptr[i], opt.sz, opt.memtype));
	}

	return ctx;
}

static void
close_ctx(struct zctap_ctx *ctx, int count, void *ptr[])
{
	int i;

	zctap_close_ctx(&ctx);

	for (i = 0; i < count; i++)
		util_free_memory(ptr[i], opt.sz, opt.memtype);
}

static void
test_one(void)
{
	struct zctap_socket_param socket_param;
	struct zctap_ctx *ctx = NULL;
	struct zctap_skq *skq = NULL;
	void *ptr[2];
	int fd;

	CHK_SYS(fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP));

	ctx = setup_ctx(array_size(ptr), ptr);
	zctap_init_socket_param(&socket_param, opt.nentries);
	CHK_ERR(zctap_attach_socket(&skq, ctx, fd, &socket_param));
	
	zctap_detach_socket(&skq);
	close_ctx(ctx, array_size(ptr), ptr);
	close(fd);
}

static void
test_ordering(void)
{
	struct zctap_socket_param socket_param;
	struct zctap_ctx *ctx = NULL;
	struct zctap_skq *skq = NULL;
	void *ptr[2];
	int fd;

	CHK_SYS(fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP));

	ctx = setup_ctx(array_size(ptr), ptr);
	zctap_init_socket_param(&socket_param, opt.nentries);
	CHK_ERR(zctap_attach_socket(&skq, ctx, fd, &socket_param));
	
	close_ctx(ctx, array_size(ptr), ptr);

	close(fd);
	zctap_detach_socket(&skq);
}

int
main(int argc, char **argv)
{
	parse_cmdline(argc, argv);

	test_one();
	test_ordering();

	return 0;
}
