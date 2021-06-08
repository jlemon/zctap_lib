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
	int fill_entries;
	int queue_id;
	int memtype;
} opt = {
	.ifname		= "eth0",
	.sz		= 1024 * 64,
	.nentries	= 1024,
	.fill_entries	= 1024,
	.queue_id	= -1,
	.memtype	= MEMTYPE_HOST,
};

static void
usage(const char *prog)
{
	error(1, 0, "Usage: %s [options]", prog);
}

#define OPTSTR "i:s:q:m"

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
		case 'q':
			opt.queue_id = atoi(optarg);
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
		CHECK(ptr[i] = util_alloc_memory(opt.sz, opt.memtype));

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
test_one(int queue_id)
{
	struct zctap_ifq_param ifq_param;
	struct zctap_ctx *ctx = NULL;
	struct zctap_ifq *ifq = NULL;
	void *ptr[2];

	ctx = setup_ctx(array_size(ptr), ptr);

	zctap_init_ifq_param(&ifq_param, false);
	ifq_param.queue_id = queue_id;
	ifq_param.fill.entries = opt.fill_entries;
	CHK_ERR(zctap_open_ifq(&ifq, ctx, &ifq_param));
	printf("returned queue %d\n", zctap_ifq_id(ifq));

	zctap_close_ifq(&ifq);
	close_ctx(ctx, array_size(ptr), ptr);
}

static void
test_ordering(void)
{
	struct zctap_socket_param socket_param;
	struct zctap_ifq_param ifq_param;
	struct zctap_ctx *ctx = NULL;
	struct zctap_ifq *ifq = NULL;
	struct zctap_skq *skq = NULL;
	void *ptr[2];
	int fd;

	ctx = setup_ctx(array_size(ptr), ptr);

	zctap_init_ifq_param(&ifq_param, false);
	ifq_param.queue_id = opt.queue_id;
	ifq_param.fill.entries = opt.fill_entries;
	CHK_ERR(zctap_open_ifq(&ifq, ctx, &ifq_param));
	printf("returned queue %d\n", zctap_ifq_id(ifq));

	CHK_SYS(fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP));
	zctap_init_socket_param(&socket_param, opt.nentries);
	CHK_ERR(zctap_attach_socket(&skq, ctx, fd, &socket_param));

	close_ctx(ctx, array_size(ptr), ptr);
	zctap_close_ifq(&ifq);

	close(fd);
	zctap_detach_socket(&skq);
}

int
main(int argc, char **argv)
{
	parse_cmdline(argc, argv);

	printf("using queue id %d\n", opt.queue_id);
	test_one(opt.queue_id);
	test_ordering();

	return 0;
}
