#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <getopt.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/epoll.h>

#include "zctap_lib.h"

struct {
	bool stop;
	unsigned long bytes_submitted;
	unsigned long bytes_reclaimed;
	unsigned long wait_ns;
	unsigned long wait_count;
	int notify_count;;
} run;

struct node {
	int family;
	int socktype;
	int protocol;
	socklen_t addrlen;
	struct sockaddr_storage addr;
};

struct {
	const char *ifname;
	size_t sz;
	int nentries;
	int fill_entries;
	int queue_id;
	int memtype;
	bool udp_proto;
} opt = {
	.ifname		= "eth0",
	.sz		= 1024 * 1024 * 2,
	.nentries	= 1024,
	.fill_entries	= 10240,
	.queue_id	= -1,
	.memtype	= MEMTYPE_HOST,
};

static void
usage(const char *prog)
{
	error(1, 0, "Usage: %s [options] hostname port", prog);
}

#define OPTSTR "i:s:q:mu"

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
		case 'u':
			opt.udp_proto = true;
			break;
		default:
			usage(basename(argv[0]));
		}
	}
}

unsigned long
nsec(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000UL + ts.tv_nsec;
}

unsigned long
elapsed(unsigned long start)
{
	return nsec() - start;
}

static struct zctap_ctx *
setup_ctx(int count, void *ptr[])
{
	struct zctap_ctx *ctx = NULL;
	int i;

	CHK_ERR(zctap_open_ctx(&ctx, opt.ifname));

	for (i = 0; i < count; i++) {
		ptr[i] = zctap_alloc_memory(opt.sz, opt.memtype);
		CHECK(ptr[i]);

		CHK_ERR(zctap_register_memory(ctx, ptr[i], opt.sz,
					       opt.memtype));
	}

	return ctx;
}

static void
close_ctx(struct zctap_ctx *ctx, int count, void *ptr[])
{
	int i;

	zctap_close_ctx(&ctx);

	for (i = 0; i < count; i++)
		zctap_free_memory(ptr[i], opt.sz, opt.memtype);
}

void
set_blocking_mode(int fd, bool on)
{
	int flag;

	CHECK((flag = fcntl(fd, F_GETFL)) != -1);

	if (on)
		flag &= ~O_NONBLOCK;
	else
		flag |= O_NONBLOCK;

	CHK_SYS(fcntl(fd, F_SETFL, flag));

	flag = fcntl(fd, F_GETFL);
	CHECK(!(flag & O_NONBLOCK) == on);
}

const char *
show_node_addr(struct node *node)
{
	static char host[NI_MAXHOST];
	int rc;

	rc = getnameinfo((struct sockaddr *)&node->addr,
	    (node->family == AF_INET) ? sizeof(struct sockaddr_in) :
					sizeof(struct sockaddr_in6),
	    host, NI_MAXHOST, NULL, 0, NI_NUMERICHOST);
	CHECK_MSG(rc == 0, "getnameinfo: %s", gai_strerror(rc));
	return host;
}

static bool
name2addr(const char *name, struct node *node, bool local)
{
	struct addrinfo hints, *result, *ai;
	int s, rc;

	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_family = node->family;
	hints.ai_socktype = node->socktype;
	node->addrlen = 0;

	rc = getaddrinfo(name, NULL, &hints, &result);
	CHECK_MSG(rc == 0, "getaddrinfo: %s", gai_strerror(rc));

	for (ai = result; ai != NULL; ai = ai->ai_next) {
		if (!local)
			break;

		s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
		if (s == -1)
			continue;

		rc = bind(s, ai->ai_addr, ai->ai_addrlen);
		close(s);

		if (rc == 0)
			break;
	}

	if (ai != NULL) {
		node->addrlen = ai->ai_addrlen;
		node->protocol = ai->ai_protocol;
		memcpy(&node->addr, ai->ai_addr, ai->ai_addrlen);
	}

	freeaddrinfo(result);

	return node->addrlen != 0;
}

void
set_port(struct node *node, int port)
{
	struct sockaddr_in *sin;
	struct sockaddr_in6 *sin6;

	if (node->family == AF_INET6) {
		sin6 = (struct sockaddr_in6 *)&node->addr;
		sin6->sin6_port = htons(port);
	} else {
		sin = (struct sockaddr_in *)&node->addr;
		sin->sin_port = htons(port);
	}
}

static void
net_connect(int fd, const char *hostname, short port)
{
	struct node node;
	int one = 1;

	node.family = AF_INET6;
	node.socktype = SOCK_STREAM;
	if (!name2addr(hostname, &node, false))
		CHECK_MSG(1, "could not get IP of %s", hostname);

	set_port(&node, port);

	CHK_SYS(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)));
	CHK_SYS(setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &one, sizeof(one)));

	CHK_SYS(connect(fd, (struct sockaddr *)&node.addr, node.addrlen));

	set_blocking_mode(fd, true);
}

#define SO_NOTIFY	71

#define N_SLICES	4

static void
send_loop(int fd, struct zctap_skq *skq, uint64_t addr)
{
	uint8_t cbuf[CMSG_SPACE(sizeof(uint64_t))];
	bool busy[N_SLICES];
	struct cmsghdr *cmsg;
	struct iovec iov;
	struct msghdr msg = {
		.msg_iov = &iov,
		.msg_iovlen = 1,
		.msg_control = &cbuf,
	};
	uint64_t *data, base, *notify;
	int i, n, count, loopc, slice;
	struct epoll_event ev;
	size_t sz;
	int ep;

	cmsg = (struct cmsghdr *)cbuf;
	cmsg->cmsg_level = SOL_SOCKET;
	cmsg->cmsg_type = SO_NOTIFY;
	cmsg->cmsg_len = CMSG_LEN(sizeof(uint64_t));
	data = (uint64_t *)CMSG_DATA(cmsg);

	sz = 10000;
	iov.iov_len = sz;
	count = (opt.sz / sz) / N_SLICES;
	loopc = 0;
	slice = 0;

	ev.events = EPOLLRDBAND;
	CHK_SYS(ep = epoll_create(1));
	CHK_SYS(epoll_ctl(ep, EPOLL_CTL_ADD, fd, &ev));

	printf("send loop\n");
	while (!run.stop) {
		unsigned long wait_ns;

		wait_ns = 0;
		while (busy[slice]) {
			if (zctap_get_cq_batch(skq, &notify, 1)) {
				n = *notify % N_SLICES;
				CHECK_MSG(busy[n], "Slice %d !busy\n", n);
				busy[n] = false;
				run.bytes_reclaimed += (count * sz);
				run.notify_count--;
			} else {
				CHECK(!wait_ns);
				wait_ns = nsec();
				CHK_INTR(n = epoll_wait(ep, &ev, 1, -1), out);
				run.wait_ns += elapsed(wait_ns);
				run.wait_count++;
				CHECK(n != 0);
			}
		}
		base = addr + (slice * count * sz);
		for (i = 0; i < count - 1; i++) {
			iov.iov_base = (void *)(base + i * sz);
			CHK_INTR(n = sendmsg(fd, &msg, MSG_ZCTAP), out);
			CHECK(n == sz);
			run.bytes_submitted += sz;
		}

		iov.iov_base = (void *)(base + i * sz);
		msg.msg_controllen = cmsg->cmsg_len;
		*data = loopc++;
		CHK_SYS(n = sendmsg(fd, &msg, MSG_ZCTAP));
		CHECK(n == sz);
		msg.msg_controllen = 0;
		run.bytes_submitted += sz;
		run.notify_count++;

		busy[slice] = true;
		slice = (slice + 1) == N_SLICES ? 0 : slice + 1;

		if (zctap_get_cq_batch(skq, &notify, 1)) {
			n = *notify % N_SLICES;
			CHECK_MSG(busy[n], "Slice %d !busy\n", n);
			busy[n] = false;
			run.bytes_reclaimed += (count * sz);
			run.notify_count--;
		}
	}
out:
	;
}

#define ns_to_sec(ns)	((long double)ns / 1000000000UL)
#define safediv(n, d)	((d) ? (long double)(n) / (d) : 0)

void zctap_debug_skq(struct zctap_skq *skq);

static void
statistics(unsigned long elapsed_ns)
{
	static const char *scale[] = {
		"bytes", "KB", "MB", "GB",
	};
	static const char *uscale[] = {
		"ns", "us", "ms", "sec",
	};
	long double seconds;
	long double rate;
	int idx;

	seconds = ns_to_sec(elapsed_ns);
	rate = run.bytes_submitted / seconds;
	for (idx = 0; rate >= 1000; idx++)
		rate /= 1000;
	printf("\n");
	printf("%ld bytes in %.4Lg seconds, %.4Lg %s/sec\n",
		run.bytes_submitted, seconds, rate, scale[idx]);

	rate = safediv(run.wait_ns, run.wait_count);
	for (idx = 0; rate >= 1000; idx++)
		rate /= 1000;
	printf("Waited %ld times %ld ns, %.4Lg %s/wait\n",
		run.wait_count, run.wait_ns, rate, uscale[idx]);
	printf("Reclaimed %ld bytes,  oustanding notifies : %d\n",
		run.bytes_reclaimed, run.notify_count);
}

static void
test_send(const char *hostname, short port)
{
	struct zctap_ctx *ctx;
	struct zctap_ifq *ifq;
	struct zctap_skq *skq;
	void *ptr[1], *pktbuf;
	unsigned long stamp;
	size_t sz;
	int fd;

	ctx = setup_ctx(array_size(ptr), ptr);

	sz = opt.fill_entries * 4096;
	pktbuf = zctap_alloc_memory(sz, opt.memtype);
	CHECK(pktbuf);
	CHK_ERR(zctap_register_memory(ctx, pktbuf, sz, opt.memtype));
	CHK_ERR(zctap_open_ifq(&ifq, ctx, opt.queue_id, opt.fill_entries));
	zctap_populate_ring(ifq, (uint64_t)pktbuf, opt.fill_entries);

	if (opt.udp_proto)
		CHK_SYS(fd = socket(AF_INET6, SOCK_DGRAM, IPPROTO_UDP));
	else
		CHK_SYS(fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP));
	CHK_ERR(zctap_attach_socket(&skq, ctx, fd, opt.nentries));

	net_connect(fd, hostname, port);

	stamp = nsec();
	send_loop(fd, skq, (uint64_t)ptr[0]);
	stamp = elapsed(stamp);
	zctap_debug_skq(skq);
	statistics(stamp);

	zctap_close_ifq(&ifq);
	close_ctx(ctx, array_size(ptr), ptr);
}

static void
handle_signal(int sig)
{
	run.stop = true;
}

static void
setup(void)
{
	struct sigaction sa = {
		.sa_handler = handle_signal,
	};
	sigaction(SIGINT, &sa, NULL);
}

int
main(int argc, char **argv)
{
	char *hostname;
	short port;

	parse_cmdline(argc, argv);
	if (argc - optind < 2)
		usage(basename(argv[0]));
	hostname = argv[optind];
	port = atoi(argv[optind + 1]);
	setup();

	test_send(hostname, port);

	return 0;
}
