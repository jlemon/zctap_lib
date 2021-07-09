#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <getopt.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <sys/epoll.h>
#include <sys/param.h>
#include <signal.h>

#include "zctap_lib.h"

struct {
	bool stop;
} run;

struct node {
	int family;
	int socktype;
	int protocol;
	socklen_t addrlen;
	struct sockaddr_storage addr;
};

struct stats {
	uint64_t rx_bytes;
	uint64_t rx_frags;
	uint64_t rx_pkts;
	unsigned long start;
} stats;

struct {
	const char *ifname;
	size_t sz;
	int nentries;
	int fill_entries;
	int queue_id;
	int memtype;
	bool udp_proto;
	bool debug;
	bool normal;
} opt = {
	.ifname		= "eth0",
	.sz		= 1024 * 1024 * 2,
	.nentries	= 1024,
	.fill_entries	= 10240,
	.queue_id	= -1,
	.memtype	= MEMTYPE_HOST,
};

unsigned long
now_nsec(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000000000UL + ts.tv_nsec;
}

unsigned long
elapsed(unsigned long start)
{
	return now_nsec() - start;
}

static void
usage(const char *prog)
{
	error(1, 0, "Usage: %s [options] hostname port", prog);
}

#define OPTSTR "di:s:q:mnu"

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
		case 's':
			opt.sz = atoi(optarg);
			break;
		case 'q':
			opt.queue_id = atoi(optarg);
			break;
		case 'm':
			opt.memtype = MEMTYPE_CUDA;
			break;
		case 'n':
			opt.normal = true;
			break;
		case 'u':
			opt.udp_proto = true;
			break;
		default:
			usage(basename(argv[0]));
		}
	}
	if (!opt.normal)
		printf("requesting queue %d\n", opt.queue_id);
	if (opt.debug)
		printf("debugging on\n");

	printf("Using %s recv\n", opt.udp_proto ? "UDP" : "TCP");
}

void
set_blocking_mode(int fd, bool on)
{
	int flag;

	CHK_FOR((flag = fcntl(fd, F_GETFL)) != -1);

	if (on)
		flag &= ~O_NONBLOCK;
	else
		flag |= O_NONBLOCK;

	CHK_SYS(fcntl(fd, F_SETFL, flag));

	flag = fcntl(fd, F_GETFL);
	CHK_FOR(!(flag & O_NONBLOCK) == on);
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
	CHK_MSG(rc == 0, "getnameinfo: %s", gai_strerror(rc));
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
	CHK_MSG(rc == 0, "getaddrinfo: %s", gai_strerror(rc));

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

static int
udp_serve(const char *hostname, short port)
{
	struct node node;
	int one = 1;
	int fd;

	node.family = AF_INET6;
	node.socktype = SOCK_DGRAM;
	if (!name2addr(hostname, &node, true))
		CHK_MSG(1, "could not get IP of %s", hostname);

	set_port(&node, port);

	CHK_SYS(fd = socket(AF_INET6, SOCK_DGRAM, IPPROTO_UDP));
	CHK_SYS(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)));
	CHK_SYS(bind(fd, (struct sockaddr *)&node.addr, node.addrlen));

	set_blocking_mode(fd, true);
	return fd;
}

static int
tcp_listen(const char *hostname, short port)
{
	struct node node;
	int one = 1;
	int fd;

	node.family = AF_INET6;
	node.socktype = SOCK_STREAM;
	if (!name2addr(hostname, &node, true))
		CHK_MSG(1, "could not get IP of %s", hostname);

	set_port(&node, port);

	CHK_SYS(fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP));
	CHK_SYS(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)));
	CHK_SYS(bind(fd, (struct sockaddr *)&node.addr, node.addrlen));

	set_blocking_mode(fd, true);

	CHK_SYS(listen(fd, 1));

	return fd;
}

static int
tcp_accept(int server_fd)
{
	int fd;

	CHK_SYS(fd = accept(server_fd, NULL, NULL));

	return fd;
}

#define SO_NOTIFY	71

#define BATCH_SIZE	32

static void
hex_dump(uint64_t pkt, size_t length, uint64_t addr)
{
	const unsigned char *address = (unsigned char *)pkt;
	const unsigned char *line = address;
	size_t line_size = 16;
	unsigned char c;
	char buf[32];
	int i = 0;

	sprintf(buf, "addr=0x%lx", addr);
	printf("length = %zu\n", length);
	printf("%s | ", buf);
	while (length-- > 0) {
		printf("%02X ", *address++);
		if (!(++i % line_size) || (length == 0 && i % line_size)) {
			if (length == 0) {
				while (i++ % line_size)
					printf("__ ");
			}
			printf(" | ");	/* right close */
			while (line < address) {
				c = *line++;
				printf("%c", (c < 33 || c == 255) ? 0x2E : c);
			}
			printf("\n");
			if (length > 0)
				printf("%s | ", buf);
		}
	}
	printf("\n");
}

static void
pkt_dump(struct zctap_iovec *iov)
{
	if (!opt.debug)
		return;

	printf("base:%llx  off:%d  len:%d\n",
	       iov->base, iov->offset, iov->length);
	return;
	hex_dump(iov->base + iov->offset, iov->length, iov->base);
}

static void
handle_metadata(struct zctap_skq *skq, struct zctap_ifq *ifq,
		struct zctap_pktdata *meta)
{
	struct zctap_iovec *miov;
	void *ptr;
	int i;

	if (opt.debug)
		printf("pkt, data:%d  iov:%d  name:%d  cmsg:%d\n",
			meta->data_len, meta->iov_count,
			meta->name_len, meta->cmsg_len);

	stats.rx_frags += meta->iov_count;
	ptr = meta->data + roundup(meta->data_len, 8);
	miov = (struct zctap_iovec *)ptr;
	for (i = 0; i < meta->iov_count; i++) {
		pkt_dump(&miov[i]);
		stats.rx_bytes += miov[i].length;
		zctap_recycle_buffer(ifq, (void *)miov[i].base);
	}
	zctap_recycle_meta(skq, meta);
}

/*
 * returns:
 *   -EAGAIN if RXq is empty, and a refill is needed,
 *   0 on EOF (only returned once!)
 *   +N for # of entries processed.
 */
static int
handle_read(struct zctap_skq *skq, struct zctap_ifq *ifq)
{
	struct zctap_iovec *iov[BATCH_SIZE];
	int i, count;

	iov[0] = NULL;
	count = zctap_get_rx_batch(skq, iov, array_size(iov));
	if (!count)
		return -EAGAIN;
	stats.rx_pkts += count;		// XXX UDP only, and fix EOF handling.

	for (i = 0; i < count; i++) {
		if (opt.udp_proto) {
			handle_metadata(skq, ifq, (void *)iov[i]->base);
			continue;
		}

		if (iov[i]->base == 0) {
			count = 0;
			break;
		}

		stats.rx_frags++;
		pkt_dump(iov[i]);
		stats.rx_bytes += iov[i]->length;
		zctap_recycle_buffer(ifq, (void *)iov[i]->base);
	}
	zctap_recycle_complete(ifq);

	if (opt.udp_proto)
		zctap_submit_meta(skq);

	return count;
}

static void
stats_start(void)
{
	memset(&stats, 0, sizeof(stats));
	stats.start = now_nsec();
}

const char *
rate(unsigned long bytes, unsigned long nsec, float *rv)
{
	static const char *scale_str[] = {
		"b/s", "Kb/s", "Mb/s", "Gb/s",
	};
	int scale;
	float val;

	val = ((float)bytes / (float)nsec) * 8 * 1000000000UL;
	for (scale = 0; val > 1000; scale++)
		val /= 1000;
	*rv = val;

	if (scale > array_size(scale_str))
		return "unknown";
	return scale_str[scale];
}

static void
stats_end(void)
{
	unsigned long interval;
	const char *scale;
	float val;

	if (!stats.rx_pkts)
		return;

	interval = elapsed(stats.start);

	printf("packets: %ld\n", stats.rx_pkts);

	printf("  frags: %ld  %ld frags/pkt\n",
		stats.rx_frags,
		stats.rx_frags / stats.rx_pkts);

	printf("  bytes: %ld  %ld bytes/frag,  %ld bytes/pkt\n",
		stats.rx_bytes,
		stats.rx_bytes / stats.rx_frags,
		stats.rx_bytes / stats.rx_pkts);

	scale = rate(stats.rx_bytes, interval, &val);
	printf("  rate: %f %s\n", val, scale);
}


static uint8_t rcvbuf[64 * 1024];

static void
fd_recv_loop(int fd)
{
	struct epoll_event ev;
	int n, ep;

	ev.events = EPOLLIN | EPOLLHUP;
	CHK_SYS(ep = epoll_create(1));
	CHK_SYS(epoll_ctl(ep, EPOLL_CTL_ADD, fd, &ev));

	printf("recv loop\n");
	while (!run.stop) {
		n = epoll_wait(ep, &ev, 1, -1);
		if (n == 0)
			continue;
		if (n == -1) {
			if (errno == EINTR)
				continue;
			ERROR_HERE(1, errno, "epoll_wait");
		}
		if (ev.events & EPOLLIN) {
			n = read(fd, rcvbuf, sizeof(rcvbuf));
			if (n == 0) {
				printf("recv EOF, break...\n");
				goto out;
			}
			if (n == -1 && errno != EAGAIN)
				ERROR_HERE(1, errno, "recv");
			stats.rx_bytes += n;
			stats.rx_frags++;
			stats.rx_pkts++;
		}
		if (ev.events & EPOLLRDHUP) {
			/* handle data before exiting */
			printf("SAW EPOLLRDHUP, break..\n");
			goto out;
		}
		if (ev.events & EPOLLHUP) {
			printf("SAW EPOLLHUP, break..\n");
			goto out;
		}
		if (ev.events & EPOLLERR) {
			printf("SAW EPOLLERR, break..\n");
			goto out;
		}
	}
out:
	printf("Exiting loop.\n");
}

static void
fd_serve_socket(int fd)
{
	stats_start();

	fd_recv_loop(fd);

	stats_end();

	close(fd);
}

static void
fd_test_recv(const char *hostname, short port)
{
	int fd;

	if (opt.udp_proto) {
		fd = udp_serve(hostname, port);
		fd_serve_socket(fd);
	} else {
		int listen = tcp_listen(hostname, port);
		for (;;) {
			fd = tcp_accept(listen);
			fd_serve_socket(fd);
		}
	}
}

static void
recv_loop(int fd, struct zctap_skq *skq, struct zctap_ifq *ifq)
{
	struct epoll_event ev;
	int n, ep;

	ev.events = EPOLLIN | EPOLLHUP;
	CHK_SYS(ep = epoll_create(1));
	CHK_SYS(epoll_ctl(ep, EPOLL_CTL_ADD, fd, &ev));

	printf("recv loop\n");
	while (!run.stop) {
		n = epoll_wait(ep, &ev, 1, -1);
		if (n == 0)
			continue;
		if (n == -1) {
			if (errno == EINTR)
				continue;
			ERROR_HERE(1, errno, "epoll_wait");
		}
		if (ev.events & EPOLLIN) {
			n = handle_read(skq, ifq);
			if (n == 0) {
				printf("RXQ iov == EOF, break...\n");
				goto out;
			}
			if (n < 0) {
				if (opt.debug)
					zctap_debug_skq(skq);
				n = recv(fd, NULL, 0, 0);
				if (n == -1 && errno != EAGAIN)
					ERROR_HERE(1, errno, "recv");

				/* epoll said POLLIN, so there is data.
				 * first handle_read indicated empty RXq,
				 * so recv() transfers the data.
				 * this second handle_read MUST see data,
				 * otherwise POLLIN is wrong.
				 */
				n = handle_read(skq, ifq);
				if (n == 0) {
					printf("RXQ iov == EOF, break...\n");
					goto out;
				}
				CHK_FOR(n != -1);
#if 0
				if (n == 0) {
					printf("recv ret 0, EOF, break...\n");
					goto out;
				}
#endif
			}
		}
		if (ev.events & EPOLLRDHUP) {
			/* handle data before exiting */
			printf("SAW EPOLLRDHUP, break..\n");
			goto out;
		}
		if (ev.events & EPOLLHUP) {
			printf("SAW EPOLLHUP, break..\n");
			goto out;
		}
		if (ev.events & EPOLLERR) {
			printf("SAW EPOLLERR, break..\n");
			goto out;
		}
	}
out:
	printf("Exiting loop.\n");
}

static void
serve_socket(int fd, struct zctap_ctx *ctx, struct zctap_ifq *ifq,
	     bool use_metadata)
{
	struct zctap_socket_param socket_param;
	struct zctap_skq *skq;
	void *pktbuf;
	size_t sz;

	zctap_init_socket_param(&socket_param, opt.nentries);
	CHK_ERR(zctap_attach_socket(&skq, ctx, fd, &socket_param));

	if (use_metadata) {
		/* add memory area for metadata */
		sz = opt.nentries * 256;
		CHK_FOR(pktbuf = util_alloc_memory(sz, MEMTYPE_HOST));
		CHK_ERR(zctap_add_meta(ctx, pktbuf, sz));

		zctap_populate_meta(skq, (uint64_t)pktbuf, opt.nentries, 256);
		printf("metadata: [%p:%p]\n", pktbuf, pktbuf + sz);
	}

	stats_start();

	recv_loop(fd, skq, ifq);

	stats_end();

	/* XXX ugh - no way to remove per-socket metadata! */
	zctap_detach_socket(&skq);
	close(fd);
}

static void
test_recv(const char *hostname, short port)
{
	struct zctap_ifq_param ifq_param;
	struct zctap_ctx *ctx;
	struct zctap_ifq *ifq;
	void *pktbuf;
	size_t sz;
	int fd;

	CHK_ERR(zctap_open_ctx(&ctx, opt.ifname));

	sz = opt.fill_entries * 4096;
	CHK_FOR(pktbuf = util_alloc_memory(sz, opt.memtype));
	CHK_ERR(util_register_memory(ctx, pktbuf, sz, opt.memtype));
	zctap_init_ifq_param(&ifq_param, !opt.udp_proto);
	ifq_param.queue_id = opt.queue_id;
	ifq_param.fill.entries = opt.fill_entries;
	if (!opt.udp_proto)
		ifq_param.split_offset -= 8;
	printf("using split %d, offset %d\n",
		ifq_param.split, ifq_param.split_offset);
	CHK_ERR(zctap_open_ifq(&ifq, ctx, &ifq_param));
	printf("actual queue %d\n", zctap_ifq_id(ifq));
	printf("pktdata:  [%p:%p]\n", pktbuf, pktbuf + sz);
	zctap_populate_ring(ifq, (uint64_t)pktbuf, opt.fill_entries);

	if (opt.udp_proto) {
		fd = udp_serve(hostname, port);
		serve_socket(fd, ctx, ifq, true);
	} else {
		int listen = tcp_listen(hostname, port);
		for (;;) {
			fd = tcp_accept(listen);
			serve_socket(fd, ctx, ifq, false);
		}
	}

	zctap_close_ifq(&ifq);
	zctap_close_ctx(&ctx);
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
	if (argc - optind < 1)
		usage(basename(argv[0]));
	hostname = argv[optind];
	port = atoi(argv[optind + 1]);

	setup();

	if (opt.normal)
		fd_test_recv(hostname, port);
	else
		test_recv(hostname, port);

	return 0;
}
