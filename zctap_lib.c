#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/sockios.h>

#include "zctap_lib.h"

#ifdef USE_CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "uapi/misc/zctap_cuda.h"
#endif

#define PAGE_SIZE	4096

struct zctap_skq {
	struct shared_queue rx;
	struct shared_queue cq;
	struct shared_queue meta;
	int socket_fd;
};

struct zctap_ifq {
	int fd;
	unsigned queue_id;
	struct shared_queue fill;
};

struct zctap_ctx {
	int fd;
	unsigned ifindex;
	struct zctap_mem *mem;
};

static void
zctap_debug_queue(const char *name, struct shared_queue *q)
{
	printf("queue %s:\n", name);
	printf("  producer: %u	%u\n", q->cached_prod, *q->prod);
	printf("  consumer: %u	%u\n", q->cached_cons, *q->cons);
}

void
zctap_debug_skq(struct zctap_skq *skq)
{
	zctap_debug_queue("RX", &skq->rx);
	zctap_debug_queue("CQ", &skq->cq);
	if (skq->meta.entries)
		zctap_debug_queue("META", &skq->meta);
}

static int
zctap_mmap_queue(int fd, struct shared_queue *q, struct zctap_user_queue *u)
{

	q->map_ptr = mmap(NULL, u->map_sz, PROT_READ | PROT_WRITE,
			  MAP_SHARED | MAP_POPULATE, fd, u->map_off);
	if (q->map_ptr == MAP_FAILED)
		return -errno;

	q->mask = u->mask;
	q->map_sz = u->map_sz;
	q->elt_sz = u->elt_sz;
	q->entries = u->entries;

	q->prod = q->map_ptr + u->off.prod;
	q->cons = q->map_ptr + u->off.cons;
	q->data = q->map_ptr + u->off.data;

	q->cached_cons = 0;
	q->cached_prod = 0;

	return 0;
}

static int
zctap_mmap_socket(int fd, struct zctap_skq *skq,
		  struct zctap_socket_param *p)
{
	int rc;

	rc = zctap_mmap_queue(fd, &skq->rx, &p->rx);
	if (rc)
		return rc;

	rc = zctap_mmap_queue(fd, &skq->cq, &p->cq);
	if (rc)
		return rc;

	rc = zctap_mmap_queue(fd, &skq->meta, &p->meta);
	if (rc)
		return rc;

	return 0;
}

static int
zctap_mmap_ifq(struct zctap_ifq *ifq, struct zctap_ifq_param *p)
{
	return zctap_mmap_queue(ifq->fd, &ifq->fill, &p->fill);
}

static void
zctap_populate(struct shared_queue *prod, uint64_t addr, int count, int size)
{
	uint64_t *addrp;
	int i;

	/* ring entries will be power of 2. */
	if (sq_prod_space(prod) < count)
		err_exit("sq_prod_space");

	for (i = 0; i < count; i++) {
		addrp = sq_prod_reserve(prod);
		*addrp = (uint64_t)addr + i * size;
	}
	sq_prod_submit(prod);
}

void
zctap_populate_ring(struct zctap_ifq *ifq, uint64_t addr, int count)
{
	zctap_populate(&ifq->fill, addr, count, PAGE_SIZE);
}

void
zctap_populate_meta(struct zctap_skq *skq, uint64_t addr, int count, int size)
{
	zctap_populate(&skq->meta, addr, count, size);
}

int
zctap_get_rx_batch(struct zctap_skq *skq, struct zctap_iovec *iov[], int count)
{
	return sq_cons_batch(&skq->rx, (void **)iov, count);
}

int
zctap_get_cq_batch(struct zctap_skq *skq, uint64_t *notify[], int count)
{
	return sq_cons_batch(&skq->cq, (void **)notify, count);
}

void
zctap_recycle_buffer(struct zctap_ifq *ifq, void *ptr)
{
	uint64_t *addrp;

	addrp = sq_prod_reserve(&ifq->fill);
	*addrp = (uint64_t)ptr;
}

bool
zctap_recycle_batch(struct zctap_ifq *ifq, struct zctap_iovec **iov, int count)
{
	uint64_t *addrp;
	int i;

	if (!sq_prod_avail(&ifq->fill, count))
		return false;

	for (i = 0; i < count; i++) {
		addrp = sq_prod_get_ptr(&ifq->fill);
		*addrp = iov[i]->base;
	}
	return true;
}

void
zctap_recycle_complete(struct zctap_ifq *ifq)
{
	sq_prod_submit(&ifq->fill);
}

void
zctap_recycle_meta(struct zctap_skq *skq, void *ptr)
{
	uint64_t *addrp;

	addrp = sq_prod_reserve(&skq->meta);
	*addrp = (uint64_t)ptr;
}

void
zctap_submit_meta(struct zctap_skq *skq)
{
	sq_prod_submit(&skq->meta);
}

void
zctap_detach_socket(struct zctap_skq **skqp)
{
	struct zctap_skq *skq = *skqp;

	if (skq->rx.map_ptr)
		munmap(skq->rx.map_ptr, skq->rx.map_sz);

	if (skq->cq.map_ptr)
		munmap(skq->cq.map_ptr, skq->cq.map_sz);

	if (skq->meta.map_ptr)
		munmap(skq->meta.map_ptr, skq->meta.map_sz);

	free(skq);
	*skqp = NULL;
}

void
zctap_init_socket_param(struct zctap_socket_param *p, int nentries)
{
	memset(p, 0, sizeof*(p));

	p->rx.entries = nentries;
	p->cq.entries = nentries;
	p->meta.entries = nentries;

	p->meta_bufsz = 256;
	p->inline_max = 0;
}

int
zctap_attach_socket(struct zctap_skq **skqp, struct zctap_ctx *ctx, int fd,
		    struct zctap_socket_param *p)
{
	struct zctap_skq *skq;
	int val = 3;
	int err;

	skq = malloc(sizeof(*skq));
	if (!skq)
		return -ENOMEM;
	memset(skq, 0, sizeof(*skq));
	skq->socket_fd = fd;

	p->zctap_fd = ctx->fd;
	p->socket_fd = fd;
	p->rx.elt_sz = sizeof(struct zctap_iovec);
	p->cq.elt_sz = sizeof(uint64_t);
	p->meta.elt_sz = sizeof(uint64_t);

	if (setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &val, sizeof(val)))
		err_exit("setsockopt(SO_ZEROCOPY)");

	/* for TX - specify outgoing device */
	if (setsockopt(fd, SOL_SOCKET, SO_BINDTOIFINDEX, &ctx->ifindex,
		       sizeof(ctx->ifindex)))
		err_exit("setsockopt(SO_BINDTOIFINDEX)");

	err = zctap(ZCTAP_ATTACH_SOCKET, p, sizeof(*p));
	if (err < 0)
		err_exit("ATTACH_SOCKET");

	err = zctap_mmap_socket(fd, skq, p);
	if (err)
		err_with(-err, "zctap_mmap_socket");

	*skqp = skq;

	return 0;
}

void
zctap_close_ifq(struct zctap_ifq **ifqp)
{
	struct zctap_ifq *ifq = *ifqp;

	close(ifq->fd);
	if (ifq->fill.map_ptr)
		munmap(ifq->fill.map_ptr, ifq->fill.map_sz);

	free(ifq);
	*ifqp = NULL;
}

int
zctap_ifq_id(struct zctap_ifq *ifq)
{
	return ifq->queue_id;
}

void
zctap_init_ifq_param(struct zctap_ifq_param *p, bool is_tcp)
{
	memset(p, 0, sizeof(*p));

	p->queue_id = -1;
	p->fill.entries = 10240;
	p->fill_bufsz = PAGE_SIZE;
	p->split = ZCTAP_SPLIT_NONE;
	if (is_tcp)
		p->split_offset = (14 + 40 + (20 + 20));	// TCP6
	else
		p->split_offset = (14 + 40 + 8);		// UDP6
}

int
zctap_open_ifq(struct zctap_ifq **ifqp, struct zctap_ctx *ctx,
	       struct zctap_ifq_param *p)
{
	struct zctap_ifq *ifq;
	int fd, err;

	ifq = malloc(sizeof(*ifq));
	if (!ifq)
		return -ENOMEM;
	memset(ifq, 0, sizeof(*ifq));

	/* not user settable */
	p->fill.elt_sz = sizeof(uint64_t);
	p->zctap_fd = ctx->fd;

	fd = zctap(ZCTAP_BIND_QUEUE, p, sizeof(*p));
	if (fd < 0) {
		err = -errno;
		free(ifq);
		return err;
	}

	ifq->fd = fd;
	ifq->queue_id = p->queue_id;

	err = zctap_mmap_ifq(ifq, p);
	if (err) {
		close(ifq->fd);
		free(ifq);
		return err;
	}

	*ifqp = ifq;
	return 0;
}

int
zctap_create_host_region(void *ptr, size_t sz)
{
	struct zctap_host_param p;
	int ret;

	p.iov.iov_base = ptr;
	p.iov.iov_len = sz;

	ret = zctap(ZCTAP_CREATE_HOST_REGION, &p, sizeof(p));
	if (ret < 0)
		ret = -errno;

	return ret;
}

int
zctap_region_from_dmabuf(int provider_fd, void *ptr)
{
	struct zctap_dmabuf_param p;
	int ret;

	p.provider_fd = provider_fd;
	p.addr = (uintptr_t)ptr;

	ret = zctap(ZCTAP_REGION_FROM_DMABUF, &p, sizeof(p));
	if (ret == -1)
		ret = -errno;

	return ret;
}

void
zctap_close_ctx(struct zctap_ctx **ctxp)
{
	struct zctap_ctx *ctx = *ctxp;

	close(ctx->fd);
	free(ctx);
	*ctxp = NULL;
}

int
zctap_open_ctx(struct zctap_ctx **ctxp, const char *ifname)
{
	struct zctap_context_param p;
	struct zctap_ctx *ctx;
	int err;

	ctx = malloc(sizeof(*ctx));
	if (!ctx)
		return -ENOMEM;
	memset(ctx, 0, sizeof(*ctx));

	ctx->ifindex = if_nametoindex(ifname);
	if (!ctx->ifindex) {
		warn("Interface %s does not exist\n", ifname);
		err = -EEXIST;
		goto out;
	}

	p.ifindex = ctx->ifindex;

	ctx->fd = zctap(ZCTAP_CREATE_CONTEXT, &p, sizeof(p));
	if (ctx->fd == -1) {
		err = -errno;
		goto out;
	}
	*ctxp = ctx;
	return 0;

out:
	free(ctx);
	return err;
}

int
zctap_attach_region(struct zctap_ctx *ctx, int region_fd)
{
	struct zctap_attach_param p;
	int err;

	p.zctap_fd = ctx->fd;
	p.region_fd = region_fd;

	err = zctap(ZCTAP_ATTACH_REGION, &p, sizeof(p));
	if (err < 0)
		err = -errno;

	return err;
}

int
zctap_attach_meta_region(struct zctap_ctx *ctx, int region_fd)
{
	struct zctap_attach_param p;
	int err;

	p.zctap_fd = ctx->fd;
	p.region_fd = region_fd;

	err = zctap(ZCTAP_ATTACH_META_REGION, &p, sizeof(p));
	if (err < 0)
		err = -errno;

	return err;
}

int
zctap_add_meta(struct zctap_ctx *ctx, void *ptr, size_t sz)
{
	int fd, err;

	printf("creating host region\n");
	fd = zctap_create_host_region(ptr, sz);
	if (fd < 0)
		return fd;

	printf("attaching host region as meta\n");
	err = zctap_attach_meta_region(ctx, fd);

	printf("Done\n");

	close(fd);
	return err;
}

int
zctap_register_host_memory(struct zctap_ctx *ctx, void *va, size_t sz)
{
	int fd, err;

	fd = zctap_create_host_region(va, sz);
	if (fd < 0)
		return fd;

	err = zctap_attach_region(ctx, fd);

	close(fd);
	return err;
}

static void *
util_alloc_host_memory(size_t size)
{
	void *addr;

	/* XXX page align size... */

	addr = mmap(NULL, size, PROT_READ | PROT_WRITE,
			MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
	if (addr == MAP_FAILED)
		err_exit("mmap");

	if (mlock(addr, size))
		err_exit("mlock");

	return addr;
}

static void
util_free_host_memory(void *area, size_t size)
{
	munmap(area, size);
}

#ifdef USE_CUDA
#define CHK_CUDA(fcn) do {						\
	CUresult err = fcn;						\
	const char *str;						\
	if (err) {							\
		cuGetErrorString(err, &str);				\
		err_exit(str);						\
	}								\
} while (0)

static uint64_t
pin_buffer(void *ptr, size_t size)
{
	uint64_t id;
	unsigned int one = 1;

	/*
	 * Disables all data transfer optimizations
	 */
	CHK_CUDA(cuPointerSetAttribute(&one,
	    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)ptr));

	CHK_CUDA(cuPointerGetAttribute(&id,
	    CU_POINTER_ATTRIBUTE_BUFFER_ID, (CUdeviceptr)ptr));

	return id;
}

static void *
util_alloc_cuda_memory(size_t size)
{
	void *gpu;
	uint64_t id;

	CHK_CUDA(cudaMalloc(&gpu, size));

	id = pin_buffer(gpu, size);

	return gpu;
}

static void *
util_free_cuda_memory(void *area, size_t size)
{
	CHK_CUDA(cudaFree(area));
}

int
util_create_cuda_dmabuf(void *ptr, size_t sz)
{
	struct zctap_cuda_dmabuf_param p;
	int fd, err;

	fd = open("/dev/zctap_cuda", O_RDWR);
	if (fd < 0)
		return -errno;

	p.iov.iov_base = ptr;
	p.iov.iov_len = sz;
	err = ioctl(fd, ZCTAP_CUDA_IOCTL_CREATE_DMABUF, &p);

	close(fd);
	return err;
}
#endif

int
util_create_region(void *ptr, size_t sz, enum zctap_memtype memtype)
{
	int err = -EOPNOTSUPP;

	if (memtype == MEMTYPE_HOST)
		err = zctap_create_host_region(ptr, sz);
#ifdef USE_CUDA
	else if (memtype == MEMTYPE_CUDA) {
		int fd;

		fd = util_create_cuda_dmabuf(ptr, sz);
		if (fd < 0)
			return fd;
		err = zctap_region_from_dmabuf(fd, ptr);
		close(fd);
	}
#endif
	return err;
}

int
util_register_memory(struct zctap_ctx *ctx, void *ptr, size_t sz,
		     enum zctap_memtype memtype)
{
	int fd, err;

	fd = util_create_region(ptr, sz, memtype);
	if (fd < 0)
		return fd;

	err = zctap_attach_region(ctx, fd);

	close(fd);
	return err;
}

void *
util_alloc_memory(size_t size, enum zctap_memtype memtype)
{
	void *area = NULL;

	if (memtype == MEMTYPE_HOST)
		area = util_alloc_host_memory(size);
#ifdef USE_CUDA
	else if (memtype == MEMTYPE_CUDA)
		area = util_alloc_cuda_memory(size);
#endif
	return area;
}

void
util_free_memory(void *area, size_t size, enum zctap_memtype memtype)
{
	if (memtype == MEMTYPE_HOST)
		util_free_host_memory(area, size);
#ifdef USE_CUDA
	else if (memtype == MEMTYPE_CUDA)
		util_free_cuda_memory(area, size);
#endif
	else
		stop_here("Unhandled memtype: %d", memtype);
}
