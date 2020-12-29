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
#endif

#define PAGE_SIZE	4096

struct zctap_skq {
	struct shared_queue rx;
	struct shared_queue cq;
	struct shared_queue meta;
	int ctx_fd;	/* XXX */
};

struct zctap_ifq {
	int fd;
	unsigned queue_id;
	struct shared_queue fill;
};

struct zctap_mem {
	int fd;
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
	printf("  producer: %u  %u\n", q->cached_prod, *q->prod);
	printf("  consumer: %u  %u\n", q->cached_cons, *q->cons);
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
zctap_get_rx_batch(struct zctap_skq *skq, struct iovec *iov[], int count)
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
	*addrp = (uint64_t)ptr & ~(PAGE_SIZE - 1);
}

bool
zctap_recycle_batch(struct zctap_ifq *ifq, struct iovec **iov, int count)
{
	uint64_t *addrp;
	int i;

	if (!sq_prod_avail(&ifq->fill, count))
		return false;

	for (i = 0; i < count; i++) {
		addrp = sq_prod_get_ptr(&ifq->fill);
		*addrp = (uint64_t)iov[i]->iov_base & ~(PAGE_SIZE - 1);
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
	*addrp = (uint64_t)ptr;		/* XXX should mask here */
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

int
zctap_attach_socket(struct zctap_skq **skqp, struct zctap_ctx *ctx, int fd,
		    int nentries)
{
	struct zctap_socket_param p;
	struct zctap_skq *skq;
	int one = 1;
	int err;

	skq = malloc(sizeof(*skq));
	if (!skq)
		return -ENOMEM;
	memset(skq, 0, sizeof(*skq));
	skq->ctx_fd = ctx->fd;

	memset(&p, 0, sizeof(p));
	p.fd = fd;

	p.rx.elt_sz = sizeof(struct iovec);
	p.rx.entries = nentries;

	p.cq.elt_sz = sizeof(uint64_t);
	p.cq.entries = nentries;

	if (setsockopt(fd, SOL_SOCKET, SO_ZEROCOPY, &one, sizeof(one)))
		err_exit("setsockopt(SO_ZEROCOPY)");

	/* for TX - specify outgoing device */
	if (setsockopt(fd, SOL_SOCKET, SO_BINDTOIFINDEX, &ctx->ifindex,
		       sizeof(ctx->ifindex)))
		err_exit("setsockopt(SO_BINDTOIFINDEX)");

	/* attaches sk to ctx and sets up custom data_ready hook */
	if (ioctl(ctx->fd, ZCTAP_CTX_IOCTL_ATTACH_SOCKET, &p))
		err_exit("ioctl(ATTACH_SOCKET)");

	err = zctap_mmap_socket(fd, skq, &p);
	if (err)
		err_with(-err, "zctap_mmap_socket");

	*skqp = skq;

	return 0;
}

int
zctap_add_meta(struct zctap_skq *skq, int fd, void *addr, size_t len,
	       int nentries, int meta_len)
{
	struct zctap_socket_param p;
	int rc;

	if (skq->meta.entries)
		return -EALREADY;

	memset(&p, 0, sizeof(p));
	p.fd = fd;
	p.resv = 1;
	p.iov.iov_base = addr;
	p.iov.iov_len = len;
	p.meta.elt_sz = sizeof(uint64_t);
	p.meta.entries = nentries;
	p.meta_len = meta_len;

	/* attaches sk to ctx and sets up custom data_ready hook */
	if (ioctl(skq->ctx_fd, ZCTAP_CTX_IOCTL_ATTACH_SOCKET, &p))
		err_exit("ioctl(ATTACH_META)");

	rc = zctap_mmap_queue(fd, &skq->meta, &p.meta);
	if (rc)
		return rc;

#define DLEN(_q) _q.data, _q.data + (_q.entries * _q.elt_sz)
	printf("skq.meta: [%p-%p]\n", DLEN(skq->meta));

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

int
zctap_open_ifq(struct zctap_ifq **ifqp, struct zctap_ctx *ctx,
	       int queue_id, int fill_entries)
{
	struct zctap_ifq_param p;
	struct zctap_ifq *ifq;
	int err;

	ifq = malloc(sizeof(*ifq));
	if (!ifq)
		return -ENOMEM;
	memset(ifq, 0, sizeof(*ifq));

	memset(&p, 0, sizeof(p));
	p.queue_id = queue_id;
	p.fill.elt_sz = sizeof(uint64_t);
	p.fill.entries = fill_entries;

	p.hdsplit = ZCTAP_SPLIT_OFFSET;
//	p.split_offset = (14 + 40 + (20 + 20));		// TCP6
	p.split_offset = (14 + 40 + 8);			// UDP6

	if (ioctl(ctx->fd, ZCTAP_CTX_IOCTL_BIND_QUEUE, &p)) {
		err = -errno;
		free(ifq);
		return err;
	}

	ifq->fd = p.ifq_fd;
	ifq->queue_id = p.queue_id;

	err = zctap_mmap_ifq(ifq, &p);
	if (err) {
		close(ifq->fd);
		free(ifq);
		return err;
	}

	*ifqp = ifq;
	return 0;
}

int
zctap_attach_region(struct zctap_ctx *ctx, struct zctap_mem *mem, int idx)
{
	struct zctap_attach_param p;

	p.mem_fd = mem->fd;
	p.mem_idx = idx;

	if (ioctl(ctx->fd, ZCTAP_CTX_IOCTL_ATTACH_REGION, &p))
		return -errno;

	return 0;
}

int
zctap_register_memory(struct zctap_ctx *ctx, void *va, size_t size,
		      enum zctap_memtype memtype)
{
	int idx, err;

	if (!ctx->mem) {
		err = zctap_open_memarea(&ctx->mem);
		if (err)
			return err;
	}
	idx = zctap_add_memarea(ctx->mem, va, size, memtype);
	if (idx < 0)
		return idx;

	return zctap_attach_region(ctx, ctx->mem, idx);
}

void
zctap_close_ctx(struct zctap_ctx **ctxp)
{
	struct zctap_ctx *ctx = *ctxp;

	if (ctx->mem)
		zctap_close_memarea(&ctx->mem);

	close(ctx->fd);
	free(ctx);
	*ctxp = NULL;
}

int
zctap_open_ctx(struct zctap_ctx **ctxp, const char *ifname)
{
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

	ctx->fd = open("/dev/zctap", O_RDWR);
	if (ctx->fd == -1)
		err_exit("open(/dev/zctap)");

	if (ioctl(ctx->fd, ZCTAP_CTX_IOCTL_ATTACH_DEV, &ctx->ifindex))
		err_exit("ioctl(ATTACH_DEV)");

	*ctxp = ctx;
	return 0;

out:
	free(ctx);
	return err;
}

int
zctap_add_memarea(struct zctap_mem *mem, void *va, size_t size,
		  enum zctap_memtype memtype)
{
	struct zctap_region_param p;
	int idx;

	p.iov.iov_base = va;
	p.iov.iov_len = size;
	p.memtype = memtype;

	idx = ioctl(mem->fd, ZCTAP_MEM_IOCTL_ADD_REGION, &p);
	if (idx < 0)
		idx = -errno;

	return idx;
}

void
zctap_close_memarea(struct zctap_mem **memp)
{
	struct zctap_mem *mem = *memp;

	close(mem->fd);
	free(mem);
	*memp = NULL;
}

/* XXX change so memory areas are always of one type? */
int
zctap_open_memarea(struct zctap_mem **memp)
{
	struct zctap_mem *mem;

	mem = malloc(sizeof(*mem));
	if (!mem)
		return -ENOMEM;
	memset(mem, 0, sizeof(*mem));

	mem->fd = open("/dev/zctap_mem", O_RDWR);
	if (mem->fd == -1)
		err_exit("open(/dev/zctap_mem)");

	*memp = mem;

	return 0;
}

static void *
zctap_alloc_host_memory(size_t size)
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
zctap_free_host_memory(void *area, size_t size)
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
zctap_alloc_cuda_memory(size_t size)
{
	void *gpu;
	uint64_t id;

	printf("allocating %ld from gpu...\n", size);
	CHK_CUDA(cudaMalloc(&gpu, size));

	id = pin_buffer(gpu, size);

	return gpu;
}

static void *
zctap_free_cuda_memory(void *area, size_t size)
{
	printf("freeing %ld from gpu...\n", size);
	CHK_CUDA(cudaFree(area));
}
#endif

void *
zctap_alloc_memory(size_t size, enum zctap_memtype memtype)
{
	void *area = NULL;

	if (memtype == MEMTYPE_HOST)
		area = zctap_alloc_host_memory(size);
#ifdef USE_CUDA
	else if (memtype == MEMTYPE_CUDA)
		area = zctap_alloc_cuda_memory(size);
#endif
	return area;
}

void
zctap_free_memory(void *area, size_t size, enum zctap_memtype memtype)
{
	if (memtype == MEMTYPE_HOST)
		zctap_free_host_memory(area, size);
#ifdef USE_CUDA
	else if (memtype == MEMTYPE_CUDA)
		zctap_free_cuda_memory(area, size);
#endif
	else
		stop_here("Unhandled memtype: %d", memtype);
}
