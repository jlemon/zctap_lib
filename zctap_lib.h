#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <sys/uio.h>
#include <linux/types.h>

#include "util_macros.h"

#include "bpf/libbpf_util.h"
#include "uapi/misc/zctap.h"
#include "uapi/misc/shqueue.h"
#include "transition.h"

#ifndef MSG_ZEROCOPY
#define MSG_ZEROCOPY	0x4000000
#endif
#define MSG_ZCTAP 	0x8000000

struct zctap_skq;
struct zctap_ifq;
struct zctap_ctx;

enum zctap_memtype {
	MEMTYPE_HOST,
	MEMTYPE_CUDA,
};

void zctap_populate_ring(struct zctap_ifq *ifq, uint64_t addr, int count);
int zctap_get_rx_batch(struct zctap_skq *skq, struct zctap_iovec *iov[],
		       int count);
int zctap_get_cq_batch(struct zctap_skq *skq, uint64_t *notify[], int count);
void zctap_recycle_buffer(struct zctap_ifq *ifq, void *ptr);
bool zctap_recycle_batch(struct zctap_ifq *ifq, struct zctap_iovec **iov,
			 int count);
void zctap_recycle_complete(struct zctap_ifq *ifq);

void zctap_populate_meta(struct zctap_skq *skq, uint64_t addr, int count,
			 int size);
void zctap_recycle_meta(struct zctap_skq *skq, void *ptr);
void zctap_submit_meta(struct zctap_skq *skq);
int zctap_add_meta(struct zctap_ctx *ctx, void *addr, size_t len);

void zctap_init_socket_param(struct zctap_socket_param *p, int nentries);
int zctap_attach_socket(struct zctap_skq **skqp, struct zctap_ctx *ctx,
			int fd, struct zctap_socket_param *p);
void zctap_detach_socket(struct zctap_skq **skqp);

int zctap_ifq_id(struct zctap_ifq *ifq);
void zctap_init_ifq_param(struct zctap_ifq_param *p, bool is_tcp);
int zctap_open_ifq(struct zctap_ifq **ifqp, struct zctap_ctx *ctx,
		   struct zctap_ifq_param *p);
void zctap_close_ifq(struct zctap_ifq **ifqp);

/* region */
int zctap_create_host_region(void *ptr, size_t sz);
int zctap_region_from_dmabuf(int provider_fd, void *ptr);

/* ctx */
int zctap_open_ctx(struct zctap_ctx **ctxp, const char *ifname);
void zctap_close_ctx(struct zctap_ctx **ctxp);
int zctap_attach_region(struct zctap_ctx *ctx, int region_fd);

/* convenience functions */
int zctap_register_host_memory(struct zctap_ctx *ctx, void *va, size_t size);

/* util functions */
void *util_alloc_memory(size_t size, enum zctap_memtype memtype);
void util_free_memory(void *area, size_t size, enum zctap_memtype memtype);
int util_create_region(void *ptr, size_t sz, enum zctap_memtype memtype);
int util_register_memory(struct zctap_ctx *ctx, void *ptr, size_t sz,
                         enum zctap_memtype memtype);

/* debug */
void zctap_debug_skq(struct zctap_skq *skq);
