#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <sys/uio.h>

#include "zctap_util.h"

#include "bpf/libbpf_util.h"
#include "uapi/misc/zctap.h"
#include "uapi/misc/shqueue.h"

#include "zctap_lib.h"


#ifndef MSG_ZEROCOPY
#define MSG_ZEROCOPY	0x4000000
#endif
#define MSG_ZCTAP 	0x8000000


struct zctap_skq;
struct zctap_ifq;
struct zctap_ctx;
struct zctap_mem;

void zctap_populate_ring(struct zctap_ifq *ifq, uint64_t addr, int count);
int zctap_get_rx_batch(struct zctap_skq *skq, struct iovec *iov[], int count);
int zctap_get_cq_batch(struct zctap_skq *skq, uint64_t *notify[], int count);
void zctap_recycle_buffer(struct zctap_ifq *ifq, void *ptr);
bool zctap_recycle_batch(struct zctap_ifq *ifq, struct iovec **iov,
			 int count);
void zctap_recycle_complete(struct zctap_ifq *ifq);

void zctap_populate_meta(struct zctap_skq *skq, uint64_t addr, int count,
			 int size);
void zctap_recycle_meta(struct zctap_skq *skq, void *ptr);
void zctap_submit_meta(struct zctap_skq *skq);
int zctap_add_meta(struct zctap_skq *skq, int fd, void *addr, size_t len,
		   int nentries, int meta_len);

int zctap_attach_socket(struct zctap_skq **skqp, struct zctap_ctx *ctx,
			int fd, int nentries);
void zctap_detach_socket(struct zctap_skq **skqp);

int zctap_ifq_id(struct zctap_ifq *ifq);
int zctap_open_ifq(struct zctap_ifq **ifqp, struct zctap_ctx *ctx,
		   int queue_id, int fill_entries);
void zctap_close_ifq(struct zctap_ifq **ifqp);

int zctap_attach_region(struct zctap_ctx *ctx, struct zctap_mem *mem, int idx);
int zctap_open_ctx(struct zctap_ctx **ctxp, const char *ifname);
void zctap_close_ctx(struct zctap_ctx **ctxp);

int zctap_open_memarea(struct zctap_mem **memp);
void zctap_close_memarea(struct zctap_mem **memp);
int zctap_add_memarea(struct zctap_mem *mem, void *va, size_t size,
		      enum zctap_memtype memtype);

void *zctap_alloc_memory(size_t size, enum zctap_memtype memtype);
void zctap_free_memory(void *area, size_t size, enum zctap_memtype memtype);


/* convenience functions */
int zctap_register_memory(struct zctap_ctx *ctx, void *va, size_t size,
			  enum zctap_memtype memtype);
