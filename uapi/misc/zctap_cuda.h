#pragma once

struct zctap_cuda_dmabuf_param {
	int fd;
	struct iovec iov;
};

#define ZCTAP_CUDA_IOCTL_CREATE_DMABUF \
	_IOR( 0, 1, struct zctap_cuda_dmabuf_param)
