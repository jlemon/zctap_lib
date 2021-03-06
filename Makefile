
include compile.mk

#TOOLS_DIR = /root/linux/lib
# UAPI_DIR = uapi
#LINUX_DIR = /root/linux/include
#KERNEL_INC = -I$(TOOLS_DIR) -I$(UAPI_DIR) -I$(LINUX_DIR)
#INC = -I$(IOU_DIR)/include $(KERNEL_INC)
#IOU_LIBS = $(IOU_DIR)/liburing.a 

INC = -I.
LIBS = libbpf.a $(EXTRA_LIBS) -lz -lelf
LIBS += -L. -lzctap
CFLAGS = -g $(OPTS) $(INC) $(DEFINES)

lib_SRCS = \
	zctap_lib.c \

TARGETS = libzctap.a

lib_OBJ = $(patsubst %.c,%.o,$(lib_SRCS))
OBJS = $(lib_OBJ)

.PHONY: all
all: $(TARGETS)

libzctap.a: $(lib_OBJ)
	$(AR) r $@ $^

clean:
	rm -f $(TARGETS) $(OBJS)
