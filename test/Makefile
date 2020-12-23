
include ../compile.mk

INC = -I..
LIBS = -L.. -lzctap $(EXTRA_LIBS)
CFLAGS = -g $(OPTS) $(INC) $(DEFINES)

targets := memarea context dmamap socket netqueue send recv

all: $(targets)

%: %.c
	$(QUIET_CC)$(CC) $(CFLAGS) -o $@ $< $(LIBS)

objs := $(patsubst %,%.o,$(targets))

clean:
	rm -f $(targets) $(objs) all

test: all
	./run.sh
