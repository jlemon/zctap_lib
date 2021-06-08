#pragma once

#include <unistd.h>
#include <sys/syscall.h>

#define __NR_zctap 447
#define zctap(cmd, arg, size)	syscall(__NR_zctap, cmd, arg, size)
