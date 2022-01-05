#pragma once

#define THREADS_MATMUTIL 512      // 每个线程块的线程数（必须是32的倍数）
#define ELEMS_MATMUTIL 4          // 每个warp处理的行数

#define THREADS_VECMUTIL 512      // 每个线程块的线程数
#define ELEMS_VECMUTIL 4          // 每个线程处理的元素个数

#define THREADS_VECADD 512
#define ELEMS_VECADD 8

#define TIME_TEST 1