#pragma once
#include "Typedef.h"
#include "MatFormat.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
				__longlong_as_double(assumed)));

		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}
#endif

inline __device__ double intraWarpScanMat(volatile double* scanTile, double val) {

	unsigned index = 2 * threadIdx.x - (threadIdx.x & (WARP_SIZE - 1));
	scanTile[index] = 0;              // 将前面一列置零
	index += WARP_SIZE;
	scanTile[index] = val;

	scanTile[index] += scanTile[index - 1];
	scanTile[index] += scanTile[index - 2];
	scanTile[index] += scanTile[index - 4];
	scanTile[index] += scanTile[index - 8];
	scanTile[index] += scanTile[index - 16];

	// 多个元素的值进行合并
	return scanTile[index] - val;
}

template <unsigned blockSize>
inline __device__ double intraWarpScan(volatile double* scanTile, double val) {

	unsigned index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));
	scanTile[index] = 0;              // 将前面一列置零
	index += min(blockSize, WARP_SIZE);
	scanTile[index] = val;

	if (blockSize >= 2)
	{
		scanTile[index] += scanTile[index - 1];
	}

	if (blockSize >= 4)
	{
		scanTile[index] += scanTile[index - 2];
	}
	if (blockSize >= 8)
	{
		scanTile[index] += scanTile[index - 4];
	}
	if (blockSize >= 16)
	{
		scanTile[index] += scanTile[index - 8];
	}
	if (blockSize >= 32)
	{
		scanTile[index] += scanTile[index - 16];
	}

	// 多个元素的值进行合并
	return scanTile[index] - val;
}

template <unsigned blockSize>
inline __device__ double intraBlockScan(double val) {
	__shared__ double scanTile[blockSize * 2];
	unsigned warpIdx = threadIdx.x / WARP_SIZE;
	unsigned laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp

	double warpResult = intraWarpScan<blockSize>(scanTile, val);
	__syncthreads();

	if (laneIdx == WARP_SIZE - 1)                 // 得到32个值的总和放在对应的warpIdx中
	{
		scanTile[warpIdx] = warpResult + val;
	}
	__syncthreads();

	if (threadIdx.x < WARP_SIZE)                  // 仅用其中一个warp进行操作
	{
		scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
	}
	__syncthreads();

	return warpResult + scanTile[warpIdx] + val;
}


template <unsigned numThreads, unsigned elemsThread>
inline __device__ void calcDataBlockLength(unsigned& offset, unsigned& dataBlockLength, unsigned arrayLength)
{
	unsigned elemsPerThreadBlock = numThreads * elemsThread;            // 计算每个线程块要处理的数据量
	offset = blockIdx.x * elemsPerThreadBlock;
	dataBlockLength = offset + elemsPerThreadBlock <= arrayLength ? elemsPerThreadBlock : arrayLength - offset;       // 对最后一个线程块的特殊处理
}


// 一个warp处理矩阵中的一行（每个线程块的线程数要被32整除）
// 这对于一行中有超过非零值的矩阵较好，否则会出现很多线程实际上并没有工作！
// 可以通过修改WARP_SIZE改善这一情况
template <uint32 threadsMatMutil>
__global__ void Matrix_multi_Vector_Kernel(CSR matrix, double* vector, double* result)
{
	// 创建共享缓存
	__shared__ double scanTile[threadsMatMutil * 2];                                      // 大小为两倍的线程数
	double localValue = 0.0;
	double WarpValue = 0.0;

	uint32 tid = blockDim.x * blockIdx.x + threadIdx.x;                                   // 线程全局id
	uint32 warpId = tid / WARP_SIZE;                                                      // 计算出要处理的行号
	uint32 laneId = tid % WARP_SIZE;                                                      // 计算出要处理的元素
	uint32 offset = blockDim.x / WARP_SIZE * gridDim.x;                                   // 计算出总的warp数

	for (uint32 row = warpId; row < matrix.m_Dimension; row += offset)
	{
		localValue = 0.0;                                                                 // 注意新的一行要刷新寄存器的值

		uint32 tempOffset = matrix.m_Fnz[row];                                            // 确定每一行的起始位置
		uint32 numPerRow = matrix.m_Fnz[row + 1] - tempOffset;                            // 计算出该行的元素个数

		for (uint32 col = laneId; col < numPerRow; col += WARP_SIZE)
		{
			localValue += matrix.m_Data[tempOffset + col] * vector[matrix.m_Col[tempOffset + col]];          // 这样写好像不能跨行
		}

		WarpValue = intraWarpScanMat(scanTile, localValue) + localValue;
		__syncthreads();

		if (laneId == WARP_SIZE - 1)                                                      // 得到一行乘积值的总和放在对应的位置中
		{
			result[row] = WarpValue;
		}
	}
}


// 每个线程处理一个或多个元素（可以保证所有的线程都处于活跃状态）
// 在使用之间要将result初始化为0
template <uint32 threadsMatMutil, uint32 elemsMatMutil>
__global__ void Matrix_multi_Vector_Kernel(COO matrix, double* vector, double* result)
{
	uint32 offset, dataBlockLength;
	calcDataBlockLength<threadsMatMutil, elemsMatMutil>(offset, dataBlockLength, matrix.m_ArrayLength);

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsMatMutil)
	{
		result[matrix.m_Row[offset + tx]] += matrix.m_Data[offset + tx] * vector[matrix.m_Col[offset + tx]];
	}
}


template <uint32 threadsVecMutil, uint32 elemsVecMutil>
__global__ void Vector_mutil_Vector_Kernel(double* vector1, double* vector2, double* result, uint32 arrayLength)
{
	uint32 offset, dataBlockLength;
	calcDataBlockLength<threadsVecMutil, elemsVecMutil>(offset, dataBlockLength, arrayLength);

	double localValue = 0.0;
	double scanValue = 0.0;

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecMutil)
	{
		localValue += vector1[offset + tx] * vector2[offset + tx];
	}

	// 得到这个线程块内，每个线程之前（包括该线程）的所有线程计算值之和
	scanValue = intraBlockScan<threadsVecMutil>(localValue);
	__syncthreads();

	if (threadIdx.x == (threadsVecMutil - 1))
	{
		atomicAdd(result, scanValue);
	}
}


template <uint32 threadsVecAdd, uint32 elemsVecAdd>
__global__ void Vector_Add_Vector_Kernel(double* vector1, double* vector2, double* result, uint32 arrayLength)
{
	uint32 offset, dataBlockLength;
	calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
	{
		result[offset + tx] = vector1[offset + tx] + vector2[offset + tx];
	}
}

template <uint32 threadsVecAdd, uint32 elemsVecAdd>
__global__ void Vector_Add_Vector_Kernel(double* vector1, double* vector2, double* molecule, double* denominator, double* result, uint32 arrayLength)
{
	uint32 offset, dataBlockLength;
	calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

	double para = *molecule / *denominator;

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
	{
		result[offset + tx] = vector1[offset + tx] + para * vector2[offset + tx];
	}
}

template <uint32 threadsVecAdd, uint32 elemsVecAdd>
__global__ void Vector_Sub_Vector_Kernel(double* vector1, double* vector2, double* result, uint32 arrayLength)
{
	uint32 offset, dataBlockLength;
	calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
	{
		result[offset + tx] = vector1[offset + tx] - vector2[offset + tx];
	}
}

template <uint32 threadsVecAdd, uint32 elemsVecAdd>
__global__ void Vector_Sub_Vector_Kernel(double* vector1, double* vector2, double* molecule, double* denominator, double* result, uint32 arrayLength)
{
	uint32 offset, dataBlockLength;
	calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

	double para = *molecule / *denominator;

	for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
	{
		result[offset + tx] = vector1[offset + tx] - para * vector2[offset + tx];
	}
}