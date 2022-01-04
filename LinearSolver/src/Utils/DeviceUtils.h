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
	scanTile[index] = 0;              // ��ǰ��һ������
	index += WARP_SIZE;
	scanTile[index] = val;

	scanTile[index] += scanTile[index - 1];
	scanTile[index] += scanTile[index - 2];
	scanTile[index] += scanTile[index - 4];
	scanTile[index] += scanTile[index - 8];
	scanTile[index] += scanTile[index - 16];

	// ���Ԫ�ص�ֵ���кϲ�
	return scanTile[index] - val;
}

template <unsigned blockSize>
inline __device__ double intraWarpScan(volatile double* scanTile, double val) {

	unsigned index = 2 * threadIdx.x - (threadIdx.x & (min(blockSize, WARP_SIZE) - 1));
	scanTile[index] = 0;              // ��ǰ��һ������
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

	// ���Ԫ�ص�ֵ���кϲ�
	return scanTile[index] - val;
}

template <unsigned blockSize>
inline __device__ double intraBlockScan(double val) {
	__shared__ double scanTile[blockSize * 2];
	unsigned warpIdx = threadIdx.x / WARP_SIZE;
	unsigned laneIdx = threadIdx.x & (WARP_SIZE - 1);  // Thread index inside warp

	double warpResult = intraWarpScan<blockSize>(scanTile, val);
	__syncthreads();

	if (laneIdx == WARP_SIZE - 1)                 // �õ�32��ֵ���ܺͷ��ڶ�Ӧ��warpIdx��
	{
		scanTile[warpIdx] = warpResult + val;
	}
	__syncthreads();

	if (threadIdx.x < WARP_SIZE)                  // ��������һ��warp���в���
	{
		scanTile[threadIdx.x] = intraWarpScan<blockSize / WARP_SIZE>(scanTile, scanTile[threadIdx.x]);
	}
	__syncthreads();

	return warpResult + scanTile[warpIdx] + val;
}


template <unsigned numThreads, unsigned elemsThread>
inline __device__ void calcDataBlockLength(unsigned& offset, unsigned& dataBlockLength, unsigned arrayLength)
{
	unsigned elemsPerThreadBlock = numThreads * elemsThread;            // ����ÿ���߳̿�Ҫ�����������
	offset = blockIdx.x * elemsPerThreadBlock;
	dataBlockLength = offset + elemsPerThreadBlock <= arrayLength ? elemsPerThreadBlock : arrayLength - offset;       // �����һ���߳̿�����⴦��
}


// һ��warp��������е�һ�У�ÿ���߳̿���߳���Ҫ��32������
// �����һ�����г�������ֵ�ľ���Ϻã��������ֺܶ��߳�ʵ���ϲ�û�й�����
// ����ͨ���޸�WARP_SIZE������һ���
template <uint32 threadsMatMutil>
__global__ void Matrix_multi_Vector_Kernel(CSR matrix, double* vector, double* result)
{
	// ����������
	__shared__ double scanTile[threadsMatMutil * 2];                                      // ��СΪ�������߳���
	double localValue = 0.0;
	double WarpValue = 0.0;

	uint32 tid = blockDim.x * blockIdx.x + threadIdx.x;                                   // �߳�ȫ��id
	uint32 warpId = tid / WARP_SIZE;                                                      // �����Ҫ������к�
	uint32 laneId = tid % WARP_SIZE;                                                      // �����Ҫ�����Ԫ��
	uint32 offset = blockDim.x / WARP_SIZE * gridDim.x;                                   // ������ܵ�warp��

	for (uint32 row = warpId; row < matrix.m_Dimension; row += offset)
	{
		localValue = 0.0;                                                                 // ע���µ�һ��Ҫˢ�¼Ĵ�����ֵ

		uint32 tempOffset = matrix.m_Fnz[row];                                            // ȷ��ÿһ�е���ʼλ��
		uint32 numPerRow = matrix.m_Fnz[row + 1] - tempOffset;                            // ��������е�Ԫ�ظ���

		for (uint32 col = laneId; col < numPerRow; col += WARP_SIZE)
		{
			localValue += matrix.m_Data[tempOffset + col] * vector[matrix.m_Col[tempOffset + col]];          // ����д�����ܿ���
		}

		WarpValue = intraWarpScanMat(scanTile, localValue) + localValue;
		__syncthreads();

		if (laneId == WARP_SIZE - 1)                                                      // �õ�һ�г˻�ֵ���ܺͷ��ڶ�Ӧ��λ����
		{
			result[row] = WarpValue;
		}
	}
}


// ÿ���̴߳���һ������Ԫ�أ����Ա�֤���е��̶߳����ڻ�Ծ״̬��
// ��ʹ��֮��Ҫ��result��ʼ��Ϊ0
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

	// �õ�����߳̿��ڣ�ÿ���߳�֮ǰ���������̣߳��������̼߳���ֵ֮��
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