#pragma once
#include "Typedef.h"
#include "MatFormat.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace BICGSTAB 
{
	template <uint32 threadsVecAdd, uint32 elemsVecAdd>
	__global__ void Vector_Add_Vector_Kernel(double* vector1, double* vector2, double* rho1, double* rho0, double* alpha, double* w, double* result, double* rr, bool* exitFlag, double tol, uint32 arrayLength)
	{
		uint32 tid = blockDim.x * blockIdx.x + threadIdx.x;

		if (tid == 0 && (*rr / arrayLength) <= tol)
		{
			*exitFlag = true;
		}

		double beta = (*rho1 / *rho0) * (*alpha / *w);
		double wTemp = *w;

		uint32 offset, dataBlockLength, index;
		calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

		for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
		{
			index = offset + tx;
			result[index] = vector1[index] + beta * (result[index] - wTemp * vector2[index]);
		}
	}

	template <uint32 threadsVecAdd, uint32 elemsVecAdd>
	__global__ void Vector_Add_Vector_Kernel(double* vector1, double* vector2, double* vector3, double* alpha, double* w, uint32 arrayLength)
	{
		uint32 offset, dataBlockLength, index;
		calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

		double alphaTemp = *alpha;
		double wTemp = *w;

		for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
		{
			index = offset + tx;
			vector1[index] = vector1[index] + alphaTemp * vector2[index] + wTemp * vector3[index];
		}
	}

	template <uint32 threadsVecAdd, uint32 elemsVecAdd>
	__global__ void Vector_Add_Vector_Kernel(double* vector1, double* vector2, double* vector3, double* alpha, double* ts, double* tt, double* w, uint32 arrayLength)
	{
		uint32 offset, dataBlockLength, index;
		calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

		double para = *ts / *tt;
		double alphaTemp = *alpha;

		if (threadIdx.x == 0)
		{
			*w = para;
		}

		for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
		{
			index = offset + tx;
			vector1[index] = vector1[index] + alphaTemp * vector2[index] + para * vector3[index];
		}
	}


	template <uint32 threadsVecAdd, uint32 elemsVecAdd>
	__global__ void Vector_Sub_Vector_Kernel(double* vector1, double* vector2, double* molecule, double* denominator, double* alpha, double* result, uint32 arrayLength)
	{
		uint32 offset, dataBlockLength;
		calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

		double para = *molecule / *denominator;

		// 这里还要再改
		if (threadIdx.x == 0)
		{
			*alpha = para;
		}

		for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
		{
			result[offset + tx] = vector1[offset + tx] - para * vector2[offset + tx];
		}
	}

	template <uint32 threadsVecAdd, uint32 elemsVecAdd>
	__global__ void Vector_Sub_Vector_Kernel(double* vector1, double* vector2, double* para, double* result, uint32 arrayLength)
	{
		uint32 offset, dataBlockLength;
		calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

		double paraTemp = *para;

		for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
		{
			result[offset + tx] = vector1[offset + tx] - paraTemp * vector2[offset + tx];
		}
	}

	template <uint32 threadsVecMutil, uint32 elemsVecMutil>
	__global__ void Vector_mutil_Vector_Kernel(double* vector1, double* vector2, double* result1, double* result2, uint32 arrayLength)
	{
		uint32 offset, dataBlockLength;
		calcDataBlockLength<threadsVecMutil, elemsVecMutil>(offset, dataBlockLength, arrayLength);

		double localValue1 = 0.0;
		double scanValue1 = 0.0;

		double localValue2 = 0.0;
		double scanValue2 = 0.0;

		for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecMutil)
		{
			localValue1 += vector1[offset + tx] * vector2[offset + tx];
			localValue2 += vector1[offset + tx] * vector1[offset + tx];
		}

		// 得到这个线程块内，每个线程之前（包括该线程）的所有线程计算值之和
		scanValue1 = intraBlockScan<threadsVecMutil>(localValue1);
		scanValue2 = intraBlockScan<threadsVecMutil>(localValue2);
		__syncthreads();

		if (threadIdx.x == (threadsVecMutil - 1))
		{
			atomicAdd(result1, scanValue1);
			atomicAdd(result2, scanValue2);
		}
	}
}

