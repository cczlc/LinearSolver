#pragma once
#include "Typedef.h"
#include "MatFormat.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace CG
{
	template <uint32 threadsVecAdd, uint32 elemsVecAdd>
	__global__ void Vector_Add_Vector_Kernel(double* vector1, double* vector2, double* molecule, double* denominator, double* result, double tol, bool* exitFlag, uint32 arrayLength)
	{
		uint32 tid = blockDim.x * blockIdx.x + threadIdx.x;
		// д�����ﱣ֤�����߳̿鶼ִ�����
		// ���ܻ���㼸�Σ��������ⲻ��
		if (tid == 0 && (*molecule / arrayLength) <= tol)
		{
			*exitFlag = true;
		}

		uint32 offset, dataBlockLength;
		calcDataBlockLength<threadsVecAdd, elemsVecAdd>(offset, dataBlockLength, arrayLength);

		double para = *molecule / *denominator;

		for (uint32 tx = threadIdx.x; tx < dataBlockLength; tx += threadsVecAdd)
		{
			result[offset + tx] = vector1[offset + tx] + para * vector2[offset + tx];
		}
	}
}
