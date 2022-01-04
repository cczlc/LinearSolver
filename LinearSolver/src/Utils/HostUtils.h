#pragma once
#include <math.h>
#include <string>
#include <iostream>
#include "Typedef.h"
#include "Constants.h"
#include "DeviceUtils.h"

void Matrix_multi_Vector(CSR& matrix, double* vector, double* result)
{
	for (uint32 i = 0; i < matrix.m_Dimension; ++i)
	{
		result[i] = 0.0;
		for (uint32 j = matrix.m_Fnz[i]; j < matrix.m_Fnz[i + 1]; ++j)
		{
			result[i] += matrix.m_Data[j] * vector[matrix.m_Col[j]];
		}
	}
}

double Vector_mutil_Vector(double* vector1, double* vector2, uint32 arrayLength) {
	double temp = 0.0;
	for (uint32 i = 0; i < arrayLength; ++i)
		temp += (vector1[i] * vector2[i]);
	return temp;
}

void Vector_Add_Vector(double* vector1, double* vector2, double* result, uint32 arrayLength)
{
	for (uint32 i = 0; i < arrayLength; ++i)
	{
		result[i] = vector1[i] + vector2[i];
	}
}

void Vector_Sub_Vector(double* vector1, double* vector2, double* result, uint32 arrayLength)
{
	for (uint32 i = 0; i < arrayLength; ++i)
	{
		result[i] = vector1[i] - vector2[i];
	}
}

double maxnorm(uint32 arrayLength, double* r)
{
	double temp = 0.0;
	for (uint32 i = 1; i < arrayLength; ++i)
		temp += r[i] * r[i];

	return temp / arrayLength;
}

void checkCudaError(cudaError_t error, const std::string& errmsg)
{
	if (error != cudaSuccess)
	{
		std::cout << "error position: " << errmsg << std::endl;
		printf("Error in CUDA function.\nError: %s\n", cudaGetErrorString(error));
		getchar();
		exit(EXIT_FAILURE);
	}
}


//void runMat_Mutil_Vec_Kernel(CSR& matrix, double* vector, double* result)
//{
//	uint32 threadsMatMutil = THREADS_MATMUTIL;
//	uint32 elemsMatMutil = ELEMS_MATMUTIL;
//
//	uint32 warpNumPerBlock = threadsMatMutil / WARP_SIZE;      // 计算出每个线程块的warp数量
//	uint32 linePerBlock = warpNumPerBlock * elemsMatMutil;     // 计算出每个线程块处理的行数
//
//	dim3 dimGrid((matrix.m_Dimension - 1) / linePerBlock + 1, 1, 1);
//	dim3 dimBlock(threadsMatMutil, 1, 1);
//
//	Matrix_multi_Vector_Kernel<THREADS_MATMUTIL> << < dimGrid, dimBlock >> > (matrix, vector, result);
//	cudaDeviceSynchronize();
//	cudaError_t error;
//	error = cudaGetLastError();
//	checkCudaError(error, "Matrix_multi_Vector_Kernel");
//
//	//double* resultHost = new double[matrix.m_Dimension];
//	//error = cudaMemcpy((void*)resultHost, result, matrix.m_Dimension * sizeof(double), cudaMemcpyDeviceToHost);
//	//checkCudaError(error, "resultHost memcpy");
//
//	//for (uint32 i = 0; i < matrix.m_Dimension; ++i)
//	//{
//	//	std::cout << resultHost[i] << " ";
//	//	parallel[i] = resultHost[i];
//	//}
//	//std::cout << std::endl;
//	//delete[] resultHost;
//}
//
//void runVec_Mutil_Vec_Kernel(double* vector1, double* vector2, double* result, uint32 arrayLength)
//{
//	cudaError_t error;
//	error = cudaMemset(result, 0.0, sizeof(double));
//	checkCudaError(error, "result memset");
//
//	uint32 threadVecMutil = THREADS_VECMUTIL;                 // 每个线程块的线程数
//	uint32 elemsVecMutil = ELEMS_VECMUTIL;                    // 每个线程处理的元素数量
//	uint32 elemsPerBlock = threadVecMutil * elemsVecMutil;    // 每个线程块处理的元素数量
//
//	dim3 dimGrid((arrayLength - 1) / elemsPerBlock + 1, 1, 1);
//	dim3 dimBlock(threadVecMutil, 1, 1);
//
//	Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <dimGrid, dimBlock >> > (vector1, vector2, result, arrayLength);
//	cudaDeviceSynchronize();
//	error = cudaGetLastError();
//	checkCudaError(error, "Vector_mutil_Vector_Kernel");
//
//	//double* resultHost = new double;
//	//cudaMemcpy((void*)resultHost, result, sizeof(double), cudaMemcpyDeviceToHost);
//	//std::cout << *resultHost << std::endl;
//
//	//delete resultHost;
//	
//}
//
//
//void runVec_Mutil_Vec_Judge_Kernel(double* vector1, double* vector2, double* result, double tol, bool* exitflag, uint32 arrayLength)
//{
//	cudaError_t error;
//	error = cudaMemset(result, 0.0, sizeof(double));
//	checkCudaError(error, "result memset");
//
//	uint32 threadVecMutil = THREADS_VECMUTIL;                 // 每个线程块的线程数
//	uint32 elemsVecMutil = ELEMS_VECMUTIL;                    // 每个线程处理的元素数量
//	uint32 elemsPerBlock = threadVecMutil * elemsVecMutil;    // 每个线程块处理的元素数量
//
//	dim3 dimGrid((arrayLength - 1) / elemsPerBlock + 1, 1, 1);
//	dim3 dimBlock(threadVecMutil, 1, 1);
//
//	Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <dimGrid, dimBlock >> > (vector1, vector2, result, arrayLength);
//	cudaDeviceSynchronize();
//	error = cudaGetLastError();
//	checkCudaError(error, "Vector_mutil_Vector_Kernel");
//
//	//double* resultHost = new double;
//	//cudaMemcpy((void*)resultHost, result, sizeof(double), cudaMemcpyDeviceToHost);
//	//std::cout << *resultHost << std::endl;
//
//	//delete resultHost;
//}
//
//
//void runVec_Add_Vec_Kernel(double* vector1, double* vector2, double* result, uint32 arrayLength, bool flag)
//{
//	uint32 threadVecAdd = THREADS_VECADD;
//	uint32 elemsVecAdd = ELEMS_VECADD;
//	uint32 elemsPerBlock = threadVecAdd * elemsVecAdd;
//
//	dim3 dimGrid((arrayLength - 1) / elemsPerBlock + 1, 1, 1);
//	dim3 dimBlock(threadVecAdd, 1, 1);
//
//	cudaError_t error;
//	if (flag) {
//		Vector_Add_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <dimGrid, dimBlock >> > (vector1, vector2, result, arrayLength);
//		cudaDeviceSynchronize();
//		error = cudaGetLastError();
//		checkCudaError(error, "Vector_Add_Vector_Kernel");
//	}
//	else  {
//		Vector_Sub_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <dimGrid, dimBlock >> > (vector1, vector2, result, arrayLength);
//		cudaDeviceSynchronize();
//		error = cudaGetLastError();
//		checkCudaError(error, "Vector_Sub_Vector_Kernel");
//	}
//
//	//double* resultHost = new double[arrayLength];
//	//error = cudaMemcpy((void*)resultHost, result, arrayLength * sizeof(double), cudaMemcpyDeviceToHost);
//	//checkCudaError(error, "resultHost memcpy");
//
//	//for (uint32 i = 0; i < arrayLength; ++i)
//	//{
//	//	std::cout << resultHost[i] << " ";
//	//}
//	//std::cout << std::endl;
//	//delete[] resultHost;
//	
//}
//
//
//void runVec_Add_Vec_Kernel(double* vector1, double* vector2, double* molecule, double* denominator, double* result, uint32 arrayLength, bool flag)
//{
//	uint32 threadVecAdd = THREADS_VECADD;
//	uint32 elemsVecAdd = ELEMS_VECADD;
//	uint32 elemsPerBlock = threadVecAdd * elemsVecAdd;
//
//	dim3 dimGrid((arrayLength - 1) / elemsPerBlock + 1, 1, 1);
//	dim3 dimBlock(threadVecAdd, 1, 1);
//
//	cudaError_t error;
//	if (flag) {
//		Vector_Add_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <dimGrid, dimBlock >> > (vector1, vector2, molecule, denominator, result, arrayLength);
//		cudaDeviceSynchronize();
//		error = cudaGetLastError();
//		checkCudaError(error, "Vector_Add_Vector_Kernel");
//	}
//	else {
//		Vector_Sub_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <dimGrid, dimBlock >> > (vector1, vector2, molecule, denominator, result, arrayLength);
//		cudaDeviceSynchronize();
//		error = cudaGetLastError();
//		checkCudaError(error, "Vector_Sub_Vector_Kernel");
//	}
//
//	//double* resultHost = new double[arrayLength];
//	//error = cudaMemcpy((void*)resultHost, result, arrayLength * sizeof(double), cudaMemcpyDeviceToHost);
//	//checkCudaError(error, "resultHost memcpy");
//
//	//for (uint32 i = 0; i < arrayLength; ++i)
//	//{
//	//	std::cout << resultHost[i] << " ";
//	//}
//	//std::cout << std::endl;
//	//delete[] resultHost;
//
//}

