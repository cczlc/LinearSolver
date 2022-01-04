#pragma once
#include "LinearSolver.h"
#include "BiCGSTAB_DeviceUtils.h"

class P_BiCGSTAB : public P_LinearSolver
{
public:
	P_BiCGSTAB(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);
	P_BiCGSTAB(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);

protected:
	void calculate() override;
};

P_BiCGSTAB::P_BiCGSTAB(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: P_LinearSolver(data, x, b, dimension, maxIter, residual) {}


P_BiCGSTAB::P_BiCGSTAB(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: P_LinearSolver(data, x, b, dimension, maxIter, residual) {}

void P_BiCGSTAB::calculate()
{
	double* deviceAx = nullptr;
	double* deviceR = nullptr;
	double* deviceR0 = nullptr;
	double* deviceV = nullptr;
	double* deviceRho1 = nullptr;
	double* deviceRho0 = nullptr;
	double* deviceAlpha = nullptr;
	double* deviceW = nullptr;
	double* deviceP = nullptr;
	double* deviceR0v = nullptr;
	double* deviceS = nullptr;
	double* deviceT = nullptr;
	double* deviceTs = nullptr;
	double* deviceTt = nullptr;
	double* deviceRr = nullptr;
	bool* exitFlag = nullptr;
	bool* deviceExitFlag = nullptr;

	cudaError_t error;
	error = cudaMalloc((void**)&deviceAx, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceAx malloc");
	error = cudaMalloc((void**)&deviceR, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceR malloc");
	error = cudaMalloc((void**)&deviceR0, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceR0 malloc");
	error = cudaMalloc((void**)&deviceV, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceV malloc");
	error = cudaMalloc((void**)&deviceRho1, sizeof(double));
	checkCudaError(error, "deviceRho1 malloc");
	error = cudaMalloc((void**)&deviceRho0, sizeof(double));
	checkCudaError(error, "deviceRho0 malloc");
	error = cudaMalloc((void**)&deviceAlpha, sizeof(double));
	checkCudaError(error, "deviceAlpha malloc");
	error = cudaMalloc((void**)&deviceW, sizeof(double));
	checkCudaError(error, "deviceW malloc");
	error = cudaMalloc((void**)&deviceP, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceP malloc");
	error = cudaMalloc((void**)&deviceR0v, sizeof(double));
	checkCudaError(error, "deviceR0v malloc");
	error = cudaMalloc((void**)&deviceS, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceS malloc");
	error = cudaMalloc((void**)&deviceT, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceT malloc");
	error = cudaMalloc((void**)&deviceTs, sizeof(double));
	checkCudaError(error, "deviceTs malloc");
	error = cudaMalloc((void**)&deviceTt, sizeof(double));
	checkCudaError(error, "deviceTt malloc");
	error = cudaMalloc((void**)&deviceRr, sizeof(double));
	checkCudaError(error, "deviceRr malloc");

	// ∆Ù”√¡„∏¥÷∆
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc((void**)&exitFlag, sizeof(bool), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	checkCudaError(error, "cudaHostAlloc data");
	cudaHostGetDevicePointer(&deviceExitFlag, exitFlag, 0);

	*exitFlag = false;

	error = cudaMemset(deviceP, 0.0, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceP memset");
	error = cudaMemset(deviceV, 0.0, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceV memset");
	error = cudaMemset(deviceRho0, 1.0, sizeof(double));
	checkCudaError(error, "deviceRho0 memset");
	error = cudaMemset(deviceAlpha, 1.0, sizeof(double));
	checkCudaError(error, "deviceAlpha memset");
	error = cudaMemset(deviceW, 1.0, sizeof(double));
	checkCudaError(error, "deviceW memset");

	// Ax
	Matrix_multi_Vector_Kernel<THREADS_MATMUTIL> << <m_DimGridMatMutil, m_DimBlockMatMutil >> > (m_DeviceDataCSR, m_DeviceX, deviceAx);

	// r - Ax
	Vector_Sub_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << < m_DimGridVecAdd, m_DimBlockVecAdd >> > (m_DeviveB, deviceAx, deviceR, m_Dimension);

	// r0_hat = r
	error = cudaMemcpy((void*)deviceR0, deviceR, m_Dimension * sizeof(double), cudaMemcpyDeviceToDevice);
	checkCudaError(error, "deviceR0 memcpy");

	error = cudaMemset(deviceRho1, 0.0, sizeof(double));
	checkCudaError(error, "deviceRho1 memset");

	// rho1 = rr
	Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceR, deviceR, deviceRho1, m_Dimension);

	//double* rho1 = new double;
	//error = cudaMemcpy((void*)rho1, deviceRho1, sizeof(double), cudaMemcpyDeviceToHost);
	//std::cout << *rho1 << std::endl;
	//delete rho1;

	//double* resultHost = new double[m_Dimension];
	//error = cudaMemcpy((void*)resultHost, deviceR, m_Dimension * sizeof(double), cudaMemcpyDeviceToHost);
	//checkCudaError(error, "resultHost memcpy");

	//for (uint32 i = 0; i < m_Dimension; ++i)
	//{
	//	std::cout << resultHost[i] << " ";
	//}
	//std::cout << std::endl;
	//delete[] resultHost;

	for (uint32 i = 0; i < m_MaxIter; ++i)
	{

		error = cudaMemset(deviceRr, 0.0, sizeof(double));
		checkCudaError(error, "deviceRr memset");

		Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceR, deviceR, deviceRr, m_Dimension);

		// p = r + beta(p - w * v)
		BICGSTAB::Vector_Add_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <m_DimGridVecAdd, m_DimBlockVecAdd >> > (deviceR, deviceV, deviceRho1, deviceRho0, deviceAlpha, deviceW, deviceP, deviceRr, deviceExitFlag, m_Residual, m_Dimension);

		//double* resultHost = new double[m_Dimension];
		//error = cudaMemcpy((void*)resultHost, deviceP, m_Dimension * sizeof(double), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "resultHost memcpy");

		//for (uint32 i = 0; i < m_Dimension; ++i)
		//{
		//	std::cout << resultHost[i] << " ";
		//}
		//std::cout << std::endl;
		//delete[] resultHost;

		// v = Ap
		Matrix_multi_Vector_Kernel<THREADS_MATMUTIL> << <m_DimGridMatMutil, m_DimBlockMatMutil >> > (m_DeviceDataCSR, deviceP, deviceV);

		//double* resultHost = new double[m_Dimension];
		//error = cudaMemcpy((void*)resultHost, deviceV, m_Dimension * sizeof(double), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "resultHost memcpy");

		//for (uint32 i = 0; i < m_Dimension; ++i)
		//{
		//	std::cout << resultHost[i] << " ";
		//}
		//std::cout << std::endl;
		//delete[] resultHost;

		error = cudaMemset(deviceR0v, 0.0, sizeof(double));
		checkCudaError(error, "deviceR0v memset");

		// R0v
		Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceR0, deviceV, deviceR0v, m_Dimension);

		//double* rho1 = new double;
		//error = cudaMemcpy((void*)rho1, deviceR0v, sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << *rho1 << std::endl;
		//delete rho1;

		// alpha = rho1 / R0v
		// s = r - alpha * v
		BICGSTAB::Vector_Sub_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <m_DimGridVecAdd, m_DimBlockVecAdd >> > (deviceR, deviceV, deviceRho1, deviceR0v, deviceAlpha, deviceS, m_Dimension);

		//double* resultHost = new double[m_Dimension];
		//error = cudaMemcpy((void*)resultHost, deviceS, m_Dimension * sizeof(double), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "resultHost memcpy");

		//for (uint32 i = 0; i < m_Dimension; ++i)
		//{
		//	std::cout << resultHost[i] << " ";
		//}
		//std::cout << std::endl;
		//delete[] resultHost;

		// As
		Matrix_multi_Vector_Kernel<THREADS_MATMUTIL> << < m_DimGridMatMutil, m_DimBlockMatMutil >> > (m_DeviceDataCSR, deviceS, deviceT);

		error = cudaMemset(deviceTs, 0.0, sizeof(double));
		checkCudaError(error, "deviceTs memset");

		error = cudaMemset(deviceTt, 0.0, sizeof(double));
		checkCudaError(error, "deviceTt memset");

		//BICGSTAB::Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << < m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceT, deviceS, deviceTs, deviceTt, m_Dimension);

		// Ts
		Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << < m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceT, deviceS, deviceTs, m_Dimension);

		//double* rho = new double;
		//error = cudaMemcpy((void*)rho, deviceTs, sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << *rho << std::endl;
		//delete rho;
		
		// Tt
		Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << < m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceT, deviceT, deviceTt, m_Dimension);

		//double* rho1= new double;
		//error = cudaMemcpy((void*)rho1, deviceTt, sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << *rho1 << std::endl;
		//delete rho1;

		//double* rho1 = new double;
		//error = cudaMemcpy((void*)rho1, deviceAlpha, sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << *rho1 << std::endl;
		//delete rho1;

		// w = ts / tt
		// x = x + alpha * p + w * s 
		BICGSTAB::Vector_Add_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <m_DimGridVecAdd, m_DimBlockVecAdd >> > (m_DeviceX, deviceP, deviceS, deviceAlpha, deviceTs, deviceTt, deviceW, m_Dimension);

		//double* rho = new double;
		//error = cudaMemcpy((void*)rho, deviceW, sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << *rho << std::endl;
		//delete rho;


		//double* resultHost = new double[m_Dimension];
		//error = cudaMemcpy((void*)resultHost, m_DeviceX, m_Dimension * sizeof(double), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "resultHost memcpy");

		//for (uint32 i = 0; i < m_Dimension; ++i)
		//{
		//	std::cout << resultHost[i] << " ";
		//}
		//std::cout << std::endl;
		//delete[] resultHost;

		// r = s - w * t
		BICGSTAB::Vector_Sub_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <m_DimGridVecAdd, m_DimBlockVecAdd >> > (deviceS, deviceT, deviceW, deviceR, m_Dimension);

		//double* resultHost = new double[m_Dimension];
		//error = cudaMemcpy((void*)resultHost, deviceR, m_Dimension * sizeof(double), cudaMemcpyDeviceToHost);
		//checkCudaError(error, "resultHost memcpy");

		//for (uint32 i = 0; i < m_Dimension; ++i)
		//{
		//	std::cout << resultHost[i] << " ";
		//}
		//std::cout << std::endl;
		//delete[] resultHost;

		double* temp = deviceRho0;
		deviceRho0 = deviceRho1;
		deviceRho1 = temp;

		error = cudaMemset(deviceRho1, 0.0, sizeof(double));
		checkCudaError(error, "deviceRho1 memset");

		// rho1 = r0 * r
		Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceR0, deviceR, deviceRho1, m_Dimension);

		//double* rho1 = new double;
		//error = cudaMemcpy((void*)rho1, deviceRho1, sizeof(double), cudaMemcpyDeviceToHost);
		//std::cout << *rho1 << std::endl;
		//delete rho1;

		if (*exitFlag)
		{
			m_Iter = i;
			break;
		}
	}

	error = cudaFree(deviceAx);
	checkCudaError(error, "deviceAx free");
	error = cudaFree(deviceR);
	checkCudaError(error, "deviceR free");
	error = cudaFree(deviceR0);
	checkCudaError(error, "deviceR0 free");
	error = cudaFree(deviceV);
	checkCudaError(error, "devideV free");
	error = cudaFree(deviceRho1);
	checkCudaError(error, "deviceRho1 free");
	error = cudaFree(deviceRho0);
	checkCudaError(error, "deviceRho0 free");
	error = cudaFree(deviceAlpha);
	checkCudaError(error, "deviceAlpha free");
	error = cudaFree(deviceW);
	checkCudaError(error, "deviceW free");
	error = cudaFree(deviceP);
	checkCudaError(error, "deviceP free");
	error = cudaFree(deviceR0v);
	checkCudaError(error, "deviceR0v free");
	error = cudaFree(deviceS);
	checkCudaError(error, "deviceS free");
	error = cudaFree(deviceT);
	checkCudaError(error, "deviceT free");
	error = cudaFree(deviceTs);
	checkCudaError(error, "deviceTs free");
	error = cudaFree(deviceTt);
	checkCudaError(error, "deviceTt free");
	error = cudaFree(deviceRr);
	checkCudaError(error, "deviceRr free");
	error = cudaFreeHost(exitFlag);
	checkCudaError(error, "exitFlag free");
}
