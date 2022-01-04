#pragma once
#include "LinearSolver.h"
#include "CG_DeviceUtils.h"

class P_CG : public P_LinearSolver
{
public:
	P_CG(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);
	P_CG(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);

	P_CG(CSR&& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);
	P_CG(COO&& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);

protected:
	void calculate() override;
};


P_CG::P_CG(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: P_LinearSolver(data, x, b, dimension, maxIter, residual) {}

P_CG::P_CG(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: P_LinearSolver(data, x, b, dimension, maxIter, residual) {}

P_CG::P_CG(CSR&& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: P_LinearSolver(std::move(data), x, b, dimension, maxIter, residual) {}

void P_CG::calculate()
{
#if TIME_TEST
	Timer tc("calculate");
#endif

	double* deviceAp = nullptr;
	double* deviceR = nullptr;
	double* deviceP = nullptr;            // p代表方向
	double* deviceNewR = nullptr;
	double* pAp = nullptr;
	double* rr = nullptr;
	double* newRnewR = nullptr;
	bool* exitFlag = nullptr;
	bool* deviceExitFlag = nullptr;

	cudaError_t error;
	error = cudaMalloc((void**)&deviceAp, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceAp malloc");
	error = cudaMalloc((void**)&deviceR, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceR malloc");
	error = cudaMalloc((void**)&deviceP, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceP malloc");
	error = cudaMalloc((void**)&deviceNewR, m_Dimension * sizeof(double));
	checkCudaError(error, "newR malloc");
	error = cudaMalloc((void**)&pAp, sizeof(double));
	checkCudaError(error, "pAp malloc");
	error = cudaMalloc((void**)&rr, sizeof(double));
	checkCudaError(error, "rr malloc");
	error = cudaMalloc((void**)&newRnewR, sizeof(double));
	checkCudaError(error, "newRnewR malloc");

	// 启用零复制
	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc((void**)&exitFlag, sizeof(bool), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	checkCudaError(error, "cudaHostAlloc data");
	cudaHostGetDevicePointer(&deviceExitFlag, exitFlag, 0);

	*exitFlag = false;

	// Ax
	Matrix_multi_Vector_Kernel<THREADS_MATMUTIL> << < m_DimGridMatMutil, m_DimBlockMatMutil >> > (m_DeviceDataCSR, m_DeviceX, deviceAp);
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	checkCudaError(error, "Ax Kernel");

	// r = b-Ax
	Vector_Sub_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <m_DimGridVecAdd, m_DimBlockVecAdd >> > (m_DeviceB, deviceAp, deviceR, m_Dimension);
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	checkCudaError(error, "r = b-Ax Kernel");

	// p = r（初始化方向为负的梯度方向）
	error = cudaMemcpy((void*)deviceP, deviceR, m_Dimension * sizeof(double), cudaMemcpyDeviceToDevice);
	checkCudaError(error, "deviceP memcpy");

	for (uint32 i = 0; i < m_MaxIter; ++i)
	{
		//TimerClock tc;
		//tc.update();

		// Ap
		Matrix_multi_Vector_Kernel<THREADS_MATMUTIL> << < m_DimGridMatMutil, m_DimBlockMatMutil >> > (m_DeviceDataCSR, deviceP, deviceAp);
		//cudaDeviceSynchronize();
		//error = cudaGetLastError();
		//checkCudaError(error, "Ap Kernel");
		//std::cout << tc.getMilliSecond() << "ms" << std::endl;
		//tc.update();

		error = cudaMemset(pAp, 0.0, sizeof(double));
		checkCudaError(error, "pAp memset");

		// pAp
		Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceP, deviceAp, pAp, m_Dimension);
		//cudaDeviceSynchronize();
		//error = cudaGetLastError();
		//checkCudaError(error, "pAp Kernel");
		//std::cout << tc.getMilliSecond() << "ms" << std::endl;
		//tc.update();

		error = cudaMemset(rr, 0.0, sizeof(double));
		checkCudaError(error, "result memset");

		// rr
		Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceR, deviceR, rr, m_Dimension);
		//Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceR, deviceR, rr, m_Residual, deviceExitFlag, m_Dimension);
		//cudaDeviceSynchronize();
		//error = cudaGetLastError();
		//checkCudaError(error, "rr Kernel");
		//std::cout << tc.getMilliSecond() << "ms" << std::endl;
		//tc.update();

		// x = x + alpha * p
		CG::Vector_Add_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <m_DimGridVecAdd, m_DimBlockVecAdd >> > (m_DeviceX, deviceP, rr, pAp, m_DeviceX, m_MinResidual, deviceExitFlag, m_Dimension);
		//cudaDeviceSynchronize();
		//error = cudaGetLastError();
		//checkCudaError(error, " x + alpha * p Kernel");
		//std::cout << tc.getMilliSecond() << "ms" << std::endl;
		//tc.update();

		// newR = r - alpha * Ap
		Vector_Sub_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <m_DimGridVecAdd, m_DimBlockVecAdd >> > (deviceR, deviceAp, rr, pAp, deviceNewR, m_Dimension);
		//cudaDeviceSynchronize();
		//error = cudaGetLastError();
		//checkCudaError(error, "r - alpha * Ap Kernel");
		//std::cout << tc.getMilliSecond() << "ms" << std::endl;
		//tc.update();

		error = cudaMemset(newRnewR, 0.0, sizeof(double));
		checkCudaError(error, "result memset");

		// newRnewR
		Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <m_DimGridVecMutil, m_DimBlockVecMutil >> > (deviceNewR, deviceNewR, newRnewR, m_Dimension);
		//cudaDeviceSynchronize();
		//error = cudaGetLastError();
		//checkCudaError(error, "newRnewR Kernel");
		//std::cout << tc.getMilliSecond() << "ms" << std::endl;
		//tc.update();

		// p = r + beta * p
		Vector_Add_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <m_DimGridVecAdd, m_DimBlockVecAdd >> > (deviceNewR, deviceP, newRnewR, rr, deviceP, m_Dimension);
		//cudaDeviceSynchronize();
		//error = cudaGetLastError();
		//checkCudaError(error, " p = r + beta * p Kernel");
		//std::cout << tc.getMilliSecond() << "ms" << std::endl;
		//tc.update();

		// 感觉在数据量较大时，交换指针比数据拷贝更快（有待验证）
		double* temp = deviceR;
		deviceR = deviceNewR;
		deviceNewR = temp;
		//error = cudaMemcpy((void*)deviceR, deviceNewR, m_Dimension * sizeof(double), cudaMemcpyDeviceToDevice);
		//checkCudaError(error, "deviceR memcpy");

		//std::cout << tc.getMilliSecond() << "ms" << std::endl;
		//std::cout << std::endl;

		// 判断退出的条件
		if (*exitFlag)
		{
			cudaMemcpy((void*)&m_Residual, rr, sizeof(double), cudaMemcpyDeviceToHost);
			m_Residual /= m_Dimension;
			std::cout << "residual: " << m_Residual << std::endl;
			m_Iter = i;
			break;
		}
	}

	error = cudaFree(deviceAp);
	checkCudaError(error, "deviceAp free");
	error = cudaFree(deviceR);
	checkCudaError(error, "deviceR free");
	error = cudaFree(deviceP);
	checkCudaError(error, "deviceP free");
	error = cudaFree(deviceNewR);
	checkCudaError(error, "deviceNewR free");
	error = cudaFree(pAp);
	checkCudaError(error, "pAp free");
	error = cudaFree(rr);
	checkCudaError(error, "rr free");
	error = cudaFree(newRnewR);
	checkCudaError(error, "newRnewR free");
	error = cudaFreeHost(exitFlag);
	checkCudaError(error, "exitFlag free");
}