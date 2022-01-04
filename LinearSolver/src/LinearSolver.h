#pragma once
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Typedef.h"
#include "Utils/HostUtils.h"
#include "Utils/DeviceUtils.h"
#include "tools/TimerClock.h"
#include "Constants.h"
// This is a personal academic project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

// Name: PVS - Studio Free
// Key : FREE - FREE - FREE - FREE

#include "MatFormat.h"



class LinearSolver
{
public:
	LinearSolver(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);
	LinearSolver(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);
	virtual ~LinearSolver();
	virtual void start();

	double getTime() const;
	uint32 getIter() const;

protected:
	virtual void calculate();

protected:
	CSR m_DataCSR;              // A（CSR格式存储）
	COO m_DataCOO;              // A（COO格式存储）
	storageFormat m_SF;         // 确定存储格式
	double* m_X;                // x
	double* m_B;                // b

	uint32 m_MaxIter;           // 最大迭代次数
	uint32 m_Iter;              // 实际迭代次数
	double m_Residual;          // 最小残差
	uint32 m_Dimension;         // 维度

	TimerClock m_TC;            // 计时器
	double m_Time;              // 时间（单位s）
};


LinearSolver::LinearSolver(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: m_DataCSR(data), m_X(x), m_B(b), m_Dimension(dimension), m_MaxIter(maxIter), m_Residual(residual), m_DataCOO()
{
	m_Time = 0.0;
	m_Iter = m_MaxIter;         // 初始化为最大迭代次数
	m_SF = CSRFORMAT;
}


LinearSolver::LinearSolver(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: m_DataCOO(data), m_X(x), m_B(b), m_Dimension(dimension), m_MaxIter(maxIter), m_Residual(residual), m_DataCSR()
{
	m_Time = 0.0;
	m_Iter = 0;
	m_SF = COOFORMAT;
}

// 这里会两次调用CSR的析构函数
LinearSolver::~LinearSolver()
{
	m_X = nullptr;
	m_B = nullptr;

	if (m_SF == CSRFORMAT)
	{
		delete[] m_DataCSR.m_Data;
		delete[] m_DataCSR.m_Col;
		delete[] m_DataCSR.m_Fnz;
	}
	else if (m_SF == COOFORMAT)
	{
		delete[] m_DataCOO.m_Data;
		delete[] m_DataCOO.m_Col;
		delete[] m_DataCOO.m_Row;
	}
}

void LinearSolver::start()
{
	m_TC.update();

	calculate();

	m_Time = m_TC.getSecond();

}

double LinearSolver::getTime() const
{
	return m_Time;
}

uint32 LinearSolver::getIter() const
{
	return m_Iter;
}

void LinearSolver::calculate()
{
	std::cout << "LinearSolver：接口类无法计算！" << std::endl;
}


class P_LinearSolver : public LinearSolver
{
public:
	P_LinearSolver(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);
	P_LinearSolver(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);
	~P_LinearSolver();

	void start() override;


protected:
	void calculate() override;
	void memcpyHostToDevice();
	void memcpyDeviceToHost();
	void destory();

protected:
	CSR m_DeviceDataCSR;
	COO m_DeviceDataCOO;

	double* m_DeviceX;
	double* m_DeviveB;

	dim3 m_DimGridMatMutil;           // 矩阵*向量的线程块数
	dim3 m_DimBlockMatMutil;          // 矩阵*向量每个线程块的线程数

	dim3 m_DimGridVecAdd;             // 向量相加的线程块数
	dim3 m_DimBlockVecAdd;            // 向量相加的每个线程块的线程数

	dim3 m_DimGridVecMutil;           // 向量相乘的线程块数
	dim3 m_DimBlockVecMutil;          // 向量相乘的每个线程块的线程数

};



P_LinearSolver::P_LinearSolver(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: LinearSolver(data, x, b, dimension, maxIter, residual)
{
	m_DeviceDataCSR.m_Data = nullptr;
	m_DeviceDataCSR.m_Col = nullptr;
	m_DeviceDataCSR.m_Fnz = nullptr;

	uint32 warpNumPerBlock = THREADS_MATMUTIL / WARP_SIZE;      // 计算出每个线程块的warp数量
	uint32 linePerBlock = warpNumPerBlock * ELEMS_MATMUTIL;     // 计算出每个线程块处理的行数

	m_DimGridMatMutil = dim3((m_Dimension - 1) / linePerBlock + 1, 1, 1);
	m_DimBlockMatMutil = dim3(THREADS_MATMUTIL, 1, 1);

	uint32 elemsPerBlock = THREADS_VECADD * ELEMS_VECADD;

	m_DimGridVecAdd = dim3((m_Dimension - 1) / elemsPerBlock + 1, 1, 1);
	m_DimBlockVecAdd = dim3(THREADS_VECADD, 1, 1);

	elemsPerBlock = THREADS_VECMUTIL * ELEMS_VECMUTIL;           // 每个线程块处理的元素数量

	m_DimGridVecMutil = dim3((m_Dimension - 1) / elemsPerBlock + 1, 1, 1);
	m_DimBlockVecMutil = dim3(THREADS_VECMUTIL, 1, 1);
}


P_LinearSolver::P_LinearSolver(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: LinearSolver(data, x, b, dimension, maxIter, residual)
{
	m_DeviceDataCOO.m_Data = nullptr;
	m_DeviceDataCOO.m_Col = nullptr;
	m_DeviceDataCOO.m_Row = nullptr;
}

P_LinearSolver::~P_LinearSolver()
{
	destory();
}


void P_LinearSolver::start()
{

	m_TC.update();

	memcpyHostToDevice();

	calculate();

	memcpyDeviceToHost();

	m_Time = m_TC.getSecond();

}

void P_LinearSolver::calculate()
{
	std::cout << "P_LinearSolver::接口类无法计算！" << std::endl;
}

void P_LinearSolver::memcpyHostToDevice()
{
	// 数据量大的时候可以考虑使用异步拷贝
	cudaError_t error;
	if (m_SF == CSRFORMAT)
	{
		error = cudaMalloc((void**)&m_DeviceDataCSR.m_Data, m_DataCSR.m_ArrayLength * sizeof(*m_DeviceDataCSR.m_Data));
		checkCudaError(error, "m_DeviceDataCSR.m_Data malloc");

		error = cudaMalloc((void**)&m_DeviceDataCSR.m_Col, m_DataCSR.m_ArrayLength * sizeof(*m_DeviceDataCSR.m_Col));
		checkCudaError(error, "m_DeviceDataCSR.m_Col malloc");

		error = cudaMalloc((void**)&m_DeviceDataCSR.m_Fnz, (m_DataCSR.m_Dimension + 1) * sizeof(*m_DeviceDataCSR.m_Fnz));
		checkCudaError(error, "m_DeviceDataCSR.m_Fnz malloc");

		error = cudaMemcpy((void*)m_DeviceDataCSR.m_Data, m_DataCSR.m_Data, m_DataCSR.m_ArrayLength * sizeof(*m_DeviceDataCSR.m_Data), cudaMemcpyHostToDevice);
		checkCudaError(error, "m_DeviceDataCSR.m_Data memcpy");

		error = cudaMemcpy((void*)m_DeviceDataCSR.m_Col, m_DataCSR.m_Col, m_DataCSR.m_ArrayLength * sizeof(*m_DeviceDataCSR.m_Col), cudaMemcpyHostToDevice);
		checkCudaError(error, "m_DeviceDataCSR.m_Col memcpy");

		error = cudaMemcpy((void*)m_DeviceDataCSR.m_Fnz, m_DataCSR.m_Fnz, (m_DataCSR.m_Dimension + 1) * sizeof(*m_DeviceDataCSR.m_Fnz), cudaMemcpyHostToDevice);
		checkCudaError(error, "m_DeviceDataCSR.m_Fnz memcpy");

		m_DeviceDataCSR.m_ArrayLength = m_DataCSR.m_ArrayLength;
		m_DeviceDataCSR.m_Dimension = m_DataCSR.m_Dimension;

	}
	else if (m_SF == COOFORMAT)
	{
		error = cudaMalloc((void**)&m_DeviceDataCOO.m_Data, m_DataCOO.m_ArrayLength * sizeof(*m_DeviceDataCOO.m_Data));
		checkCudaError(error, "m_DeviceDataCOO.m_Data malloc");

		error = cudaMalloc((void**)&m_DeviceDataCOO.m_Col, m_DataCOO.m_ArrayLength * sizeof(*m_DeviceDataCOO.m_Col));
		checkCudaError(error, "m_DeviceDataCOO.m_Col malloc");

		error = cudaMalloc((void**)&m_DeviceDataCOO.m_Row, m_DataCOO.m_ArrayLength * sizeof(*m_DeviceDataCOO.m_Row));
		checkCudaError(error, "m_DeviceDataCOO.m_Row malloc");

		error = cudaMemcpy((void*)m_DeviceDataCOO.m_Data, m_DataCOO.m_Data, m_DataCOO.m_ArrayLength * sizeof(*m_DeviceDataCOO.m_Data), cudaMemcpyHostToDevice);
		checkCudaError(error, "m_DeviceDataCOO.m_Data memcpy");

		error = cudaMemcpy((void*)m_DeviceDataCOO.m_Col, m_DataCOO.m_Col, m_DataCOO.m_ArrayLength * sizeof(*m_DeviceDataCOO.m_Col), cudaMemcpyHostToDevice);
		checkCudaError(error, "m_DeviceDataCOO.m_Col memcpy");

		error = cudaMemcpy((void*)m_DeviceDataCOO.m_Row, m_DataCOO.m_Row, m_DataCOO.m_ArrayLength * sizeof(*m_DeviceDataCOO.m_Row), cudaMemcpyHostToDevice);
		checkCudaError(error, "m_DeviceDataCOO.m_Row memcpy");

		m_DeviceDataCOO.m_ArrayLength = m_DataCOO.m_ArrayLength;         // 直接赋值即可
	}

	error = cudaMalloc((void**)&m_DeviceX, m_Dimension * sizeof(double));
	checkCudaError(error, "m_DeviceX malloc");

	error = cudaMalloc((void**)&m_DeviveB, m_Dimension * sizeof(double));
	checkCudaError(error, "m_DeviceB malloc");

	error = cudaMemcpy((void*)m_DeviceX, m_X, m_Dimension * sizeof(double), cudaMemcpyHostToDevice);
	checkCudaError(error, "m_DeviceX memcpy");

	error = cudaMemcpy((void*)m_DeviveB, m_B, m_Dimension * sizeof(double), cudaMemcpyHostToDevice);
	checkCudaError(error, "m_DeviceB memcpy");
}

void P_LinearSolver::memcpyDeviceToHost()
{
	cudaError_t error;
	error = cudaMemcpy((void*)m_X, m_DeviceX, m_Dimension * sizeof(double), cudaMemcpyDeviceToHost);
	checkCudaError(error, "m_X memcpy");
}

void P_LinearSolver::destory()
{
	cudaError_t error;
	if (m_SF == CSRFORMAT)
	{
		error = cudaFree(m_DeviceDataCSR.m_Data);
		checkCudaError(error, "m_DeviceDataCSR.m_Data free");

		error = cudaFree(m_DeviceDataCSR.m_Col);
		checkCudaError(error, "m_DeviceDataCSR.m_Col free");

		error = cudaFree(m_DeviceDataCSR.m_Fnz);
		checkCudaError(error, "m_DeviceDataCSR.m_Fnz free");
	}
	else if (m_SF == COOFORMAT)
	{
		error = cudaFree(m_DeviceDataCOO.m_Data);
		checkCudaError(error, "m_DeviceDataCOO.m_Data free");

		error = cudaFree(m_DeviceDataCOO.m_Col);
		checkCudaError(error, "m_DeviceDataCOO.m_Col free");

		error = cudaFree(m_DeviceDataCOO.m_Row);
		checkCudaError(error, "m_DeviceDataCOO.m_Row free");
	}

	error = cudaFree(m_DeviceX);
	checkCudaError(error, "m_DeviceX free");

	error = cudaFree(m_DeviveB);
	checkCudaError(error, "m_DeviveB free");
}