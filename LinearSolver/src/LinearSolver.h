#pragma once
#include <string>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Typedef.h"
#include "Utils/HostUtils.h"
#include "Utils/DeviceUtils.h"
#include "tools/TimerClock.h"
#include "tools/BenchMark.h"
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

	LinearSolver(CSR&& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);         // 使用移动构造函数代替拷贝构造函数
	LinearSolver(COO&& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);         // 使用移动构造函数代替拷贝构造函数

	virtual ~LinearSolver();
	virtual void start();
	virtual void test();

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
	double m_MinResidual;       // 最小残差
	double m_Residual;          // 实际残差
	uint32 m_Dimension;         // 维度

	TimerClock m_TC;            // 计时器
	double m_Time;              // 时间（单位s）
};


LinearSolver::LinearSolver(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: m_DataCSR(data), m_X(x), m_B(b), m_Dimension(dimension), m_MaxIter(maxIter), m_MinResidual(residual), m_Residual(0.0), m_DataCOO()
{
	m_Time = 0.0;
	m_Iter = m_MaxIter;         // 初始化为最大迭代次数
	m_SF = CSRFORMAT;
}


LinearSolver::LinearSolver(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: m_DataCOO(data), m_X(x), m_B(b), m_Dimension(dimension), m_MaxIter(maxIter), m_MinResidual(residual), m_Residual(0.0), m_DataCSR()
{
	m_Time = 0.0;
	m_Iter = 0;
	m_SF = COOFORMAT;
}


LinearSolver::LinearSolver(CSR&& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: m_DataCSR(std::move(data)), m_X(x), m_B(b), m_Dimension(dimension), m_MaxIter(maxIter), m_MinResidual(residual), m_Residual(0.0), m_DataCOO()
{
	m_Time = 0.0;
	m_Iter = m_MaxIter;
	m_SF = CSRFORMAT;
}


// 这里会两次调用CSR的析构函数
LinearSolver::~LinearSolver()
{
	m_X = nullptr;
	m_B = nullptr;
}

void LinearSolver::start()
{
#if TIME_TEST
	Timer tc("total");
#endif

	m_TC.update();

	calculate();

	m_Time = m_TC.getSecond();

}

void LinearSolver::test()
{
	double* resultMat = nullptr;

	//{
	//	Timer tc("Matrix_multi_Vector");
	//	Matrix_multi_Vector(m_DataCSR, m_X, resultMat);

	//	extern double sequence[5000];
	//	for (uint32 i = 0; i < m_Dimension; ++i)
	//	{
	//		sequence[i] = resultMat[i];
	//		std::cout << resultMat[i] << " ";
	//	}
	//	std::cout << std::endl;
	//	std::cout << std::endl;
	//	std::cout << std::endl;
	//}


	//{
	//	Timer tc("Vector_mutil_Vector");
	//	double resultVecMul = Vector_mutil_Vector(m_X, m_X, m_Dimension);

	//	std::cout << resultVecMul << std::endl;
	//	std::cout << std::endl;
	//	std::cout << std::endl;
	//}

	{
		Timer tc("memory malloc");
		resultMat = new double[m_Dimension];
	}

	{
		Timer tc("Vector_Add_Vector");
		Vector_Add_Vector(m_X, m_X, resultMat, m_Dimension);

		//extern double sequence[5000];
		//for (uint32 i = 0; i < m_Dimension; ++i)
		//{
		//	//sequence[i] = resultMat[i];
		//	std::cout << resultMat[i] << " ";
		//}
		//std::cout << std::endl;
		//std::cout << std::endl;
		//std::cout << std::endl;
	}

	delete[] resultMat;
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

	P_LinearSolver(CSR&& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);           // 使用移动构造函数代替拷贝构造函数
	P_LinearSolver(COO&& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);           // 使用移动构造函数代替拷贝构造函数

	~P_LinearSolver();

	void start() override;
	void test() override;


protected:
	void calculate() override;
	void memcpyHostToDevice();
	void memcpyDeviceToHost();
	void destory();

protected:
	DeviceCSR m_DeviceDataCSR;
	DeviceCOO m_DeviceDataCOO;

	double* m_DeviceX;
	double* m_DeviceB;

	dim3 m_DimGridMatMutil;           // 矩阵*向量的线程块数
	dim3 m_DimBlockMatMutil;          // 矩阵*向量每个线程块的线程数

	dim3 m_DimGridVecAdd;             // 向量相加的线程块数
	dim3 m_DimBlockVecAdd;            // 向量相加的每个线程块的线程数

	dim3 m_DimGridVecMutil;           // 向量相乘的线程块数
	dim3 m_DimBlockVecMutil;          // 向量相乘的每个线程块的线程数

};



P_LinearSolver::P_LinearSolver(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: LinearSolver(data, x, b, dimension, maxIter, residual), m_DeviceDataCSR(), m_DeviceDataCOO(), m_DeviceX(nullptr), m_DeviceB(nullptr)
{
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
	: LinearSolver(data, x, b, dimension, maxIter, residual), m_DeviceDataCSR(), m_DeviceDataCOO(), m_DeviceX(nullptr), m_DeviceB(nullptr)
{
	// 等待完善
}


P_LinearSolver::P_LinearSolver(CSR&& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: LinearSolver(std::move(data), x, b, dimension, maxIter, residual), m_DeviceDataCSR(), m_DeviceDataCOO(), m_DeviceX(nullptr), m_DeviceB(nullptr)
{
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


P_LinearSolver::~P_LinearSolver()
{
	destory();
}


void P_LinearSolver::start()
{
#if TIME_TEST
	Timer tc("total");
#endif

	m_TC.update();

	memcpyHostToDevice();

	calculate();

	memcpyDeviceToHost();

	m_Time = m_TC.getSecond();

}


void P_LinearSolver::test()
{
	constexpr uint32 TEST = 3;

	cudaError_t error;

#if TEST == 1

	Timer tc("Matrix_multi_Vector_Kernel");

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

	error = cudaMalloc((void**)&m_DeviceX, m_Dimension * sizeof(double));
	checkCudaError(error, "m_DeviceX malloc");

	error = cudaMemcpy((void*)m_DeviceX, m_X, m_Dimension * sizeof(double), cudaMemcpyHostToDevice);
	checkCudaError(error, "m_DeviceX memcpy");

	double* deviceResult = nullptr;
	error = cudaMalloc((void**)&deviceResult, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceResult malloc");

	Matrix_multi_Vector_Kernel<THREADS_MATMUTIL> << <m_DimGridMatMutil, m_DimBlockMatMutil >> > (m_DeviceDataCSR, m_DeviceX, deviceResult);

	error = cudaMemcpy((void*)result, deviceResult, m_Dimension * sizeof(double), cudaMemcpyDeviceToDevice);
	checkCudaError(error, "result memcpy");

#elif TEST == 2

	Timer tc("Vector_mutil_Vector_Kernel");

	double* deviceResult = nullptr;
	error = cudaMalloc((void**)&deviceResult, m_Dimension * sizeof(double));
	checkCudaError(error, "deviceResult malloc");

	error = cudaMalloc((void**)&m_DeviceX, m_Dimension * sizeof(double));
	checkCudaError(error, "m_DeviceX malloc");

	error = cudaMemcpy((void*)m_DeviceX, m_X, m_Dimension * sizeof(double), cudaMemcpyHostToDevice);
	checkCudaError(error, "m_DeviceX memcpy");

	Vector_mutil_Vector_Kernel<THREADS_VECMUTIL, ELEMS_VECMUTIL> << <m_DimGridVecMutil, m_DimBlockVecMutil >> > (m_DeviceX, m_DeviceX, deviceResult, m_Dimension);

	error = cudaMemcpy((void*)result2, deviceResult, sizeof(double), cudaMemcpyDeviceToHost);
	checkCudaError(error, "result2 memcpy");

#else 

	Timer tc("Vector_Add_Vector_Kernel");
	double* deviceResult = nullptr;
	double* result = nullptr;

	{
		Timer tc("memory malloc");
		result = new double[m_Dimension];

		error = cudaMalloc((void**)&deviceResult, m_Dimension * sizeof(double));
		checkCudaError(error, "deviceResult malloc");

		error = cudaMalloc((void**)&m_DeviceX, m_Dimension * sizeof(double));
		checkCudaError(error, "m_DeviceX malloc");

		error = cudaMalloc((void**)&m_DeviceB, m_Dimension * sizeof(double));
		checkCudaError(error, "m_DeviceB malloc");
	} 

	{

		Timer tc("cpu-gpu");

		error = cudaMemcpy((void*)m_DeviceX, m_X, m_Dimension * sizeof(double), cudaMemcpyHostToDevice);
		checkCudaError(error, "m_DeviceX memcpy");

		error = cudaMemcpy((void*)m_DeviceB, m_B, m_Dimension * sizeof(double), cudaMemcpyHostToDevice);
		checkCudaError(error, "m_DeviceB memcpy");
	}

	{
		Timer tc("calculate");
		Vector_Add_Vector_Kernel<THREADS_VECADD, ELEMS_VECADD> << <m_DimGridVecAdd, m_DimBlockVecAdd >> > (m_DeviceB, m_DeviceX, deviceResult, m_Dimension);
		cudaDeviceSynchronize();
	}

	{
		Timer tc("gpu-cpu");
		error = cudaMemcpy((void*)result, deviceResult, m_Dimension * sizeof(double), cudaMemcpyDeviceToHost);
		checkCudaError(error, "result memcpy");
	}

	//extern double parallel[5000];
	//for (uint32 i = 0; i < m_Dimension; ++i)
	//{
	//	std::cout << result[i] << " ";
	//}

#endif
}


void P_LinearSolver::calculate()
{
	std::cout << "P_LinearSolver::接口类无法计算！" << std::endl;
}

void P_LinearSolver::memcpyHostToDevice()
{
#if TIME_TEST
	Timer tc("HtoD");
#endif

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

	error = cudaMalloc((void**)&m_DeviceB, m_Dimension * sizeof(double));
	checkCudaError(error, "m_DeviceB malloc");

	error = cudaMemcpy((void*)m_DeviceX, m_X, m_Dimension * sizeof(double), cudaMemcpyHostToDevice);
	checkCudaError(error, "m_DeviceX memcpy");

	error = cudaMemcpy((void*)m_DeviceB, m_B, m_Dimension * sizeof(double), cudaMemcpyHostToDevice);
	checkCudaError(error, "m_DeviceB memcpy");
}

void P_LinearSolver::memcpyDeviceToHost()
{
#if TIME_TEST
	Timer tc("DtoH");
#endif

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

	error = cudaFree(m_DeviceB);
	checkCudaError(error, "m_DeviveB free");
}