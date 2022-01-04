#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Typedef.h"
#include "LinearSolver.h"

#include "CG/Parallel/P_CG.h"
#include "CG/Sequence/S_CG.h"
#include "BiCGSTAB/Parallel/P_BiCGSTAB.h"
#include "BiCGSTAB/Sequence/S_BICGSTAB.h"


double parallel[5000];
double sequence[5000];

int main()
{
	CSR data;
	data.m_Dimension = 5000;
	data.m_Data = new double[3 * data.m_Dimension - 2];
	data.m_ArrayLength = 3 * data.m_Dimension - 2;
	data.m_Data[0] = 1.0;
	for (uint32 i = 0; i < 3 * data.m_Dimension - 2; ++i)
	{
		if (i % 3 == 1)
			data.m_Data[i] = (int)(i / 3) + 2;
		else if (i % 3 == 2)
			data.m_Data[i] = (int)(i / 3) + 2;
		else
			data.m_Data[i] = (int)(i / 3) + 1;
	}

	data.m_Col = new uint32[3 * data.m_Dimension - 2];
	data.m_Col[0] = 0;
	data.m_Col[1] = 1;
	for (uint32 i = 0; i < 3 * data.m_Dimension - 4; ++i)
	{
		if (i % 3 == 2)
			data.m_Col[i] = (int)(i / 3);
		else if (i % 3 == 0)
			data.m_Col[i] = (int)(i / 3);
		else
			data.m_Col[i] = (int)(i / 3) + 1;
	}
	data.m_Col[3 * data.m_Dimension - 4] = data.m_Dimension - 2;
	data.m_Col[3 * data.m_Dimension - 3] = data.m_Dimension - 1;

	data.m_Fnz = new uint32[data.m_Dimension + 1];
	data.m_Fnz[0] = 0;
	data.m_Fnz[1] = 2;
	for (uint32 i = 2; i < data.m_Dimension; ++i)
	{
		data.m_Fnz[i] = data.m_Fnz[i - 1] + 3;
	}
	data.m_Fnz[data.m_Dimension] = 3 * data.m_Dimension - 2;

	double* x = new double[data.m_Dimension];
	for (uint32 i = 0; i < data.m_Dimension; ++i)
	{
		x[i] = 1.0 / (i + 3.14);
	}

	double* b = new double[data.m_Dimension];
	for (uint32 i = 0; i < data.m_Dimension; ++i)             // 初始化b为0
		b[i] = 0;

	Matrix_multi_Vector(data, x, b);

	for (uint32 i = 0; i < data.m_Dimension; ++i)
	{
		x[i] = 0;
	}

	CSR data1;
	data1.m_Dimension = 3;
	data1.m_Data = new double[9];
	data1.m_Data[0] = 10.0;
	data1.m_Data[1] = 1.0;
	data1.m_Data[2] = -5;
	data1.m_Data[3] = -20.0;
	data1.m_Data[4] = 3.0;
	data1.m_Data[5] = 20.0;
	data1.m_Data[6] = 5.0;
	data1.m_Data[7] = 3.0;
	data1.m_Data[8] = 5.0;

	data1.m_ArrayLength = 9;

	data1.m_Col = new uint32[9];
	data1.m_Col[0] = 0;
	data1.m_Col[1] = 1;
	data1.m_Col[2] = 2;
	data1.m_Col[3] = 0;
	data1.m_Col[4] = 1;
	data1.m_Col[5] = 2;
	data1.m_Col[6] = 0;
	data1.m_Col[7] = 1;
	data1.m_Col[8] = 2;


	data1.m_Fnz = new uint32[4];
	data1.m_Fnz[0] = 0;
	data1.m_Fnz[1] = 3;
	data1.m_Fnz[2] = 6;
	data1.m_Fnz[3] = 9;

	double* x1 = new double[3];
	x1[0] = 1.0;
	x1[1] = 1.0;
	x1[2] = 1.0;

	double* b1 = new double[3];
	b1[0] = 1.0;
	b1[1] = 2.0;
	b1[2] = 6.0;


	//P_CG solverP_CG(data, x, b, data.m_Dimension, 50000, 0.000001);
	//solverP_CG.start();
	//std::cout << solverP_CG.getTime() << std::endl;
	//std::cout << solverP_CG.getIter() << std::endl;

	//S_BICGSTAB solverS_BICGSTAB(data1, x1, b1, data1.m_Dimension, 50000, 0.000001);
	//solverS_BICGSTAB.start();
	//std::cout << solverS_BICGSTAB.getTime() << std::endl;
	//std::cout << solverS_BICGSTAB.getIter() << std::endl;

	CSR data2("../res/A.txt");

	for (uint32 i = 0; i < data2.m_ArrayLength; ++i)
	{
		std::cout << data2.m_Data[i] << " ";
	}
	std::cout << std::endl;

	for (uint32 i = 0; i < data2.m_ArrayLength; ++i)
	{
		std::cout << data2.m_Col[i] << " ";
	}
	std::cout << std::endl;

	for (uint32 i = 0; i < data2.m_Dimension; ++i)
	{
		std::cout << data2.m_Fnz[i] << " ";
	}
	std::cout << std::endl;

	std::cout << data2.m_ArrayLength << std::endl;
	std::cout << data2.m_Dimension << std::endl;

	double* b2 = new double[data2.m_Dimension];

	std::ifstream file("../res/b.txt");
	std::string str;
	getline(file, str);
	getline(file, str);
	std::stringstream stream(str);
	uint32 i = 0;
	while (stream >> b2[i++]) {}

	double* x2 = new double[data2.m_Dimension];
	for (uint32 i = 0; i < data2.m_Dimension; ++i)
	{
		x2[i] = 0.0;
	}


	//S_BICGSTAB solverS_BICGSTAB(data2, x2, b2, data2.m_Dimension, 50000, 0.000001);
	//solverS_BICGSTAB.start();
	//std::cout << solverS_BICGSTAB.getTime() << std::endl;
	//std::cout << solverS_BICGSTAB.getIter() << std::endl;

	P_BiCGSTAB solverP_BICGSTAB(data2, x2, b2, data2.m_Dimension, 50000, 1e-12);
	solverP_BICGSTAB.start();
	std::cout << solverP_BICGSTAB.getTime() << std::endl;
	std::cout << solverP_BICGSTAB.getIter() << std::endl;

	//uint32 error = 0;
	//for (uint32 i = 0; i < 5000; ++i)
	//{
	//	if (parallel[i] != sequence[i])
	//	{
	//		std::cout << parallel[i] << "   " << sequence[i] << std::endl;
	//		++error;
	//	}
	//}
	//std::cout << "error: " << error << std::endl;

	for (uint32 i = 0; i < data.m_Dimension; ++i)
	{
		std::cout << x[i] << " ";
	}
	std::cout << std::endl;

	delete[] x;
	delete[] b;
	delete[] x1;
	delete[] b1;
	delete[] x2;
	delete[] b2;

	return 0;
}