#pragma once
#include "Typedef.h"
#include <string>
#include <fstream>
#include <sstream>

enum storageFormat
{
	CSRFORMAT,
	COOFORMAT
};


struct DeviceCSR
{
	double* m_Data;           // 矩阵的非零元素值
	uint32* m_Fnz;            // 每一行的起始索引
	uint32* m_Col;            // 每个元素的列坐标
	uint32 m_ArrayLength;     // 存放数据长度
	uint32 m_Dimension;       // 矩阵维度  
};

class CSR
{
public:
	double* m_Data;           // 矩阵的非零元素值
	uint32* m_Fnz;            // 每一行的起始索引
	uint32* m_Col;            // 每个元素的列坐标
	uint32 m_ArrayLength;     // 存放数据长度
	uint32 m_Dimension;       // 矩阵维度    

public:
	CSR() = default;
	CSR(const std::string& filePathA);
	CSR(const CSR& csr);                          // 拷贝构造函数（使用默认的浅拷贝即可，使用深拷贝的话在调用GPU的 kernel 函数会出错）
	CSR(CSR&& csr) noexcept;                      // 移动构造函数（使用默认的浅拷贝即可，使用深拷贝的话在调用GPU的 kernel 函数会出错）
	CSR& operator=(const CSR& csr);               // 拷贝赋值运算符
	CSR& operator=(CSR&& csr) noexcept;           // 移动赋值运算符
	~CSR();

};

CSR::CSR(const std::string& filePathA)
{
	std::ifstream file(filePathA);

	if (!file.is_open())
	{
		std::cout << "path:" << filePathA << " file open failed!" << std::endl;
	}

	std::string str;
	getline(file, str);
	std::stringstream stream(str);
	stream >> m_ArrayLength;

	m_Data = new double[m_ArrayLength];
	m_Col = new uint32[m_ArrayLength];

	getline(file, str);
	std::stringstream stream2(str);
	stream2 >> m_Dimension;

	m_Fnz = new uint32[m_Dimension + 1];

	getline(file, str);
	std::stringstream stream3(str);
	uint32 i = 0;
	while (stream3 >> m_Data[i++]) {}

	getline(file, str);
	std::stringstream stream4(str);
	i = 0;
	while (stream4 >> m_Col[i++]) {}

	getline(file, str);
	std::stringstream stream5(str);
	i = 0;
	while (stream5 >> m_Fnz[i++]) {}
}

// 拷贝构造函数
CSR::CSR(const CSR& csr)
{
	m_ArrayLength = csr.m_ArrayLength;
	m_Dimension = csr.m_Dimension;
	m_Data = new double[m_ArrayLength];
	m_Col = new uint32[m_ArrayLength];
	m_Fnz = new uint32[m_Dimension + 1];

	// 由于都是基础类型，所以直接使用memcpy
	memcpy(m_Data, csr.m_Data, m_ArrayLength * sizeof(double));
	memcpy(m_Col, csr.m_Col, m_ArrayLength * sizeof(uint32));
	memcpy(m_Fnz, csr.m_Fnz, (m_Dimension + 1) * sizeof(uint32));
}

// 拷贝赋值运算符
CSR& CSR::operator=(const CSR& csr)
{
	if (this != &csr)
	{
		// 这里不要清空，直接用 memcpy 覆盖是不是也可以？
		// 这样就不用重新 new 了，但是要保证两个对象的 m_Dimension 和 m_ArrayLength 完全一致，不然会出错
		delete[] m_Data;         // 清空原始的内存
		delete[] m_Col;
		delete[] m_Fnz;

		m_ArrayLength = csr.m_ArrayLength;
		m_Dimension = csr.m_Dimension;
		m_Data = new double[m_ArrayLength];
		m_Col = new uint32[m_ArrayLength];
		m_Fnz = new uint32[m_Dimension + 1];

		// 由于都是基础类型，所以直接使用memcpy
		// 如果是自己写的类，并且含有指针，需要利用循环一次次调用构造函数
		memcpy(m_Data, csr.m_Data, m_ArrayLength * sizeof(double));
		memcpy(m_Col, csr.m_Col, m_ArrayLength * sizeof(uint32));
		memcpy(m_Fnz, csr.m_Fnz, (m_Dimension + 1) * sizeof(uint32));

	}

	return *this;
}

// 移动构造函数
CSR::CSR(CSR&& csr) noexcept
{
	m_Data = csr.m_Data;
	m_Col = csr.m_Col;
	m_Fnz = csr.m_Fnz;
	m_ArrayLength = csr.m_ArrayLength;
	m_Dimension = csr.m_Dimension;

	csr.m_Data = nullptr;
	csr.m_Col = nullptr;
	csr.m_Fnz = nullptr;
	csr.m_ArrayLength = 0;
	csr.m_Dimension = 0;

}

// 移动赋值运算符
CSR& CSR::operator=(CSR&& csr) noexcept
{
	if (this != &csr)
	{
		delete[] m_Data;
		delete[] m_Col;
		delete[] m_Fnz;

		m_Data = csr.m_Data;
		m_Col = csr.m_Col;
		m_Fnz = csr.m_Fnz;
		m_ArrayLength = csr.m_ArrayLength;
		m_Dimension = csr.m_Dimension;

		csr.m_Data = nullptr;
		csr.m_Col = nullptr;
		csr.m_Fnz = nullptr;
		csr.m_ArrayLength = 0;
		csr.m_Dimension = 0;
	}

	return *this;
}

// 析构函数释放分配的内存（如果分配了的话）
CSR::~CSR()
{
	delete[] m_Data;
	delete[] m_Col;
	delete[] m_Fnz;
}




class COO
{
public:
	double* m_Data;           // 矩阵的非零元素值
	uint32* m_Row;            // 每个元素的行坐标
	uint32* m_Col;            // 每个元素的列坐标
	uint32 m_ArrayLength;     // 存放数据长度
};


struct DeviceCOO
{
public:
	double* m_Data;           // 矩阵的非零元素值
	uint32* m_Row;            // 每个元素的行坐标
	uint32* m_Col;            // 每个元素的列坐标
	uint32 m_ArrayLength;    // 存放数据长度
};

