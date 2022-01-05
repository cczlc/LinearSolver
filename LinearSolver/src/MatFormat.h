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
	double* m_Data;           // ����ķ���Ԫ��ֵ
	uint32* m_Fnz;            // ÿһ�е���ʼ����
	uint32* m_Col;            // ÿ��Ԫ�ص�������
	uint32 m_ArrayLength;     // ������ݳ���
	uint32 m_Dimension;       // ����ά��  
};

class CSR
{
public:
	double* m_Data;           // ����ķ���Ԫ��ֵ
	uint32* m_Fnz;            // ÿһ�е���ʼ����
	uint32* m_Col;            // ÿ��Ԫ�ص�������
	uint32 m_ArrayLength;     // ������ݳ���
	uint32 m_Dimension;       // ����ά��    

public:
	CSR() = default;
	CSR(const std::string& filePathA);
	CSR(const CSR& csr);                          // �������캯����ʹ��Ĭ�ϵ�ǳ�������ɣ�ʹ������Ļ��ڵ���GPU�� kernel ���������
	CSR(CSR&& csr) noexcept;                      // �ƶ����캯����ʹ��Ĭ�ϵ�ǳ�������ɣ�ʹ������Ļ��ڵ���GPU�� kernel ���������
	CSR& operator=(const CSR& csr);               // ������ֵ�����
	CSR& operator=(CSR&& csr) noexcept;           // �ƶ���ֵ�����
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

// �������캯��
CSR::CSR(const CSR& csr)
{
	m_ArrayLength = csr.m_ArrayLength;
	m_Dimension = csr.m_Dimension;
	m_Data = new double[m_ArrayLength];
	m_Col = new uint32[m_ArrayLength];
	m_Fnz = new uint32[m_Dimension + 1];

	// ���ڶ��ǻ������ͣ�����ֱ��ʹ��memcpy
	memcpy(m_Data, csr.m_Data, m_ArrayLength * sizeof(double));
	memcpy(m_Col, csr.m_Col, m_ArrayLength * sizeof(uint32));
	memcpy(m_Fnz, csr.m_Fnz, (m_Dimension + 1) * sizeof(uint32));
}

// ������ֵ�����
CSR& CSR::operator=(const CSR& csr)
{
	if (this != &csr)
	{
		// ���ﲻҪ��գ�ֱ���� memcpy �����ǲ���Ҳ���ԣ�
		// �����Ͳ������� new �ˣ�����Ҫ��֤��������� m_Dimension �� m_ArrayLength ��ȫһ�£���Ȼ�����
		delete[] m_Data;         // ���ԭʼ���ڴ�
		delete[] m_Col;
		delete[] m_Fnz;

		m_ArrayLength = csr.m_ArrayLength;
		m_Dimension = csr.m_Dimension;
		m_Data = new double[m_ArrayLength];
		m_Col = new uint32[m_ArrayLength];
		m_Fnz = new uint32[m_Dimension + 1];

		// ���ڶ��ǻ������ͣ�����ֱ��ʹ��memcpy
		// ������Լ�д���࣬���Һ���ָ�룬��Ҫ����ѭ��һ�δε��ù��캯��
		memcpy(m_Data, csr.m_Data, m_ArrayLength * sizeof(double));
		memcpy(m_Col, csr.m_Col, m_ArrayLength * sizeof(uint32));
		memcpy(m_Fnz, csr.m_Fnz, (m_Dimension + 1) * sizeof(uint32));

	}

	return *this;
}

// �ƶ����캯��
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

// �ƶ���ֵ�����
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

// ���������ͷŷ�����ڴ棨��������˵Ļ���
CSR::~CSR()
{
	delete[] m_Data;
	delete[] m_Col;
	delete[] m_Fnz;
}




class COO
{
public:
	double* m_Data;           // ����ķ���Ԫ��ֵ
	uint32* m_Row;            // ÿ��Ԫ�ص�������
	uint32* m_Col;            // ÿ��Ԫ�ص�������
	uint32 m_ArrayLength;     // ������ݳ���
};


struct DeviceCOO
{
public:
	double* m_Data;           // ����ķ���Ԫ��ֵ
	uint32* m_Row;            // ÿ��Ԫ�ص�������
	uint32* m_Col;            // ÿ��Ԫ�ص�������
	uint32 m_ArrayLength;    // ������ݳ���
};

