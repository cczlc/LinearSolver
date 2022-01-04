#pragma once
#include "Typedef.h"
class DataGenration
{
public:
	DataGenration();
	~DataGenration();

	float* getData();
	uint32* getFnz();
	uint32* getClm();

private:
	float* m_Data;
	uint32* m_Fnz;
	uint32* m_Clm;

	uint32 m_NumPerCow;
};

DataGenration::DataGenration()
{
}

DataGenration::~DataGenration()
{
}