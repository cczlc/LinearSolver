#pragma once
#include "LinearSolver.h"

class S_CG : public LinearSolver
{
public:
	S_CG(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);
	S_CG(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);

protected:
	void calculate() override;
};


S_CG::S_CG(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: LinearSolver(data, x, b, dimension, maxIter, residual) {}

S_CG::S_CG(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: LinearSolver(data, x, b, dimension, maxIter, residual) {}

void S_CG::calculate() 
{
#if TIME_TEST
	Timer tc("calculate");
#endif

	double* r = new double[m_Dimension];
	double* ap = new double[m_Dimension];
	double* newR = new double[m_Dimension];
	double* p = new double[m_Dimension];

	// ����Ax
	Matrix_multi_Vector(m_DataCSR, m_X, ap);

	// r = b - Ax��r�ǲв����
	// ��ʼ�в�ֵ
	for (uint32 i = 0; i < m_Dimension; ++i)
	{
		r[i] = m_B[i] - ap[i];
	}

	// ��ʼ������Ϊ���ݶȷ���
	for (uint32 i = 0; i < m_Dimension; ++i)
		p[i] = r[i];

	double pAp = 0.0;
	double rr = 0;
	double alpha = 0.0;
	double newRnewR = 0.0;
	double beta = 0.0;

	// �����ݶȷ�����Ӧ���ǹ̶��ĵ���������
	for (uint32 i = 0; i < m_MaxIter; ++i)
	{
		// ����Ap
		Matrix_multi_Vector(m_DataCSR, p, ap);

		// ����pAp
		pAp = Vector_mutil_Vector(ap, p, m_Dimension);

		// ����rr
		rr = Vector_mutil_Vector(r, r, m_Dimension);

		alpha = rr / pAp;

		for (uint32 i = 0; i < m_Dimension; ++i)
		{
			// x = x + alpha * d
			m_X[i] += alpha * p[i];

			// newR = r - alpha * Ad
			newR[i] = r[i] - alpha * ap[i];
		}

		if ((maxnorm(m_Dimension, r) <= m_MinResidual))
		{
			m_Iter = i;
			break;
		}

		newRnewR = Vector_mutil_Vector(newR, newR, m_Dimension);
		beta = newRnewR / rr;

		for (uint32 i = 0; i < m_Dimension; ++i)
		{
			// p = r + beta * p
			p[i] = newR[i] + beta * p[i];
			r[i] = newR[i];
		}
	}

	delete[] r;
	delete[] ap;
	delete[] newR;

}