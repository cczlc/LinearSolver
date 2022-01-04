#pragma once
#include "LinearSolver.h"

class S_BICGSTAB : public LinearSolver
{
public:
	S_BICGSTAB(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);
	S_BICGSTAB(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual);

protected:
	void calculate() override;
};

S_BICGSTAB::S_BICGSTAB(CSR& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: LinearSolver(data, x, b, dimension, maxIter, residual) {}


S_BICGSTAB::S_BICGSTAB(COO& data, double* x, double* b, uint32 dimension, uint32 maxIter, double residual)
	: LinearSolver(data, x, b, dimension, maxIter, residual) {}

void S_BICGSTAB::calculate()
{
	double* r = new double[m_Dimension];
	double* r0_hat = new double[m_Dimension];
	double* ax = new double[m_Dimension];
	double* p = new double[m_Dimension];
	double* v = new double[m_Dimension];
	double* s = new double[m_Dimension];
	double* t = new double[m_Dimension];
	double rho0 = 1.0, alpha = 1.0, w = 1.0;
	double rho1 = 0.0, beta = 0.0;

	memset(p, 0.0, m_Dimension * sizeof(double));
	memset(v, 0.0, m_Dimension * sizeof(double));

	// Ax
	Matrix_multi_Vector(m_DataCSR, m_X, ax);

	// r - Ax
	Vector_Sub_Vector(m_B, ax, r, m_Dimension);

	//memccpy(r0_hat, r, m_Dimension * sizeof(double));
	for (uint32 i = 0; i < m_Dimension; ++i)
	{
		r0_hat[i] = r[i];
	}

	// rho1
	rho1 = Vector_mutil_Vector(r, r, m_Dimension);

	// 没有应用预处理技术
	for(uint32 i = 0; i < m_MaxIter; ++i)
	{
		if (maxnorm(m_Dimension, r) <= m_Residual)
		{
			m_Iter = i;
			break;
		}
		
		beta = (rho1 / rho0) * (alpha / w);

		for (uint32 i = 0; i < m_Dimension; ++i)
		{
			p[i] = r[i] + beta * (p[i] - w * v[i]);
		}

		// v = Ap
		Matrix_multi_Vector(m_DataCSR, p, v);

		alpha = rho1 / Vector_mutil_Vector(r0_hat, v, m_Dimension);

		for (uint32 i = 0; i < m_Dimension; ++i)
		{
			s[i] = r[i] - alpha * v[i];
		}

		// As
		Matrix_multi_Vector(m_DataCSR, s, t);

		// w
		w = Vector_mutil_Vector(t, s, m_Dimension) / Vector_mutil_Vector(t, t, m_Dimension);

		rho0 = rho1;

		for (uint32 i = 0; i < m_Dimension; ++i)
		{
			m_X[i] = m_X[i] + alpha * p[i] + w * s[i];
			r[i] = s[i] - w * t[i];
		} 

		rho1 = Vector_mutil_Vector(r0_hat, r, m_Dimension);

		for (uint32 i = 0; i < m_Dimension; i+= 1000)
		{
			std::cout << m_X[i] << " ";
		}
		std::cout << std::endl;
	}

	delete[] r;
	delete[] r0_hat;
	delete[] ax;
	delete[] p;
	delete[] v;
	delete[] s;
	delete[] t;
}