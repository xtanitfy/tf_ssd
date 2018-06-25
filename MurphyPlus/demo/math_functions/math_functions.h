#ifndef __MATH_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cblas.h"

/*
	功能： C=alpha*A*B+beta*C 
	A,B,C 是输入矩阵（一维数组格式） 
	CblasRowMajor :数据是行主序的（二维数据也是用一维数组储存的） 
	TransA, TransB：是否要对A和B做转置操作（CblasTrans CblasNoTrans） 
	M： A、C 的行数 
	N： B、C 的列数 
	K： A 的列数， B 的行数 
	lda ： A的列数（不做转置）行数（做转置） 
	ldb： B的列数（不做转置）行数（做转置）
*/
void Murphy_gemm_f( enum CBLAS_TRANSPOSE TransA,
     enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);
	
void Murphy_gemm_d(enum CBLAS_TRANSPOSE TransA,
     enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C);

/*
	功能： y=alpha*A*x+beta*y 
	其中X和Y是向量，A 是矩阵 
	M：A 的行数 
	N：A 的列数 
	cblas_sgemv 中的 参数1 表示对X和Y的每个元素都进行操作
*/

void Murphy_gemv_f(enum CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y);

void Murphy_gemv_d(enum CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y);
	

/*
	功能： Y=alpha*X+Y 
	N：为X和Y中element的个数
*/
void Murphy_axpy_f(const int N, const float alpha, const float* X,float* Y);
void Murphy_axpy_d(const int N, const double alpha, const double* X,double* Y);

void Murphy_set_f(const int N, const float alpha, float* Y);
void Murphy_set_d(const int N, const double alpha, double* Y);


#endif




