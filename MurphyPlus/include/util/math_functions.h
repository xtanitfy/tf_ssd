#ifndef __MATH_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cblas.h"
#include <math.h>

#define FLT_MIN 1.175494351e-38F
#define FLT_MAX 3.402823466e+38F 

#if  defined(ACCURACY_FLOAT)
#define Murphy_gemm Murphy_gemm_f
#define Murphy_gemv Murphy_gemv_f
#define Murphy_axpy Murphy_axpy_f
#define Murphy_set  Murphy_set_f
#define Murphy_sum  Murphy_sum_f 
#define Murphy_strided_dot Murphy_strided_dot_f 
#define Murphy_cpu_dot Murphy_cpu_dot_f 
#define Murphy_scale Murphy_scale_f
#define Murphy_copy Murphy_copy_f
#define Murphy_exp Murphy_exp_f
#define Murphy_div Murphy_div_f
#define Murphy_sqr Murphy_sqr_f
#define Murphy_powx Murphy_powx_f
#define Murphy_pow Murphy_pow_f
#define Murphy_mul Murphy_mul_f

#elif defined(ACCURACY_DOUBLE)
#define Murphy_gemm Murphy_gemm_d
#define Murphy_gemv Murphy_gemv_d
#define Murphy_axpy Murphy_axpy_d
#define Murphy_set  Murphy_set_d
#define Murphy_sum  Murphy_sum_d 
#define Murphy_strided_dot Murphy_strided_dot_d
#define Murphy_cpu_dot Murphy_cpu_dot_d
#define Murphy_scale Murphy_scale_d
#define Murphy_copy Murphy_copy_d
#define Murphy_exp Murphy_exp_d
#define Murphy_div Murphy_div_d
#define Murphy_sqr Murphy_sqr_d
#define Murphy_powx Murphy_powx_d
#define Murphy_pow Murphy_pow_d
#define Murphy_mul Murphy_mul_d

#else
	//not support
#endif

void Murphy_gemm_f(  CBLAS_TRANSPOSE TransA,
      CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);
void Murphy_gemm_d( CBLAS_TRANSPOSE TransA,
      CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C);

void Murphy_gemv_f( CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y);
void Murphy_gemv_d( CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y);
	
void Murphy_axpy_f(const int N, const float alpha, const float* X,float* Y);
void Murphy_axpy_d(const int N, const double alpha, const double* X,double* Y);

void Murphy_set_f(const int N, const float alpha, float* Y);
void Murphy_set_d(const int N, const double alpha, double* Y);

unsigned int Murphy_rng_rand();

float Murphy_sum_f(const int n, const float* x);
double Murphy_sum_d(const int n, const double* x);

float Murphy_strided_dot_f(const int n, const float* x, const int incx,
                                    const float* y, const int incy);
double Murphy_strided_dot_d(const int n, const double* x,
                            const int incx, const double* y, const int incy);

float Murphy_cpu_dot_f(const int n, const float* x, const float* y);
double Murphy_cpu_dot_d(const int n, const double* x, const double* y);

void Murphy_scale_f(const int n, const float alpha, float *x);
void Murphy_scale_d(const int n, const double alpha, double *x) ;

void Murphy_copy_f(const int N, const float* X, float* Y);
void Murphy_copy_d(const int N, const double* X, double* Y); 
int Murphy_ceil(double val);
int Murphy_floor(double val);

void Murphy_exp_f(const int n, const float* a, float* y);
void Murphy_exp_d(const int n, const double* a, double* y);

void Murphy_div_f(const int n, const float* a, const float* b,float* y) ;
void Murphy_div_d(const int n, const double* a, const double* b,double* y);

void Murphy_sqr_f(const int n, const float* a, float* y);
void Murphy_sqr_d(const int n, const double* a, double* y);

void Murphy_powx_f(const int n, const float* a, const float b,float* y);
void Murphy_powx_d(const int n, const double* a, const double b,double* y);

void Murphy_mul_f(const int n, const float* a, const float* b,float* y);
void Murphy_mul_d(const int n, const double* a, const double* b,double* y);

float Murphy_pow_f(float a,float b);
double Murphy_pow_d(double a,double b);

double Murphy_fabs(double val);
double Murphy_sqrt(double val);


#endif

