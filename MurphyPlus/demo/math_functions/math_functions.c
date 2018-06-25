#include "math_functions.h"
#include <math.h>

//²Î¿¼ÍøÖ·http://blog.csdn.net/seven_first/article/details/47378697

void Murphy_gemm_f( CBLAS_TRANSPOSE TransA,
     CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

void Murphy_gemm_d(CBLAS_TRANSPOSE TransA,
     CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
		ldb, beta, C, N);
}

void Murphy_gemv_f(CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void Murphy_gemv_d(CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
	cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void Murphy_axpy_f(const int N, const float alpha, const float* X,float* Y) 
{
	cblas_saxpy(N, alpha, X, 1, Y, 1); 
}

void Murphy_axpy_d(const int N, const double alpha, const double* X,double* Y) 
{ 
	cblas_daxpy(N, alpha, X, 1, Y, 1); 
}

void Murphy_set_f(const int N, const float alpha, float* Y) 
{
  if (alpha == 0) {
    memset(Y, 0, sizeof(float) * N);  // NOLINT(Murphy/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

void Murphy_set_d(const int N, const double alpha, double* Y) 
{
  if (alpha == 0) {
    memset(Y, 0, sizeof(double) * N);  // NOLINT(Murphy/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

void Murphy_add_scalar_f(const int N, const float alpha, float* Y)
{
	for (int i = 0; i < N; ++i) {
		Y[i] += alpha;
	}
}

void Murphy_add_scalar_d(const int N, const double alpha, double* Y) 
{
	for (int i = 0; i < N; ++i) {
		Y[i] += alpha;
	}
}

void Murphy_copy_f(const int N, const float* X, float* Y) 
{
	if (X != Y) {
		memcpy(Y, X, sizeof(float) * N);  // NOLINT(Murphy/alt_fn)
	}
}

void Murphy_copy_d(const int N, const double* X, double* Y) 
{
	if (X != Y) {
		memcpy(Y, X, sizeof(double) * N);  // NOLINT(Murphy/alt_fn)
	}
}

void Murphy_scal_f(const int N, const float alpha, float *X) 
{
	cblas_sscal(N, alpha, X, 1);
}

void Murphy_scal_d(const int N, const double alpha, double *X) 
{
	cblas_dscal(N, alpha, X, 1);
}

void Murphy_saxpby(const int N, const float alpha, const float* X,
                            const float beta, float* Y) 
{
	cblas_sscal(N, beta, Y, 1);
	cblas_saxpy(N, alpha, X, 1, Y, 1);
}

void Murphy_daxpby(const int N, const double alpha, const double* X,
                             const double beta, double* Y) 
{
	cblas_dscal(N, beta, Y, 1);
	cblas_daxpy(N, alpha, X, 1, Y, 1);
}

void Murphy_add_f(const int n, const float* a, const float* b,
    float* y) {
	//vsAdd(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = a[i] + b[i];
	}
}

void Murphy_add_d(const int n, const double* a, const double* b,
    double* y) {
	//vdAdd(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = a[i] + b[i];
	}
}

void Murphy_sub_f(const int n, const float* a, const float* b,
    float* y) {
	//vsSub(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = a[i] - b[i];
	}
}

void Murphy_sub_d(const int n, const double* a, const double* b,
    double* y) {
	//vdSub(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = a[i] - b[i];
	}
}

void Murphy_mul_f(const int n, const float* a, const float* b,
    float* y) {
	//vsMul(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = a[i] * b[i];
	}
}

void Murphy_mul_d(const int n, const double* a, const double* b,
    double* y) {
	//vdMul(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = a[i] * b[i];
	}
}

void Murphy_div_f(const int n, const float* a, const float* b,
    float* y) {
	//vsDiv(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = a[i] / b[i];
	}
}

void Murphy_div_d(const int n, const double* a, const double* b,
    double* y) {
	//vdDiv(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = a[i] / b[i];
	}
}

void Murphy_powx_f(const int n, const float* a, const float b,
    float* y) {
	//vsPowx(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = powf(a[i],b);
	}
}

void Murphy_powx_d(const int n, const double* a, const double b,
    double* y) {
	//vdPowx(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = pow(a[i],b);
	}
}

void Murphy_sqr_f(const int n, const float* a, float* y) {
	//vsSqr(n, a, y);
	for (int i = 0;i < n;i++) {
		y[i] = powf(a[i],2);
	}
}


void Murphy_sqr_d(const int n, const double* a, double* y) {
	//vdSqr(n, a, y);
	for (int i = 0;i < n;i++) {
		y[i] = pow(a[i],2);
	}
}


void Murphy_exp_f(const int n, const float* a, float* y) {
	//vsExp(n, a, y);
	for (int i = 0;i < n;i++) {
		y[i] = expf(a[i]);
	}
}

void Murphy_exp_d(const int n, const double* a, double* y) {
	//vdExp(n, a, y);
	for (int i = 0;i < n;i++) {
		y[i] = exp(a[i]);
	}
}

void Murphy_log_f(const int n, const float* a, float* y) {
	//vsLn(n, a, y);
	for (int i = 0;i < n;i++) {
		y[i] = logf(a[i]);
	}
}

void Murphy_log_d(const int n, const double* a, double* y) {
	//vdLn(n, a, y);
	for (int i = 0;i < n;i++) {
		y[i] = log(a[i]);
	}
}

void Murphy_abs_f(const int n, const float* a, float* y) {
    for (int i = 0;i < n;i++) {
		y[i] = fabsf(a[i]);
	}
}

void Murphy_abs_d(const int n, const double* a, double* y) {
    //vdAbs(n, a, y);
	for (int i = 0;i < n;i++) {
		y[i] = fabs(a[i]);
	}
}

void Murphy_scale_f(const int n, const float alpha, const float *x,
                            float* y) {
	cblas_scopy(n, x, 1, y, 1);
	cblas_sscal(n, alpha, y, 1);
}

void Murphy_scale_d(const int n, const double alpha, const double *x,
                             double* y) {
	cblas_dcopy(n, x, 1, y, 1);
	cblas_dscal(n, alpha, y, 1);
}


unsigned int Murphy_rng_rand() {
	random();
}

