#include "math_functions.h"
#include <math.h>

static void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

static void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

static void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

static void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}

static void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
	for (i = 0;i < M*N;i++) {
		 C[i] *= BETA;
	}
	#if 0
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
	#endif
	
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

static void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

//²Î¿¼ÍøÖ·http://blog.csdn.net/seven_first/article/details/47378697
#if 1
void Murphy_gemm_f(  CBLAS_TRANSPOSE TransA,
     CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}
#else

void Murphy_gemm_f(  CBLAS_TRANSPOSE TransA,
     CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  const int TA = (TransA == CblasNoTrans) ? 0: 1;
  const int TB = (TransB == CblasNoTrans) ? 0: 1;
  int ldc = N;
  gemm(TA,TB,M,N,K,alpha,(float *)A,lda,(float *)B,ldb,beta,C,ldc);
}

#endif

void Murphy_gemm_d( CBLAS_TRANSPOSE TransA,
     CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
	int lda = (TransA == CblasNoTrans) ? K : M;
	int ldb = (TransB == CblasNoTrans) ? N : K;
	cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
		ldb, beta, C, N);
}

void Murphy_gemv_f( CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

void Murphy_gemv_d( CBLAS_TRANSPOSE TransA, const int M,
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
    double* y) 
{
	//vdDiv(n, a, b, y);
	for (int i = 0;i < n;i++) {
		y[i] = a[i] / b[i];
	}
}

float Murphy_pow_f(float a,float b)
{
	return powf(a,b);
}

double Murphy_pow_d(double a,double b)
{
	return pow(a,b);
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
		y[i] = exp(a[i]);
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

void Murphy_scale_f(const int n, const float alpha, float *x) 
{
	cblas_sscal(n, alpha, x, 1);
//	cblas_scopy(n, x, 1, y, 1);
//	cblas_sscal(n, alpha, y, 1);
}

void Murphy_scale_d(const int n, const double alpha,double *x) 
{
	cblas_dscal(n, alpha, x, 1);
	//cblas_dcopy(n, x, 1, y, 1);
	//cblas_dscal(n, alpha, y, 1);
}

float Murphy_sum_f(const int n, const float* x) 
{
	return cblas_sasum(n, x, 1);
}

double Murphy_sum_d(const int n, const double* x) 
{
	return cblas_dasum(n, x, 1);
}

unsigned int Murphy_rng_rand() 
{
	//return random();
	return rand();
}


float Murphy_strided_dot_f(const int n, const float* x, const int incx,
                                    const float* y, const int incy) 
{
  return cblas_sdot(n, x, incx, y, incy);
}

float Murphy_cpu_dot_f(const int n, const float* x, const float* y) 
{
 	 return Murphy_strided_dot_f(n, x, 1, y, 1);
}


double Murphy_strided_dot_d(const int n, const double* x,
                            const int incx, const double* y, const int incy) 
{
	return cblas_ddot(n, x, incx, y, incy);
}

double Murphy_cpu_dot_d(const int n, const double* x, const double* y) 
{
	return Murphy_strided_dot_d(n, x, 1, y, 1);
}


int Murphy_ceil(double val)
{
	return (int) (val + 0.5);
}

int Murphy_floor(double val)
{
	return (int) val;
}

double Murphy_fabs(double val)
{
	return fabs(val);
}

double Murphy_sqrt(double val)
{
	return sqrt(val);
}


