#include "math_functions.h"

//#define DATA_TYPE_FLOAT 1
#define DATA_TYPE_DOUBLE 1

#if defined(DATA_TYPE_FLOAT)
typedef float DATA_TYPE;
#else
typedef double DATA_TYPE;	
#endif

void printMatrix(char *title,DATA_TYPE *pData,int m,int n)
{
	int i = 0,j = 0;
	printf("%s\n",title);
	DATA_TYPE (*ptr)[n] = (DATA_TYPE(*)[n])pData;
	for (i = 0;i < m;i++) {
		for (j = 0;j < n;j++) {
			printf("%.8f ",ptr[i][j]);
		}
		printf("\n");
	}
	
}

DATA_TYPE A[4] = {2,3,4,5};
DATA_TYPE B[2] = {1,2};
DATA_TYPE C[2];
DATA_TYPE Y[4];

int main(void)
{	
	#if defined(DATA_TYPE_FLOAT)
		Murphy_gemm_f(CblasNoTrans,CblasNoTrans,2,1,2,1,A,B,0,C);
		printMatrix("After Murphy_sgemm",C,2,1);
		
		Murphy_gemv_f(CblasNoTrans,2,2,1,A,B,0,C);
		printMatrix("After Murphy_sgemv",C,2,1);
		
		Murphy_axpy_f(4,0.1,A,Y);
		printMatrix("After Murphy_saxpy",Y,2,2);
		
		
		
	#else
		Murphy_gemm_d(CblasNoTrans,CblasNoTrans,2,1,2,1,A,B,0,C);
		printMatrix("After Murphy_dgemm",C,2,1);
		
		Murphy_gemv_d(CblasNoTrans,2,2,1,A,B,0,C);
		printMatrix("After Murphy_dgemv",C,2,1);
		
		Murphy_axpy_d(4,0.1,A,Y);
		printMatrix("After Murphy_daxpy",Y,2,2);
	#endif

	
	return 0;
}



