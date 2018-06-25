#ifndef __PUBLIC_H__
#define __PUBLIC_H__

#include <stdio.h>
#include <string.h> 
#include <stdlib.h>
#include "data_type.h"

#define ACCURACY_FLOAT //ACCURACY_DOUBLE //  //ACCURACY_BONARY

#if defined(ACCURACY_FLOAT)
typedef float DATA_TYPE;

#elif defined(ACCURACY_DOUBLE)
typedef double DATA_TYPE;	
#else 
//not support
#endif

#define OUT_DIR "out"

#define CHECK_EXPR_RET(expr,errNo) if(expr) { printf("[%s][%s:%d] error!\n",__FILE__,__FUNCTION__,__LINE__); \
											getchar(); \
											return errNo; \
										}

#define CHECK_EXPR_NORET(expr) if(expr) \
										{ \
											printf("[%s][%s:%d] error!\n",__FILE__,__FUNCTION__,__LINE__); \
											getchar(); \
											return; \
										}
#define DIM_OF(a) (sizeof(a)/sizeof(a[0]))
#define VOS_MAX(x,y) ((x > y) ? x: y)
#define VOS_MIN(x,y) ((x > y) ? y: x)

#endif
