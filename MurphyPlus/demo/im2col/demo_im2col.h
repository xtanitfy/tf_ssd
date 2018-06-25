#ifndef __DEMO_IM2COL_H__
#define __DEMO_IM2COL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef unsigned char BOOL;
#define TRUE 1
#define FALSE 0


void im2col_cpu(const int* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    int* data_col);




#endif