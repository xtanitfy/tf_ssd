#ifndef __IM2COL_H__
#define __IM2COL_H__

#include "public.h"


void im2col_cpu(const DATA_TYPE* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    DATA_TYPE* data_col);




#endif