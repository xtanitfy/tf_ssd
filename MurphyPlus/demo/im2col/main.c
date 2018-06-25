#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/*
void im2col_cpu(const DATA_TYPE* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    DATA_TYPE* data_col) 
*/
/*
//oc=3 ic=2 oh=ow=2 ih=iw=3 kh=kw=2 weights:[oc,ic*kh*kw] col:[ic*kh*kw,oh*ow]
原数据：
	1 2 3 
	4 5 6
	7 8 9
	
	10 11 12
	13 14 15 
	16 17 18
	
展开之后：
  1  2  4  5 ;
  2  3  5  6; 
  4  5  7  8; 
  5  6  8  9; 
  10 11 13 14;
  11 12 14 15; 
  13 14 16 17; 
  14 15 17 18
*/


int data[2][3][3] = {
	{1,2,3, 4,5,6, 7,8,9},
	{10,11,12, 13,14,15, 16,17,18}
};


int main(void)
{
	int height = 3;
	int width = 3;
	int kernel_h = 2;
	int kernel_w = 2;
	int pad_h = 0;
	int pad_w = 0;
	int stride_h = 1;
	int stride_w = 1;
	int dilation_h = 1;
	int dilation_w = 1;
	int channels = 2;
	
	const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	int channel_size = height * width;
	unsigned long outSize = channels * kernel_h * kernel_w * output_h * output_w;
	int *outData = (int *)malloc(sizeof(int) * outSize);
	
	im2col_cpu(data,channels,height,width,kernel_h,kernel_w, \
					pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,outData);
	
	for (int i = 0;i < outSize;i++) {
		printf("%d ",outData[i]);
	}
	printf("\n");
	return 0;
}

