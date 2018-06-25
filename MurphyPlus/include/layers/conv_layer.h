#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "blob.h"
#include "layer.h"

typedef struct
{
	int channel_axis_;
	int num_spatial_axes_;
	int kernel_shape_[BLOB_MAX_AXES];
	int kernel_shape_size_;
	int stride_[BLOB_MAX_AXES];
	int stride_size_;
	int pad_[BLOB_MAX_AXES];
	int pad_size_ ;
	int dilation_[BLOB_MAX_AXES];
	int dilation_size_;
	BOOL is_1x1_;
	int channels_;
	int num_output_;
	int group_;
	int conv_out_channels_;
	int conv_in_channels_;
	BOOL bias_term_;
	int kernel_dim_;
	int weight_offset_;
	int num_;
	int bottom_shape_[BLOB_MAX_AXES];
	int bottom_shape_size_;
	int output_shape_[BLOB_MAX_AXES];
	int output_shape_size_;
	int conv_out_spatial_dim_;
	int col_offset_;
	int output_offset_;
	int conv_input_shape_[BLOB_MAX_AXES];
	int conv_input_shape_size_;
	int col_buffer_shape_[BLOB_MAX_AXES];
	int col_buffer_shape_size_;
	BLOB_t col_buffer_;
	int bottom_dim_;
	int top_dim_;
	int num_kernels_im2col_;
	int num_kernels_col2im_;
	int out_spatial_dim_;
	BLOB_t bias_multiplier_;	
}CONV_INNER_PARAM_t;

int ConvolutionLayer_setUp(LAYER_t *pLayer);
int ConvolutionLayer_reshape(LAYER_t *pLayer);
int ConvolutionLayer_forward(LAYER_t *pLayer);
int ConvolutionLayer_backward(LAYER_t *pLayer);

#endif