#ifndef __POOLING_LAYER_H__
#define __POOLING_LAYER_H__

#include "blob.h"
#include "layer.h"

typedef struct
{
	BOOL global_pooling_;
	int kernel_h_;
	int kernel_w_;
	int pad_h_;
	int pad_w_;
	int stride_h_;
	int stride_w_;
	int height_;
	int width_;
	int pooled_height_;
	int pooled_width_;
	int channels_;
}POOL_INNER_PARAM_t;

int PoolingLayer_setUp(LAYER_t *pLayer);
int PoolingLayer_reshape(LAYER_t *pLayer);
int PoolingLayer_forward(LAYER_t *pLayer);
int PoolingLayer_backward(LAYER_t *pLayer);


#endif