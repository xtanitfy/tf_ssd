#ifndef __TYPE_LAYER_REG_H__
#define __TYPE_LAYER_REG_H__

#include "public.h"
#include "blob.h"
#include "layer.h"
#include "conv_layer.h"
#include "input_layer.h"
#include "pooling_layer.h"
#include "inner_product_layer.h"
#include "relu_layer.h"
#include "softmax_layer.h"
#include "normalize_layer.h"
#include "permute_layer.h"
#include "flatten_layer.h"
#include "prior_box_layer.h"
#include "concat_layer.h"
#include "reshape_layer.h"
#include "detection_output_layer.h"

typedef struct
{	
	char *typename;
	int (*setUp)(LAYER_t *pLayer);
	int (*reshape)(LAYER_t *pLayer);
	int (*forward)(LAYER_t *pLayer);
	int (*backward)(LAYER_t *pLayer);	
}LAYER_REGISTER_t;


LAYER_REGISTER_t gTypeLayerRegister[] = {
	{"Input",InputLayer_setUp,InputLayer_reshape,
					InputLayer_forward,InputLayer_backward},
	{"Convolution",ConvolutionLayer_setUp,ConvolutionLayer_reshape,
					ConvolutionLayer_forward,ConvolutionLayer_backward},
	{"Pooling",PoolingLayer_setUp,PoolingLayer_reshape,
					PoolingLayer_forward,PoolingLayer_backward},
	{"InnerProduct",InnerProductLayer_setUp,InnerProductLayer_reshape,
				InnerProductLayer_forward,InnerProductLayer_backward},
	{"ReLU",NULL,NULL,ReLULayer_forward,ReLULayer_backward},
	{"Softmax",NULL,SoftmaxLayer_reshape,SoftmaxLayer_forward,SoftmaxLayer_backward},
	{"Normalize",NormalizeLayer_setUp,NormalizeLayer_reshape,
					NormalizeLayer_forward,NormalizeLayer_backward},
	{"Permute",PermuteLayer_setUp,PermuteLayer_reshape,
				PermuteLayer_forward,PermuteLayer_backward},
	{"Flatten",NULL,FlattenLayer_reshape,FlattenLayer_forward,FlattenLayer_backward},
	{"PriorBox",PriorBoxLayer_setUp,PriorBoxLayer_reshape,PriorBoxLayer_forward,NULL},
	{"Concat",ConcatLayer_setUp,ConcatLayer_reshape,ConcatLayer_forward,ConcatLayer_backward},
	{"Reshape",ReshapeLayer_setUp,ReshapeLayer_reshape,ReshapeLayer_forward,ReshapeLayer_backward},
	{"DetectionOutput",DetectionOutputLayer_setUp,DetectionOutputLayer_reshape,DetectionOutputLayer_forward},
};

#endif
