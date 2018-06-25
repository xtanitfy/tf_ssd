#ifndef __INNER_PRODUCT_LAYER_H__
#define __INNER_PRODUCT_LAYER_H__

#include "blob.h"
#include "layer.h"


typedef struct
{
	BOOL bias_term_;
	BOOL transpose_;
	int N_;
	int K_;
	int M_;
	BLOB_t bias_multiplier_;
}IP_INNER_PARAM_t;

int InnerProductLayer_setUp(LAYER_t *pLayer);
int InnerProductLayer_reshape(LAYER_t *pLayer);
int InnerProductLayer_forward(LAYER_t *pLayer);
int InnerProductLayer_backward(LAYER_t *pLayer);

#endif