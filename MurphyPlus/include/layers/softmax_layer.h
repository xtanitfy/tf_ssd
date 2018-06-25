#ifndef __SOFTMAX_LAYER_H__
#define __SOFTMAX_LAYER_H__

#include "blob.h"
#include "layer.h"

typedef struct {
	int softmax_axis_;
	BLOB_t sum_multiplier_;
	int outer_num_;
	int inner_num_;
	BLOB_t scale_;
}SOFTMAX_INNER_PARAM_t;

int SoftmaxLayer_reshape(LAYER_t *pLayer);
int SoftmaxLayer_forward(LAYER_t *pLayer);
int SoftmaxLayer_backward(LAYER_t *pLayer);

#endif