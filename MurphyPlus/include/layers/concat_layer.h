#ifndef __CONCAT_LAYER_H__
#define __CONCAT_LAYER_H__
#include "blob.h"
#include "layer.h"
#include "public.h"
#include "math_functions.h"

typedef struct
{
	int count_;
	int num_concats_;
	int concat_input_size_;
	int concat_axis_;
}CONCAT_THIS;


int ConcatLayer_setUp(LAYER_t *pLayer);
int ConcatLayer_reshape(LAYER_t *pLayer);
int ConcatLayer_forward(LAYER_t *pLayer);
int ConcatLayer_backward(LAYER_t *pLayer);

#endif