#ifndef __RELU_LAYER_H__
#define __RELU_LAYER_H__

#include "blob.h"
#include "layer.h"

int ReLULayer_forward(LAYER_t *pLayer);
int ReLULayer_backward(LAYER_t *pLayer);

#endif