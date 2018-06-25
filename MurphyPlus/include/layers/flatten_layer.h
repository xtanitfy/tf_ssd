#ifndef __FLATTEN_LAYER_H__
#define __FLATTEN_LAYER_H__

#include "blob.h"
#include "layer.h"
#include "public.h"
#include "math_functions.h"

int FlattenLayer_reshape(LAYER_t *pLayer);
int FlattenLayer_forward(LAYER_t *pLayer);
int FlattenLayer_backward(LAYER_t *pLayer);

#endif