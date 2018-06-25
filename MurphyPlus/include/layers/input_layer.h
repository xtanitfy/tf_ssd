#ifndef __INPUT_LAYER_H__
#define __INPUT_LAYER_H__
#include "layer.h"
#include "dlist.h"

int InputLayer_setUp(LAYER_t *pLayer);
int InputLayer_reshape(LAYER_t *pLayer);
int InputLayer_forward(LAYER_t *pLayer);
int InputLayer_backward(LAYER_t *pLayer);

#endif
