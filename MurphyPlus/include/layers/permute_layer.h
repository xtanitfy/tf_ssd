#ifndef __PERMUTE_LAYER_H__
#define __PERMUTE_LAYER_H__
#include "blob.h"
#include "layer.h"
#include "public.h"
#include "math_functions.h"

typedef struct
{
	int num_axes_;
	BOOL need_permute_;
	int permute_order_[BLOB_MAX_AXES];
	int permute_order_size;
	int old_steps_[BLOB_MAX_AXES];
	int old_steps_size;
	int new_steps_[BLOB_MAX_AXES];
	int new_steps_size;
}PERMUTE_THIS;

int PermuteLayer_setUp(LAYER_t *pLayer);
int PermuteLayer_reshape(LAYER_t *pLayer);
int PermuteLayer_forward(LAYER_t *pLayer);
int PermuteLayer_backward(LAYER_t *pLayer);

#endif