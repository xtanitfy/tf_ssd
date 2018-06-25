#ifndef __NORMALIZE_LAYER_H__
#define __NORMALIZE_LAYER_H__

#include "blob.h"
#include "layer.h"

typedef struct
{
	BLOB_t buffer_;
	BLOB_t buffer_channel_;
	BLOB_t buffer_spatial_;
	BOOL across_spatial_;
	BLOB_t norm_;
	float eps_ ;
	BLOB_t sum_channel_multiplier_;
	BLOB_t sum_spatial_multiplier_;
	BOOL channel_shared_;
}NORMALIZE_INNER_PARAM_t;


int NormalizeLayer_setUp(LAYER_t *pLayer);
int NormalizeLayer_reshape(LAYER_t *pLayer);
int NormalizeLayer_forward(LAYER_t *pLayer);
int NormalizeLayer_backward(LAYER_t *pLayer);

#endif