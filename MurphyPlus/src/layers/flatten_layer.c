#include "flatten_layer.h"


int FlattenLayer_reshape(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->top[0] == pLayer->bottom[0],-1);
	FlattenParameter *pFlattenParam = &pLayer->pLayerParam->flatten_param;
	int start_axis = BLOB_CanonicalAxisIndex(pLayer->bottom[0],pFlattenParam->axis);
	int end_axis = BLOB_CanonicalAxisIndex(pLayer->bottom[0],pFlattenParam->end_axis);

	int top_shape[BLOB_MAX_AXES];
	int top_shape_size = 0;
	for (int i = 0;i < start_axis;i++) {
		top_shape[top_shape_size++] = BLOB_shapeByIndex(pLayer->bottom[0],i);
	}
	int flattened_dim = BLOB_countByStartAndEnd(pLayer->bottom[0],start_axis,end_axis+1);
	top_shape[top_shape_size++] = flattened_dim;
	for (int i = end_axis + 1;i < BLOB_num_axes(pLayer->bottom[0]);i++) {
		top_shape[top_shape_size++] = BLOB_shapeByIndex(pLayer->bottom[0],i);
	}
	BLOB_reshapeByArray(pLayer->top[0],top_shape,top_shape_size);
	CHECK_EXPR_RET(BLOB_count(pLayer->top[0]) != BLOB_count(pLayer->bottom[0]),-1);
	
	return 0;
}

int FlattenLayer_forward(LAYER_t *pLayer)
{
	BLOB_shareData(pLayer->top[0],pLayer->bottom[0]);

	return 0;
}

int FlattenLayer_backward(LAYER_t *pLayer)\
{
	return 0;
}

