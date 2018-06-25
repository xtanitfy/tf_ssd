#include "concat_layer.h"

int ConcatLayer_setUp(LAYER_t *pLayer)
{
	CONCAT_THIS *this = (CONCAT_THIS *)malloc(sizeof(CONCAT_THIS));
	CHECK_EXPR_RET(this == NULL,-1);
	pLayer->innerParam = this;
	
#if 0
	ConcatParameter *pConcatParam = &pLayer->pLayerParam->concat_param;
	printf("pConcatParam->concat_dim:%d\n",pConcatParam->concat_dim);
	printf("pConcatParam->axis:%d\n",pConcatParam->axis);
	CHECK_EXPR_RET(pConcatParam->axis != 0 && pConcatParam->concat_dim != 0,-1);
#endif

	return 0;
}

int ConcatLayer_reshape(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	CONCAT_THIS *this = (CONCAT_THIS *)pLayer->innerParam;

	int num_axes = BLOB_num_axes(pLayer->bottom[0]);
	ConcatParameter *pConcatParam = &pLayer->pLayerParam->concat_param;

	this->concat_axis_ = BLOB_CanonicalAxisIndex(pLayer->bottom[0],pConcatParam->axis);

	int top_shape[BLOB_MAX_AXES];
	int top_shape_size = 0;
	BLOB_shape(pLayer->bottom[0],top_shape,&top_shape_size);
	this->num_concats_ = BLOB_countByStartAndEnd(pLayer->bottom[0],0,this->concat_axis_);
	this->concat_input_size_ = BLOB_countByStart(pLayer->bottom[0],this->concat_axis_+1);
	int bottom_count_sum = BLOB_count(pLayer->bottom[0]);
	for (int i = 1;i < pLayer->bottomCnt;i++) {
		CHECK_EXPR_RET(num_axes != BLOB_num_axes(pLayer->bottom[i]),-1);
		for (int j = 0;j < num_axes;j++) {
			if (j == this->concat_axis_) {
				continue;
			}
			CHECK_EXPR_RET(top_shape[j] != BLOB_shapeByIndex(pLayer->bottom[i],j),-1);
		}
		bottom_count_sum += BLOB_count(pLayer->bottom[i]);
		top_shape[this->num_concats_] += BLOB_shapeByIndex(pLayer->bottom[i],
											this->num_concats_);
	}
	
	BLOB_reshapeByArray(pLayer->top[0],top_shape,top_shape_size);
	CHECK_EXPR_RET(bottom_count_sum != BLOB_count(pLayer->top[0]),-1);
	if (pLayer->bottomCnt == 1) {
		BLOB_shareData(pLayer->top[0],pLayer->bottom[0]);
		BLOB_shareDiff(pLayer->top[0],pLayer->bottom[0]);
		
	}
	
	return 0;
}

int ConcatLayer_forward(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	CONCAT_THIS *this = (CONCAT_THIS *)pLayer->innerParam;
	if (pLayer->bottomCnt == 1) {
		printf("pLayer->bottomCnt == 1\n");
		getchar();
		return 0;
	}
	DATA_TYPE *top_data = BLOB_data(pLayer->top[0]);
	int offset_concat_axis = 0;
	int top_concat_axis = BLOB_shapeByIndex(pLayer->top[0],this->concat_axis_);

	for (int i = 0; i < pLayer->bottomCnt; i++) {
		DATA_TYPE *bottom_data = BLOB_data(pLayer->bottom[i]);
		int bottom_concat_axis = BLOB_shapeByIndex(pLayer->bottom[i],this->concat_axis_);
		for (int n = 0; n < this->num_concats_; n++) {
			Murphy_copy(bottom_concat_axis * this->concat_input_size_,
							bottom_data + n * bottom_concat_axis * this->concat_input_size_,
			top_data + (n * top_concat_axis + offset_concat_axis)* this->concat_input_size_);
		}
		offset_concat_axis += bottom_concat_axis;
	}

	return 0;
}

int ConcatLayer_backward(LAYER_t *pLayer)
{
	return 0;
}

