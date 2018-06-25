#include "reshape_layer.h"


static void initReshapeInnerParam(RESHAPE_THIS *this)
{
	memset(this,'\0',sizeof(RESHAPE_THIS));
}

int ReshapeLayer_setUp(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->bottom[0] == pLayer->top[0],-1);
	RESHAPE_THIS *this = (RESHAPE_THIS *)malloc(sizeof(RESHAPE_THIS));
	CHECK_EXPR_RET(this == NULL,-1);
	initReshapeInnerParam(this);
	pLayer->innerParam = this;
	this->inferred_axis_ = -1;
	this->copy_axes_size = 0;

	ReshapeParameter *pReshapeParameter = &pLayer->pLayerParam->reshape_param;
	BlobShape *pTopBlobShape = &pReshapeParameter->shape;
	int top_num_axes = pTopBlobShape->dim_size;
	this->constant_count_ = 1;
	for (int i = 0;i < top_num_axes;i++) {
		int top_dim = pTopBlobShape->dim[i];
		if (top_dim == 0) {
			this->copy_axes_[this->copy_axes_size++] = i;
		} else if (top_dim == -1) {
			this->inferred_axis_ = i;
		} else {
			this->constant_count_ *= top_dim;
		}
	}
	
	return 0;
}

int ReshapeLayer_reshape(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	RESHAPE_THIS *this = pLayer->innerParam;
	
	int input_start_axis = pLayer->pLayerParam->reshape_param.axis;
	int start_axis =  (input_start_axis >= 0) ? input_start_axis :
		BLOB_num_axes(pLayer->bottom[0]) + input_start_axis + 1;
	CHECK_EXPR_RET(start_axis < 0,-1);
	CHECK_EXPR_RET(start_axis > BLOB_num_axes(pLayer->bottom[0]),-1);

	int num_axes = pLayer->pLayerParam->reshape_param.num_axes;
	int end_axis = (num_axes == -1) ? BLOB_num_axes(pLayer->bottom[0]) : (start_axis + num_axes);
	CHECK_EXPR_RET(end_axis > BLOB_num_axes(pLayer->bottom[0]),-1);

	int num_axes_replaced = end_axis - start_axis;
	int num_axes_retained = BLOB_num_axes(pLayer->bottom[0]) - num_axes_replaced;
	BlobShape *pTopBlobShape = &pLayer->pLayerParam->reshape_param.shape;
	int num_new_axes = pTopBlobShape->dim_size;
	int top_shape[BLOB_MAX_AXES];
	int top_shape_size = 0;

	for (int i = 0;i < start_axis;i++) {
		top_shape[top_shape_size++] = BLOB_shapeByIndex(pLayer->bottom[0],i);
	}
	for (int i = 0;i < num_new_axes;i++) {
		top_shape[top_shape_size++] = pTopBlobShape->dim[i];
	}
	for (int i = end_axis; i < BLOB_num_axes(pLayer->bottom[0]); ++i) {
		top_shape[top_shape_size++] = BLOB_shapeByIndex(pLayer->bottom[0],i);
	}
	CHECK_EXPR_RET(top_shape_size != (num_axes_retained + num_new_axes),-1);

	for (int i = 0;i < this->copy_axes_size;i++) {
		int copy_axis_index = this->copy_axes_[i];
		CHECK_EXPR_RET(BLOB_num_axes(pLayer->bottom[0]) < start_axis + copy_axis_index,-1);
		 top_shape[start_axis + copy_axis_index] =
		 		BLOB_shapeByIndex(pLayer->bottom[0],start_axis + copy_axis_index);
	}

	if (this->inferred_axis_ > 0) {
		int explicit_count = this->constant_count_;
		explicit_count *= BLOB_countByStartAndEnd(pLayer->bottom[0],0,start_axis);
		explicit_count *= BLOB_countByStart(pLayer->bottom[0],end_axis);
		for (int i = 0; i < this->copy_axes_size; ++i) {
			const int copy_axis_index = this->copy_axes_[i];
			explicit_count *= top_shape[start_axis + copy_axis_index];
		}
		CHECK_EXPR_RET(BLOB_count(pLayer->bottom[0]) % explicit_count != 0,-1);
		int inferred_dim = BLOB_count(pLayer->bottom[0]) / explicit_count;
		  top_shape[start_axis + this->inferred_axis_] = inferred_dim;
	}

	BLOB_reshapeByArray(pLayer->top[0],top_shape,top_shape_size);
	CHECK_EXPR_RET(BLOB_count(pLayer->top[0]) != BLOB_count(pLayer->bottom[0]),-1);
	
	
	
	return 0;
}

int ReshapeLayer_forward(LAYER_t *pLayer)
{
	BLOB_shareData(pLayer->top[0],pLayer->bottom[0]);
	
	return 0;
}

int ReshapeLayer_backward(LAYER_t *pLayer)
{
	BLOB_shareDiff(pLayer->top[0],pLayer->bottom[0]);
	return 0;
}

