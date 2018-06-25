#include "permute_layer.h"

static void Permute(const int count, DATA_TYPE *bottom_data, const bool forward,
	const int* permute_order, const int* old_steps, const int* new_steps,
		const int num_axes, DATA_TYPE *top_data);

static void initPermuteInnerParam(PERMUTE_THIS *this)
{
	memset(this,'\0',sizeof(PERMUTE_THIS));
}

int PermuteLayer_setUp(LAYER_t *pLayer)
{
	PERMUTE_THIS *this = (PERMUTE_THIS *)malloc(sizeof(PERMUTE_THIS));
	CHECK_EXPR_RET(this == NULL,-1);
	initPermuteInnerParam(this);
	pLayer->innerParam = this;
	PermuteParameter *pPermuteParam = &pLayer->pLayerParam->permute_param;
	CHECK_EXPR_RET(pLayer->bottomCnt != 1,-1);
	
	this->num_axes_ = BLOB_num_axes(pLayer->bottom[0]);

	int orders[BLOB_MAX_AXES];
	int orders_size = 0;

	BOOL isExist = FALSE;
	for (int i = 0;i < pPermuteParam->order_size;i++) {
		isExist = FALSE;
		for (int j = 0;j < orders_size;j++) {
			if (orders[j] == pPermuteParam->order[i]) {
				isExist = TRUE;
				break;
			}
		} 
		CHECK_EXPR_RET(isExist == TRUE,-1);
		if (isExist == FALSE) {
			orders[orders_size++] = pPermuteParam->order[i];
		}
	}

	for (int i = 0;i < this->num_axes_;i++) {
		isExist = FALSE;
		for (int j = 0;j < orders_size;j++) {
			if (orders[j] == i) {
				isExist = TRUE;
				break;
			}
		}
		if (isExist == FALSE) {
			orders[orders_size++] = i;
		}
	}
	CHECK_EXPR_RET(this->num_axes_ != orders_size,-1);

	this->need_permute_ = FALSE;
	for (int i = 0;i < this->num_axes_;i++) {
		if (i != orders[i]) {
			this->need_permute_ = TRUE;
			break;
		}
	}

	int top_shape[BLOB_MAX_AXES];
	int top_shape_size = 0;
	this->permute_order_size = this->num_axes_;
	this->old_steps_size = this->num_axes_;
	this->new_steps_size = this->num_axes_;
	for (int i = 0;i < this->num_axes_;i++) {
		this->permute_order_[i] = orders[i];
		top_shape[top_shape_size++] = orders[i];
	}
	BLOB_reshapeByArray(pLayer->top[0],top_shape,top_shape_size);
		
	return 0;
}

int PermuteLayer_reshape(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	PERMUTE_THIS *this = pLayer->innerParam;
	int top_shape[BLOB_MAX_AXES];
	int top_shape_size = 0;
	for (int i = 0;i < this->num_axes_;i++) {
		if (i == this->num_axes_ - 1) {
			this->old_steps_[i] = 1;
		} else {
			this->old_steps_[i] = BLOB_countByStart(pLayer->bottom[0],i+1);
		}
		top_shape[top_shape_size++] = BLOB_shapeByIndex(pLayer->bottom[0],
										this->permute_order_[i]);
	}
	BLOB_reshapeByArray(pLayer->top[0],top_shape,top_shape_size);

	for (int i = 0;i < this->num_axes_;i++) {
		if (i == this->num_axes_ - 1) {
			this->new_steps_[i] = 1;
		} else {
			this->new_steps_[i] = BLOB_countByStart(pLayer->top[0],i+1);
		}
	}

	return 0;
}

int PermuteLayer_forward(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	PERMUTE_THIS *this = pLayer->innerParam;

	if (this->need_permute_) {
		DATA_TYPE *bottom_data = BLOB_data(pLayer->bottom[0]);
		DATA_TYPE *top_data = BLOB_data(pLayer->top[0]);
		const int top_count = BLOB_count(pLayer->top[0]);
		const int* permute_order = this->permute_order_;
		const int* old_steps = this->old_steps_;
		const int* new_steps = this->new_steps_;
		bool forward = true;
		Permute(top_count, bottom_data, forward, permute_order, old_steps,
					new_steps, this->num_axes_, top_data);
	} else {
		// If there is no need to permute, we share data to save memory.
		BLOB_shareData(pLayer->top[0],pLayer->bottom[0]);
	}

	return 0;
}


int PermuteLayer_backward(LAYER_t *pLayer)
{
	//CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	//PERMUTE_THIS *this = pLayer->innerPara;

	return 0;
}

static void Permute(const int count, DATA_TYPE *bottom_data, const bool forward,
	const int* permute_order, const int* old_steps, const int* new_steps,
		const int num_axes, DATA_TYPE *top_data) 
{
	for (int i = 0; i < count; ++i) {
		int old_idx = 0;
		int idx = i;
		for (int j = 0; j < num_axes; ++j) {
			int order = permute_order[j];
			old_idx += (idx / new_steps[j]) * old_steps[order];
			idx %= new_steps[j];
		}
		if (forward) {
			top_data[i] = bottom_data[old_idx];
		} else {
			bottom_data[old_idx] = top_data[i];
		}
	}
}

