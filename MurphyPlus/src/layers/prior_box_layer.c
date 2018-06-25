#include "prior_box_layer.h"

static void initPriorParam(PRIORBOX_THIS *this)
{
	memset(this,'\0',sizeof(PRIORBOX_THIS));
}

int PriorBoxLayer_setUp(LAYER_t *pLayer)
{
	PRIORBOX_THIS *this = (PRIORBOX_THIS *)malloc(sizeof(PRIORBOX_THIS));
	CHECK_EXPR_RET(this == NULL,-1);
	pLayer->innerParam = this;
	initPriorParam(this);

	PriorBoxParameter *pPriorBoxParam = &pLayer->pLayerParam->prior_box_param;
	CHECK_EXPR_RET(pPriorBoxParam->min_size <= 0,-1)

	this->min_size_ = pPriorBoxParam->min_size;
	this->max_size_ = -1;
	this->aspect_ratios_size = 0;
	this->aspect_ratios_[this->aspect_ratios_size++] = 1.;
	this->flip_ = pPriorBoxParam->flip;

	BOOL isExist = FALSE;
	for (int i = 0;i < pPriorBoxParam->aspect_ratio_size;i++) {
		float ar = 	pPriorBoxParam->aspect_ratio[i];
		isExist = FALSE;
		for (int j = 0;j < this->aspect_ratios_size;j++) {
			if (Murphy_fabs(ar - this->aspect_ratios_[j]) < 1e-6) {
				isExist = TRUE;
				break;
			}
		}
		if (isExist == FALSE) {
			this->aspect_ratios_[this->aspect_ratios_size++] = ar;
			if (this->flip_ == TRUE) {
				this->aspect_ratios_[this->aspect_ratios_size++] = 1./ar;
			}
		}
	}
	
	this->num_priors_ = this->aspect_ratios_size;
	if (pPriorBoxParam->max_size > 0) {
		this->max_size_ = pPriorBoxParam->max_size;
		CHECK_EXPR_RET(this->min_size_ >= this->max_size_,-1);
		this->num_priors_ += 1;
	}
	this->clip_ = pPriorBoxParam->clip;
	if (pPriorBoxParam->variance_size > 1) {
		CHECK_EXPR_RET(pPriorBoxParam->variance_size != 4,-1);
		for (int i = 0;i < pPriorBoxParam->variance_size;i++) {
			CHECK_EXPR_RET(pPriorBoxParam->variance[i] <= 0,-1);
			this->variance_[this->variance_size++] = pPriorBoxParam->variance[i];
		}
	} else if (pPriorBoxParam->variance_size == 1) {
		this->variance_[this->variance_size++] = pPriorBoxParam->variance[0];
	} else {
		this->variance_[this->variance_size++] = 0.1;
	}
	
	return 0;
}

int PriorBoxLayer_reshape(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	PRIORBOX_THIS *this = pLayer->innerParam;

	int layer_width = BLOB_width(pLayer->bottom[0]);
	int layer_height = BLOB_height(pLayer->bottom[0]);
	int top_shape[3];
	top_shape[0] = 1;
	top_shape[1] = 2;
	top_shape[2] = layer_width * layer_height * this->num_priors_* 4;
	CHECK_EXPR_RET(top_shape[2] <= 0,-1);
	BLOB_reshapeByArray(pLayer->top[0],top_shape,3);

	return 0;
}

int PriorBoxLayer_forward(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	PRIORBOX_THIS *this = pLayer->innerParam;
	int layer_width = BLOB_width(pLayer->bottom[0]);
	int layer_height = BLOB_height(pLayer->bottom[0]);
	int img_width = BLOB_width(pLayer->bottom[1]);
	int img_height = BLOB_height(pLayer->bottom[1]);

	float step_x = (float)img_width / (float)layer_width;
	float step_y = (float)img_height / (float)layer_height;

	DATA_TYPE *top_data = BLOB_data(pLayer->top[0]);
	int dim = layer_height * layer_width * this->num_priors_ * 4;
	int idx = 0;
	bool bprint_once = true;
	for (int h = 0; h < layer_height; ++h) {
		for (int w = 0; w < layer_width; ++w) {
			float center_x = (w + 0.5) * step_x;
			float center_y = (h + 0.5) * step_y;
			float box_width, box_height;
			// first prior: aspect_ratio = 1, size = min_size
			box_width = box_height = this->min_size_;
			// xmin
			top_data[idx++] = (center_x - box_width / 2.) / (float)img_width;
			// ymin
			top_data[idx++] = (center_y - box_height / 2.) / (float)img_height;
			// xmax
			top_data[idx++] = (center_x + box_width / 2.) / (float)img_width;
			// ymax
			top_data[idx++] = (center_y + box_height / 2.) / (float)img_height;
			
			if (bprint_once == true) { 
				printf("[0 PriorBoxLayer] box_width:%.12f box_height:%.12f\n",box_width,box_height);
	  		}
			if (this->max_size_ > 0) {
				// second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
				box_width = box_height = Murphy_sqrt(this->min_size_ * this->max_size_);
				//printf("box_width:%.12f\n",box_width);
			//	getchar();

				// xmin
				top_data[idx++] = (center_x - box_width / 2.) / (float)img_width;
				// ymin
				top_data[idx++] = (center_y - box_height / 2.) / (float)img_height;
				// xmax
				top_data[idx++] = (center_x + box_width / 2.) / (float)img_width;
				// ymax
				top_data[idx++] = (center_y + box_height / 2.) / (float)img_height;

				if (bprint_once == true) { 
					printf("[1 PriorBoxLayer] box_width:%.12f box_height:%.12f\n",box_width,box_height);
	  			}
			}

		// rest of priors
			for (int r = 0; r < this->aspect_ratios_size; ++r) {
				float ar = this->aspect_ratios_[r];
				if (Murphy_fabs(ar - 1.) < 1e-6) {
					continue;
				}
				box_width = this->min_size_ * Murphy_sqrt(ar);
				box_height = this->min_size_ / Murphy_sqrt(ar);
				
				// xmin
				top_data[idx++] = (center_x - box_width / 2.) / (float)img_width;
				// ymin
				top_data[idx++] = (center_y - box_height / 2.) / (float)img_height;
				// xmax
				top_data[idx++] = (center_x + box_width / 2.) / (float)img_width;
				// ymax
				top_data[idx++] = (center_y + box_height / 2.) / (float)img_height;

				if (bprint_once == true) { 
					printf("[2 PriorBoxLayer] box_width:%.12f box_height:%.12f\n",box_width,box_height);
	  			}
			}
			if (bprint_once == true) {
				bprint_once = false; 
		 	}
		}
	}
	if (this->clip_) {
	   for (int d = 0; d < dim; ++d) {
		 top_data[d] = VOS_MIN(VOS_MAX(top_data[d], 0.), 1.);
	   }
	 }
	 // set the variance.
	int pIndices[2];
	pIndices[0] = 0;
	pIndices[1] = 1;
	top_data += BLOB_offsetByIndices(pLayer->top[0],pIndices,2);
	//top_data += pLayer->top[0]->offset(0, 1);
	if (this->variance_size == 1) {
		Murphy_set(dim, (DATA_TYPE)(this->variance_[0]), top_data);
	} else {
		int count = 0;
		for (int h = 0; h < layer_height; ++h) {
			for (int w = 0; w < layer_width; ++w) {
				for (int i = 0; i < this->num_priors_; ++i) {
					for (int j = 0; j < 4; ++j) {
						top_data[count] = this->variance_[j];
						++count;
					}
				}
			}
		}
	}

	return  0;
}

