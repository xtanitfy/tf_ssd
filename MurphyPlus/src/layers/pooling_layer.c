#include "pooling_layer.h"
#include "math_functions.h"

static void initInnerParam(POOL_INNER_PARAM_t *pParam)
{
	memset(pParam,'\0',sizeof(POOL_INNER_PARAM_t));
	pParam->global_pooling_ = FALSE;
}

int PoolingLayer_setUp(LAYER_t *pLayer)
{
	printf("%s setup\n",pLayer->pLayerParam->name);

	POOL_INNER_PARAM_t *this = (POOL_INNER_PARAM_t *)malloc(sizeof(POOL_INNER_PARAM_t));
	CHECK_EXPR_RET(this == NULL,-1);
	pLayer->innerParam = this;
	initInnerParam(this);

	LayerParameter *pLayerParameter = pLayer->pLayerParam;
	PoolingParameter *pPoolParam = &pLayerParameter->pooling_param;
	 
  	if (pPoolParam->global_pooling == TRUE) {
		//With Global_pooling: true Filter size cannot specified
		CHECK_EXPR_RET(pPoolParam->kernel_size != 0  \
							|| pPoolParam->kernel_h != 0  \
								|| pPoolParam->kernel_w != 0 , -1);
		
	} else {
		//Filter size is kernel_size OR kernel_h and kernel_w; not both
		CHECK_EXPR_RET(pPoolParam->kernel_size != 0  \
							&& (pPoolParam->kernel_h != 0 || pPoolParam->kernel_w != 0),-1);
		CHECK_EXPR_RET(pPoolParam->kernel_size == 0 \
						&& ((pPoolParam->kernel_h != 0 && pPoolParam->kernel_w == 0)  \
							|| (pPoolParam->kernel_h == 0 && pPoolParam->kernel_w != 0)),-1);
	}

	CHECK_EXPR_RET(pPoolParam->pad != 0  \
					&& (pPoolParam->pad_h != 0 || pPoolParam->pad_w != 0),-1);
	CHECK_EXPR_RET(pPoolParam->pad == 0  \
					&&	((pPoolParam->pad_h != 0 && pPoolParam->pad_w == 0) || \
							(pPoolParam->pad_h == 0 && pPoolParam->pad_w != 0)),-1);

	CHECK_EXPR_RET(pPoolParam->stride != 0  \
					&& (pPoolParam->stride_h != 0 || pPoolParam->stride_w != 0),-1);
	CHECK_EXPR_RET(pPoolParam->stride == 0  \
					&&	((pPoolParam->stride_h != 0 && pPoolParam->stride_w == 0) || \
							(pPoolParam->stride_h == 0 && pPoolParam->stride_w != 0)),-1);

	this->global_pooling_ = pPoolParam->global_pooling;	
	if (this->global_pooling_) {				
		this->kernel_h_ = BLOB_height(pLayer->bottom[0]);
		this->kernel_w_ = BLOB_width(pLayer->bottom[0]);
	} else {
		if (pPoolParam->kernel_size != 0) {
			this->kernel_h_ = this->kernel_w_ = pPoolParam->kernel_size;
		} else {
			this->kernel_h_ = pPoolParam->kernel_h;
			this->kernel_w_ = pPoolParam->kernel_w;
		}
	}
	CHECK_EXPR_RET(this->kernel_h_ <= 0, -1);
	CHECK_EXPR_RET(this->kernel_w_ <= 0, -1);
	if (pPoolParam->pad_h == 0) {
		this->pad_h_ = this->pad_w_ = pPoolParam->pad;
	} else {
		this->pad_h_ = pPoolParam->pad_h;
		this->pad_w_ = pPoolParam->pad_w;
	}
	if (pPoolParam->stride_h == 0) {
		this->stride_h_ = this->stride_w_ = pPoolParam->stride;
	} else {
		this->stride_h_ = pPoolParam->stride_h;
		this->stride_w_ = pPoolParam->stride_w;
	}

	if (this->global_pooling_ == TRUE) {
		CHECK_EXPR_RET(!(this->pad_h_ == 0 \
							&& this->pad_w_ == 0 \
								&& this->stride_h_ == 1 \
									&& this->stride_w_ == 1), -1);
	}

	if (this->pad_h_ != 0 || this->pad_w_ != 0) {
		//Padding implemented only for average and max pooling.
		CHECK_EXPR_RET(!(pPoolParam->pool == PoolingParameter_PoolMethod_AVE || \
							pPoolParam->pool == PoolingParameter_PoolMethod_MAX), -1);
		CHECK_EXPR_RET(this->pad_h_ >= this->kernel_h_, -1);
		CHECK_EXPR_RET(this->pad_w_ >= this->kernel_w_, -1);
	}
	
	return 0;
}

int PoolingLayer_reshape(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer == NULL,-1);
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);

	//printf("%s reshape\n",pLayer->pLayerParam->name);

	POOL_INNER_PARAM_t *this = pLayer->innerParam;
	CHECK_EXPR_RET(4 != BLOB_num_axes(pLayer->bottom[0]), -1);
	this->channels_ = BLOB_channels(pLayer->bottom[0]);
	this->height_ = BLOB_height(pLayer->bottom[0]);
	this->width_ = BLOB_width(pLayer->bottom[0]);
	if (this->global_pooling_ == TRUE) {
		this->kernel_h_ = BLOB_height(pLayer->bottom[0]);
		this->kernel_w_ = BLOB_width(pLayer->bottom[0]);
	}
	#if 0
	printf("[PoolingLayer_reshape]height_:%d\n",this->height_);
	printf("[PoolingLayer_reshape]width_:%d\n",this->width_);
	printf("[PoolingLayer_reshape]pad_h_:%d\n",this->pad_h_);
	printf("[PoolingLayer_reshape]pad_w_:%d\n",this->pad_w_);
	printf("[PoolingLayer_reshape]kernel_h_:%d\n",this->kernel_h_);
	printf("[PoolingLayer_reshape]kernel_w_:%d\n",this->kernel_w_);
	printf("[PoolingLayer_reshape]stride_h_:%d\n",this->stride_h_);
	printf("[PoolingLayer_reshape]stride_w_:%d\n",this->stride_w_);
	#endif
	this->pooled_height_ = Murphy_ceil((float)(this->height_ 
					+ 2 * this->pad_h_ - this->kernel_h_) \
							/ (float)this->stride_h_) + 1;
	this->pooled_width_ = Murphy_ceil((float)(this->width_ 
					+ 2 * this->pad_w_ - this->kernel_w_) \
							/ (float)this->stride_w_) + 1;
	
	if (this->pad_h_ || this->pad_w_) {
		if ((this->pooled_height_ - 1) * this->stride_h_ \
					>= this->height_ + this->pad_h_) {
	    	--this->pooled_height_;
	    }
	    if ((this->pooled_width_ - 1) * this->stride_w_ \
					>= this->width_ + this->pad_w_) {
	    	--this->pooled_width_;
	    }
		CHECK_EXPR_RET((this->pooled_height_ - 1) * this->stride_h_ \
						>= this->height_ + this->pad_h_, -1);
		CHECK_EXPR_RET((this->pooled_width_ - 1) * this->stride_w_ \
						>= this->width_ + this->pad_w_, -1);
	}

	BLOB_reshapeByNCHW(pLayer->top[0], BLOB_num(pLayer->bottom[0]), 
						this->channels_, this->pooled_height_,
								this->pooled_width_);
	if (pLayer->topCnt > 1) {
		BLOB_reshapeLike(pLayer->top[1], pLayer->top[0]);
	}
	//BLOB_printShapeString(pLayer->top[0], "PoolingLayer_reshape top 0");
	
	return 0;
}

int PoolingLayer_forward(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer == NULL,-1);
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);

	POOL_INNER_PARAM_t *this = pLayer->innerParam;
	
	//printf("%s forward\n",pLayer->pLayerParam->name);
	DATA_TYPE *bottom_data = BLOB_data(pLayer->bottom[0]);
	DATA_TYPE *top_data = BLOB_data(pLayer->top[0]);
	int top_count = BLOB_count(pLayer->top[0]);

	LayerParameter *pLayerParameter = pLayer->pLayerParam;
	PoolingParameter *pPoolParam = &pLayerParameter->pooling_param;
	
	int bottomChannelOffset = BLOB_offsetByNCHW(pLayer->bottom[0], 0, 1, 0, 0);
	int topChannelOffset = BLOB_offsetByNCHW(pLayer->top[0], 0, 1, 0, 0);

	int channels_ = this->channels_;
	int pooled_height_ = this->pooled_height_;
	int pooled_width_ = this->pooled_width_;
	int stride_h_ = this->stride_h_;
	int stride_w_ = this->stride_w_;
	int pad_h_ = this->pad_h_;
	int pad_w_ = this->pad_w_;
	int kernel_h_ = this->kernel_h_;
	int kernel_w_ = this->kernel_w_;
	int height_ = this->height_;
	int width_ = this->width_;
	
	if (pPoolParam->pool == PoolingParameter_PoolMethod_MAX) {
		//printf("PoolingParameter_PoolMethod_MAX\n");
		Murphy_set(top_count,(DATA_TYPE)(-FLT_MAX), top_data);
		for (int n = 0; n < BLOB_num(pLayer->bottom[0]); ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int ph = 0; ph < pooled_height_; ++ph) {
					for (int pw = 0; pw < pooled_width_; ++pw) {
						int hstart = ph * stride_h_ - pad_h_;
						int wstart = pw * stride_w_ - pad_w_;
						int hend = VOS_MIN(hstart + kernel_h_, height_);
						int wend = VOS_MIN(wstart + kernel_w_, width_);
						hstart = VOS_MAX(hstart, 0);
						wstart = VOS_MAX(wstart, 0);
						const int pool_index = ph * pooled_width_ + pw;
						for (int h = hstart; h < hend; ++h) {
							for (int w = wstart; w < wend; ++w) {
								int index = h * width_ + w;
								if (bottom_data[index] > top_data[pool_index]) {
									top_data[pool_index] = bottom_data[index];
								}
							}
						}
					}
				}

				bottom_data += bottomChannelOffset;
				top_data += topChannelOffset;
			}
		}
	} else if (pPoolParam->pool == PoolingParameter_PoolMethod_AVE) {
		for (int i = 0; i < top_count; ++i) {
			top_data[i] = 0;
		}
		
		for (int n = 0; n < BLOB_num(pLayer->bottom[0]); ++n) {
			for (int c = 0; c < channels_; ++c) {
				for (int ph = 0; ph < pooled_height_; ++ph) {
					for (int pw = 0; pw < pooled_width_; ++pw) {
						int hstart = ph * stride_h_ - pad_h_;
						int wstart = pw * stride_w_ - pad_w_;
						int hend = VOS_MIN(hstart + kernel_h_, height_ + pad_h_);
						int wend = VOS_MIN(wstart + kernel_w_, width_ + pad_w_);
						int pool_size = (hend - hstart) * (wend - wstart);
						hstart = VOS_MAX(hstart, 0);
						wstart = VOS_MAX(wstart, 0);
						hend = VOS_MIN(hend, height_);
						wend = VOS_MIN(wend, width_);
						for (int h = hstart; h < hend; ++h) {
							for (int w = wstart; w < wend; ++w) {
								top_data[ph * pooled_width_ + pw] +=
								    bottom_data[h * width_ + w];
							}
						}
						top_data[ph * pooled_width_ + pw] /= pool_size;
					}
				}
				// compute offset
				bottom_data += bottomChannelOffset;
				top_data += topChannelOffset;
			}
		}
	
	} else {
		//not support
	}
	//BLOB_writeTopBlobToTxtFile(pLayer->pLayerParam->name,pLayer->top,pLayer->topCnt);

	return 0;
}

int PoolingLayer_backward(LAYER_t *pLayer)
{
	printf("%s backward\n",pLayer->pLayerParam->name);

	return 0;
}


