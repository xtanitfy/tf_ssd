#include "normalize_layer.h"
#include "math_functions.h"
#include "public.h"

//static NORMALIZE_INNER_PARAM_t *this = NULL;

static void initNormInnerParam(NORMALIZE_INNER_PARAM_t *this)
{
	memset(this,'\0',sizeof(NORMALIZE_INNER_PARAM_t));
	BLOB_init(&this->buffer_spatial_);
	BLOB_init(&this->buffer_channel_);
	BLOB_init(&this->sum_channel_multiplier_);
	BLOB_init(&this->sum_spatial_multiplier_);
	BLOB_init(&this->buffer_);
}

int NormalizeLayer_setUp(LAYER_t *pLayer)
{
	NORMALIZE_INNER_PARAM_t *this = (NORMALIZE_INNER_PARAM_t *)malloc(sizeof(NORMALIZE_INNER_PARAM_t));
	CHECK_EXPR_RET(this == NULL,-1);
	
	initNormInnerParam(this);
		
	pLayer->innerParam = this;
	
	BLOB_t *pBottom = pLayer->bottom[0];
	CHECK_EXPR_RET(BLOB_num_axes(pBottom) < 2,-1);
	
	BLOB_reshapeByNCHW(&this->buffer_,1,
						BLOB_channels(pBottom),BLOB_height(pBottom),BLOB_width(pBottom));
	BLOB_reshapeByNCHW(&this->buffer_channel_,1,BLOB_channels(pBottom),1,1);
	BLOB_reshapeByNCHW(&this->buffer_spatial_,1,1,
									BLOB_height(pBottom),BLOB_width(pBottom));
	
	NormalizeParameter *pNormParam = &pLayer->pLayerParam->norm_param;
	this->across_spatial_ = pNormParam->across_spatial;
	if (this->across_spatial_) {
		BLOB_reshapeByNCHW(&this->norm_,BLOB_num(pBottom),1,1,1);
	} else {
		BLOB_reshapeByNCHW(&this->norm_,BLOB_num(pBottom),1,
								BLOB_height(pBottom),BLOB_width(pBottom));
	}

	this->eps_ = pNormParam->eps;
	
	int channels = BLOB_channels(pBottom);
	int spatial_dim = BLOB_height(pBottom) * BLOB_width(pBottom);
	BLOB_reshapeByNCHW(&this->sum_channel_multiplier_,1,channels,1,1);
	Murphy_set(channels,(DATA_TYPE)(1),BLOB_data(&this->sum_channel_multiplier_));
	
	BLOB_reshapeByNCHW(&this->sum_spatial_multiplier_,1,1,
						BLOB_height(pBottom),BLOB_width(pBottom));
	Murphy_set(spatial_dim,(DATA_TYPE)(1),BLOB_data(&this->sum_spatial_multiplier_));
	this->channel_shared_ = pNormParam->channel_shared;
	
	return 0;
}

int NormalizeLayer_reshape(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	NORMALIZE_INNER_PARAM_t *this = pLayer->innerParam;

	BLOB_t *pBottom = pLayer->bottom[0];
	CHECK_EXPR_RET(BLOB_num_axes(pBottom) < 2,-1);
	BLOB_t *pTop = pLayer->top[0];
	BLOB_reshapeLike(pTop,pBottom);
	BLOB_reshapeByNCHW(&this->buffer_,1,
					BLOB_channels(pBottom),BLOB_height(pBottom),BLOB_width(pBottom));
	if (this->across_spatial_ == FALSE) {
		BLOB_reshapeByNCHW(&this->norm_,BLOB_num(pBottom),1,
								BLOB_height(pBottom),BLOB_width(pBottom));
	}
	int spatial_dim = BLOB_height(pBottom) * BLOB_width(pBottom);
	if (spatial_dim != BLOB_count(&this->sum_spatial_multiplier_)) {
		BLOB_reshapeByNCHW(&this->sum_spatial_multiplier_,1,1,
						BLOB_height(pBottom),BLOB_width(pBottom));
		Murphy_set(spatial_dim,(DATA_TYPE)(1),BLOB_data(&this->sum_spatial_multiplier_));
		BLOB_reshapeByNCHW(&this->buffer_spatial_,1,1,
											BLOB_height(pBottom),BLOB_width(pBottom));
	}
	
	return 0;
}

int NormalizeLayer_forward(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	NORMALIZE_INNER_PARAM_t *this = pLayer->innerParam;

	DATA_TYPE *bottom_data = BLOB_data(pLayer->bottom[0]);
	DATA_TYPE *top_data = BLOB_data(pLayer->top[0]);
	DATA_TYPE *scale = BLOB_data(&pLayer->pWeigtsBlobs[0]);
	DATA_TYPE *buffer_data = BLOB_data(&this->buffer_);
	DATA_TYPE *norm_data = BLOB_data(&this->norm_);

	Murphy_set(BLOB_count(&this->norm_),(DATA_TYPE)this->eps_,norm_data);

	DATA_TYPE* sum_channel_multiplier = BLOB_data(&this->sum_channel_multiplier_);
	DATA_TYPE* sum_spatial_multiplier = BLOB_data(&this->sum_spatial_multiplier_);
	
	int num = BLOB_num(pLayer->bottom[0]);
	int dim = BLOB_count(pLayer->bottom[0]) / num;
	int spatial_dim = BLOB_height(pLayer->bottom[0]) * BLOB_width(pLayer->bottom[0]);
	int channels = BLOB_channels(pLayer->bottom[0]);
	BOOL across_spatial_ = this->across_spatial_;
	BOOL channel_shared_ = this->channel_shared_;
	float eps_ = this->eps_;
	
	for (int n = 0; n < num; ++n) {
		Murphy_sqr(dim, bottom_data, buffer_data);
		if (across_spatial_ == TRUE) {
			// add eps to avoid overflow
			norm_data[n] = Murphy_pow(Murphy_sum(dim, buffer_data)+eps_,(DATA_TYPE)(0.5));
			Murphy_copy(dim,bottom_data,top_data);
			Murphy_scale(dim, (DATA_TYPE)(1.0 / norm_data[n]), top_data);
		} else {
			Murphy_gemv(CblasTrans, channels, spatial_dim, (DATA_TYPE)(1),
			buffer_data, sum_channel_multiplier, (DATA_TYPE)(1),
			norm_data);
			// compute norm
			Murphy_powx(spatial_dim, norm_data, (DATA_TYPE)(0.5), norm_data);
			// scale the layer
			Murphy_gemm(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
					1, (DATA_TYPE)(1), sum_channel_multiplier, norm_data,
						(DATA_TYPE)(0), buffer_data);
			Murphy_div(dim, bottom_data, buffer_data, top_data);
			norm_data += spatial_dim;
		}
		// scale the output
		if (channel_shared_ == TRUE) {
			Murphy_scale(dim, scale[0], top_data);
		} else {
			Murphy_gemm(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
				1, (DATA_TYPE)(1), scale, sum_spatial_multiplier,
				(DATA_TYPE)(0),
				buffer_data);
			Murphy_mul(dim, top_data, buffer_data, top_data);
		}
		bottom_data += dim;
		top_data += dim;
  	}
	return 0;
}

int NormalizeLayer_backward(LAYER_t *pLayer)
{
	return 0;
}

