#include "softmax_layer.h"
#include "math_functions.h"

static void initSoftmaxInnerParam(SOFTMAX_INNER_PARAM_t *this)
{
	memset(this,'\0',sizeof(SOFTMAX_INNER_PARAM_t));
}

int SoftmaxLayer_reshape(LAYER_t *pLayer)
{

	SOFTMAX_INNER_PARAM_t *this = (SOFTMAX_INNER_PARAM_t *)malloc(sizeof(SOFTMAX_INNER_PARAM_t));
	CHECK_EXPR_RET(this == NULL,-1);
	pLayer->innerParam = this;

	initSoftmaxInnerParam(this);
	
	this->softmax_axis_ = BLOB_CanonicalAxisIndex(pLayer->bottom[0],
								pLayer->pLayerParam->softmax_param.axis);
	BLOB_reshapeLike(pLayer->top[0],pLayer->bottom[0]);
	
	int mult_dims[BLOB_MAX_AXES];
	int mult_dims_size = 0;

	int softmax_axis = this->softmax_axis_;
	BLOB_t *psum_multiplier = &this->sum_multiplier_;
	mult_dims[mult_dims_size++] = BLOB_shapeByIndex(pLayer->bottom[0],softmax_axis);
	BLOB_init(psum_multiplier);
	BLOB_reshapeByArray(psum_multiplier,mult_dims,mult_dims_size);
	DATA_TYPE *multiplier_data = BLOB_data(psum_multiplier);
	Murphy_set(BLOB_count(psum_multiplier),(DATA_TYPE)1.,multiplier_data);
	
	this->outer_num_ = BLOB_countByStartAndEnd(pLayer->bottom[0],
								0,softmax_axis);
	this->inner_num_ = BLOB_countByStart(pLayer->bottom[0],
								softmax_axis+1);
	int scale_dims[BLOB_MAX_AXES];
	int scale_dims_size = 0;
	BLOB_shape(pLayer->bottom[0],scale_dims,&scale_dims_size);
	scale_dims[softmax_axis] = 1;

	BLOB_reshapeByArray(&this->scale_,scale_dims,scale_dims_size);
  
	return 0;
}

int SoftmaxLayer_forward(LAYER_t *pLayer)
{
	SOFTMAX_INNER_PARAM_t *this = (SOFTMAX_INNER_PARAM_t *)pLayer->innerParam;
	CHECK_EXPR_RET(this == NULL,-1);
	
	DATA_TYPE* bottom_data = BLOB_data(pLayer->bottom[0]);
	DATA_TYPE* top_data = BLOB_data(pLayer->top[0]);
	DATA_TYPE* scale_data = BLOB_data(&this->scale_);
	int channels = BLOB_shapeByIndex(pLayer->bottom[0],
									this->softmax_axis_);
	int dim = BLOB_count(pLayer->bottom[0]) / this->outer_num_;
	Murphy_copy(BLOB_count(pLayer->bottom[0]),bottom_data,top_data);
	int outer_num = this->outer_num_;
	int inner_num = this->inner_num_;
	BLOB_t *psum_multiplier = &this->sum_multiplier_;
#if 0
	printf("outer_num:%d\n",outer_num);
	printf("inner_num:%d\n",inner_num);
	printf("channels:%d\n",channels);
	printf("dim:%d\n",dim);
	getchar();
#endif

	for (int i = 0; i < outer_num; ++i) {
		// initialize scale_data to the first plane
		Murphy_copy(inner_num, bottom_data + i * dim, scale_data);
		//scale_data中存放每一层中的最大的值
		for (int j = 0; j < channels; j++) {
			for (int k = 0; k < inner_num; k++) {
				scale_data[k] = VOS_MAX(scale_data[k],
						bottom_data[i * dim + j * inner_num + k]);
			}
		}

		Murphy_gemm(CblasNoTrans, CblasNoTrans, channels, inner_num,
        	1, -1., BLOB_data(psum_multiplier), scale_data, 1., top_data);

		Murphy_exp(dim, top_data, top_data);

		Murphy_gemv(CblasTrans, channels, inner_num, 1.,
        		top_data, BLOB_data(psum_multiplier), 0., scale_data);

		for (int j = 0; j < channels; j++) {
			Murphy_div(inner_num, top_data, scale_data, top_data);
			top_data += inner_num;
		}
	}
	
	//BLOB_writeTopBlobToTxtFile(pLayer->pLayerParam->name,pLayer->top,pLayer->topCnt);	
	return 0;
}

int SoftmaxLayer_backward(LAYER_t *pLayer)
{
	return 0;
}


