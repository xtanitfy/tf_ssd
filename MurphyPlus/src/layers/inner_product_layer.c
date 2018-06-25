#include "inner_product_layer.h"
#include "math_functions.h" 

//IP_INNER_PARAM_t *this = NULL;

static void ipInnerParamInit(IP_INNER_PARAM_t *pParam)
{
	memset(pParam,'\0',sizeof(IP_INNER_PARAM_t));
}

int InnerProductLayer_setUp(LAYER_t *pLayer)
{
	//printf("%s layer setup!\n",pLayer->pLayerParam->name);

	IP_INNER_PARAM_t *this = (IP_INNER_PARAM_t *)malloc(sizeof(IP_INNER_PARAM_t));
	CHECK_EXPR_RET(this == NULL,-1);
	ipInnerParamInit(this);
	pLayer->innerParam = this;
	
	LayerParameter *pLayerParam = pLayer->pLayerParam;
	InnerProductParameter *pIPParam = &pLayerParam->inner_product_param;
	int num_output = pIPParam->num_output;
	this->bias_term_ = pIPParam->bias_term;
	this->transpose_ = pIPParam->transpose;
	this->N_ = num_output;

	int axis = BLOB_CanonicalAxisIndex(pLayer->bottom[0],pIPParam->axis);
	this->K_ = BLOB_countByStart(pLayer->bottom[0],axis);

#if 0 //only for trainning
	if (pLayer->weigtsBlobsSize < 0) {
		if (this->bias_term_ == TRUE) {
			pLayer->weigtsBlobsSize = 2;
		} else {
			pLayer->weigtsBlobsSize = 1;
		}
		pLayer->pWeigtsBlobs = (BLOB_t *)malloc(sizeof(BLOB_t) \
											* pLayer->weigtsBlobsSize);
		CHECK_EXPR_RET(pLayer->pWeigtsBlobs == NULL,-1);
		for (int i = 0;i < pLayer->weigtsBlobsSize;i++) {
			BLOB_init(&pLayer->pWeigtsBlobs[i]);
		}
		
		int weight_shape[2];
		if (this->transpose_ == TRUE) {
			weight_shape[0] = this->K_;
			weight_shape[1] = this->N_;
		} else {
			weight_shape[0] = this->N_;
			weight_shape[1] = this->K_;
		}
		BLOB_reshapeByArray(&pLayer->pWeigtsBlobs[0],weight_shape,2);	
	}
#endif

	return 0;
}

int InnerProductLayer_reshape(LAYER_t *pLayer)
{
	//printf("%s layer reshape!\n",pLayer->pLayerParam->name);

	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	IP_INNER_PARAM_t *this = (IP_INNER_PARAM_t *)pLayer->innerParam;

	LayerParameter *pLayerParam = pLayer->pLayerParam;
	InnerProductParameter *pIPParam = &pLayerParam->inner_product_param;
	
	int axis = BLOB_CanonicalAxisIndex(pLayer->bottom[0],pIPParam->axis);
	int new_K = BLOB_countByStart(pLayer->bottom[0],axis);
	CHECK_EXPR_RET(new_K != this->K_,-1);
	this->M_ = BLOB_countByStartAndEnd(pLayer->bottom[0],0,axis);

	int top_shape[BLOB_MAX_AXES];
	int top_shape_size = 0;
	BLOB_shape(pLayer->bottom[0],top_shape,&top_shape_size);
	top_shape_size = 2;
	top_shape[axis] = this->N_;
	BLOB_reshapeByArray(pLayer->top[0],top_shape,top_shape_size);
	if (this->bias_term_ == TRUE) {
		int bias_shape[BLOB_MAX_AXES];
		int bias_shape_size = 0;
		bias_shape[bias_shape_size++] = this->M_;
		BLOB_init(&this->bias_multiplier_);
		BLOB_reshapeByArray(&this->bias_multiplier_,bias_shape,bias_shape_size);
		Murphy_set(this->M_,(DATA_TYPE)1,BLOB_data(&this->bias_multiplier_));
	}
	 
	return 0;
}

int InnerProductLayer_forward(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	IP_INNER_PARAM_t *this = (IP_INNER_PARAM_t *)pLayer->innerParam;

	DATA_TYPE *bottom_data = BLOB_data(pLayer->bottom[0]);
	DATA_TYPE *top_data = BLOB_data(pLayer->top[0]);
	DATA_TYPE *weight = BLOB_data(&pLayer->pWeigtsBlobs[0]);
	DATA_TYPE *bias = BLOB_data(&pLayer->pWeigtsBlobs[1]);
	BOOL transpose = this->transpose_;
	int M = this->M_;
	int N = this->N_;
	int K = this->K_;

	Murphy_gemm(CblasNoTrans,(transpose==TRUE)? CblasNoTrans : CblasTrans,
			M,N,K,(DATA_TYPE)1,bottom_data,weight,(DATA_TYPE)0.,top_data);

	if (this->bias_term_ == TRUE) {
		Murphy_gemm(CblasNoTrans, CblasNoTrans, M, N, 1, (DATA_TYPE)1.,
			BLOB_data(&this->bias_multiplier_),bias,(DATA_TYPE)1.,top_data);
	}
	
	return 0;
}

int InnerProductLayer_backward(LAYER_t *pLayer)
{
	return 0;
}



