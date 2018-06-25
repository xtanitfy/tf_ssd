#include "relu_layer.h"


int ReLULayer_forward(LAYER_t *pLayer)
{
	//printf("%s forward!\n",pLayer->pLayerParam->name);
	DATA_TYPE *bottom_data = BLOB_data(pLayer->bottom[0]);
	DATA_TYPE *top_data = BLOB_data(pLayer->top[0]);
	int count = BLOB_count(pLayer->bottom[0]);
	float negative_slope = pLayer->pLayerParam->relu_param.negative_slope;
	for (int i = 0; i < count; ++i) {
		top_data[i] = VOS_MAX(bottom_data[i], (DATA_TYPE)0)
				+ negative_slope * VOS_MIN(bottom_data[i], (DATA_TYPE)0);
	}
	//BLOB_writeTopBlobToTxtFile(pLayer->pLayerParam->name,pLayer->top,pLayer->topCnt);
	return 0;
}

int ReLULayer_backward(LAYER_t *pLayer)
{
	return 0;
}



