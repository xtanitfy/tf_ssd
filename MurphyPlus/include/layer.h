#ifndef __LAYER_H__
#define __LAYER_H__

#include "blob.h"
#include "dlist.h"

typedef struct LAYER_ LAYER_t;

#define LAYER_SETUP(layerTypeName) layerTypeName##Layer_setUp
#define LAYER_RESHAPE(layerTypeName) layerTypeName##Layer_reshape
#define LAYER_FORWARD(layerTypeName) layerTypeName##Layer_forward
#define LAYER_BACKWARD(layerTypeName) layerTypeName##Layer_backward

struct  LAYER_
{
	int (*setUp)(LAYER_t *pLayer);
	int (*reshape)(LAYER_t *pLayer);
	int (*forward)(LAYER_t *pLayer);
	int (*backward)(LAYER_t *pLayer);
	
	LayerParameter *pLayerParam;

	BLOB_t **bottom;
	int bottomCnt;
	
	BLOB_t **top;
	int topCnt;
	
	BLOB_t *pWeigtsBlobs;
	int weigtsBlobsSize;

	BOOL *propagate_down;

	void *innerParam;

	BOOL isOuput;
	struct list_head list;
};


#endif