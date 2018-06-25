#ifndef __NET_H__
#define __NET_H__

#include "public.h"
#include "dlist.h"
#include "parameter.h"
#include "layer.h"

typedef struct
{
	Phase phase;
	
	NetParameter netParam;

	BLOB_t **inputBlob;
	char (*inputBlobName)[PARSE_STR_NAME_SIZE];
	int inputBlobNum;

	LAYER_t *pInputLayer;
	BLOB_t *pAllBlobs;
	char (*blobName)[PARSE_STR_NAME_SIZE];
	int *blobNameInx;
	
	int blobSizeTmp;
	int blobsNum;
 
	struct list_head layerList;
}NET_t;

typedef struct
{
	BLOB_t **pOut;
	int outNum;
}NET_RES_t;

NET_t *NET_create(Phase phase,char *weightsFile);
int NET_feedData(NET_t *pNet,BLOB_t *pblob,int blobCnt);
int NET_Init(NET_t *pNet);
int NET_forward(NET_t *pNet,NET_RES_t *pRes);
int NET_backward(NET_t *pNet);
void NET_printAllBlobs(NET_t *pNet);
void NET_setTestLayerName(char **layerNameArr,int len);
int  NET_GetOutputRes(NET_RES_t *pRes);
void NET_setFilterLayerName(char **layerNameArr,int len);
#endif