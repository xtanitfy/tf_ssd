#include "net.h"
#include "wb_decode.h"
#include "type_layer_reg.h"
#include "util.h"

extern int parseNetParameter(NetParameter * netParameter);
static int NETAddBlobs(NET_t *pNet);
static void NETGetAllBlobsNum(NET_t *pNet);
static int NETAddLayer(NET_t *pNet,LayerParameter *pLayerParam);
static int NETAddLayers(NET_t *pNet);
static void NETAddInputBlobName(NET_t *pNet);
static int NETAddLayerWeigtsBlobs(LayerParameter *pLayerParam,LAYER_t *pLayer);
static int NETInit(NET_t *pNet);
static void NETInitLayerNode(LAYER_t *pLayer);
static int NETJudgeAllLayersIsOutput(NET_t *pNet);
static int NETGetBlobIdxByName(NET_t *pNet,char *name);
static BOOL NETFilterLayer(LAYER_t *layer);

static char (*gTestLayerNameArr)[PARSE_STR_NAME_SIZE] = NULL;
static int gTestLayerNum = 0;

static char (*gFilterLayerNameArr)[PARSE_STR_NAME_SIZE] = NULL;
static int gFilterLayerNum = 0;


NET_t *NET_create(Phase phase,char *weightsFile)
{
	NET_t *pNet = (NET_t *)malloc(sizeof(NET_t));
	CHECK_EXPR_RET(pNet == NULL, NULL);

	pNet->phase = phase;

	int ret = parseNetParameter(&pNet->netParam);
	CHECK_EXPR_RET(ret < 0, NULL);
	
	ret = WB_loadToNet(weightsFile, &pNet->netParam);
	CHECK_EXPR_RET(ret < 0, NULL);	

	INIT_LIST_HEAD(&pNet->layerList);

	ret = NETAddBlobs(pNet);
	CHECK_EXPR_RET(ret < 0, NULL);	
		
	ret = NETAddLayers(pNet);
	CHECK_EXPR_RET(ret < 0, NULL);	

	//NETJudgeAllLayersIsOutput(pNet);
	
	ret = NETInit(pNet);
	CHECK_EXPR_RET(ret < 0, NULL);	
		
	return pNet;
}

#if 0
LAYER_t *NET_GetOutputLayerByName(char *layername)
{


	return 0;
}


LAYER_t *NET_GetLayerByName(char *layername)
{
	

	return 0;
}
#endif

//It looks like is no need.
int NETJudgeAllLayersIsOutput(NET_t *pNet)
{

#if 1
	LAYER_t *pLocalLayer = NULL;
	LAYER_t *pOtherLayer = NULL;
	char *bottomName = NULL;
	char *topName = NULL;
	
	//judge every layer's topname is exist in all other layers' bottoms,
	//if at least one not exists then it's output layer.
	list_for_each_entry(pLocalLayer, &pNet->layerList, list, LAYER_t) {
		pLocalLayer->isOuput = FALSE;
		for (int i = 0;i < pLocalLayer->topCnt;i++) {
			BOOL isExist = FALSE;
			list_for_each_entry(pOtherLayer, &pNet->layerList, list, LAYER_t) {
				if (pLocalLayer != pOtherLayer) {
					for (int j = 0;j < pOtherLayer->bottomCnt;j++) {
						bottomName = pOtherLayer->pLayerParam->bottom[j];
						topName = pLocalLayer->pLayerParam->top[i];
						if (strcmp(topName,bottomName) == 0) {
							isExist = TRUE;
							break;
						}
					}
					if (isExist == TRUE) {
						break;
					}
				}
			}
			if (isExist == FALSE) {
				pLocalLayer->isOuput = TRUE;
				printf("%s is Output layer!\n",pLocalLayer->pLayerParam->name);
				break;
			}
		}		
	}
#endif

	return 0;
}

int NETGetBlobIdxByName(NET_t *pNet,char *name)
{
	for (int i = 0;i < pNet->blobSizeTmp;i++) {
		if (strcmp(pNet->blobName[i],name) == 0) {
			return i;
		}
	}
	return -1;
}

int NET_printAllBlobsName(NET_t *pNet)
{
	printf("NET_printAllBlobsName:\n");
	for (int i = 0;i < pNet->blobSizeTmp;i++) {
		printf("\t%s indx:%d\n",pNet->blobName[i],i);
	}

	return 0;
}

int NETAddBlobsName(NET_t *pNet,LayerParameter *pLayerParam)
{
	BOOL isFind = FALSE;

	for (int i = 0;i < pLayerParam->top_size;i++) {

		isFind = FALSE;
		for (int j = 0;j < pNet->blobSizeTmp;j++) {
			if (strcmp(pNet->blobName[j],pLayerParam->top[i]) == 0) {
				isFind = TRUE;
				break;
			}
		}
		if (isFind == FALSE) {
			strcpy(pNet->blobName[pNet->blobSizeTmp++],pLayerParam->top[i]);
		}
	}

	return 0;
}


void NETGetAllBlobsNum(NET_t *pNet)
{
	NetParameter *pNetParam = &pNet->netParam;
	LayerParameter *pLayerParam = NULL;
	int blobsNum = 0;
	
	blobsNum += pNetParam->input_size;
	for (int i = 0;i < pNetParam->layer_size;i++) {
		pLayerParam = &pNetParam->layer[i];
		blobsNum += pLayerParam->top_size;
	}
	pNet->blobsNum = blobsNum;
}


void NETAddInputBlobName(NET_t *pNet)
{
	for (int i = 0;i < pNet->netParam.input_size;i++) {
		strcpy(pNet->blobName[pNet->blobSizeTmp++],pNet->netParam.input[i]);
	}
}

int NETAddInputBlobs(NET_t *pNet)
{
	BOOL isFind = FALSE; 

	if (pNet->netParam.input_size <= 0) {
		//printf("NETAddInputBlobs no input parameter!\n");
		return 0;
	}
	
	pNet->inputBlobNum = pNet->netParam.input_size;
	pNet->inputBlobName = (char (*)[PARSE_STR_NAME_SIZE])malloc(PARSE_STR_NAME_SIZE *\
								pNet->inputBlobNum);
	CHECK_EXPR_RET(pNet->inputBlobName == NULL, -1);
	for (int i = 0;i < pNet->inputBlobNum;i++) {
		strcpy(pNet->inputBlobName[i],pNet->netParam.input[i]);
	}
	
	pNet->inputBlob = (BLOB_t **)malloc(sizeof(BLOB_t*) * pNet->inputBlobNum);
	CHECK_EXPR_RET(pNet->inputBlob == NULL, -1);
	
	for (int i = 0;i < pNet->inputBlobNum;i++) {
		isFind = FALSE;
		for (int j = 0;j < pNet->blobSizeTmp;j++) {
			if (strcmp(pNet->inputBlobName[i],pNet->blobName[j]) == 0) {
				 isFind = TRUE;
				 pNet->inputBlob[i] = &pNet->pAllBlobs[j];
				 break;
			}
		}	
		CHECK_EXPR_RET(isFind==FALSE,-1);
	}

	return 0;
}


int NETAddLayerBottomBlobs(NET_t *pNet,LayerParameter *pLayerParam,LAYER_t *pLayer)
{
	pLayer->bottomCnt = pLayerParam->bottom_size;
	pLayer->bottom = (BLOB_t **)malloc(sizeof(BLOB_t *) * pLayer->bottomCnt);
	CHECK_EXPR_RET(pLayer->bottom == NULL, -1);
	//printf("----[%s]\n",pLayerParam->name);
	CHECK_EXPR_RET(pLayerParam->bottom_size != pLayerParam->bottom_size,-1);
	for (int i = 0;i < pLayerParam->bottom_size;i++) {
		int idx = NETGetBlobIdxByName(pNet,pLayerParam->bottom[i]);
		CHECK_EXPR_RET(idx < 0,-1);
		pLayer->bottom[i] = &pNet->pAllBlobs[idx];
		//printf("\t----[%s]\n",pLayerParam->bottom[i]);
	}

	return 0;
}

int NETAddLayerWeigtsBlobs(LayerParameter *pLayerParam,LAYER_t *pLayer)
{
	pLayer->weigtsBlobsSize = pLayerParam->blobs_size;
	pLayer->pWeigtsBlobs = (BLOB_t *)malloc(sizeof(BLOB_t) * pLayer->weigtsBlobsSize);
	CHECK_EXPR_RET(pLayer->pWeigtsBlobs == NULL, -1);

	BLOB_t *pBlob = NULL;
	BlobProto *pBlobProto = NULL;
	for (int i = 0;i < pLayerParam->blobs_size;i++) {
		pBlob = &pLayer->pWeigtsBlobs[i];
		pBlobProto = &pLayerParam->blobs[i];
		BLOB_init(pBlob);
		int ret = BLOB_fromProto(pBlob, pBlobProto, TRUE);
		CHECK_EXPR_RET(ret < 0, -1);
		BLOB_freeProtoMemory(pBlobProto);
	}
	
	return 0;
}


int NETAddLayerTopBlobs(NET_t *pNet,LayerParameter *pLayerParam,LAYER_t *pLayer)
{
	pLayer->topCnt = pLayerParam->top_size;
	pLayer->top = (BLOB_t **)malloc(sizeof(BLOB_t *) * pLayer->topCnt);
	CHECK_EXPR_RET(pLayer->top == NULL, -1);
	for (int i = 0;i < pLayerParam->top_size;i++) {
		int idx = NETGetBlobIdxByName(pNet,pLayerParam->top[i]);
		CHECK_EXPR_RET(idx < 0,-1);
		pLayer->top[i] = &pNet->pAllBlobs[idx];
	}

	return 0;
}

static void NETInitLayerNode(LAYER_t *pLayer)
{
	pLayer->forward = NULL;
	pLayer->setUp = NULL;
	pLayer->reshape = NULL;
	pLayer->backward = NULL;
	pLayer->pLayerParam = NULL;
	pLayer->bottom = NULL;
	pLayer->bottomCnt = 0;
	pLayer->top = NULL;
	pLayer->topCnt = 0;
	pLayer->pWeigtsBlobs = 0;
	pLayer->weigtsBlobsSize = 0;
	pLayer->propagate_down = NULL;
	pLayer->innerParam = NULL;
	pLayer->list.next = NULL;
	pLayer->list.prev = NULL;
}

int NETAddLayer(NET_t *pNet,LayerParameter *pLayerParam)
{
	if (pLayerParam->include_size > 0) {
		BOOL isInclude = FALSE;
		for (int i = 0;i < pLayerParam->include_size;i++) {
			if (pLayerParam->include[i].phase == pNet->phase) {
				isInclude = TRUE;
			}
		}
		if (isInclude == FALSE) {
			return 0;
		}
	}
	
	LAYER_t *pLayer = (LAYER_t *)malloc(sizeof(LAYER_t));
	CHECK_EXPR_RET(pLayer == NULL, -1);
	NETInitLayerNode(pLayer);
	

	pLayer->pLayerParam = pLayerParam;
	BOOL isFind = FALSE; 
	int i = 0;
	for (i = 0;i < DIM_OF(gTypeLayerRegister);i++) {
		if (strcmp(gTypeLayerRegister[i].typename,pLayerParam->type) == 0) {
			isFind = TRUE;
			break;
		}
	}
	
	if (isFind == TRUE) {
		pLayer->setUp = gTypeLayerRegister[i].setUp;
		pLayer->reshape = gTypeLayerRegister[i].reshape;
		pLayer->forward = gTypeLayerRegister[i].forward;
		pLayer->backward = gTypeLayerRegister[i].backward;

	} else {
		printf("Not support %s layer\n",pLayerParam->type);
		return -1;
		//CHECK_EXPR_RET(TRUE, -1);
	}

	int ret = NETAddInputBlobs(pNet);
	CHECK_EXPR_RET(ret < 0, -1);
	
	ret = NETAddLayerBottomBlobs(pNet, pLayerParam, pLayer);
	CHECK_EXPR_RET(ret < 0, -1);
	
	ret = NETAddLayerTopBlobs(pNet, pLayerParam, pLayer);
	CHECK_EXPR_RET(ret < 0, -1);

	//printf("layername:%s\n",pLayerParam->name);
	
	ret = NETAddLayerWeigtsBlobs(pLayerParam, pLayer);
	CHECK_EXPR_RET(ret < 0, -1);
	
	list_add_tail(&pLayer->list,&pNet->layerList);
	
	return 0;
}

int NETAddLayers(NET_t *pNet)
{
	int ret = -1;
	LayerParameter *pLayerParam = NULL;
	NetParameter *pNetParam = &pNet->netParam;
	for (int i = 0;i < pNetParam->layer_size;i++) {
		pLayerParam = &pNetParam->layer[i];
		ret = NETAddLayer(pNet,pLayerParam);
		//CHECK_EXPR_RET(ret < 0, -1);
	}
	return 0;
}


int NETAddBlobs(NET_t *pNet)
{
	NetParameter *pNetParam = &pNet->netParam;
	LayerParameter *pLayerParam = NULL;

	NETGetAllBlobsNum(pNet);
	
	pNet->pAllBlobs = (BLOB_t *)malloc(sizeof(BLOB_t) * pNet->blobsNum);
	CHECK_EXPR_RET(pNet->pAllBlobs == NULL, -1);
	for (int i = 0;i < pNet->blobsNum;i++) {
		BLOB_init(&pNet->pAllBlobs[i]); 
	}
	
	pNet->blobName = (char (*)[PARSE_STR_NAME_SIZE])malloc(PARSE_STR_NAME_SIZE * pNet->blobsNum);
	CHECK_EXPR_RET(pNet->blobName == NULL, -1);
	pNet->blobNameInx = (int *)malloc(sizeof(int) *pNet->blobsNum);
	CHECK_EXPR_RET(pNet->blobNameInx == NULL, -1);

	pNet->blobSizeTmp = 0;	
	NETAddInputBlobName(pNet);
	for (int i = 0;i < pNetParam->layer_size;i++) {
		pLayerParam = &pNetParam->layer[i];
		NETAddBlobsName(pNet,pLayerParam);
	}

	NET_printAllBlobsName(pNet);

	return 0;
}

int NET_feedData(NET_t *pNet,BLOB_t *pblob,int blobCnt)
{
	CHECK_EXPR_RET(pNet == NULL, -1);
	CHECK_EXPR_RET(pblob == NULL, -1);
	LAYER_t *pInputLayer = pNet->pInputLayer;
	printf("NET_feedData:%s\n",pInputLayer->pLayerParam->name);
	
	CHECK_EXPR_RET(blobCnt != pInputLayer->topCnt, -1);
	for (int i = 0;i < blobCnt;i++) {
		BLOB_CopyFrom(pInputLayer->top[i], &pblob[i], FALSE, TRUE);
		printf("[NET_feedData]shape_cnt_:%d\n",pInputLayer->top[i]->shape_cnt_);
	}
	BLOB_writeTopBlobToTxtFile(pInputLayer->pLayerParam->name,
					pInputLayer->top,pInputLayer->topCnt);
	
	//BLOB_writeTxt("input_blob.txt",&blob);
	return 0;
}

void NET_printAllBlobs(NET_t *pNet)
{
	LAYER_t *pLayer = NULL;
	list_for_each_entry(pLayer, &pNet->layerList, list, LAYER_t) {
		printf("%s:\n",pLayer->pLayerParam->name);
		for (int i = 0;i < pLayer->topCnt;i++) {
			printf("	%s\n",pLayer->pLayerParam->top[i]);
			printf("	shape_cnt_:%d\n",pLayer->top[i]->shape_cnt_);
		}
	}
}

void NET_setFilterLayerName(char **layerNameArr,int len)
{
	gFilterLayerNameArr = (char (*)[PARSE_STR_NAME_SIZE])malloc(PARSE_STR_NAME_SIZE * sizeof(char) * len);
	for (int i = 0;i < len;i++) {
		strcpy(gFilterLayerNameArr[gFilterLayerNum++],layerNameArr[i]);
	}
}


void NET_setTestLayerName(char **layerNameArr,int len)
{
	gTestLayerNameArr = (char (*)[PARSE_STR_NAME_SIZE])malloc(PARSE_STR_NAME_SIZE * sizeof(char) * len);
	for (int i = 0;i < len;i++) {
		strcpy(gTestLayerNameArr[gTestLayerNum++],layerNameArr[i]);
	}
	
}

BOOL NETFilterLayer(LAYER_t *layer)
{
#if 1
	char *name = layer->pLayerParam->name;
	if (gFilterLayerNum == 0) {
		return FALSE;
	}

	BOOL isFiltrLayer = FALSE;
	for (int i = 0;i < gFilterLayerNum;i++) {
		if (strcmp(name,gFilterLayerNameArr[i]) == 0) {
			isFiltrLayer = TRUE;
			break;
		}
	}

	return isFiltrLayer;
#else
	return TRUE;
#endif

}



BOOL NETIsTestLayer(LAYER_t *layer)
{
#if 1
	char *name = layer->pLayerParam->name;
	if (gTestLayerNum == 0) {
		return TRUE;
	}

	BOOL isTestLayer = FALSE;
	for (int i = 0;i < gTestLayerNum;i++) {
		if (strcmp(name,gTestLayerNameArr[i]) == 0) {
			isTestLayer = TRUE;
			break;
		}
	}

	return isTestLayer;
#else
	return TRUE;
#endif

}

int NETInit(NET_t *pNet)
{
	printf("NETInit\n");

	struct list_head *phead = &pNet->layerList;
	struct list_head *pFirst = phead->next;
	pNet->pInputLayer = list_entry(pFirst, LAYER_t, list);
	LAYER_t *pInputLayer = pNet->pInputLayer;

	LayerParameter *pLayerParam = pInputLayer->pLayerParam;
	for (int i = 0;i < pInputLayer->topCnt;i++) {
		int shape[BLOB_MAX_AXES];
		int shape_size = 0;
		for (int i = 0;i < pLayerParam->input_param.shape[0].dim_size;i++) {
			shape[shape_size++] = pLayerParam->input_param.shape[0].dim[i];
		}
		BLOB_reshapeByArray(pInputLayer->top[i],shape,shape_size);
	}
	
	LAYER_t *pLayer = NULL;
	list_for_each_entry(pLayer, &pNet->layerList, list, LAYER_t) {
		if (NETIsTestLayer(pLayer) == TRUE && NETFilterLayer(pLayer) == FALSE) {

			if (pLayer->setUp != NULL) {
				pLayer->setUp(pLayer);
			}

			if (pLayer->reshape != NULL) {
				pLayer->reshape(pLayer);
				for (int i = 0;i < pLayer->topCnt;i++) {
					//BLOB_printShapeString(pLayer->top[i],pLayer->pLayerParam->name);
				}
			}
		}
	}
	list_del(pFirst);
	
	return 0;
}



int NET_Init(NET_t *pNet)
{
	printf("NETInit\n");
	LAYER_t *pLayer = NULL;
	list_for_each_entry(pLayer, &pNet->layerList, list, LAYER_t) {
		if (NETIsTestLayer(pLayer) == TRUE && NETFilterLayer(pLayer) == FALSE) {
			if (pLayer->setUp != NULL) {
				printf("%s setUp!\n",pLayer->pLayerParam->name);
				pLayer->setUp(pLayer);
			}
			if (pLayer->reshape != NULL) {
				printf("%s reshape!\n",pLayer->pLayerParam->name);
				pLayer->reshape(pLayer);
			}
			
		}
	}
	return 0;
}

void printBlobIdx(NET_t *pNet,LAYER_t *pLayer)
{
	printf("-----[%s]\n",pLayer->pLayerParam->name);
	for (int i = 0;i < pLayer->topCnt;i++) {
		int idx = NETGetBlobIdxByName(pNet,pLayer->pLayerParam->top[i]);
		printf("----\ttop[%d] idx:%d\n",i,idx);
	}
	
	for (int i = 0;i < pLayer->bottomCnt;i++) {
		int idx = NETGetBlobIdxByName(pNet,pLayer->pLayerParam->bottom[i]);
		printf("----\tbottom[%d] idx:%d\n",i,idx);
	}
}

int NET_forward(NET_t *pNet,NET_RES_t *pRes)
{
	LAYER_t *pLayer = NULL;
	
	list_for_each_entry(pLayer, &pNet->layerList, list, LAYER_t) {
		if (NETIsTestLayer(pLayer) == TRUE && NETFilterLayer(pLayer) == FALSE) {
			if (pLayer->forward != NULL) {
				printf("%s forward!\n",pLayer->pLayerParam->name);	
			//	CalTimeStart();
				pLayer->forward(pLayer);
			//	CalTimeEnd(pLayer->pLayerParam->name);
				BLOB_writeTopBlobToTxtFile(pLayer->pLayerParam->name,
											pLayer->top,pLayer->topCnt);	
			}
		}
	}
	
	return 0;
}

int NET_backward(NET_t *pNet)
{
	return 0;
}


