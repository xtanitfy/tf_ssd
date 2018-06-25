#include "wb_decode.h"

struct list_head *WB_Decode(char *wbFile)
{
	printf("WB_Decode:%s\n",wbFile);
	FILE *fp = fopen(wbFile,"rb+");
	CHECK_EXPR_RET(fp == NULL, NULL);

	struct list_head *pHead = (struct list_head *)malloc(sizeof(struct list_head));
	CHECK_EXPR_RET(pHead == NULL, NULL);

	INIT_LIST_HEAD(pHead);
	
	WB_HEAD_t head;
	int nread = fread(&head,sizeof(WB_HEAD_t),1,fp);
	CHECK_EXPR_RET(nread != 1, NULL);

	int itemNum = head.layersNum;
	WB_HEADT_BLOB_ITEM_t *pBlobInfo = (WB_HEADT_BLOB_ITEM_t *)malloc(\
					sizeof(WB_HEADT_BLOB_ITEM_t) * itemNum);
	CHECK_EXPR_RET(pBlobInfo == NULL, NULL);
	nread = fread(pBlobInfo,sizeof(WB_HEADT_BLOB_ITEM_t),itemNum,fp);
	CHECK_EXPR_RET(nread != itemNum, NULL);
	
	WB_HEADT_BLOB_ITEM_t *pBlobItem = NULL;
	WB_BLOB_t *pBlob = NULL;

	for (int i = 0;i < itemNum;i++) {
		pBlobItem = &pBlobInfo[i];

		int blobSize = pBlobItem->blob_size;
		WB_LAYER_BLOB_t *pLayerBlob = (WB_LAYER_BLOB_t *)malloc(sizeof(WB_LAYER_BLOB_t));
		CHECK_EXPR_RET(pLayerBlob == NULL, NULL);
		pLayerBlob->blobSize = blobSize;
		strcpy(pLayerBlob->layerName,pBlobItem->layerName);
		
		for (int j = 0;j < blobSize;j++) {
			pBlob = &pBlobItem->blob[j];
			fseek(fp,pBlob->blobFileOffset,SEEK_SET);
			
			unsigned long dataSize = 1;
			pLayerBlob->blobProto[j].shape.dim_size = pBlob->dim_size;
			pLayerBlob->blobProto[j].shape.dim = (INT64 *)malloc(sizeof(INT64)*pBlob->dim_size);
			CHECK_EXPR_RET(pLayerBlob->blobProto[j].shape.dim == NULL, NULL);
			for (int k = 0;k < pBlob->dim_size;k++) {
				dataSize *= pBlob->dim[k];
				pLayerBlob->blobProto[j].shape.dim[k] = (INT64)pBlob->dim[k];
			}
			pLayerBlob->blobProto[j].data_size = dataSize;
			pLayerBlob->blobProto[j].data = (float *)malloc(sizeof(float) * dataSize);
			CHECK_EXPR_RET(pLayerBlob->blobProto[j].data == NULL, NULL);

			float *data = pLayerBlob->blobProto[j].data;	
			nread = fread(data,sizeof(float),dataSize,fp);
			CHECK_EXPR_RET(nread != dataSize, NULL);
		}
		list_add_tail(&pLayerBlob->list,pHead);
		
	}
	
	return pHead;
}


int WB_loadToNet(char *wbFile,NetParameter *pNet)
{
	CHECK_EXPR_RET(wbFile == NULL, -1);
	CHECK_EXPR_RET(pNet == NULL, -1);
	
	struct list_head *pHead = WB_Decode(wbFile);
	CHECK_EXPR_RET(pHead == NULL, -1);

	WB_LAYER_BLOB_t *pLayerBlob = NULL;
	LayerParameter *pLayer = NULL;
	BOOL isExist = FALSE;
	list_for_each_entry(pLayerBlob, pHead, list, WB_LAYER_BLOB_t) {
		isExist = FALSE;
		for (int i = 0;i < pNet->layer_size;i++) {
			pLayer =  &pNet->layer[i];
			if (strcmp(pLayerBlob->layerName,pLayer->name) == 0) {
				//load weigts blobs
				pLayer->blobs_size = pLayerBlob->blobSize;
				pLayer->blobs = pLayerBlob->blobProto;
				//printf("dim_size:%d\n",pLayerBlob->blobProto)
				printf("-->layerName:%s\n",pLayer->name);
				isExist = TRUE;
				break;
			}
		}
		CHECK_EXPR_RET(isExist == FALSE,-1);
	}
	printf("[WB_loadToNet] finish!\n");
	
	return 0;
}
