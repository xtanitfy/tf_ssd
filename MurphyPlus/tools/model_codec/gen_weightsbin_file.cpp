#include "gen_weightsbin_file.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "public.h"
#include "data_type.h"

#define MODEL_WEIGHTS_FILE "/home/samba/CNN/caffe_ssd/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_56539.caffemodel"
#define MODEL_PROTOTXT_FILE "model/ssd300_deploy.prototxt"
#define WEIGHTS_BIN_FILE "weigts.bin"

using namespace caffe;
using namespace std;

int WB_init(NetParameter *pNetParam,char *modelProtoTxtFile,char *modelWeightsFile);
static int WB_encode(NetParameter *pNetParam);
static int WB_writeHead(NetParameter *pNetParam);
static int WB_writeBlobInfo(NetParameter *pNetParam);
static int WB_writeBlob(BlobProto &blobProto,unsigned long Offset);
static int getWeightsBinFilePath(char *srcFile,char *dstFile);

static FILE *fp = NULL;
static WB_HEAD_t *pHead = NULL;
static WB_HEADT_BLOB_ITEM_t *pBlobItem = NULL;
static unsigned long allBlobSize = 0;
static char weightBinPath[256];

int main(int argc,char **argv)
{
	int ret = -1;
#if 0
	Net<float> caffe_net(MODEL_FILE, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(WEIGHTS_FILE);
    
    float iter_loss;
    caffe_net.Forward(&iter_loss);
	cout << "iter_loss:" << iter_loss << endl;
#endif
	NetParameter netParam;

	if (argc != 3) {
		printf("argv[1]:caffe protofile\n");
		printf("argv[2]:model weights file\n");
		CHECK_EXPR_RET(TRUE, -1);
		return -1;
	}
	ret= getWeightsBinFilePath(argv[2],weightBinPath);
	CHECK_EXPR_RET(ret < 0, -1);
	
	ret = WB_init(&netParam,argv[1],argv[2]);
	CHECK_EXPR_RET(ret < 0, -1);
	
	ret = WB_encode(&netParam);
	CHECK_EXPR_RET(ret < 0, -1);

	cout << "allBlobSize:" << allBlobSize << endl;
	return 0;
}


int getWeightsBinFilePath(char *srcFile,char *dstFile)
{
	int len = 0;
	CHECK_EXPR_RET(srcFile == NULL || dstFile == NULL,-1);
	strcpy(dstFile,srcFile);
	char *ptr = strrchr(dstFile,'/');
	ptr++;
	strcpy(ptr,WEIGHTS_BIN_FILE);

	return 0;
}

int WB_init(NetParameter *pNetParam,char *modelProtoTxtFile,char *modelWeightsFile)
{
	ReadNetParamsFromTextFileOrDie(modelProtoTxtFile,pNetParam);
	cout<< "netParam.name():" << pNetParam->name() <<  endl;
	cout<< "layer_size:" << pNetParam->layer_size() <<  endl;
	
	ReadNetParamsFromBinaryFileOrDie(modelWeightsFile, pNetParam);
	printf("sizeof(WB_HEADT_BLOB_ITEM_t):%ld\n",sizeof(WB_HEADT_BLOB_ITEM_t));

	fp = fopen(weightBinPath,"wb+");
	CHECK_EXPR_RET(fp == NULL, -1);
	
	return 0;
}

int WB_writeHead(NetParameter *pNetParam)
{
	pHead = (WB_HEAD_t *)malloc(sizeof(WB_HEAD_t));
	CHECK_EXPR_RET(pHead == NULL, -1);

	int cnt = 0;
	for (int i = 0;i < pNetParam->layer_size();i++) {
		LayerParameter layer = pNetParam->layer(i);
		int blob_size = layer.blobs_size();
		if (blob_size > 0) {
			cnt++;
		}
	}
	pHead->layersNum = cnt;
	int nwrite = fwrite(pHead,sizeof(WB_HEAD_t),1,fp);
	CHECK_EXPR_RET(nwrite != 1, -1);
	
	printf("pHead->layersNum:%d\n",pHead->layersNum);
	
	return 0;
}

int WB_writeContent(NetParameter *pNetParam)
{
	CHECK_EXPR_RET(pNetParam == NULL, -1);
	CHECK_EXPR_RET(pHead == NULL, -1);

	int blobInfoSize = sizeof(WB_HEADT_BLOB_ITEM_t) * pHead->layersNum;
	pBlobItem = (WB_HEADT_BLOB_ITEM_t *)malloc(blobInfoSize);
	CHECK_EXPR_RET(pBlobItem == NULL, -1);

	LayerParameter layerParameter;
	unsigned long offset = 0;

	offset += sizeof(WB_HEAD_t);
	unsigned long blobInfoOffset = offset;
	offset += blobInfoSize;

	int index = 0;
	int ret = -1;
	for (int i = 0;i < pNetParam->layer_size();i++) {
		LayerParameter layer = pNetParam->layer(i);
		int blob_size = layer.blobs_size();
		if (blob_size <= 0) {
			continue;
		}
		strcpy(pBlobItem[index].layerName,layer.name().c_str());
		pBlobItem[index].blob_size = blob_size;
		cout << "layerName:" << pBlobItem[index].layerName << endl;
		for (int j = 0;j < blob_size;j++) {
			
			BlobProto blob = layer.blobs(j);
			int dim_size = blob.shape().dim_size();
		
			WB_BLOB_t *pBlobInfo = &pBlobItem[index].blob[j];
			pBlobInfo->dim_size = dim_size;
			cout << "\tdim_size:" << dim_size << endl;
			CHECK_EXPR_RET(dim_size > WB_ONE_BLOB_MAX_DIM, -1)
				
			pBlobItem[index].blob[j].blobFileOffset = offset;

			ret = WB_writeBlob(blob,offset);
			CHECK_EXPR_RET(ret < 0, -1);
			
			int blobBytesSize = 1; 
			for (int k = 0;k < dim_size;k++) {
				pBlobInfo->dim[k] = blob.shape().dim(k);
				cout << "\t\tdim["<< k <<"]:" << blob.shape().dim(k) << endl;
				blobBytesSize *= pBlobInfo->dim[k];
			}
			blobBytesSize *= sizeof(float);
			offset += blobBytesSize;
		}
		index++;
	}
	
	fseek(fp,blobInfoOffset,SEEK_SET);
	int nwrite = fwrite(pBlobItem,sizeof(WB_HEADT_BLOB_ITEM_t),pHead->layersNum,fp);
	CHECK_EXPR_RET(nwrite != pHead->layersNum, -1);
	free(pBlobItem);
	
	return 0;
}

int WB_encode(NetParameter *pNetParam)
{
	WB_writeHead(pNetParam);
	
	WB_writeContent(pNetParam);

	fclose(fp);
	
	return 0;
}

static int WB_writeBlob(BlobProto &blobProto,unsigned long offset)
{
	fseek(fp,offset,SEEK_SET);

	//not support 
 	if (blobProto.double_data_size() > 0)  {
		printf("double_data_size:%d\n",blobProto.double_data_size());
		return -1;
	}
	
	if (blobProto.data_size() > 0)  {
		printf("data_size:%d\n",blobProto.data_size());
	}

	//not support 
	if (blobProto.diff_size() > 0 ) {
		printf("diff_size:%d\n",blobProto.diff_size());
		return -1;
	}
	
	if (blobProto.has_num() || blobProto.has_channels() ||
        blobProto.has_height() || blobProto.has_width()) {
      	printf("has n c h w!\n");
		//getchar();
    } 
		
	unsigned long cnt = blobProto.data_size();
	float *pData = (float *)malloc(sizeof(float) * cnt);
	CHECK_EXPR_RET(pData == NULL, -1);
	
	for (int i = 0;i < cnt;i++) {
		pData[i] = blobProto.data(i);
	}

	int nwrite = fwrite(pData,sizeof(float),cnt,fp);
	CHECK_EXPR_RET(nwrite != cnt, -1);

	allBlobSize += cnt*sizeof(float);
	
	free(pData);
	return 0;
}
