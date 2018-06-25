#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#define WEIGHTS_FILE "/disk/caffe/caffe-ssd/jobs/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel"
#define MODEL_FILE "model/ssd300_deploy.prototxt"

using namespace std;
using namespace caffe;

static int testBlob();

//WriteProtoToBinaryFile
int main(int argc,char **argv)
{
	cout << "model encode!" << endl; 
	
	testBlob();

#if 0
	Net<float> caffe_net(MODEL_FILE, caffe::TEST);
    caffe_net.CopyTrainedLayersFrom(WEIGHTS_FILE);
    
    float iter_loss;
    caffe_net.Forward(&iter_loss);
#endif
#if 1
	NetParameter netParam;
	ReadNetParamsFromTextFileOrDie(MODEL_FILE,&netParam);
	cout<< "netParam.name():" << netParam.name() <<  endl;
	cout<< "layer_size:" << netParam.layer_size() <<  endl;

	ReadNetParamsFromBinaryFileOrDie(WEIGHTS_FILE, &netParam);

	LayerParameter layerParameter;
	for (int i = 0;i < netParam.layer_size();i++) {
		cout<< netParam.layer(i).name() << ":" << endl;
		cout<< "	[" << netParam.layer(i).type() << "]" << endl;
		cout<< "	[netParam.blobs_size():" << netParam.layer(i).blobs_size() << "]" << endl;
		/*
		char *str = "PriorBox";
		layerParameter = netParam.layer(i);
		if (strcmp(netParam.layer(i).type().c_str(),str) == 0) {
			//cout<< str << endl;
			//cout<< layerParameter.prior_box_param().aspect_ratio_size() << endl;
			getchar();
		}
		*/
	}
#endif
	return 0;
}


int testBlob()
{

	caffe::Blob<int > blob(3,4,5,6);
	cout << blob.count() << endl;
	
	BlobProto blob_proto;
	blob_proto.set_num(1);
	blob_proto.set_channels(1);
	blob_proto.set_height(3);
	blob_proto.set_width(2);
	blob.ShapeEquals(blob_proto);
	WriteProtoToBinaryFile(blob_proto,"blob.proto");

	BlobProto blob_read;
	ReadProtoFromBinaryFile("blob.proto",&blob_read);
	cout << "blob_read.num:" << blob_read.num() << endl;
	cout << "blob_read.channels:" << blob_read.channels() << endl;
	cout << "blob_read.height:" << blob_read.height() << endl;
	cout << "blob_read.width:" << blob_read.width() << endl;
	cout << "testBlob OK!" << endl;
	
	
	return 0;
}
/*
int encodeBlob(BlobProto *blob_proto)
{
	FILE *fp = fopen("blob.bin","wb+");
	CHECK_EXPR_RET(fp == NULL,-1);
	
	int size = 0,sizePtrMem = 0;
	unsigned long fileOffset = 0;
	BlobProto_t sBlobProto;
	sBlobProto.num = blob_proto->num();
	sBlobProto.channels = blob_proto->channels();
	sBlobProto.height = blob_proto->height();
	sBlobProto.width = blob_proto->width();
	//fileOffset += sizeof(sBlobProto);
	
	size = blob_proto->shape().dim().size();
	sBlobProto.shape.dimN = size;
	sizePtrMem = sizeof(VOS_INT64)*size;
	sBlobProto.shape.dim = (VOS_INT64 *)malloc(sizePtrMem);
	memcpy(sBlobProto.shape.dim,&blob_proto->shape().dim(),sizePtrMem);
	
	int nwrite = fwrite(&sBlobProto,sizeof(sBlobProto),1,fp);
	CHECK_EXPR_RET(nwrite != 1,-1);

	//nwrite = fwrite(&blob_proto->shape()->dim(),blob_proto->shape()->dim()->size(),1,fp);
	//CHECK_EXPR_RET(nwrite != 1,-1);
	
	return 0;
}
*/




