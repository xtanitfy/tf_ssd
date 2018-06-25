#include "blob.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
static int saveBlobProtoToBinaryfile(BlobProto *pBlobProto,char *file);
static int convertMatToBlobProto(BlobProto *pBlobProto,Mat &img);
static int convertBlobprotoToMat(Mat &img,BLOB_t *pBlob);
static char  *getProtofilename(char *src);
static int preprocess(Mat &img);
static int Mat_print(Mat &img);
static int blobProtoPrint(BlobProto *pBlobProto);

static int gChannel = 0;
static int gHeight = 0;
static int gWidth = 0;

int main(int argc,char **argv)
{
	if (argc != 5) {
		printf("argc:%d != 5\n",argc);
		printf("argv[1]:image file\n");
		printf("argv[2]:channel\n");
		printf("argv[3]:height\n");
		printf("argv[4]:width\n");
		return -1;
	}

	//BLOB_t blob;
	//BLOB_init(&blob);
	
	Mat img = imread(argv[1]);
	if (img.empty()) {
		cout << "open" << argv[1] << "failed!" << endl;
		return -1;
	}
	cout << "img.channels():" << img.channels() << endl;

	gChannel = atoi(argv[2]);
	gHeight = atoi(argv[3]);
	gWidth = atoi(argv[4]);

	preprocess(img);

	//Mat_print(img);
	
	BlobProto blobProto;
	convertMatToBlobProto(&blobProto,img);

	//blobProtoPrint(&blobProto);
	//Mat_print(img);
	printf("getProtofilename(argv[1]):%s\n",getProtofilename(argv[1]));


	BLOB_saveBlobProtoToBinaryfile(&blobProto, getProtofilename(argv[1]));
	//saveBlobProtoToBinaryfile(&blobProto,getProtofilename(argv[1]));

	printf("convert image to blobproto finish!\n");
	return 0;
}

int blobProtoPrint(BlobProto *pBlobProto)
{
	float *pData = pBlobProto->data;
	for (int i = 0;i < pBlobProto->data_size;i++) {
		if (i % gWidth == 0) {
			printf("\n");
		}
		printf("%f ",pBlobProto->data[i]);
	}

	return 0;
}

int Mat_print(Mat &img)
{
	/*only print one row*/
	if (gChannel == 3) {
		printf("channel B:\n");
	 	for(int v = 0;v < img.rows;v++){
			for(int u = 0;u < img.cols;u++){
				printf("%f ",img.ptr<float >(v,u)[0]);
			}
			printf("\n");
		}
		
		
		printf("channel G:\n");
	 	for(int v = 0;v < img.rows;v++){
			for(int u = 0;u < img.cols;u++){
				printf("%f ",img.ptr<float >(v,u)[1]);
			}
			printf("\n");
		}
		
		
		printf("channel R:\n");
	 	for(int v = 0;v < img.rows;v++){
			for(int u = 0;u < img.cols;u++){
				printf("%f ",img.ptr<float >(v,u)[2]);
			}
			printf("\n");
		}
		getchar();
		
	} else {
		printf("channel gray:\n");
	 	for(int v = 0;v < img.rows;v++){
			for(int u = 0;u < img.cols;u++){
				printf("%f ",img.ptr<float >(v,u)[0]);
			}
			printf("\n");
		}
		
	}
}

int saveBlobProtoToBinaryfile(BlobProto *pBlobProto,char *file)
{
	FILE *fp = fopen(file,"wb+");
	CHECK_EXPR_RET(fp == NULL, -1);
	
	int nwrite = fwrite(pBlobProto,sizeof(BlobProto),1,fp);
	CHECK_EXPR_RET(nwrite != 1, -1);

	unsigned long long size = 1;
#if 0
	for (int i = 0;i < pBlobProto->shape.dim_size;i++) {
		size *= pBlobProto->shape.dim[i];
	}
#else
	size = pBlobProto->num * pBlobProto->channels * pBlobProto->height * pBlobProto->width;
#endif

	if (pBlobProto->data_size > 0) {
		printf("pBlobProto->data_size:%d\n",pBlobProto->data_size);
		int nwrite = fwrite(pBlobProto->data,sizeof(float),pBlobProto->data_size,fp);
		CHECK_EXPR_RET(nwrite != pBlobProto->data_size, -1);
	}

	if (pBlobProto->diff_size > 0) {
		printf("pBlobProto->diff_size:%d\n",pBlobProto->diff_size);
		int nwrite = fwrite(pBlobProto->diff,sizeof(float),pBlobProto->diff_size,fp);
		CHECK_EXPR_RET(nwrite != pBlobProto->diff_size, -1);
	}
	
	if (pBlobProto->double_data_size > 0) {
		printf("pBlobProto->double_data_size:%d\n",pBlobProto->double_data_size);
		int nwrite = fwrite(pBlobProto->double_data,sizeof(float),pBlobProto->double_data_size,fp);
		CHECK_EXPR_RET(nwrite != pBlobProto->double_data_size, -1);
	}

	if (pBlobProto->double_diff_size > 0) {
		printf("pBlobProto->double_data_size:%d\n",pBlobProto->double_diff_size);
		int nwrite = fwrite(pBlobProto->double_diff,sizeof(float),pBlobProto->double_diff_size,fp);
		CHECK_EXPR_RET(nwrite != pBlobProto->double_diff_size, -1);
	}
	
	fclose(fp);
	return 0;

}

int preprocess(Mat &img)
{
	cv::Mat sample;

	if (img.channels() == 3 && gChannel == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && gChannel == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && gChannel == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && gChannel == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != Size())
		cv::resize(sample, sample_resized, Size(gWidth,gHeight));
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (gChannel == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);
  
	img = sample_float;

	return 0;
}

int convertMatToBlobProto(BlobProto *pBlobProto,Mat &img)
{
	CHECK_EXPR_RET(img.channels() != gChannel, -1);
	CHECK_EXPR_RET(img.rows != gHeight, -1);
	CHECK_EXPR_RET(img.cols != gWidth, -1);

	memset(pBlobProto,'\0',sizeof(BlobProto));
	unsigned long long size = 1;
	pBlobProto->num = 1;
	pBlobProto->channels = gChannel;
	pBlobProto->height = gHeight;
	pBlobProto->width = gWidth;

	size = 1 * gChannel * gHeight * gWidth;
	cout  << "gChannel:" <<gChannel << endl;
	#if 0
	for (int i = 0;i < pBlobProto->shape.dim_size;i++) {
		size *= pBlobProto->shape.dim[i];
	}
	#endif

		
	pBlobProto->data_size = size;
	pBlobProto->data = (float *)malloc(sizeof(float)*size);

	int cnt = 0;
	if (gChannel == 3) {
	  	for(int v = 0;v < img.rows;v++){
			for(int u = 0;u < img.cols;u++){
				pBlobProto->data[cnt++] = img.ptr<float >(v,u)[0];				
			}
		}
		for(int v = 0;v < img.rows;v++){
			for(int u = 0;u < img.cols;u++){
				pBlobProto->data[cnt++] = img.ptr<float >(v,u)[1];				
			}
		}
		for(int v = 0;v < img.rows;v++){
			for(int u = 0;u < img.cols;u++){
				pBlobProto->data[cnt++] = img.ptr<float >(v,u)[2];				
			}
		}

	} else {
		for(int v = 0;v < img.rows;v++){
			for(int u = 0;u < img.cols;u++){
				pBlobProto->data[cnt++] = img.ptr<float >(v,u)[0];		
			}
		}		
	}

	return 0;
}


char  *getProtofilename(char *src)
{
	char buf[128];
	strcpy(buf,src);
	
	char *ptr = strrchr(buf,'.');
	if (ptr == NULL) {
		return NULL;
	}
	strcpy(ptr,".blobproto");
	
	char *dst = (char *)malloc(sizeof(char) * (strlen(buf) + 1));
	strcpy(dst,buf);
	
	return dst;
}


int convertBlobprotoToMat(Mat &img,BLOB_t *pBlob)
{
	
}





