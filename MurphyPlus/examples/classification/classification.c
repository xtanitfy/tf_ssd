#include "classification.h"

#define WEIGHTS_BIN_FILE "model/weigts.bin"

static char  *getProtofilename(char *src);
static float getMean(char *filename);
static char  *getMeanfilename(char *src);

static char *model_file = NULL;
static char *trained_file = NULL;
static char *mean_file = NULL;
static char *label_file = NULL;
static char *image_file = NULL;

char *gTestLayersName[] = {
	"conv1","pool1","conv2","pool2","ip1","relu1","ip2","prob"
};

void usage()
{
	printf("argv[1]:network.caffemodel\n");
	printf("argv[2]:mean.binaryproto\n");
	printf("argv[3]:labels.txt\n");
	printf("argv[4]:image  file\n");
}

int main(int argc,char **argv)
{
	if (argc != 5) {
		usage();
		return 1;
	}
	trained_file = argv[1];
	mean_file = argv[2];
	label_file = argv[3];
	image_file = argv[4];

	//NET_setTestLayerName(gTestLayersName,DIM_OF(gTestLayersName));
	
	NET_t *pNet = NET_create(TEST, trained_file);
	CHECK_EXPR_RET(pNet == NULL, -1);
	printf("build network!\n");

	BLOB_t blob;
	BLOB_loadBlobProtoFromBinaryfile(&blob, getProtofilename(image_file));
	//BLOB_writeTxt("input_blob.txt",&blob);
	
	BLOB_subtract(&blob,getMean(getMeanfilename(mean_file)));
	//BLOB_writeTxt("input_blob.txt",&blob);

	//NET_printAllBlobs(pNet);
	NET_feedData(pNet, &blob, 1);
	
	//NET_printAllBlobs(pNet);
	NET_RES_t res;
	NET_forward(pNet,&res);
	
	return 0;
}


float getMean(char *filename)
{
	FILE *fp = fopen(filename,"rb");
	CHECK_EXPR_RET(fp == NULL, -1);

	char buf[128];
	int nread = fread(buf,sizeof(buf)-1,1,fp);
	buf[nread-1] = '\0';

	fclose(fp);

	float meanValue = atof(buf);
	printf("meanValue:%f\n",meanValue);
	
	return meanValue;
}

char  *getProtofilename(char *src)
{
	char buf[128];
	strcpy(buf,src);
	
	char *ptr = strrchr(buf,'.');
	CHECK_EXPR_RET(ptr== NULL, NULL);
	strcpy(ptr,".blobproto");
	
	char *dst = (char *)malloc(sizeof(char) * (strlen(buf) + 1));
	strcpy(dst,buf);

	printf("protofilename:%s\n",dst);

	return dst;
}

char  *getMeanfilename(char *src)
{
	char buf[128];
	strcpy(buf,src);
	
	char *ptr = strrchr(buf,'.');
	if (ptr == NULL) {
		return NULL;
	}
	strcpy(ptr,".txt");
	
	char *dst = (char *)malloc(sizeof(char) * (strlen(buf) + 1));
	strcpy(dst,buf);
	
	return dst;
}

