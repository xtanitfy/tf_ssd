#include "detect.h"
#include "util.h"

#define MEAN_MAX_SIZE 3

static char  *getProtofilename(char *src);
static int getMean(char *filename);
static char  *getMeanfilename(char *src);
static int readFileOneline(FILE *fp,char *buf,int *len);

static char *model_file = NULL;
static char *trained_file = NULL;
static char *mean_file = NULL;
static char *label_file = NULL;
static char *image_file = NULL;
static float mean[MEAN_MAX_SIZE];
static int gmean_size = 0;

char *gFilterLayersName[] = {
	"detection_out"
};


char *gTestLayersName[] = {
	"conv1_1","relu1_1","conv1_2","relu1_2","pool1","conv2_1","relu2_1",
	"conv2_2","relu2_2","pool2","conv3_1","relu3_1","conv3_2","relu3_2",
	"conv3_3","relu3_3","pool3","conv4_1","relu4_1","conv4_2","relu4_2",
	"conv4_3","relu4_3","pool4","conv5_1","relu5_1","conv5_2","relu5_2",
	"conv5_3","relu5_3","pool5","fc6","relu6","fc7","relu7","conv6_1",
	"conv6_1_relu","conv6_2","conv6_2_relu","conv7_1","conv7_1_relu",
	"conv7_2","conv7_2_relu","conv8_1","conv8_1_relu","conv8_2","conv8_2_relu",
	"pool6","conv4_3_norm","conv4_3_norm_mbox_loc","conv4_3_norm_mbox_loc_perm",
	"conv4_3_norm_mbox_loc_flat","conv4_3_norm_mbox_conf","conv4_3_norm_mbox_conf_perm",
	"conv4_3_norm_mbox_conf_flat","conv4_3_norm_mbox_priorbox","fc7_mbox_loc","fc7_mbox_loc_perm",
	"fc7_mbox_loc_flat","fc7_mbox_conf","fc7_mbox_conf_perm","fc7_mbox_conf_flat","fc7_mbox_priorbox",
	"conv6_2_mbox_loc","conv6_2_mbox_loc_perm","conv6_2_mbox_loc_flat","conv6_2_mbox_conf","conv6_2_mbox_conf_perm",
	"conv6_2_mbox_conf_flat","conv6_2_mbox_priorbox","conv7_2_mbox_loc","conv7_2_mbox_loc_perm",
	"conv7_2_mbox_loc_flat","conv7_2_mbox_conf","conv7_2_mbox_conf_perm","conv7_2_mbox_conf_flat",
	"conv7_2_mbox_priorbox","conv8_2_mbox_loc","conv8_2_mbox_loc_perm","conv8_2_mbox_loc_flat",
	"conv8_2_mbox_conf","conv8_2_mbox_conf_perm","conv8_2_mbox_conf_flat","conv8_2_mbox_priorbox",
	"pool6_mbox_loc","pool6_mbox_loc_perm","pool6_mbox_loc_flat","pool6_mbox_conf","pool6_mbox_conf_perm",
	"pool6_mbox_conf_flat","pool6_mbox_priorbox","mbox_loc","mbox_conf","mbox_priorbox",
	
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
	//NET_setFilterLayerName(gFilterLayersName,DIM_OF(gFilterLayersName));
	
	int ret = getMean(mean_file);
	CHECK_EXPR_RET(ret < 0,-1);

	NET_t *pNet = NET_create(TEST, trained_file);
	CHECK_EXPR_RET(pNet == NULL, -1);
	printf("build network!\n");

	BLOB_t blob;
	BLOB_loadBlobProtoFromBinaryfile(&blob, getProtofilename(image_file));
	
	ret = BLOB_subtractArray(&blob,1,mean,gmean_size);
	CHECK_EXPR_RET(ret < 0,-1);

	
	NET_feedData(pNet, &blob, 1);

	//NET_printAllBlobs(pNet);
	NET_RES_t res;

	CalTimeStart();
	NET_forward(pNet,&res);
	CalTimeEnd("detect");
	
	return 0;
}


int readFileOneline(FILE *fp,char *buf,int *len)
{
	char ch = 0;
	int n = 0;
	do {
		int nread = fread(&ch,1,1,fp);
		if (nread <= 0) {
			break;
		}
		buf[n++] = ch;
				
	}while(ch != '\n');
	*len = n;

	if (n == 0) {
		return -1;
	}
	return 0;
}


int getMean(char *filename)
{
	FILE *fp = fopen(filename,"r");
	CHECK_EXPR_RET(fp == NULL, -1);

	char line[128];
	int len = 0;
	int ret = readFileOneline(fp,line,&len);
	CHECK_EXPR_RET(ret < 0,-1);
	
	int mean_size = atoi(line);

	int readLineNum = 0;
	gmean_size = 0;
	for (int i = 0;i < mean_size;i++) {
		ret = readFileOneline(fp,line,&len);
		if (ret < 0) {
			break;
		}
		mean[gmean_size++] = atof(line);
	}
	CHECK_EXPR_RET(gmean_size != mean_size,-1);

	for (int i = 0;i < gmean_size;i++) {
		printf("mean:%f\n",mean[i]);
	}
	//getchar();
	
	return 0;
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

