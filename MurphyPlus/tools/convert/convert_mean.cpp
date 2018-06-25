#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <stdio.h>
#include "public.h"

using namespace cv;
using namespace caffe;
using namespace std;

static float getMeanValue(const string& mean_file,int num_channels_,
							Size input_geometry_,float *meanval);

static int writeMeanToFile(char *filename,float meanVal);
static char  *getMeanfilename(char *src);

int main(int argc,char **argv)
{
	if (argc != 5) {
		printf("argv[1]:mean file!\n");
		printf("argv[2]:channel size\n");
		printf("argv[3]:height\n");
		printf("argv[4]:width\n");
		return -1;
	}
	string meanFile = argv[1];
	int num_channels_ = atoi(argv[2]);
	Size input_geometry_(atoi(argv[4]),atoi(argv[3]));
	
	float meanval;
	meanval = getMeanValue(meanFile,num_channels_,input_geometry_,&meanval);
	//cout << "meanval:" << meanval << endl;
	writeMeanToFile(getMeanfilename(argv[1]),meanval);
	return 0;
}

int writeMeanToFile(char *filename,float meanVal)
{
	FILE *fp = fopen(filename,"w+");
	CHECK_EXPR_RET(fp == NULL, -1);
#if 0
	char *str = "1\n";
	int nwrite = fwrite(str,strlen(str),1,fp);
	CHECK_EXPR_RET(nwrite != 1,-1);
#endif
	char buf[128];
	snprintf(buf,sizeof(buf),"%f\n",meanVal);
	int nwrite = fwrite(buf,strlen(buf),1,fp);
	CHECK_EXPR_RET(nwrite != 1,-1);

	fclose(fp);
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



/* Load the mean file in binaryproto format. */
float getMeanValue(const string& mean_file,int num_channels_, Size input_geometry_,float *meanval) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  cout << "channel_mean:" << channel_mean(0) << endl;
  return channel_mean(0);
 //  Mat mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
//	*meanval = channel_mean;
}

