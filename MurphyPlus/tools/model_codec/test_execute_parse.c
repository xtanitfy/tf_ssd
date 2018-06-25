#include "parameter.h"
#include "public.h"

extern int parseNetParameter(NetParameter * netParameter);

int main(void)
{
	NetParameter net;
	int ret = parseNetParameter(&net);
	CHECK_EXPR_RET(ret < 0,-1);
	
	printf("net.name:%s\n",net.name);
	printf("net.layer_size:%d\n",net.layer_size);
	//printf("net.layer[0].transform_param.mean_value[0]:%f\n",
	//		net.layer[0].transform_param.mean_value[0]);
	//printf("netParameter->layer[0].video_data_param.video_file:%s\n",
		//	net.layer[0].video_data_param.video_file);		
	
	printf("Well done! Test execute parseNetParameter prototxt OK!\n");
	return 0;
}
