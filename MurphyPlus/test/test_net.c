#include "net.h"

#define WEIGHTS_BIN_FILE "model/weigts.bin"


#define LAYER_FORWARD_FUNC(layername) layername##_layer_forward 

int conv_layer_forward()
{
	printf("conv_layer_forward\n");
	return 0;
}

int main(int argc,char **argv)
{
	NET_t *pNet = NET_create(TEST, WEIGHTS_BIN_FILE);
	CHECK_EXPR_RET(pNet == NULL, -1);

	NET_RES_t res;
	NET_forward(pNet,&res);
	
	return 0;
}

