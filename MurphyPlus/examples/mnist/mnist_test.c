#include "mnist_test.h"

#define WEIGHTS_BIN_FILE "model/weigts.bin"


int main(int agrc,char **argv)
{
	NET_t *pNet = NET_create(TEST, WEIGHTS_BIN_FILE);
	CHECK_EXPR_RET(pNet == NULL, -1);

	NET_RES_t res;
	NET_forward(pNet,&res);
	
	
	return 0;
}
