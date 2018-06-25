#include "wb_decode.h"

#define WB_BIN_FILE "model/weigts.bin"
extern int parseNetParameter(NetParameter * netParameter);

int main(int argc,char **argv)
{	
#if 0
	struct list_head *pHead = WB_Decode(WB_BIN_FILE);
	CHECK_EXPR_RET(pHead == NULL, -1);
	
	WB_LAYER_BLOB_t *pBlob = NULL;
	int i = 0;
	list_for_each_entry(pBlob, pHead, list, WB_LAYER_BLOB_t) {
		printf("[%d] layername:%s\n",i++,pBlob->layerName);
	}
#endif
	NetParameter net;
	printf("test wbdecode!\n");
	int ret = parseNetParameter(&net);
	CHECK_EXPR_RET(ret < 0, -1);
	printf("parseNetParameter finish!\n");
	
	ret = WB_loadToNet(WB_BIN_FILE, &net);
	CHECK_EXPR_RET(ret < 0, -1);
	
	return 0;
}
