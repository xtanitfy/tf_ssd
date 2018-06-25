#ifndef __WB_DECODE_H__
#define __WB_DECODE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "public.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "data_type.h"
#include "dlist.h"
#include "wb_codec.h"
#include "parameter.h"

typedef struct
{
	char layerName[WB_LAYER_NAME_MAX_SIZE];
	BlobProto blobProto[WB_ONE_LAYER_MAX_BLOB];
	int blobSize;
	struct list_head list;
}WB_LAYER_BLOB_t;


struct list_head *WB_Decode(char *wbFile);
int WB_loadToNet(char *wbFile,NetParameter *pNet);

#endif