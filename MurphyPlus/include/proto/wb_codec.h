#ifndef __WB_CODEC_H__
#define __WB_CODEC_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//WB :weights binary file
#define WB_LAYER_NAME_MAX_SIZE 64
#define WB_ONE_LAYER_MAX_BLOB 4
#define WB_ONE_BLOB_MAX_DIM 4

typedef struct
{
	int layersNum;
}WB_HEAD_t;

typedef struct
{
	INT64 dim[WB_ONE_BLOB_MAX_DIM];
	int dim_size; 
	unsigned long blobFileOffset;
}WB_BLOB_t;

typedef struct
{
	char layerName[WB_LAYER_NAME_MAX_SIZE];
	WB_BLOB_t blob[WB_ONE_LAYER_MAX_BLOB];
	int blob_size; 
}WB_HEADT_BLOB_ITEM_t;

#endif