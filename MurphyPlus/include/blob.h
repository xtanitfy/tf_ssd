#ifndef __BLOB_H__
#define __BLOB_H__

#include "public.h"
#include "data_type.h"
#include "parameter.h"

#define BLOB_MAX_AXES 32

typedef struct
{ 
	DATA_TYPE *data_;
	BOOL dataMemMalloced;
	
	DATA_TYPE *diff_;
	BOOL diffMemMalloced;
	
	int shape_[BLOB_MAX_AXES];
	int shape_cnt_;
	int count_;
	int capacity_;
}BLOB_t;

#if defined(ACCURACY_FLOAT)
#define BLOB_toProto BLOB_toProto_f

#elif defined(ACCURACY_DOUBLE)
#define BLOB_toProto BLOB_toProto_d

#else 
//not support
#endif

void BLOB_init(BLOB_t *pBlob);
void BLOB_initByNCHW(BLOB_t *pBlob,int num,int channels,int height,int width);
void BLOB_initByShape(BLOB_t *pBlob,int *shape,int cnt);

BLOB_t *BLOB_create();
BLOB_t *BLOB_createByNCHW(int num,int channels,int height,int width);
BLOB_t *BLOB_createByShape(int *shape,int cnt);
int BLOB_reshapeByNCHW(BLOB_t *pBlob,int number,int channel,int height,int width);
int BLOB_reshapeByArray(BLOB_t *pBlob,int *shape,int num);
int BLOB_reshapeByArrayKeepMemory(BLOB_t *pBlob,int *shape,int num);
int BLOB_reshapeByBlobShape(BLOB_t *pBlob,BlobShape *shape);
void BLOB_reshapeLike(BLOB_t *pBlob,BLOB_t *pOtherBlob);
int BLOB_shape(BLOB_t *pBlob,int *shape,int *pCnt);
DATA_TYPE* BLOB_data(BLOB_t *pBlob);
DATA_TYPE* BLOB_diff(BLOB_t *pBlob);
int BLOB_setCpuData(BLOB_t *pBlob,DATA_TYPE * data);
int BLOB_shareData(BLOB_t *pBlob,BLOB_t *other);
int BLOB_shareDiff(BLOB_t *pBlob,BLOB_t *other);
DATA_TYPE BLOB_asumDiff(BLOB_t *pBlob);
DATA_TYPE BLOB_asumData(BLOB_t *pBlob);
void BLOB_scale_data(BLOB_t *pBlob,DATA_TYPE scale_factor);
void BLOB_scale_diff(BLOB_t *pBlob,DATA_TYPE scale_factor);
BOOL BLOB_shapeEquals(BLOB_t *pBlob,const BlobProto *pOther);
int BLOB_CopyFrom(BLOB_t *pBlobDst,BLOB_t *pBlobSrc, BOOL copy_diff, BOOL reshape);
BOOL BLOB_shapeEqualsBlob(BLOB_t *pBlobDst,BLOB_t *pBlobSrc);
int BLOB_fromProto(BLOB_t *pBlob,BlobProto *proto, BOOL reshape);
int BLOB_toProto_f(BLOB_t *pBlob,BlobProto *proto, BOOL write_diff);
int BLOB_toProto_d(BLOB_t *pBlob,BlobProto *proto, BOOL write_diff);
void BLOB_shapeString(BLOB_t *pBlob,char *buf,int len);
int BLOB_num_axes(BLOB_t *pBlob);
int BLOB_CanonicalAxisIndex(BLOB_t *pBlob,int axis_index);
int BLOB_count(BLOB_t *pBlob);
int BLOB_countByStart(BLOB_t *pBlob,int start_axis);
int BLOB_countByStartAndEnd(BLOB_t *pBlob,int start_axis, int end_axis);
int BLOB_num(BLOB_t *pBlob);
int BLOB_channels(BLOB_t *pBlob);
int BLOB_height(BLOB_t *pBlob);
int BLOB_width(BLOB_t *pBlob);
int BLOB_shapeByIndex(BLOB_t *pBlob,int index);
int BLOB_legacyShape(BLOB_t *pBlob,int index);
int BLOB_offsetByNCHW(BLOB_t *pBlob,int n,int c,int h,int w);
int BLOB_offsetByIndices(BLOB_t *pBlob,int *pIndices,int num);
DATA_TYPE BLOB_data_at(BLOB_t *pBlob,int n, int c, int h,int w);
DATA_TYPE BLOB_diff_at(BLOB_t *pBlob,int n, int c, int h,int w);
DATA_TYPE BLOB_data_at_byIndices(BLOB_t *pBlob,int *pIndices,int num);
DATA_TYPE BLOB_diff_at_byIndices(BLOB_t *pBlob,int *pIndices,int num);
DATA_TYPE *BLOB_data(BLOB_t *pBlob);
DATA_TYPE *BLOB_diff(BLOB_t *pBlob);
void BLOB_freeProtoMemory(BlobProto *proto);
int BLOB_saveBlobProtoToBinaryfile(BlobProto *pBlobProto,char *file);
int BLOB_loadBlobProtoFromBinaryfile(BLOB_t *pBlob,char *blobProtoFile);
void BLOB_subtract(BLOB_t *pBlob,DATA_TYPE value);
int BLOB_writeTxt(char *txtfile,BLOB_t *pBlob);
void BLOB_printShapeString(BLOB_t *pBlob,char *title);
BOOL BLOB_shapeEqualsByArr(BLOB_t *pBlobSrc,int *pShape,int len);
void  BLOB_writeTopBlobToTxtFile(char *layername,BLOB_t **pBlob,int len);
int BLOB_subtractArray(BLOB_t *pBlob,int channelAxes,DATA_TYPE *arr,int len);

#endif
