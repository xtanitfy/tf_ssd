#include "blob.h"
#include "muti_tree.h"
#include "math_functions.h"

BLOB_t *BLOB_create()
{
	BLOB_t *pBlob = (BLOB_t *)malloc(sizeof(BLOB_t));
	CHECK_EXPR_RET(pBlob == NULL, NULL);

	pBlob->data_ = NULL;
	pBlob->diff_ = NULL;
	pBlob->shape_cnt_ = 0;
	pBlob->capacity_ = 0;
	pBlob->count_ = 0;
	pBlob->dataMemMalloced = FALSE;
	pBlob->diffMemMalloced = FALSE;
	return pBlob;
}

void BLOB_init(BLOB_t *pBlob)
{
	CHECK_EXPR_NORET(pBlob == NULL);
	pBlob->data_ = NULL;
	pBlob->diff_ = NULL;
	pBlob->shape_cnt_ = 0;
	pBlob->capacity_ = 0;
	pBlob->count_ = 0;
	pBlob->dataMemMalloced = FALSE;
	pBlob->diffMemMalloced = FALSE;
}

void BLOB_initByNCHW(BLOB_t *pBlob,int num,int channels,int height,int width)
{
	BLOB_init(pBlob);
	BLOB_reshapeByNCHW(pBlob,num,channels,height,width);
}

void BLOB_initByShape(BLOB_t *pBlob,int *shape,int cnt)
{
	BLOB_init(pBlob);
	BLOB_createByShape(shape,cnt);
}

BLOB_t *BLOB_createByNCHW(int num,int channels,int height,int width)
{
	BLOB_t *pBlob = BLOB_create();
	int ret = BLOB_reshapeByNCHW(pBlob,num,channels,height,width);
	CHECK_EXPR_RET(ret < 0, NULL);

	return pBlob;
}

BLOB_t *BLOB_createByShape(int *shape,int cnt)
{
	BLOB_t *pBlob = BLOB_create();
	int ret = BLOB_reshapeByArray(pBlob,shape,cnt);
	CHECK_EXPR_RET(ret < 0, NULL);
	return pBlob;
}


int BLOB_reshapeByNCHW(BLOB_t *pBlob,int num,int channels,int height,int width)
{
	CHECK_EXPR_RET(pBlob == NULL, -1);

	int shape[4];
	
	shape[0] = num;
	shape[1] = channels;
	shape[2] = height;
	shape[3] = width;
	BLOB_reshapeByArray(pBlob,shape,4);
	
	return 0;
}

int BLOB_reshapeByArrayKeepMemory(BLOB_t *pBlob,int *shape,int num)
{
	if (pBlob->dataMemMalloced == TRUE) {
		int count = 1;
		for (int i = 0;i < num;i++) {
			count *= shape[i];
		}
		if (BLOB_count(pBlob) < count) {
			CHECK_EXPR_RET(pBlob->data_ == NULL,-1);
			free(pBlob->data_);
			pBlob->dataMemMalloced = FALSE;
		}
	}
	
	BLOB_reshapeByArray(pBlob,shape,num);

	return 0;
}

int BLOB_reshapeByArray(BLOB_t *pBlob,int *shape,int num)
{
	CHECK_EXPR_RET(pBlob == NULL, -1);
	CHECK_EXPR_RET(shape == NULL, -1);
	CHECK_EXPR_RET(num > BLOB_MAX_AXES, -1);

	int *shapeArr = pBlob->shape_;

	pBlob->shape_cnt_ = num;
	pBlob->count_ = 1;
	for (int i = 0;i < num;i++) {
		shapeArr[i] = shape[i];
		pBlob->count_ *= shape[i];
	}

	if (pBlob->count_ > pBlob->capacity_) {
		pBlob->capacity_ = pBlob->count_;
	}

	return 0;
}

int BLOB_reshapeByBlobShape(BLOB_t *pBlob,BlobShape *shape) 
{
	CHECK_EXPR_RET(pBlob == NULL, -1);
	CHECK_EXPR_RET(shape == NULL,-1);
	CHECK_EXPR_RET(shape->dim_size > BLOB_MAX_AXES, -1);

	int shapeTmp[BLOB_MAX_AXES];
	int cnt = 0;
	for (int i = 0;i < shape->dim_size;i++) {
		shapeTmp[cnt++] =  shape->dim[i];
	}
	BLOB_reshapeByArray(pBlob,shapeTmp,cnt);

	return 0;
}

void BLOB_reshapeLike(BLOB_t *pBlob,BLOB_t *pOtherBlob) 
{
	CHECK_EXPR_NORET(pBlob == NULL);
	BLOB_reshapeByArray(pBlob,pOtherBlob->shape_,pOtherBlob->shape_cnt_);
}

int BLOB_shape(BLOB_t *pBlob,int *shape,int *pCnt)
{
	CHECK_EXPR_RET(pBlob == NULL, -1);
	CHECK_EXPR_RET(shape == NULL, -1);
	CHECK_EXPR_RET(pCnt == NULL, -1);
	
	int shapeCnt = pBlob->shape_cnt_;
	*pCnt = shapeCnt;
	for (int i = 0;i < shapeCnt;i++) {
		shape[i] = pBlob->shape_[i];
	}
	return 0;
}

DATA_TYPE *BLOB_data(BLOB_t *pBlob) 
{
	CHECK_EXPR_RET(pBlob == NULL, NULL);
	if (pBlob->dataMemMalloced == FALSE) {
		pBlob->data_ = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * pBlob->count_);
		CHECK_EXPR_RET(pBlob->data_ == NULL, NULL);
		memset(pBlob->data_,'\0',sizeof(DATA_TYPE) * pBlob->count_);
		pBlob->dataMemMalloced = TRUE;
	}
	return (DATA_TYPE *)pBlob->data_;
}

DATA_TYPE* BLOB_diff(BLOB_t *pBlob) 
{
	CHECK_EXPR_RET(pBlob == NULL, NULL);
	if (pBlob->diffMemMalloced == FALSE) {
		pBlob->diff_ = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * pBlob->count_);
		CHECK_EXPR_RET(pBlob->diff_ == NULL, NULL);
		pBlob->diffMemMalloced = TRUE;
	}
	return (DATA_TYPE *)pBlob->diff_;
}


int BLOB_setCpuData(BLOB_t *pBlob,DATA_TYPE * data) 
{
	CHECK_EXPR_RET(pBlob == NULL, -1);
	CHECK_EXPR_RET(data == NULL, -1);
	pBlob->data_ = data;

	return 0;
}

int BLOB_shareData(BLOB_t *pBlob,BLOB_t *other)
{
	CHECK_EXPR_RET(pBlob == NULL, -1);
	CHECK_EXPR_RET(other == NULL, -1);

	pBlob->data_ = other->data_;
	pBlob->dataMemMalloced = other->dataMemMalloced;
	return 0;
}

int BLOB_shareDiff(BLOB_t *pBlob,BLOB_t *other)
{
	CHECK_EXPR_RET(pBlob == NULL, -1);
	CHECK_EXPR_RET(other == NULL, -1);

	pBlob->diff_ = other->diff_;
	pBlob->diffMemMalloced = other->diffMemMalloced;
	return 0;
}

int BLOB_subtractArray(BLOB_t *pBlob,int channelAxes,DATA_TYPE *arr,int len)
{
	CHECK_EXPR_RET(BLOB_shapeByIndex(pBlob,channelAxes) != len,-1);

	int outNum = BLOB_countByStartAndEnd(pBlob,0,channelAxes);
	int spaceNum = BLOB_countByStart(pBlob,channelAxes+1);
	
	int k = 0;
	for (int i = 0;i < outNum;i++) {
		for (int c = 0;c < len;c++) {
			for (int j = 0;j < spaceNum;j++) {
				BLOB_data(pBlob)[k++] -= arr[c];
			}
		}
	}

	return 0;
}


void BLOB_subtract(BLOB_t *pBlob,DATA_TYPE value)
{
	for (int i = 0;i < pBlob->count_;i++) {
		pBlob->data_[i] -= value;
	}
}

DATA_TYPE BLOB_asumData(BLOB_t *pBlob)
{
	return Murphy_sum(pBlob->count_, pBlob->data_);
}

DATA_TYPE BLOB_asumDiff(BLOB_t *pBlob)
{
	return Murphy_sum(pBlob->count_, pBlob->diff_);
}


void BLOB_scale_data(BLOB_t *pBlob,DATA_TYPE scale_factor)
{
	Murphy_scale(pBlob->count_, scale_factor, pBlob->data_);
}


void BLOB_scale_diff(BLOB_t *pBlob,DATA_TYPE scale_factor)
{
	Murphy_scale(pBlob->count_, scale_factor, pBlob->diff_);
}

BOOL BLOB_shapeEquals(BLOB_t *pBlob,const BlobProto *pOther)
{
	int dimSize = pOther->shape.dim_size;
	if (dimSize != pBlob->shape_cnt_) {
		return FALSE;
	}
	
	BOOL equal = TRUE;
	for (int i = 0;i < dimSize;i++) {
		if (pOther->shape.dim[i] != pBlob->shape_[i]) {
			equal = FALSE;
			break;
		}
	}

	return equal;
}

BOOL BLOB_shapeEqualsByArr(BLOB_t *pBlobSrc,int *pShape,int len)
{
	BOOL equal = TRUE;
	if (pBlobSrc->shape_cnt_ != len) {
		equal = FALSE;
	} else {
		for (int i = 0;i < pBlobSrc->shape_cnt_;i++) {
			if (pBlobSrc->shape_[i] != pShape[i]) {
				equal = FALSE;
				break;
			}
		}
	}
	
	return equal;
}


BOOL BLOB_shapeEqualsBlob(BLOB_t *pBlobDst,BLOB_t *pBlobSrc)
{
	BOOL equal = TRUE;
	if (pBlobSrc->shape_cnt_ != pBlobDst->shape_cnt_) {
		equal = FALSE;
	} else {
		for (int i = 0;i < pBlobSrc->shape_cnt_;i++) {
			if (pBlobSrc->shape_[i] != pBlobDst->shape_[i]) {
				equal = FALSE;
				break;
			}
		}
	}
	
	return equal;
}

int BLOB_CopyFrom(BLOB_t *pBlobDst,BLOB_t *pBlobSrc, BOOL copy_diff, BOOL reshape)
{
	CHECK_EXPR_RET(pBlobDst == NULL, -1);
	CHECK_EXPR_RET(pBlobSrc == NULL, -1);

	printf("pBlobSrc->count_:%d\n",pBlobSrc->count_);
	printf("pBlobDst->count_:%d\n",pBlobDst->count_);

	if (pBlobSrc->count_ != pBlobDst->count_ || BLOB_shapeEqualsBlob(pBlobDst,pBlobSrc) == FALSE) {
		CHECK_EXPR_RET(reshape == FALSE, -1);
		printf("BLOB_CopyFrom reshape!\n");
		if (reshape == TRUE) {
			BLOB_reshapeLike(pBlobDst,pBlobSrc);
		}
	}

	if (copy_diff == TRUE)  {
		Murphy_copy(pBlobSrc->count_, BLOB_diff(pBlobSrc), BLOB_diff(pBlobDst));
	} else {
		Murphy_copy(pBlobSrc->count_, BLOB_data(pBlobSrc), BLOB_data(pBlobDst));
	}
	return 0;
}

void BLOB_freeProtoMemory(BlobProto *proto)
{
	CHECK_EXPR_NORET(proto == NULL);
	if (proto->data != NULL && proto->data_size > 0) {
		free(proto->data);	
		proto->data = NULL;
		proto->data_size = 0;
	}
	
	if (proto->diff != NULL && proto->diff_size > 0) {
		free(proto->diff);	
		proto->diff = NULL;
		proto->diff_size = 0;
	}
	
	if (proto->double_data != NULL && proto->double_data_size > 0) {
		free(proto->double_data);	
		proto->double_data = NULL;
		proto->double_data_size = 0;
	}
	
	if (proto->double_diff != NULL && proto->double_diff_size > 0) {
		free(proto->double_diff);
		proto->double_diff = NULL;
		proto->double_diff_size = 0;
	}
}

int BLOB_fromProto(BLOB_t *pBlob,BlobProto *proto, BOOL reshape)
{
	int shape[BLOB_MAX_AXES];
	int shapeNum = 0;
	
	if (reshape == TRUE) {
		//printf("proto->shape.dim_size:%d\n",proto->shape.dim_size);
		if (proto->shape.dim_size == 0) {
			shape[shapeNum++] = proto->num;
			shape[shapeNum++] = proto->channels;
			shape[shapeNum++] = proto->height;
			shape[shapeNum++] = proto->width;
		} else {
			CHECK_EXPR_RET(proto->shape.dim == NULL, -1);
			for (int i = 0;i < proto->shape.dim_size;i++) {
				shape[shapeNum++] = proto->shape.dim[i];
			}
		}
		BLOB_reshapeByArray(pBlob, shape, shapeNum);
	} else {
		CHECK_EXPR_RET(BLOB_shapeEquals(pBlob, proto) == FALSE,-1);
	}

	DATA_TYPE *pData = BLOB_data(pBlob);
	if (proto->double_data_size > 0) {
		CHECK_EXPR_RET(proto->double_data_size != pBlob->count_,-1);
		for (int i = 0;i < proto->double_data_size;i++) {
			pData[i] = proto->double_data[i];
		}
	} else {
		//printf("proto->data_size:%d pBlob->count_:%d\n",proto->data_size,pBlob->count_);
		CHECK_EXPR_RET(proto->data_size != pBlob->count_,-1);
		for (int i = 0;i < proto->data_size;i++) {
			pData[i] = proto->data[i];
		}
	}

	DATA_TYPE *pDataDiff = BLOB_diff(pBlob);
	if (proto->double_diff_size > 0) {
		if (proto->double_diff_size != pBlob->count_) {
			return 0;
		}
		for (int i = 0;i < proto->double_diff_size;i++) {
			pDataDiff[i] = proto->double_diff[i];
		}
	} else {
		if (proto->diff_size != pBlob->count_) {
			return 0;
		}
		for (int i = 0;i < proto->diff_size;i++) {
			pDataDiff[i] = proto->diff[i];
		}
	}
	pBlob->dataMemMalloced = TRUE;
	
	return 0;
}


int BLOB_toProto_f(BLOB_t *pBlob,BlobProto *proto, BOOL write_diff) 
{
	proto->shape.dim_size = pBlob->shape_cnt_;
	for (int i = 0;i < pBlob->shape_cnt_;i++) {
		proto->shape.dim[i] = pBlob->shape_[i];
	}

	if (proto->data != NULL) {
		free(proto->data);
	}
	proto->data = (float *)malloc(sizeof(float) * pBlob->count_);
	CHECK_EXPR_RET(proto->data == NULL, -1);
	proto->data_size = pBlob->count_;
	
	float *pDst = proto->data;
	DATA_TYPE *pData = pBlob->data_;
	for (int i = 0;i < pBlob->count_;i++) {
		pDst[i] = pData[i];
	}

	if (write_diff == TRUE) {
		proto->diff = (float *)malloc(sizeof(float) * pBlob->count_);
		CHECK_EXPR_RET(proto->diff == NULL, -1);
		proto->diff_size = pBlob->count_;

		float *pDst = proto->diff;
		DATA_TYPE *pSrc = pBlob->diff_;
		for (int i = 0;i < pBlob->count_;i++) {
			pDst[i] = pSrc[i];
		}
	} 

	return 0;
}


int BLOB_toProto_d(BLOB_t *pBlob,BlobProto *proto, BOOL write_diff) 
{
	proto->shape.dim_size = pBlob->shape_cnt_;
	for (int i = 0;i < pBlob->shape_cnt_;i++) {
		proto->shape.dim[i] = pBlob->shape_[i];
	}

	if (proto->double_data != NULL) {
		free(proto->double_data);
	}
	proto->double_data = (double *)malloc(sizeof(double) * pBlob->count_);
	CHECK_EXPR_RET(proto->data == NULL, -1);
	proto->double_data_size = pBlob->count_;
	
	double *pDst = proto->double_data;
	DATA_TYPE *pData = pBlob->data_;
	for (int i = 0;i < pBlob->count_;i++) {
		pDst[i] = pData[i];
	}

	if (write_diff == TRUE) {
		proto->double_diff = (double *)malloc(sizeof(double) * pBlob->count_);
		CHECK_EXPR_RET(proto->double_diff == NULL, -1);
		proto->double_diff_size = pBlob->count_;

		double *pDst = proto->double_diff;
		DATA_TYPE *pSrc = pBlob->diff_;
		for (int i = 0;i < pBlob->count_;i++) {
			pDst[i] = pSrc[i];
		}
	} 

	return 0;
}

void BLOB_printShapeString(BLOB_t *pBlob,char *title)
{
	char buf[1024];
	int len = sizeof(buf);

	BLOB_shapeString(pBlob,buf,len);

	printf("%s:%s\n",title,buf);
}

void BLOB_shapeString(BLOB_t *pBlob,char *buf,int len)
{
	char tmp[128];
	int cnt = 0;
	int tmpLen = 0;
	for (int i = 0;i < pBlob->shape_cnt_;i++) {
		snprintf(tmp,sizeof(tmp),"%d ",pBlob->shape_[i]);
		tmpLen = strlen(tmp);
		len -= tmpLen;
		CHECK_EXPR_NORET(len <= 0);
		strcpy(&buf[cnt],tmp);
		cnt += tmpLen;
	}
	
	snprintf(tmp,len,"(%d)",pBlob->count_);
	tmpLen = strlen(tmp);
	len -= tmpLen;
	CHECK_EXPR_NORET(len <= 0);
	strcpy(&buf[cnt],tmp);
}

int BLOB_num_axes(BLOB_t *pBlob)
{
	return pBlob->shape_cnt_;
}

int BLOB_CanonicalAxisIndex(BLOB_t *pBlob,int axis_index) 
{
	CHECK_EXPR_RET(axis_index < -BLOB_num_axes(pBlob), -1);
	CHECK_EXPR_RET(axis_index > BLOB_num_axes(pBlob), -1);

	if (axis_index < 0) {
		return axis_index + BLOB_num_axes(pBlob);
	}
	return axis_index;
}

int BLOB_count(BLOB_t *pBlob)
{
	return pBlob->count_;
}


int BLOB_countByStart(BLOB_t *pBlob,int start_axis)
{
	return BLOB_countByStartAndEnd(pBlob,start_axis, BLOB_num_axes(pBlob));
}

int BLOB_countByStartAndEnd(BLOB_t *pBlob,int start_axis, int end_axis)
{
	CHECK_EXPR_RET(start_axis > end_axis, -1);
	CHECK_EXPR_RET(start_axis < 0, -1);
	CHECK_EXPR_RET(end_axis < 0, -1);
	CHECK_EXPR_RET(start_axis > BLOB_num_axes(pBlob), -1);
	CHECK_EXPR_RET(end_axis > BLOB_num_axes(pBlob), -1);

	int count = 1;
	for (int i = start_axis; i < end_axis; ++i) {
		count *= pBlob->shape_[i];
	}
	return count;
	
}

int BLOB_num(BLOB_t *pBlob)
{
	return pBlob->shape_[0];
}

int BLOB_channels(BLOB_t *pBlob)
{
	return pBlob->shape_[1];
}

int BLOB_height(BLOB_t *pBlob)
{
	return pBlob->shape_[2];
}

int BLOB_width(BLOB_t *pBlob)
{
	return pBlob->shape_[3];
}

int BLOB_shapeByIndex(BLOB_t *pBlob,int index) 
{
	return pBlob->shape_[BLOB_CanonicalAxisIndex(pBlob,index)];
}
 

int BLOB_legacyShape(BLOB_t *pBlob,int index)  
{
	CHECK_EXPR_RET(BLOB_num_axes(pBlob) > 4, -1);
	CHECK_EXPR_RET(index > 4, -1);
	CHECK_EXPR_RET(index < -4, -1);
	if (index >= BLOB_num_axes(pBlob) || index < -BLOB_num_axes(pBlob)) {
      return 1;
    }
    return BLOB_shapeByIndex(pBlob,index);
}

int BLOB_offsetByNCHW(BLOB_t *pBlob,int n,int c,int h,int w)
{
	CHECK_EXPR_RET(n < 0, -1);
	CHECK_EXPR_RET(n > BLOB_num(pBlob), -1);
	CHECK_EXPR_RET(BLOB_channels(pBlob) < 0, -1);
	CHECK_EXPR_RET(c > BLOB_channels(pBlob), -1);
	CHECK_EXPR_RET(BLOB_height(pBlob) < 0, -1);
	CHECK_EXPR_RET(h > BLOB_height(pBlob), -1);
	CHECK_EXPR_RET(BLOB_width(pBlob) < 0, -1);
	CHECK_EXPR_RET(w > BLOB_width(pBlob), -1);
	
	 return ((n * BLOB_channels(pBlob) + c) * BLOB_height(pBlob) + h) * BLOB_width(pBlob) + w;
}


int BLOB_offsetByIndices(BLOB_t *pBlob,int *pIndices,int num)
{
	CHECK_EXPR_RET(num > BLOB_num_axes(pBlob), -1);
	int offset = 0;

	for (int i = 0; i < BLOB_num_axes(pBlob); ++i) {
		offset *= BLOB_shapeByIndex(pBlob,i);
		if (num > i) {
			CHECK_EXPR_RET(num > BLOB_num_axes(pBlob), -1);
			CHECK_EXPR_RET(pIndices[i] <  0,-1);
			CHECK_EXPR_RET(pIndices[i] >= BLOB_shapeByIndex(pBlob,i),-1);
			offset += pIndices[i];
		}
	}
	return offset;
}

DATA_TYPE BLOB_data_at(BLOB_t *pBlob,int n, int c, int h,int w)
{
	return pBlob->data_[BLOB_offsetByNCHW(pBlob, n,  c,  h,  w)];
}

DATA_TYPE BLOB_diff_at(BLOB_t *pBlob,int n, int c, int h,int w)
{
	return pBlob->diff_[BLOB_offsetByNCHW(pBlob, n,  c,  h,  w)];
}

DATA_TYPE BLOB_data_at_byIndices(BLOB_t *pBlob,int *pIndices,int num)
{
	return pBlob->data_[BLOB_offsetByIndices(pBlob, pIndices,  num)];
}


DATA_TYPE BLOB_diff_at_byIndices(BLOB_t *pBlob,int *pIndices,int num)
{
	return pBlob->diff_[BLOB_offsetByIndices(pBlob, pIndices,  num)];
}


int BLOB_saveBlobProtoToBinaryfile(BlobProto *pBlobProto,char *file)
{
	FILE *fp = fopen(file,"wb+");
	CHECK_EXPR_RET(fp == NULL, -1);
	
	int nwrite = fwrite(pBlobProto,sizeof(BlobProto),1,fp);
	CHECK_EXPR_RET(nwrite != 1, -1);

	unsigned long long size = 1;
	for (int i = 0;i < pBlobProto->shape.dim_size;i++) {
		size *= pBlobProto->shape.dim[i];
	}

	printf("pBlobProto->data_size:%d\n",pBlobProto->data_size);
	
	if (pBlobProto->data_size > 0) {
		int nwrite = fwrite(pBlobProto->data,sizeof(float),pBlobProto->data_size,fp);
		printf("1----\n");
		CHECK_EXPR_RET(nwrite != pBlobProto->data_size, -1);
	}

	if (pBlobProto->diff_size > 0) {
		int nwrite = fwrite(pBlobProto->diff,sizeof(float),pBlobProto->diff_size,fp);
		printf("2----\n");
		CHECK_EXPR_RET(nwrite != pBlobProto->diff_size, -1);
	}
	
	if (pBlobProto->double_data_size > 0) {
		int nwrite = fwrite(pBlobProto->double_data,sizeof(float),pBlobProto->double_data_size,fp);
		CHECK_EXPR_RET(nwrite != pBlobProto->double_data_size, -1);
	}

	if (pBlobProto->data_size > 0) {
		int nwrite = fwrite(pBlobProto->double_diff,sizeof(float),pBlobProto->double_diff_size,fp);
		CHECK_EXPR_RET(nwrite != pBlobProto->double_diff_size, -1);
	}
	
	fclose(fp);
	return 0;

}

void  BLOB_writeTopBlobToTxtFile(char *layername,BLOB_t **pBlob,int len)
{
	for (int i = 0;i < len;i++) {
		char filename[256];
		snprintf(filename,sizeof(filename),(char *)"%s_top%d.txt",layername,i);
		BLOB_writeTxt(filename,pBlob[i]);
	}
}

int BLOB_writeTxt(char *txtfile,BLOB_t *pBlob)
{
	char fullPath[256];
	snprintf(fullPath,sizeof(fullPath),"%s/%s",OUT_DIR,txtfile);
	FILE *fp = fopen(fullPath,"w+");
	CHECK_EXPR_RET(fp == NULL, -1);

	CHECK_EXPR_RET(pBlob->dataMemMalloced == FALSE,-1);
	
	char buf[64];
	int nwrite = 0;
	for (int i = 0;i < pBlob->count_;i++) {
		if (i % 32 == 0) {
			nwrite = fwrite("\n",1,1,fp);
			CHECK_EXPR_RET(nwrite != 1, -1);
		}
		snprintf(buf,sizeof(buf),"%.10f ",pBlob->data_[i]);
		nwrite = fwrite(buf,strlen(buf),1,fp);
		CHECK_EXPR_RET(nwrite != 1, -1);
	}

	fclose(fp);
	return 0;
}


int BLOB_loadBlobProtoFromBinaryfile(BLOB_t *pBlob,char *blobProtoFile)
{
	FILE *fp = fopen(blobProtoFile,"rb");
	CHECK_EXPR_RET(fp == NULL, -1);

	BlobProto blobProto;
	int nread = fread(&blobProto,sizeof(BlobProto),1,fp);
	CHECK_EXPR_RET(nread  != 1, -1);

	BlobProto *pBlobProto = &blobProto;
	printf("pBlobProto->data_size:%d\n",pBlobProto->data_size);
	if (pBlobProto->data_size > 0) {
		pBlobProto->data = (float *)malloc(sizeof(float) * pBlobProto->data_size);
		CHECK_EXPR_RET(pBlobProto->data == NULL, -1);
	
		int nread = fread(pBlobProto->data,sizeof(float),pBlobProto->data_size,fp);
		CHECK_EXPR_RET(nread != pBlobProto->data_size, -1);
	}

	if (pBlobProto->diff_size > 0) {
		pBlobProto->diff = (float *)malloc(sizeof(float) * pBlobProto->diff_size);
		CHECK_EXPR_RET(pBlobProto->diff == NULL, -1);
	
		int nread = fread(pBlobProto->diff,sizeof(float),pBlobProto->diff_size,fp);
		CHECK_EXPR_RET(nread != pBlobProto->diff_size, -1);
	}
	
	if (pBlobProto->double_data_size > 0) {
		pBlobProto->double_data = (double *)malloc(sizeof(double) * pBlobProto->double_data_size);
		CHECK_EXPR_RET(pBlobProto->double_data == NULL, -1);
	
		int nread = fread(pBlobProto->double_data,sizeof(float),pBlobProto->double_data_size,fp);
		CHECK_EXPR_RET(nread != pBlobProto->double_data_size, -1);
	}

	if (pBlobProto->data_size > 0) {
		pBlobProto->double_diff = (double *)malloc(sizeof(double) * pBlobProto->double_diff_size);
		CHECK_EXPR_RET(pBlobProto->double_diff == NULL, -1);
	
		int nread = fread(pBlobProto->double_diff,sizeof(float),pBlobProto->double_diff_size,fp);
		CHECK_EXPR_RET(nread != pBlobProto->double_diff_size, -1);
	}

	fclose(fp);
	printf("read finsih!\n");
	
	BLOB_init(pBlob);
	BLOB_fromProto(pBlob, pBlobProto, TRUE);
	BLOB_freeProtoMemory(pBlobProto);
	
	return 0;
}
