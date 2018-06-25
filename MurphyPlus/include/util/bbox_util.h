#ifndef __BBOX_UTIL_H__
#define __BBOX_UTIL_H__

#include "public.h"
#include "parameter.h"
#include "math_functions.h"

typedef struct
{
	NormalizedBBox *pBBox;
	int bbox_size;
}VEC_BBOX_t;

typedef struct
{
	int label;
	float *conf;
	int conf_size;
}MAP_LABEL_CONF_BBOX_t;

typedef struct
{
	MAP_LABEL_CONF_BBOX_t *pMapLabelConf;
	int map_size;
}VEC_CONF_t;

typedef struct
{
	float *data;
	int size;
}VEC_VARIANCE_t;

typedef struct
{
	int label;
	int *bboxIdx;
	int bboxSize;
}MAP_LABEL_BBOX_IDX_t;

void GetLocPredictions(DATA_TYPE * loc_data, int num,
      	int num_preds_per_class, VEC_BBOX_t *pVecBBox);
void GetConfidenceScores(DATA_TYPE *conf_data, int num,
     	int num_preds_per_class,int num_classes,VEC_CONF_t *conf_preds);
float BBoxSize(NormalizedBBox *pBbox, BOOL normalized);
void GetPriorBBoxes(DATA_TYPE *prior_data, const int num_priors,
      VEC_BBOX_t *prior_bboxes,VEC_VARIANCE_t *prior_variances);
void DecodeBBox(NormalizedBBox *prior_bbox, float *prior_variance,
    PriorBoxParameter_CodeType code_type, BOOL variance_encoded_in_target,
    NormalizedBBox *bbox, NormalizedBBox* decode_bbox);
void DecodeBBoxesAll(VEC_BBOX_t *all_loc_preds,VEC_BBOX_t *prior_bboxes,
    VEC_VARIANCE_t *prior_variances,int num,PriorBoxParameter_CodeType code_type, 
    BOOL variance_encoded_in_target,VEC_BBOX_t *all_decode_bboxes);
void IntersectBBox(NormalizedBBox *bbox1, NormalizedBBox *bbox2,
                   NormalizedBBox* intersect_bbox);
float JaccardOverlap(NormalizedBBox *bbox1, NormalizedBBox *bbox2,BOOL normalized);
void ApplyKeepTopScore(float *scores,int priorbox_size,float score_threshold,
				int top_k, int *indices,int *pIndicesSize);
void ApplyNMS(NormalizedBBox *bboxes,float nms_threshold, 
				int *srcIndices,int srcIndicesSize,
						int *dstIndices,int *pDstIndicesSize);
void ClipBBox(NormalizedBBox *bbox, NormalizedBBox *clip_bbox);
void sortIntArrInverseByScore(int *arr,int len,float *score);

#endif