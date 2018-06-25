#ifndef __DETECTION_OUTPUT_LAYER_H__
#define __DETECTION_OUTPUT_LAYER_H__
#include "blob.h"
#include "layer.h"
#include "public.h"
#include "math_functions.h"
#include "io.h"
#include "bbox_util.h"

typedef struct
{
	int label;
	int bboxIdx;
	float score;
}OUTPUT_ITEM_t;

typedef struct
{
	OUTPUT_ITEM_t *pItem;
	int itemSize;
}BATCH_OUTPUT_INFO_t;

typedef struct
{
	int num_classes_;
	BOOL share_location_;
	int num_loc_classes_;
	int background_label_id_;
	PriorBoxParameter_CodeType code_type_;
	BOOL variance_encoded_in_target_;
	int keep_top_k_;
	float confidence_threshold_;

	int num_;
	int num_priors_;

	float nms_threshold_;
	int top_k_;

	BOOL need_save_;
	char *output_directory_;
	char *output_name_prefix_;
	char *output_format_;
	MAP_LABEL_NAME_t *label_to_name_;
	int label_to_name_size;
	
	MAP_LABEL_DISPLAYNAME_t *label_to_display_name_;
	int label_to_display_name_size;
	
	char (*names_)[PARSE_STR_NAME_SIZE];
	int names_size;
	
	int num_test_image_;
	int name_count_;

	BOOL visualize_;
	float visualize_threshold_;
	
	VEC_BBOX_t *all_loc_preds;
	int all_loc_preds_size;

	VEC_CONF_t *conf_preds;
	int conf_preds_size;

	VEC_BBOX_t prior_bboxes;

	VEC_VARIANCE_t *variance;	
	int variance_size;

	VEC_BBOX_t *all_decode_bboxes;
	int all_decode_bboxes_size;

	MAP_LABEL_BBOX_IDX_t *scoreKeepBboxIdx;
	int scoreKeepBboxIdxSize;

	MAP_LABEL_BBOX_IDX_t *nmsKeepBboxIdx;
	int nmsKeepBboxIdxSize;

	BATCH_OUTPUT_INFO_t *pBatchOutputInfo;
}DETECTION_OUTPUT_THIS;


int DetectionOutputLayer_setUp(LAYER_t *pLayer);
int DetectionOutputLayer_reshape(LAYER_t *pLayer);
int DetectionOutputLayer_forward(LAYER_t *pLayer);


#endif