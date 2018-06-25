#include "bbox_util.h"

void GetLocPredictions(DATA_TYPE * loc_data, int num,
      	int num_preds_per_class, VEC_BBOX_t *pVecBBox)
{	
	for (int i = 0; i < num; i++) {
		CHECK_EXPR_NORET(pVecBBox[i].bbox_size != num_preds_per_class);
		for (int j = 0;j < num_preds_per_class;j++) {
			int start_idx = j * 4;
			pVecBBox[i].pBBox[j].xmin = loc_data[start_idx];
			pVecBBox[i].pBBox[j].ymin = loc_data[start_idx+1];
			pVecBBox[i].pBBox[j].xmax = loc_data[start_idx+2];
			pVecBBox[i].pBBox[j].ymax = loc_data[start_idx+3];
		}
		loc_data += num_preds_per_class*4;
	}
}

void GetConfidenceScores(DATA_TYPE *conf_data, int num,
     	int num_preds_per_class,int num_classes,VEC_CONF_t *conf_preds)
{
	for (int i = 0;i < num;i++) {
		CHECK_EXPR_NORET(conf_preds[i].map_size != num_classes);
		for (int p = 0;p < num_preds_per_class;p++) {
			int start_idx = p * num_classes;
			for (int c = 0;c < conf_preds[i].map_size;c++) {
				MAP_LABEL_CONF_BBOX_t *pLabelConf = &conf_preds[i].pMapLabelConf[c];
				pLabelConf->conf[p] = conf_data[start_idx + c];
			}
		}
		conf_data += num_preds_per_class * num_classes;
	}
}

float BBoxSize(NormalizedBBox *pBbox, BOOL normalized) 
{
	if (pBbox->xmax < pBbox->xmin || pBbox->ymax < pBbox->ymin) {
		// If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		return 0;
	} else {
		if (pBbox->size > 0) {
			return pBbox->size;
		} else {
			float width = pBbox->xmax - pBbox->xmin;
			float height = pBbox->ymax - pBbox->ymin;
			if (normalized) {
				return width * height;
			} else {
				// If bbox is not within range [0, 1].
				return (width + 1) * (height + 1);
			}
		}
	}
}


void GetPriorBBoxes(DATA_TYPE *prior_data, const int num_priors,
      VEC_BBOX_t *prior_bboxes,VEC_VARIANCE_t *prior_variances)
{
	for (int i = 0; i < num_priors; ++i) {
		int start_idx = i * 4;
		NormalizedBBox bbox;
		bbox.xmin = prior_data[start_idx];
		bbox.ymin = prior_data[start_idx+1];
		bbox.xmax = prior_data[start_idx+2];
		bbox.ymax = prior_data[start_idx+3];
		float bbox_size = BBoxSize(&bbox,TRUE);
		bbox.size = bbox_size;

		NormalizedBBox *pBbox = &prior_bboxes->pBBox[i];
		memcpy(pBbox,&bbox,sizeof(bbox));
	}
	for (int i = 0; i < num_priors; ++i) {
		int start_idx = (num_priors + i) * 4;
		prior_variances[i].size = 4;
		for (int j = 0; j < 4; ++j) {
			prior_variances[i].data[j] = prior_data[start_idx + j];
		}
	}
}



void DecodeBBox(NormalizedBBox *prior_bbox, float *prior_variance,
    PriorBoxParameter_CodeType code_type, BOOL variance_encoded_in_target,
    NormalizedBBox *bbox, NormalizedBBox* decode_bbox) 
{
	if (code_type == PriorBoxParameter_CodeType_CORNER) {
		if (variance_encoded_in_target) {
			// variance is encoded in target, we simply need to add the offset
			// predictions.
			decode_bbox->xmin = prior_bbox->xmin + bbox->xmin;
			decode_bbox->ymin = prior_bbox->ymin + bbox->ymin;
			decode_bbox->xmax = prior_bbox->xmax + bbox->xmax;
			decode_bbox->ymax = prior_bbox->ymax + bbox->ymax;
		} else {
			decode_bbox->xmin = prior_bbox->xmin + prior_variance[0] * bbox->xmin;
			decode_bbox->ymin = prior_bbox->ymin + prior_variance[0] * bbox->ymin;
			decode_bbox->xmax = prior_bbox->xmax + prior_variance[0] * bbox->xmax;
			decode_bbox->ymax = prior_bbox->ymax + prior_variance[0] * bbox->ymax;
		}
	} else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
			
		float prior_width = prior_bbox->xmax - prior_bbox->xmin;
		CHECK_EXPR_NORET(prior_width <= 0);

		float prior_height = prior_bbox->ymax - prior_bbox->ymin;
		CHECK_EXPR_NORET(prior_height <= 0);

		float prior_center_x = (prior_bbox->xmin + prior_bbox->xmax) / 2.;
		float prior_center_y = (prior_bbox->ymin + prior_bbox->ymax) / 2.;

		float decode_bbox_center_x, decode_bbox_center_y;
		float decode_bbox_width, decode_bbox_height;
		if (variance_encoded_in_target) {
			// variance is encoded in target, we simply need to retore the offset
			// predictions.
			decode_bbox_center_x = bbox->xmin * prior_width + prior_center_x;
			decode_bbox_center_y = bbox->ymin * prior_height + prior_center_y;
			decode_bbox_width = exp(bbox->xmax) * prior_width;
			decode_bbox_height = exp(bbox->ymax) * prior_height;
		} else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decode_bbox_center_x =
			prior_variance[0] * bbox->xmin * prior_width + prior_center_x;
			decode_bbox_center_y =
			prior_variance[1] * bbox->ymin * prior_height + prior_center_y;
			decode_bbox_width =
			exp(prior_variance[2] * bbox->xmax) * prior_width;
			decode_bbox_height =
			exp(prior_variance[3] * bbox->ymax) * prior_height;
		}
		decode_bbox->xmin = decode_bbox_center_x - decode_bbox_width / 2.;
		decode_bbox->ymin = decode_bbox_center_y - decode_bbox_height / 2.;
		decode_bbox->xmax = decode_bbox_center_x + decode_bbox_width / 2.;
		decode_bbox->ymax = decode_bbox_center_y + decode_bbox_height / 2.;
	} else {
		CHECK_EXPR_NORET(TRUE);
	}
	
	float bbox_size = BBoxSize(decode_bbox,TRUE);
	decode_bbox->size = bbox_size;
}

void DecodeBBoxesAll(VEC_BBOX_t *all_loc_preds,VEC_BBOX_t *prior_bboxes,
    VEC_VARIANCE_t *prior_variances,int num,PriorBoxParameter_CodeType code_type, 
    BOOL variance_encoded_in_target,VEC_BBOX_t *all_decode_bboxes) 
{
	if (prior_bboxes->bbox_size > 1) {
		CHECK_EXPR_NORET(prior_variances[0].size != 4);
	}
	for (int i = 0;i < num;i++) {
		for (int p = 0;p < prior_bboxes->bbox_size;p++) {
			DecodeBBox(&prior_bboxes->pBBox[p],prior_variances[p].data,code_type,
					variance_encoded_in_target,&all_loc_preds[i].pBBox[p],
					&all_decode_bboxes[i].pBBox[p]);
		}
	}
}

void IntersectBBox(NormalizedBBox *bbox1, NormalizedBBox *bbox2,
                   NormalizedBBox* intersect_bbox) 
{
	if (bbox2->xmin > bbox1->xmax || bbox2->xmax < bbox1->xmin ||
			bbox2->ymin > bbox1->ymax || bbox2->ymax < bbox1->ymin) {
		// Return [0, 0, 0, 0] if there is no intersection.
		intersect_bbox->xmin = 0;
		intersect_bbox->ymin = 0;
		intersect_bbox->xmax = 0;
		intersect_bbox->ymax = 0;
	} else {
		intersect_bbox->xmin = VOS_MAX(bbox1->xmin, bbox2->xmin);
		intersect_bbox->ymin = VOS_MAX(bbox1->ymin, bbox2->ymin);
		intersect_bbox->xmax = VOS_MIN(bbox1->xmax, bbox2->xmax);
		intersect_bbox->ymax = VOS_MIN(bbox1->ymax, bbox2->ymax);
	}
}

float JaccardOverlap(NormalizedBBox *bbox1, NormalizedBBox *bbox2,BOOL normalized) 
{
	NormalizedBBox intersect_bbox;
	IntersectBBox(bbox1, bbox2, &intersect_bbox);
	float intersect_width, intersect_height;
	if (normalized) {
		intersect_width = intersect_bbox.xmax - intersect_bbox.xmin;
		intersect_height = intersect_bbox.ymax - intersect_bbox.ymin;
	} else {
		intersect_width = intersect_bbox.xmax - intersect_bbox.xmin + 1;
		intersect_height = intersect_bbox.ymax - intersect_bbox.ymin + 1;
	}
	if (intersect_width > 0 && intersect_height > 0) {
		float intersect_size = intersect_width * intersect_height;
		float bbox1_size = BBoxSize(bbox1,TRUE);
		float bbox2_size = BBoxSize(bbox2,TRUE);
		return intersect_size / (bbox1_size + bbox2_size - intersect_size);
	} else {
		return 0.;
	}
}


void sortIntArrInverseByScore(int *arr,int len,float *score)
{
	int tmp;
	BOOL flag = FALSE;
	for (int i = 0;i < len;i++) {
		flag = FALSE;
		for (int j = 0;j < len - i- 1;j++) {
			if (score[arr[j]] < score[arr[j+1]]) {
				tmp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = tmp;
				flag = TRUE;
			}
		}
		if (flag == FALSE) {
			break;
		}
	}
	
}

void ApplyKeepTopScore(float *scores,int priorbox_size,float score_threshold,
				int top_k, int *indices,int *pIndicesSize)
{
	int indiceSize = 0;
	for (int i = 0;i < priorbox_size;i++) {
		if (scores[i] > score_threshold) {
			indices[indiceSize++] = i;
		}
	}
	sortIntArrInverseByScore(indices,indiceSize,scores);
	
	if (top_k > -1 && indiceSize > top_k) {
		indiceSize = top_k;
	}

	*pIndicesSize = indiceSize;
	
}

void ApplyNMS(NormalizedBBox *bboxes,float nms_threshold, 
				int *srcIndices,int srcIndicesSize,
						int *dstIndices,int *pDstIndicesSize)
{	
	int dstIndicesSize = 0;
	for (int i = 0;i < srcIndicesSize;i++) {
		BOOL keep = TRUE;
		int Idx = srcIndices[i];
		for (int j = 0;j < dstIndicesSize;j++) {
			if (keep == TRUE) {
				int keepIdx = dstIndices[j];
				float overlap = JaccardOverlap(&bboxes[Idx],&bboxes[keepIdx],TRUE);
				keep = overlap <= nms_threshold;
			}
			
		}
		if (keep == TRUE) {
			dstIndices[dstIndicesSize++] = Idx;
		}
	}
	
	*pDstIndicesSize = dstIndicesSize;
	
}

	  
void ClipBBox(NormalizedBBox *bbox, NormalizedBBox *clip_bbox)
{
	clip_bbox->xmin = VOS_MAX(VOS_MIN(bbox->xmin,1.f),0.f);
	clip_bbox->ymin = VOS_MAX(VOS_MIN(bbox->ymin,1.f),0.f);
	clip_bbox->xmax = VOS_MAX(VOS_MIN(bbox->xmax,1.f),0.f);
	clip_bbox->ymax = VOS_MAX(VOS_MIN(bbox->ymax,1.f),0.f);
	clip_bbox->size = BBoxSize(clip_bbox,TRUE);
	clip_bbox->difficult = bbox->difficult;
}


