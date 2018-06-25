#include "detection_output_layer.h"
#include "io.h"

extern int parseLabelMap(LabelMap * labelMap);

void initDetectOutputInnerParam(DETECTION_OUTPUT_THIS *this)
{
	memset(this,'\0',sizeof(DETECTION_OUTPUT_THIS));
}

int DetectionOutputLayer_setUp(LAYER_t *pLayer)
{
	DETECTION_OUTPUT_THIS *this = (DETECTION_OUTPUT_THIS *)malloc(sizeof(DETECTION_OUTPUT_THIS));
	CHECK_EXPR_RET(this == NULL,-1);
	initDetectOutputInnerParam(this);
	pLayer->innerParam = this;
	
	DetectionOutputParameter  *pParam = &pLayer->pLayerParam->detection_output_param;
	CHECK_EXPR_RET(pParam->num_classes == 0,-1);
	this->num_classes_ = pParam->num_classes;
	this->share_location_ = pParam->share_location;
	this->num_loc_classes_ = this->share_location_ ? 1 : this->num_classes_;
	this->background_label_id_ = pParam->background_label_id;
	this->code_type_ = pParam->code_type;
	this->variance_encoded_in_target_ = pParam->variance_encoded_in_target;
	this->keep_top_k_ = pParam->keep_top_k;
	this->confidence_threshold_ = (pParam->confidence_threshold != 0) 
						? pParam->confidence_threshold : -FLT_MAX;
	this->nms_threshold_ = pParam->nms_param.nms_threshold;
	CHECK_EXPR_RET(this->nms_threshold_ < 0,-1);
	this->top_k_ = -1;
	if (pParam->nms_param.top_k != 0) {
		this->top_k_ = pParam->nms_param.top_k;
	}
	
	SaveOutputParameter *pSaveOutputParameter = &pParam->save_output_param;
	this->output_directory_ = pSaveOutputParameter->output_directory;
	if (this->output_directory_[0] != '\0') {
		IO_createdDirectory(this->output_directory_);
	}
	this->output_name_prefix_ = pSaveOutputParameter->output_name_prefix;
	this->need_save_ = this->output_directory_[0] == '\0' ? FALSE : TRUE;
	this->output_format_ = pSaveOutputParameter->output_format;
#if 1
	if (pSaveOutputParameter->label_map_file[0] != '\0') {
		LabelMap label_map;
		parseLabelMap(&label_map);
		IO_MapLabelToName(&label_map,
				TRUE,&this->label_to_name_,&this->label_to_name_size);
		IO_MapLabelToDisplayName(&label_map,
				TRUE,&this->label_to_display_name_,&this->label_to_display_name_size);
	} else {
		this->need_save_ = FALSE;
	}
#endif
	
	return 0;
}

int allocLabelBBoxIdxMemory(DETECTION_OUTPUT_THIS *this,
						MAP_LABEL_BBOX_IDX_t **pBBoxIdx,int *pSize)
{
	if (*pBBoxIdx != NULL) {
		return 0;
	}
	
	MAP_LABEL_BBOX_IDX_t *bboxIdx = NULL;
	int size = this->num_classes_; 
	bboxIdx = (MAP_LABEL_BBOX_IDX_t *)malloc(sizeof(MAP_LABEL_BBOX_IDX_t) *	size);
	CHECK_EXPR_RET(bboxIdx == NULL,-1);

	for (int i = 0;i < size;i++) {
		bboxIdx[i].label = i;
		bboxIdx[i].bboxIdx = (int *)malloc(sizeof(int) * this->num_priors_);
		CHECK_EXPR_RET(bboxIdx[i].bboxIdx == NULL,-1);
		bboxIdx[i].bboxSize = 0;
	}
	*pBBoxIdx = bboxIdx;
	*pSize = size;
	return 0;
}

int DetectionOutputLayer_reshape(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	DETECTION_OUTPUT_THIS *this = (DETECTION_OUTPUT_THIS *)pLayer->innerParam;
	
	CHECK_EXPR_RET(BLOB_num(pLayer->bottom[0]) != BLOB_num(pLayer->bottom[1]),-1);
	
	this->num_priors_ = BLOB_height(pLayer->bottom[2]) / 4;
	CHECK_EXPR_RET(this->num_priors_ * this->num_loc_classes_ * 4 != 
						BLOB_channels(pLayer->bottom[0]),-1);
	CHECK_EXPR_RET(this->num_priors_ * this->num_classes_  != 
						BLOB_channels(pLayer->bottom[1]),-1);

	int num = BLOB_num(pLayer->bottom[0]);
	int top_shape[2];
	int top_shape_size = 0;
	if (this->keep_top_k_ > 0) {
		top_shape[top_shape_size++] = num * this->keep_top_k_;
	} else {
		int nms_top_k = pLayer->pLayerParam->detection_output_param.nms_param.top_k;
		if (nms_top_k > 0) {
			top_shape[top_shape_size++] = num * this->num_classes_ * nms_top_k;
		} else {
			top_shape[top_shape_size++] = num * this->num_classes_ * this->num_priors_;
		}
	}

	printf("this->keep_top_k_:%d\n",this->keep_top_k_);
	printf("this->num_priors_:%d\n",this->num_priors_);
	// [image_id, label, confidence, xmin, ymin, xmax, ymax]
	top_shape[top_shape_size++] = 7;
	BLOB_reshapeByArray(pLayer->top[0],top_shape,top_shape_size);
	BLOB_data(pLayer->top[0]);//malloc the memory
#if 1
	
	//malloc the memory needed
	if (this->all_loc_preds == NULL) {
		//prediction bbox
		this->all_loc_preds_size = num;
		this->all_loc_preds = (VEC_BBOX_t *)malloc(sizeof(VEC_BBOX_t) * 
								this->all_loc_preds_size);
		CHECK_EXPR_RET(this->all_loc_preds == NULL,-1);
		for (int i = 0;i < this->all_loc_preds_size;i++) {
			this->all_loc_preds[i].pBBox = (NormalizedBBox *)malloc(this->num_priors_ *
									sizeof(NormalizedBBox));
			CHECK_EXPR_RET(this->all_loc_preds[i].pBBox == NULL,1);
			this->all_loc_preds[i].bbox_size = this->num_priors_;
		}
	}

	//confidence prediction
	if (this->conf_preds == NULL) {
		this->conf_preds_size = num;
		this->conf_preds = (VEC_CONF_t *)malloc(sizeof(VEC_CONF_t) * this->conf_preds_size);
		CHECK_EXPR_RET(this->conf_preds == NULL,-1);
		for (int i = 0;i < this->conf_preds_size;i++) {
			VEC_CONF_t *pConf = &this->conf_preds[i];
			pConf->map_size = this->num_classes_;
			pConf->pMapLabelConf = (MAP_LABEL_CONF_BBOX_t *)malloc(sizeof(MAP_LABEL_CONF_BBOX_t) * 
							pConf->map_size);
			CHECK_EXPR_RET(pConf->pMapLabelConf == NULL,-1);
			for (int j = 0;j < pConf->map_size;j++) {
				pConf->pMapLabelConf[j].conf_size = this->num_priors_;
				pConf->pMapLabelConf[j].conf = (float *)malloc(sizeof(float) * this->num_priors_);
				CHECK_EXPR_RET(pConf->pMapLabelConf[j].conf == NULL,-1);
			}
		}
	}
	
	//priorbox
	if (this->prior_bboxes.pBBox == NULL) {
		this->prior_bboxes.bbox_size = this->num_priors_;
		this->prior_bboxes.pBBox = (NormalizedBBox *)malloc(sizeof(NormalizedBBox) *
									this->prior_bboxes.bbox_size);
	}
	//prior variance
	if (this->variance == NULL) {
		this->variance_size = this->num_priors_;
		this->variance = (VEC_VARIANCE_t *)malloc(sizeof(VEC_VARIANCE_t) * 
								this->variance_size);
		CHECK_EXPR_RET(this->variance == NULL,-1);
		for (int i = 0;i < this->variance_size;i++) {
			this->variance[i].size =  4;
			this->variance[i].data = (float *)malloc(sizeof(float) * 4);
			CHECK_EXPR_RET(this->variance[i].data == NULL,-1);
		}
	}
#endif

	//all decode bbox
	if (this->all_decode_bboxes  == NULL) {
		this->all_decode_bboxes_size = num;
		this->all_decode_bboxes = (VEC_BBOX_t *)malloc(sizeof(VEC_BBOX_t) * 
									this->all_decode_bboxes_size);
		CHECK_EXPR_RET(this->all_decode_bboxes == NULL,-1);
		for (int i = 0;i < this->all_decode_bboxes_size;i++) {
			this->all_decode_bboxes[i].pBBox = (NormalizedBBox *)malloc(this->num_priors_ *
									sizeof(NormalizedBBox));
			CHECK_EXPR_RET(this->all_decode_bboxes[i].pBBox == NULL,1);
			this->all_decode_bboxes[i].bbox_size = this->num_priors_;
		}
	}

	//scoreKeepBboxIdx
	allocLabelBBoxIdxMemory(this,&this->scoreKeepBboxIdx,&this->scoreKeepBboxIdxSize);
	
	//nmsKeepBboxIdx
	allocLabelBBoxIdxMemory(this,&this->nmsKeepBboxIdx,&this->nmsKeepBboxIdxSize);
	
	//all output info
	if (this->pBatchOutputInfo == NULL) {
		this->pBatchOutputInfo = (BATCH_OUTPUT_INFO_t *)malloc(sizeof(BATCH_OUTPUT_INFO_t) * num);
		CHECK_EXPR_RET(this->pBatchOutputInfo == NULL,-1);
		for (int i = 0;i < num;i++) {
			BATCH_OUTPUT_INFO_t *pOutput = &this->pBatchOutputInfo[i];
			pOutput->itemSize = this->num_classes_ * this->num_priors_;
			pOutput->pItem = (OUTPUT_ITEM_t *)malloc(sizeof(OUTPUT_ITEM_t) * pOutput->itemSize);
			CHECK_EXPR_RET(pOutput->pItem == NULL,-1);
		}
	}

	return 0;
}

void sortOutputInfoByScoreInverse(BATCH_OUTPUT_INFO_t *pOutput)
{
	int len = pOutput->itemSize;
	BOOL flag = FALSE;
	OUTPUT_ITEM_t tmp;
	for (int i = 0;i < len;i++) {
		flag = FALSE;
		for (int j = 0;j < len-i-1;j++) {
			if (pOutput->pItem[j].score < pOutput->pItem[j+1].score) {
				tmp = pOutput->pItem[j];
				pOutput->pItem[j] = pOutput->pItem[j+1];
				pOutput->pItem[j+1] = tmp;
				flag = TRUE;
			}
		}
		if (flag == FALSE) {
			break;
		}
	}
}
void writeIdx(DETECTION_OUTPUT_THIS *this)
{
	FILE *fp = fopen("out/idx.txt","w+");
	CHECK_EXPR_NORET(fp == NULL);

	char buf[128];
	for (int i = 0;i < this->nmsKeepBboxIdxSize;i++) {
		
		for (int j = 0;j < this->nmsKeepBboxIdx[i].bboxSize;j++) {
			snprintf(buf,sizeof(buf),"%d ",this->nmsKeepBboxIdx[i].bboxIdx[j]);
			fwrite(buf,strlen(buf),1,fp);
		}
	}
	fclose(fp);
}


void writeDecodeBbox(DETECTION_OUTPUT_THIS *this)
{
	FILE *fp = fopen("out/decode_bbox.txt","w+");
	CHECK_EXPR_NORET(fp == NULL);

	char buf[128];
	NormalizedBBox *bbox = this->all_decode_bboxes[0].pBBox;
	for (int i = 0;i < this->all_decode_bboxes[0].bbox_size;i++) {
		if (i % 32 == 0) {
			fwrite("\n",1,1,fp);
		}
		snprintf(buf,sizeof(buf),"%f ",bbox[i].xmin);
		fwrite(buf,strlen(buf),1,fp);
		
		snprintf(buf,sizeof(buf),"%f ",bbox[i].xmax);
		fwrite(buf,strlen(buf),1,fp);
		
		snprintf(buf,sizeof(buf),"%f ",bbox[i].ymin);
		fwrite(buf,strlen(buf),1,fp);
		
		snprintf(buf,sizeof(buf),"%f ",bbox[i].ymax);
		fwrite(buf,strlen(buf),1,fp);
	}

	fclose(fp);
}

void writeScores(DETECTION_OUTPUT_THIS *this)
{
	FILE *fp = fopen("out/scores.txt","w+");
	CHECK_EXPR_NORET(fp == NULL);

	char buf[128];
	int cnt = 0;
	MAP_LABEL_CONF_BBOX_t *score = this->conf_preds[0].pMapLabelConf;
	for (int i = 0;i < this->conf_preds[0].map_size;i++) {
		for (int j = 0;j < score[i].conf_size;j++) {
			if ((cnt++) % 32 == 0) {
				fwrite("\n",1,1,fp);
			}
			snprintf(buf,sizeof(buf),"%f ",score[i].conf[j]);
			fwrite(buf,strlen(buf),1,fp);
		}
	}
	fclose(fp);
}

int DetectionOutputLayer_forward(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer->innerParam == NULL,-1);
	DETECTION_OUTPUT_THIS *this = (DETECTION_OUTPUT_THIS *)pLayer->innerParam;
	DATA_TYPE *loc_data = BLOB_data(pLayer->bottom[0]);
	DATA_TYPE *conf_data = BLOB_data(pLayer->bottom[1]);
	DATA_TYPE *prior_data = BLOB_data(pLayer->bottom[2]);
	int num = BLOB_num(pLayer->bottom[0]);

	GetLocPredictions(loc_data,num,this->num_priors_,this->all_loc_preds);
	
	GetConfidenceScores(conf_data,num,this->num_priors_,
									this->num_classes_,this->conf_preds);
	//writeScores(this);
	
	GetPriorBBoxes(prior_data,this->num_priors_,&this->prior_bboxes,
									this->variance);

	DecodeBBoxesAll(this->all_loc_preds,&this->prior_bboxes,this->variance,
					num,this->code_type_,this->variance_encoded_in_target_,
							this->all_decode_bboxes);
	//writeDecodeBbox(this);
	int num_kept = 0;
	for (int i = 0;i < num;i++) {
		VEC_BBOX_t *pDecodeBbox = &this->all_decode_bboxes[i];
		VEC_CONF_t *pConf = &this->conf_preds[i];
		BATCH_OUTPUT_INFO_t *pOutput = &this->pBatchOutputInfo[i];
		pOutput->itemSize = 0;
		for (int c = 0;c < this->num_classes_;c++) {
			if (c == 0) {
				continue;
			}
			ApplyKeepTopScore(pConf->pMapLabelConf[c].conf,this->num_priors_,
				this->confidence_threshold_,this->top_k_,
						this->scoreKeepBboxIdx[c].bboxIdx,&this->scoreKeepBboxIdx[c].bboxSize);
			ApplyNMS(pDecodeBbox->pBBox,this->nms_threshold_,
						this->scoreKeepBboxIdx[c].bboxIdx,this->scoreKeepBboxIdx[c].bboxSize,
							this->nmsKeepBboxIdx[c].bboxIdx,&this->nmsKeepBboxIdx[c].bboxSize);
		}
		//writeIdx(this);
	
		for (int c = 0;c < this->num_classes_;c++) {
			for (int p = 0;p < this->nmsKeepBboxIdx[c].bboxSize;p++) {
				int idx = this->nmsKeepBboxIdx[c].bboxIdx[p];
				pOutput->pItem[pOutput->itemSize].label = c;
				pOutput->pItem[pOutput->itemSize].score = pConf->pMapLabelConf[c].conf[idx];
				pOutput->pItem[pOutput->itemSize].bboxIdx = idx;
				pOutput->itemSize++;
			}
		}
		if (this->keep_top_k_ > -1 && pOutput->itemSize > this->keep_top_k_) {
			sortOutputInfoByScoreInverse(pOutput);
			pOutput->itemSize = this->keep_top_k_;
		} 
		num_kept += pOutput->itemSize;
	}
	
	int top_shape[2];
	top_shape[0] = num_kept;
	top_shape[1] = 7;
	//BLOB_reshapeByArrayKeepMemory(pLayer->top[0],top_shape,2);
	BLOB_reshapeByArray(pLayer->top[0],top_shape,2);
	DATA_TYPE *top_data = BLOB_data(pLayer->top[0]);
	
	int count = 0;
	for (int i = 0;i < num;i++) {
		BATCH_OUTPUT_INFO_t *pOutput = &this->pBatchOutputInfo[i];
		VEC_BBOX_t *pDecodeBox = &this->all_decode_bboxes[i];
		for (int j = 0;j < pOutput->itemSize;j++) {
			top_data[count * 7] = i;
			top_data[count * 7 + 1] = pOutput->pItem[j].label;
			top_data[count * 7 + 2] = pOutput->pItem[j].score;
			NormalizedBBox clip_bbox;
			int idx = pOutput->pItem[j].bboxIdx;
			ClipBBox(&pDecodeBox->pBBox[idx],&clip_bbox);
			top_data[count * 7 + 3] = clip_bbox.xmin;
		    top_data[count * 7 + 4] = clip_bbox.ymin;
		    top_data[count * 7 + 5] = clip_bbox.xmax;
		    top_data[count * 7 + 6] = clip_bbox.ymax;
			count++;
		}
	}

	return 0;
}


