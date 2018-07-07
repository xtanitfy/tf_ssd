#include "match_boxes.h"
#include "common.h"

static float iou(float *bbox1, float *bbox2);

void print_batch_gt_boxes(BATCH_BOXES_t *pBatchBoxes)
{
    for (int i = 0;i < pBatchBoxes->len;i++) {
        for (int j = 0;j < pBatchBoxes->pBoxes[i].len;j++) {
            for (int k = 0;k < 4;k++) {
                printf("%f..",pBatchBoxes->pBoxes[i].pBoxes[j].axis[k]);
            }
            printf("\n");
        }
    }
}

float cal_dist(float *box1,float *box2)
{
    float dist = 0.0;
    for (int i = 0;i < 4;i++) {
        dist += (box1[i] - box2[i]) * (box1[i] - box2[i]);
    }
    return dist;
}

int do_match(BATCH_BOXES_t *pBatchGtBoxes,float *prior_boxes,
            PyObject *obj_all_match_indices,PyObject *obj_all_match_overlaps,
                int batch_size,int anchor_size,float overlap_threshold)
{
    int size = batch_size * anchor_size;
    float *local_indices = malloc(sizeof(float)*size);
    ASSERT(local_indices != NULL);
    
    float *local_overlaps = malloc(sizeof(float)*size);
    ASSERT(local_overlaps != NULL);
    
    float (*pLocal_indices)[anchor_size] = (float (*)[anchor_size])local_indices;
    float (*pLocal_overlaps)[anchor_size] = (float (*)[anchor_size])local_overlaps;
    
    for (int i = 0;i < batch_size;i++) {
        for (int j = 0;j < anchor_size;j++) {
            pLocal_indices[i][j] = -1.;
            pLocal_overlaps[i][j] = 0.0;
        }
    }
    
    
    float (*pPriorboxes)[4] = (float (*)[4])prior_boxes;
    for (int n = 0;n < batch_size;n++) {
        int img_gt_num = pBatchGtBoxes->pBoxes[n].len;
        float (*overlaps)[img_gt_num] = (float (*)[img_gt_num])malloc(sizeof(float)*anchor_size*img_gt_num);
        ASSERT(overlaps != NULL);
        for (int i = 0;i < anchor_size;i++) {
            for (int j = 0;j < img_gt_num;j++) {
                overlaps[i][j] = 0.0;
            }
        }
        
        for (int i = 0;i < anchor_size;i++) {
            for (int j = 0;j < img_gt_num;j++) { 
                float overlap = iou(pBatchGtBoxes->pBoxes[n].pBoxes[j].axis,pPriorboxes[i]);
                if (overlap > 1e-6) {
                    pLocal_overlaps[n][i] = VOS_MAX(pLocal_overlaps[n][i],overlap);
                    overlaps[i][j] = overlap;
                } 
            }
        }
        
        int max_anchor_idx = -1;
        int max_gt_idx = -1;
        float max_overlap = 0.0;
        for (int i = 0;i < img_gt_num;i++) {
            max_anchor_idx = -1;
            max_gt_idx = -1;
            max_overlap = 0.0;

            for (int j = 0;j < anchor_size;j++) {
                if (overlaps[j][i] > max_overlap) {
                    max_anchor_idx = j;
                    max_overlap = overlaps[j][i];
                    max_gt_idx = i;
                }
            } 

            if (max_overlap == 0.0) {    
                float dist_min = 0.0;
                for (int j = 0;j < anchor_size;j++) {
                    float dist = cal_dist(pBatchGtBoxes->pBoxes[n].pBoxes[i].axis,pPriorboxes[j]);
                    if (j == 0) {
                        dist_min = dist;
                        max_anchor_idx = 0;
                    } else if (dist < dist_min) {
                        dist_min = dist;
                        max_anchor_idx = j;
                    }
                }
                max_gt_idx = i;
                max_overlap = overlaps[max_anchor_idx][i];
            }
        }
        pLocal_overlaps[n][max_anchor_idx] = max_overlap;
        pLocal_indices[n][max_anchor_idx] = max_gt_idx;
     
        for (int i = 0;i < anchor_size;i++) {
            max_overlap = 0.;
            max_gt_idx = -1.;
            
            for (int j = 0;j < img_gt_num;j++) {
                if (pLocal_indices[n][i] != -1) {
                    continue;
                }
                float overlap = overlaps[i][j];
                if (overlap > overlap_threshold && overlap > max_overlap) {
                    max_gt_idx = j;
                    max_overlap = overlap;
                }
                
            }
            if (max_gt_idx != -1) {
                pLocal_overlaps[n][i] = max_overlap; 
                pLocal_indices[n][i] = max_gt_idx;
            }
        }
        free(overlaps);
    }
   
    for (int i = 0;i < size;i++) {
        PyList_SetItem(obj_all_match_indices,i,Py_BuildValue("f",(float)local_indices[i]));
        PyList_SetItem(obj_all_match_overlaps,i,Py_BuildValue("f",(float)local_overlaps[i]));
    }
    free(local_overlaps);
    free(local_indices);
    
    return 0;
}

static float iou(float *bbox1, float *bbox2)
{
    if ((bbox2[0] > bbox1[2]) || (bbox2[2] < bbox1[0]) || (bbox2[1] > bbox1[3]) || (bbox2[3] < bbox1[1])) {
        return 0.;
    }
    
    float inter_xmin = VOS_MAX(bbox1[0], bbox2[0]);
    float inter_ymin = VOS_MAX(bbox1[1], bbox2[1]);
    float inter_xmax = VOS_MIN(bbox1[2], bbox2[2]);
    float inter_ymax = VOS_MIN(bbox1[3], bbox2[3]);
    
    float inter_width = inter_xmax - inter_xmin;
    float inter_height = inter_ymax - inter_ymin;
    float inter_size = inter_width * inter_height;
    
    float bbox1_size = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    float bbox2_size = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    
    return inter_size / (bbox1_size + bbox2_size - inter_size);
}

    

    
    

   
    
    
    
    
    
     
