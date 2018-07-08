#ifndef __SPARDE_TO_DENSE_H__
#define __SPARDE_TO_DENSE_H__


#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "match_boxes.h"

typedef struct
{
   float *label;
   int len;
}IMG_LABEL_t;

typedef struct
{
   IMG_LABEL_t *data;
   int len;
}BATCH_LABEL_t;

int do_sparse_to_dense(BATCH_BOXES_t *batch_gt_boxes,BATCH_LABEL_t *batch_gt_lables,float *all_match_indices,
                    PyObject *gt_boxes_dense,PyObject *gt_labels_dense,PyObject *input_mask,
                    int batch_size,int anchor_size,int cls_num,int background_label_id);
                    
#endif