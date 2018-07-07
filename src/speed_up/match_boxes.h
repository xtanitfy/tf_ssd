#ifndef __MATCH_BOXES_H__
#define __MATCH_BOXES_H__

#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    float axis[4];
}BOX_t;

typedef struct
{
   BOX_t *pBoxes;
   int len;
}IMG_BOXES_t;

typedef struct
{
   IMG_BOXES_t *pBoxes;
   int len;
}BATCH_BOXES_t;

void print_batch_gt_boxes(BATCH_BOXES_t *pBatchBoxes);
int do_match(BATCH_BOXES_t *pBatchGtBoxes,float *prior_boxes,
            PyObject *obj_all_match_indices,PyObject *obj_all_match_overlaps,
            int batch_size,int anchor_size,float overlap_threshold);


#endif

