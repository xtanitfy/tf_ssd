#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "match_boxes.h"
#include "sparse_to_dense.h"
#include "common.h"

static int batch_size = 0;
static int anchor_size = 0;
static float overlap_threshold = 0.0;
static int cls_num = 12;
static int background_id = 0;

void destroy_memory(BATCH_BOXES_t *pBatchBoxes,float *prior_boxes)
{
    ASSERT(prior_boxes != NULL);
    free(prior_boxes);
    
    ASSERT(pBatchBoxes != NULL);
    ASSERT(pBatchBoxes->pBoxes != NULL);
    for (int i = 0;i < pBatchBoxes->len;i++) {
        ASSERT(pBatchBoxes->pBoxes[i].pBoxes != NULL);
        free(pBatchBoxes->pBoxes[i].pBoxes);
    }  
    free(pBatchBoxes->pBoxes);
    free(pBatchBoxes);
}

void parse_batch_gt_boxes(PyObject *p,BATCH_BOXES_t *pBatchBoxes)
{
    float item = 0.0;
    int dim0_size = 0;
    int dim1_size = 0;
    int dim2_size = 0;
    int n, result,j,m;
        
    PyObject *dim1_obj = NULL,*dim2_obj = NULL,*sub_obj=NULL;
    
    dim0_size = PyList_Size(p);
    //printf("dim0_size:%d\n",dim0_size);
    
    pBatchBoxes->len = dim0_size;
    pBatchBoxes->pBoxes = (IMG_BOXES_t *)malloc(sizeof(IMG_BOXES_t)*pBatchBoxes->len);
    ASSERT(pBatchBoxes->pBoxes != NULL);
    
    for(j = 0;j < dim0_size;j++) { /*dim1*/
        dim1_obj = PyList_GetItem(p, j);
        dim1_size = PyList_Size(dim1_obj);
        //printf("    dim1_size:%d\n",dim1_size);
        IMG_BOXES_t *pImgBoxes = &pBatchBoxes->pBoxes[j];
        pImgBoxes->pBoxes = (BOX_t *)malloc(sizeof(BOX_t)*dim1_size);
        pImgBoxes->len = dim1_size;
        
        for (m = 0;m < dim1_size;m++) {
            dim2_obj = PyList_GetItem(dim1_obj, m);
            dim2_size = PyList_Size(dim2_obj);

            BOX_t *pBox = &pImgBoxes->pBoxes[m];
            for (n = 0;n < dim2_size;n++) {
                sub_obj = PyList_GetItem(dim2_obj, n);
                PyArg_Parse(sub_obj,"f",&pBox->axis[n]);
                //printf("%f ",item);
            }
            
        }
        //printf("\n");
    }   
}




PyObject* macth_boxes(PyObject* self, PyObject* args)
{
    BATCH_BOXES_t *pBatchGtBoxes = NULL;
    int arg_size = 0,i = 0;
    PyObject *dim0_obj = NULL;
    float *prior_boxes = NULL;
    PyObject *obj_all_match_indices = NULL;
    PyObject *obj_all_match_overlaps = NULL;
    
    //printf("speed_up\n");
    arg_size = PyTuple_Size(args);
    //printf("arg_size:%d\n",arg_size);
    
    ASSERT(arg_size == 7);
    for (i = 0;i < arg_size;i++) { /*arg*/
        
        dim0_obj = PyTuple_GetItem(args, i);
        if (i == 0) {
            pBatchGtBoxes = (BATCH_BOXES_t *)malloc(sizeof(BATCH_BOXES_t));
            pBatchGtBoxes->len = 0;
            ASSERT(pBatchGtBoxes != NULL);
            parse_batch_gt_boxes(dim0_obj,pBatchGtBoxes); 
            //print_batch_gt_boxes(pBatchGtBoxes);
            
        } else if (i == 1) {
            int size = PyList_Size(dim0_obj);
            prior_boxes = (float *)malloc(sizeof(float) * size);
            ASSERT(prior_boxes != NULL);
            for (int k = 0;k < size;k++) {
                PyObject *obj = PyList_GetItem(dim0_obj, k);
                PyArg_Parse(obj,"f",&prior_boxes[k]);
            }

        } else if (i == 2) {
            obj_all_match_indices = dim0_obj;
        
        } else if (i == 3) {
            obj_all_match_overlaps = dim0_obj;
            
        } else if (i == 4) {
            PyArg_Parse(dim0_obj,"i",&batch_size);
            //printf("========batch_size:%d\n",batch_size);
            
        } else if (i == 5) {
            PyArg_Parse(dim0_obj,"i",&anchor_size);
            //printf("========anchor_size:%d\n",anchor_size);
            
        } else if (i == 6) {
            PyArg_Parse(dim0_obj,"f",&overlap_threshold);
            //printf("========overlap_threshold:%f\n",overlap_threshold);
        }
    }
    
    do_match(pBatchGtBoxes,prior_boxes,obj_all_match_indices,obj_all_match_overlaps,
                batch_size,anchor_size,overlap_threshold);
    
    destroy_memory(pBatchGtBoxes,prior_boxes);
    
    return Py_BuildValue("i", 0);
}



void parse_batch_gt_labels(PyObject *p,BATCH_LABEL_t *pBatchlabels)
{
    float item = 0.0;
    int batch_size = 0;
    int dim1_size = 0;
	int sub_size = 0;
    int n, j,m;
        
    PyObject *dim1_obj = NULL,*sub_obj=NULL;
    
    batch_size = PyList_Size(p);

    pBatchlabels->len = batch_size;
    pBatchlabels->data = (IMG_LABEL_t *)malloc(sizeof(IMG_LABEL_t)*pBatchlabels->len);
    ASSERT(pBatchlabels->data != NULL);
    #if 1
    for(j = 0;j < batch_size;j++) { /*dim1*/
        dim1_obj = PyList_GetItem(p, j);
        dim1_size = PyList_Size(dim1_obj);
		
        IMG_LABEL_t *img_label = &pBatchlabels->data[j];
		img_label->label = (float *)malloc(sizeof(float)*dim1_size);
        img_label->len = dim1_size;
        
        for (m = 0;m < img_label->len;m++) {
            sub_obj = PyList_GetItem(dim1_obj, m);
			PyArg_Parse(sub_obj,"f",&img_label->label[m]);
        }
        //printf("\n");
    }   
	#endif
}


PyObject* sparse_to_dense(PyObject* self, PyObject* args)
{
    BATCH_BOXES_t *pBatchGtBoxes = NULL;
    BATCH_LABEL_t *pBatchLabel = NULL;
    int arg_size = 0,i = 0;
    PyObject *dim0_obj = NULL;
    float *match_indices = NULL;
	PyObject *gt_boxes_dense = NULL,*gt_labels_dense=NULL,*input_mask=NULL;

    arg_size = PyTuple_Size(args);
    
    ASSERT(arg_size == 10);
    for (i = 0;i < arg_size;i++) {
        dim0_obj = PyTuple_GetItem(args, i);
        if (i == 0) {
            pBatchGtBoxes = (BATCH_BOXES_t *)malloc(sizeof(BATCH_BOXES_t));
			ASSERT(pBatchGtBoxes != NULL);
            pBatchGtBoxes->len = 0;
            parse_batch_gt_boxes(dim0_obj,pBatchGtBoxes); 
            
        } else if (i == 1) {
        	pBatchLabel =(BATCH_LABEL_t *)malloc(sizeof(BATCH_LABEL_t));
			ASSERT(pBatchLabel != NULL);
			pBatchLabel->len = 0;
            parse_batch_gt_labels(dim0_obj,pBatchLabel);
            
        } else if (i == 2) {
            int size = PyList_Size(dim0_obj);
            match_indices = (float *)malloc(sizeof(float) * size);
            ASSERT(match_indices != NULL);
            for (int k = 0;k < size;k++) {
                PyObject *obj = PyList_GetItem(dim0_obj, k);
                PyArg_Parse(obj,"f",&match_indices[k]);
			}      
			
        } else if (i == 3) {
            gt_boxes_dense = dim0_obj;
            
        } else if (i == 4) {
            gt_labels_dense = dim0_obj;
            
        } else if (i == 5) {
            input_mask = dim0_obj;
            
        } else if (i == 6) {
            PyArg_Parse(dim0_obj,"i",&batch_size);
            //printf("batch_size:%d\n",batch_size);
            
        } else if (i == 7) {
            PyArg_Parse(dim0_obj,"i",&anchor_size);
            //printf("anchor_size:%d\n",anchor_size);
            
        } else if (i == 8) {
            PyArg_Parse(dim0_obj,"i",&cls_num);
            //printf("cls_num:%d\n",cls_num);
            
        } else if (i == 9) {
            PyArg_Parse(dim0_obj,"i",&background_id);
            //printf("background_id:%d\n",background_id);
            
        }
    }

	do_sparse_to_dense(pBatchGtBoxes,pBatchLabel,match_indices,
                    gt_boxes_dense,gt_labels_dense,input_mask,
                    batch_size,anchor_size,cls_num,background_id);
	
	/*free match_indices*/
	ASSERT(match_indices != NULL);
	free(match_indices);
	
	/*free pBatchLabel memory*/
	ASSERT(pBatchLabel != NULL);
	ASSERT(pBatchLabel->data != NULL);
	for (int i = 0;i < pBatchLabel->len;i++) {
		ASSERT(pBatchLabel->data[i].label != NULL);
		free(pBatchLabel->data[i].label);
	}
	free(pBatchLabel->data);
    free(pBatchLabel);
	
	/*free pBatchGtBoxes memory*/
    ASSERT(pBatchGtBoxes != NULL);
    ASSERT(pBatchGtBoxes->pBoxes != NULL);
    for (int i = 0;i < pBatchGtBoxes->len;i++) {
        ASSERT(pBatchGtBoxes->pBoxes[i].pBoxes != NULL);
        free(pBatchGtBoxes->pBoxes[i].pBoxes);
    }  
    free(pBatchGtBoxes->pBoxes);
    free(pBatchGtBoxes);

    return Py_BuildValue("i", 0);
}


static PyMethodDef colinMethods[] =
{
    {"macth_boxes", macth_boxes, METH_VARARGS, "Just a test"},
    {"sparse_to_dense", sparse_to_dense, METH_VARARGS, "Just a test"},
};


void initspeed_up()
{
    PyObject *m;
    m = Py_InitModule("speed_up", colinMethods);
}
