#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "match_boxes.h"
#include "common.h"
    
static int batch_size = 0;
static int anchor_size = 0;
static float overlap_threshold = 0.0;

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
                //PyList_SetItem(dim2_obj,n,Py_BuildValue("f",(float)n));
                
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

static PyMethodDef colinMethods[] =
{
    {"macth_boxes", macth_boxes, METH_VARARGS, "Just a test"},  
};


void initspeed_up()
{
    PyObject *m;
    m = Py_InitModule("speed_up", colinMethods);
}
