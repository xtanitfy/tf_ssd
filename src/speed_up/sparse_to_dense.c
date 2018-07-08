#include "sparse_to_dense.h"
#include "common.h"

int do_sparse_to_dense(BATCH_BOXES_t *batch_gt_boxes,
								BATCH_LABEL_t *batch_gt_lables,
									float *all_match_indices,
                    			PyObject *gt_boxes_dense,PyObject *gt_labels_dense,PyObject *input_mask,
                    				int batch_size,int anchor_size,int cls_num,int background_label_id)
{
    
    int size = batch_size * anchor_size * 4;
   	float *local_gt_boxes_dense = (float *)malloc(sizeof(float)*size);
    ASSERT(local_gt_boxes_dense != NULL);
	memset(local_gt_boxes_dense,'\0',sizeof(float)*size);
		
	size = batch_size * anchor_size * cls_num;
    float *local_gt_labels_dense = (float *)malloc(sizeof(float)*size);
    ASSERT(local_gt_labels_dense != NULL);
	memset(local_gt_labels_dense,'\0',sizeof(float)*size);

    size = batch_size * anchor_size;
	float *local_input_mask = (float *)malloc(sizeof(float)*size);
    ASSERT(local_input_mask != NULL);	
	memset(local_input_mask,'\0',sizeof(float)*size);

	float (*p_match_indices)[anchor_size] = (float (*)[anchor_size])all_match_indices;
	float (*p_local_gt_boxes_dense)[anchor_size][4] = 
									(float (*)[anchor_size][4])local_gt_boxes_dense;
	float (*p_local_gt_labels_dense)[anchor_size][cls_num] = 
									(float (*)[anchor_size][cls_num])local_gt_labels_dense;
	float (*p_local_input_mask)[anchor_size] = 
									(float (*)[anchor_size])local_input_mask;

	float *one_label = (float *)malloc(sizeof(float)*cls_num);
	ASSERT(one_label != NULL);	
	float *one_box = (float *)malloc(sizeof(float)*4);
	ASSERT(one_box != NULL);
	
	for (int n = 0;n < batch_size;n++) {
		for (int i = 0;i < anchor_size;i++) {
			for (int k = 0;k < cls_num;k++) {
				one_label[k] = 0.0;
			}
			for (int k = 0;k < 4;k++) {
				one_box[k] = 0.0;
			}
			if (p_match_indices[n][i] == -1.) {
				one_label[background_label_id] = 1.;
			} else {
				p_local_input_mask[n][i] = 1.;
				int gt_idx = (int)p_match_indices[n][i];
				int label_idx = (int)batch_gt_lables->data[n].label[gt_idx];
				one_label[label_idx] = 1;
				for (int k = 0;k < 4;k++) {
					one_box[k] = batch_gt_boxes->pBoxes[n].pBoxes[gt_idx].axis[k];
				}
			}
			
			for (int k = 0;k < 4;k++) {
				p_local_gt_boxes_dense[n][i][k] = one_box[k];
			}

			for (int k = 0;k < cls_num;k++) {
				p_local_gt_labels_dense[n][i][k] = one_label[k];
			}	
			
		}
	}
	
	free(one_box);
	free(one_label);
		
	size = batch_size * anchor_size * 4;
	for (int i = 0;i < size;i++) {
        PyList_SetItem(gt_boxes_dense,i,Py_BuildValue("f",(float)local_gt_boxes_dense[i]));
    }

	size = batch_size * anchor_size * cls_num;
	for (int i = 0;i < size;i++) {
        PyList_SetItem(gt_labels_dense,i,Py_BuildValue("f",(float)local_gt_labels_dense[i]));
    }
	
	size = batch_size * anchor_size;
	for (int i = 0;i < size;i++) {
        PyList_SetItem(input_mask,i,Py_BuildValue("f",(float)local_input_mask[i]));
    }

	free(local_input_mask);
	free(local_gt_labels_dense);
	free(local_gt_boxes_dense);
	
    return 0;
}