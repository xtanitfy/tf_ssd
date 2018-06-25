#ifndef __RESHAPE_LAYER_H__
#define __RESHAPE_LAYER_H__

#include "blob.h"
#include "layer.h"
#include "public.h"
#include "math_functions.h"

typedef struct
{
  int copy_axes_[BLOB_MAX_AXES];
  int copy_axes_size;
  
  /// @brief the index of the axis whose dimension we infer, or -1 if none
  int inferred_axis_;
  /// @brief the product of the "constant" output dimensions
  int constant_count_;

}RESHAPE_THIS;


int ReshapeLayer_setUp(LAYER_t *pLayer);
int ReshapeLayer_reshape(LAYER_t *pLayer);
int ReshapeLayer_forward(LAYER_t *pLayer);
int ReshapeLayer_backward(LAYER_t *pLayer);
#endif