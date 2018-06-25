#ifndef __PRIOR_BOX_LAYER_H__
#define __PRIOR_BOX_LAYER_H__
#include "blob.h"
#include "layer.h"
#include "public.h"
#include "math_functions.h"

#define PRIORBOX_MAX_AR 32
#define PROIORBOX_MAX_VARIANCE 16

typedef struct
{
  float min_size_;
  float max_size_;
  float aspect_ratios_[PRIORBOX_MAX_AR];
  int aspect_ratios_size;
  BOOL flip_;
  int num_priors_;
  BOOL clip_;
  float variance_[PROIORBOX_MAX_VARIANCE];
  int variance_size;
}PRIORBOX_THIS;

int PriorBoxLayer_setUp(LAYER_t *pLayer);
int PriorBoxLayer_reshape(LAYER_t *pLayer);
int PriorBoxLayer_forward(LAYER_t *pLayer);

#endif