#include "conv_layer.h"
#include "layer.h"
#include "dlist.h"
#include "parameter.h"
#include "math_functions.h"
#include "im2col.h"

static int input_shape(CONV_INNER_PARAM_t *this,int i);
static void compute_output_shape(CONV_INNER_PARAM_t *this);
static void forward_cpu_gemm(CONV_INNER_PARAM_t *this,DATA_TYPE* input,
    const DATA_TYPE* weights, DATA_TYPE* output, BOOL skip_im2col);
static void conv_im2col_cpu(CONV_INNER_PARAM_t *this,DATA_TYPE* data, DATA_TYPE* col_buff);
static void forward_cpu_bias(CONV_INNER_PARAM_t *this,DATA_TYPE* output,const DATA_TYPE* bias);


static BOOL reverse_dimensions()
{
	return FALSE;
}

static void convInitInnerParam(CONV_INNER_PARAM_t *pParam)
{
	memset(pParam,'\0',sizeof(CONV_INNER_PARAM_t));
}

int ConvolutionLayer_setUp(LAYER_t *pLayer)
{

	//printf("ConvolutionLayer_setUp 0\n");
	CONV_INNER_PARAM_t *this = (CONV_INNER_PARAM_t *)malloc(sizeof(CONV_INNER_PARAM_t));
	CHECK_EXPR_RET(this == NULL, -1);
	pLayer->innerParam = (void *)this;

	convInitInnerParam(this);

	LayerParameter *pLayerParam = pLayer->pLayerParam;
	ConvolutionParameter *pConvParam = &pLayerParam->convolution_param;
	BLOB_t **pBottoms = pLayer->bottom;
	this->channel_axis_ = BLOB_CanonicalAxisIndex(pBottoms[0], pConvParam->axis);
	
	//printf("ConvolutionLayer_setUp 1\n");
	//BLOB_printShapeString(pBottoms[0],pLayer->pLayerParam->name);
	const int first_spatial_axis = this->channel_axis_ + 1;
	const int num_axes = BLOB_num_axes(pBottoms[0]);
	this->num_spatial_axes_ = num_axes - first_spatial_axis;

	//printf("ConvolutionLayer_setUp 11\n");
	//kernel 
	this->kernel_shape_size_ = 0;
	if (pConvParam->kernel_h != 0 || pConvParam->kernel_w != 0) {
		this->kernel_shape_[this->kernel_shape_size_++] = pConvParam->kernel_h;
		this->kernel_shape_[this->kernel_shape_size_++] = pConvParam->kernel_w;
	} else {
		const int num_kernel_dims = pConvParam->kernel_size_size;
		CHECK_EXPR_RET(!(num_kernel_dims == 0 || num_kernel_dims == 1 ||
          num_kernel_dims == this->num_spatial_axes_),-1);
		
		for (int i = 0; i < this->num_spatial_axes_; ++i) {
			this->kernel_shape_[this->kernel_shape_size_++] =
				pConvParam->kernel_size[(num_kernel_dims == 1) ? 0 : i];
		}
	}
	for (int i = 0;i < this->num_spatial_axes_;i++) {
		CHECK_EXPR_RET(this->kernel_shape_[i] <= 0, -1)
	}
	//printf("ConvolutionLayer_setUp 2\n");
	//stride 
	this->stride_size_ = 0;
	if (pConvParam->stride_h != 0 || pConvParam->stride_w != 0) {
		this->stride_[this->stride_size_++] = pConvParam->stride_h;
		this->stride_[this->stride_size_++] = pConvParam->stride_w;
	} else {
		const int num_stride_dims = pConvParam->stride_size;
		CHECK_EXPR_RET(!(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == this->num_spatial_axes_),-1);
		const int kDefaultStride = 1;
		for (int i = 0; i < this->num_spatial_axes_; ++i) {
			if (num_stride_dims == 0) {
				this->stride_[this->stride_size_++] = kDefaultStride;
			} else {
				this->stride_[this->stride_size_++] =
					pConvParam->stride[(num_stride_dims == 1) ? 0 : i];
			}
		}
	}

	//pad 
	this->pad_size_ = 0;
	if (pConvParam->pad_h != 0 || pConvParam->pad_w != 0) {
		this->pad_[this->pad_size_++] = pConvParam->pad_h;
		this->pad_[this->pad_size_++] = pConvParam->pad_w;
	} else {
		const int num_pad_dims = pConvParam->pad_size;
		//printf("num_pad_dims:%d\n",num_pad_dims);
		CHECK_EXPR_RET(!(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == this->num_spatial_axes_),-1);
		const int kDefaultPad = 0;
		for (int i = 0; i < this->num_spatial_axes_; ++i) {
			if (num_pad_dims == 0) {
				this->pad_[this->pad_size_++] = kDefaultPad;
			} else {
				this->pad_[this->pad_size_++] =
					pConvParam->pad[(num_pad_dims == 1) ? 0 : i];
			}
		}
	}
	
	//dilation 
	const int num_dilation_dims = pConvParam->dilation_size;
	CHECK_EXPR_RET(!(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == this->num_spatial_axes_),-1);

	const int kDefaultDilation = 1;
	this->dilation_size_ = 0;
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		this->dilation_[this->dilation_size_++] = (num_dilation_dims == 0) ? kDefaultDilation :
					pConvParam->dilation[(num_dilation_dims == 1) ? 0 : i];
	}

	this->is_1x1_ = TRUE;
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		this->is_1x1_ &=
			this->kernel_shape_[i] == 1 && this->stride_[i] == 1 && this->pad_[i] == 0;
		if (!this->is_1x1_) {
			break; 
		}
	}
	
	this->channels_ = BLOB_shapeByIndex(pBottoms[0], this->channel_axis_);
	this->num_output_ = pLayerParam->convolution_param.num_output;
	CHECK_EXPR_RET(this->num_output_ <= 0, -1);
	this->group_ = pLayerParam->convolution_param.group;
	CHECK_EXPR_RET(this->channels_ % this->group_ != 0, -1);
	CHECK_EXPR_RET(this->num_output_ % this->group_ != 0, -1);

	if (reverse_dimensions()) {
		this->conv_out_channels_ = this->channels_;
		this->conv_in_channels_ = this->num_output_;
	} else {
		this->conv_out_channels_ = this->num_output_;
		this->conv_in_channels_ = this->channels_;
	}

	int weight_shape[BLOB_MAX_AXES];
	int weight_shape_size = 0;
	weight_shape[weight_shape_size++] = this->conv_out_channels_;
	weight_shape[weight_shape_size++] = this->conv_in_channels_ / this->group_;
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		weight_shape[weight_shape_size++] = this->kernel_shape_[i];
	}

	this->bias_term_ = pLayerParam->convolution_param.bias_term;
	int bias_shape[1];
	bias_shape[0] = this->num_output_;
	if (pLayer->weigtsBlobsSize > 0 ) { 
		CHECK_EXPR_RET(1 + this->bias_term_ != pLayer->weigtsBlobsSize, -1);
		CHECK_EXPR_RET(BLOB_shapeEqualsByArr(&pLayer->pWeigtsBlobs[0],weight_shape,weight_shape_size) == FALSE, -1);
		if (this->bias_term_) {
			CHECK_EXPR_RET(BLOB_shapeEqualsByArr(&pLayer->pWeigtsBlobs[1],bias_shape,1) == FALSE, -1);
		}
	} else {
		//fill the weights intial value that not implemented, because it's for training

	}

	this->kernel_dim_ = BLOB_countByStart(&pLayer->pWeigtsBlobs[0],1);
	this->weight_offset_ = this->conv_out_channels_ * this->kernel_dim_ / this->group_;

	
	//printf("%s setup finish!\n",pLayerParam->name);
	return 0;
}

int ConvolutionLayer_reshape(LAYER_t *pLayer)
{
	//printf("%s reshape!\n",pLayer->pLayerParam->name);
	CHECK_EXPR_RET(pLayer == NULL, -1);
	CHECK_EXPR_RET(pLayer->innerParam == NULL, -1);

	CONV_INNER_PARAM_t *this = (CONV_INNER_PARAM_t *)pLayer->innerParam;
	
	const int first_spatial_axis = this->channel_axis_ + 1;
	CHECK_EXPR_RET(first_spatial_axis+this->num_spatial_axes_ != 
					BLOB_num_axes(pLayer->bottom[0]),-1);
	
	this->num_ = BLOB_countByStartAndEnd(pLayer->bottom[0], 0, this->channel_axis_);
	for (int bottom_id = 1; bottom_id < pLayer->bottomCnt; ++bottom_id) {
		CHECK_EXPR_RET(BLOB_shapeEqualsBlob(pLayer->bottom[0],pLayer->bottom[bottom_id]) == FALSE,-1);
	}

	BLOB_shape(pLayer->bottom[0],this->bottom_shape_,&this->bottom_shape_size_);
	compute_output_shape(this);

	int top_shape[BLOB_MAX_AXES];
	int top_shape_size = 0;
	for (int i = 0;i <this->channel_axis_;i++) {
		top_shape[top_shape_size++] = BLOB_shapeByIndex(pLayer->bottom[0],0);
	}
	top_shape[top_shape_size++] = this->num_output_;
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		top_shape[top_shape_size++] = this->output_shape_[i];
	}
	for (int top_id = 0; top_id < pLayer->topCnt; ++top_id) {
		BLOB_reshapeByArray(pLayer->top[top_id],top_shape,top_shape_size);
	}
	
	//printf("channel_axis_:%d\n",this->channel_axis_);
	//printf("first_spatial_axis:%d\n",first_spatial_axis);
	if (reverse_dimensions() == TRUE) {
		this->conv_out_spatial_dim_ = BLOB_countByStart(pLayer->bottom[0],first_spatial_axis);
	} else {
		this->conv_out_spatial_dim_ = BLOB_countByStart(pLayer->top[0],first_spatial_axis);
	}
	//printf("=============conv_out_spatial_dim_:%d\n",this->conv_out_spatial_dim_);
	this->col_offset_ = this->kernel_dim_ * this->conv_out_spatial_dim_;
	this->output_offset_ = this->conv_out_channels_ * this->conv_out_spatial_dim_ / this->group_;

	this->conv_input_shape_size_ = this->num_spatial_axes_ + 1;
	for (int i = 0; i < this->num_spatial_axes_ + 1; ++i) {
		if (reverse_dimensions()) {
			this->conv_input_shape_[i] = BLOB_shapeByIndex(pLayer->top[0],this->channel_axis_ + i);
		} else {
			this->conv_input_shape_[i] = BLOB_shapeByIndex(pLayer->bottom[0],this->channel_axis_ + i);
		}
	}

	this->col_buffer_shape_size_ = 0;
	this->col_buffer_shape_[this->col_buffer_shape_size_++] = this->kernel_dim_ * this->group_;
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		if (reverse_dimensions()) {
			this->col_buffer_shape_[this->col_buffer_shape_size_++] = input_shape(this,i + 1); 
		} else {
			this->col_buffer_shape_[this->col_buffer_shape_size_++] = this->output_shape_[i];
		}
	}
	
	BLOB_init(&this->col_buffer_);
	BLOB_reshapeByArray(&this->col_buffer_,this->col_buffer_shape_,this->col_buffer_shape_size_);

	this->bottom_dim_ = BLOB_countByStart(pLayer->bottom[0],this->channel_axis_);
	this->top_dim_ = BLOB_countByStart(pLayer->top[0],this->channel_axis_);
  	this->num_kernels_im2col_ = this->conv_in_channels_ * this->conv_out_spatial_dim_;
  	this->num_kernels_col2im_ = reverse_dimensions() ? this->top_dim_ : this->bottom_dim_;

	this->out_spatial_dim_ = BLOB_countByStart(pLayer->top[0],first_spatial_axis);
	if (this->bias_term_) {
		int bias_multiplier_shape[BLOB_MAX_AXES];
		int bias_multiplier_shape_size = 0;
		bias_multiplier_shape[bias_multiplier_shape_size++] = this->out_spatial_dim_;
		BLOB_init(&this->bias_multiplier_);
		BLOB_reshapeByArray(&this->bias_multiplier_,bias_multiplier_shape,bias_multiplier_shape_size);
		Murphy_set(this->bias_multiplier_.count_, (DATA_TYPE)1,BLOB_data(&this->bias_multiplier_));
	}
#if 0
	printf("[--murphy]conv_out_channels_:%d\n",this->conv_out_channels_);
	printf("[murphy]conv_out_spatial_dim_:%d\n",this->conv_out_spatial_dim_);
	printf("[murphy]kernel_dim_:%d\n",this->kernel_dim_);
	printf("[murphy]weight_offset_:%d\n",this->weight_offset_);
	printf("[murphy]col_offset_:%d\n",this->col_offset_);
	printf("[murphy]output_offset_:%d\n",this->output_offset_);

	getchar();
	printf("%s reshape finish!\n",pLayer->pLayerParam->name);
#endif

	return 0;
}

int ConvolutionLayer_forward(LAYER_t *pLayer)
{
	CHECK_EXPR_RET(pLayer == NULL, -1);
	CHECK_EXPR_RET(pLayer->innerParam == NULL, -1);
	CONV_INNER_PARAM_t *this = (CONV_INNER_PARAM_t *)pLayer->innerParam;
	
	//printf("%s forward!\n",pLayer->pLayerParam->name);
	DATA_TYPE *weight = BLOB_data(&pLayer->pWeigtsBlobs[0]);
	for (int i = 0;i < pLayer->bottomCnt;i++) {
		DATA_TYPE *bottom_data = BLOB_data(pLayer->bottom[i]);
		DATA_TYPE *top_data = BLOB_data(pLayer->top[i]);
		
		for (int n = 0; n < this->num_; ++n) {
			forward_cpu_gemm(this,bottom_data + n * this->bottom_dim_, weight,
								top_data + n * this->top_dim_,FALSE);
			if (this->bias_term_ == TRUE) {
				const DATA_TYPE* bias = BLOB_data(&pLayer->pWeigtsBlobs[1]);
				forward_cpu_bias(this,top_data + n * this->top_dim_, bias);
			}
		}
	}
	
	return 0;
}

void forward_cpu_gemm(CONV_INNER_PARAM_t *this,DATA_TYPE* input,
    const DATA_TYPE* weights, DATA_TYPE* output, BOOL skip_im2col) 
{
#if 0
	printf("[murphy]conv_out_channels_:%d\n",this->conv_out_channels_);
	printf("[murphy]conv_out_spatial_dim_:%d\n",this->conv_out_spatial_dim_);
	printf("[murphy]kernel_dim_:%d\n",this->kernel_dim_);
	printf("[murphy]weight_offset_:%d\n",this->weight_offset_);
	printf("[murphy]col_offset_:%d\n",this->col_offset_);
	printf("[murphy]output_offset_:%d\n",this->output_offset_);
	// getchar();
#endif

	DATA_TYPE* col_buff = input;
  if (!this->is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(this,input, BLOB_data(&this->col_buffer_));
    }
    col_buff = BLOB_data(&this->col_buffer_);
  }
	for (int g = 0; g < this->group_; ++g) {
		Murphy_gemm(CblasNoTrans, CblasNoTrans, this->conv_out_channels_ /this->group_,
			this->conv_out_spatial_dim_, this->kernel_dim_,
		(DATA_TYPE)1., weights + this->weight_offset_ * g, col_buff + this->col_offset_ * g,
		(DATA_TYPE)0., output + this->output_offset_ * g);
	}

}

void conv_im2col_cpu(CONV_INNER_PARAM_t *this,DATA_TYPE* data, DATA_TYPE* col_buff) 
{
	im2col_cpu(data, this->conv_in_channels_,
		this->conv_input_shape_[1], this->conv_input_shape_[2],
		this->kernel_shape_[0], this->kernel_shape_[1],this->pad_[0], this->pad_[1],
		this->stride_[0], this->stride_[1],this->dilation_[0], this->dilation_[1], col_buff);
}

  
void forward_cpu_bias(CONV_INNER_PARAM_t *this,DATA_TYPE* output,const DATA_TYPE* bias) 
{
	Murphy_gemm(CblasNoTrans, CblasNoTrans, this->num_output_,
		this->out_spatial_dim_, 1, (DATA_TYPE)1., bias, BLOB_data(&this->bias_multiplier_),
			(DATA_TYPE)1., output);
}


int ConvolutionLayer_backward(LAYER_t *pLayer)
{
	return 0;
}



int input_shape(CONV_INNER_PARAM_t *this,int i) 
{
  	return this->bottom_shape_[this->channel_axis_ + i];
}


void compute_output_shape(CONV_INNER_PARAM_t *this) 
{
	const int* kernel_shape_data = this->kernel_shape_;
	const int* stride_data = this->stride_;
	const int* pad_data = this->pad_;
	const int* dilation_data = this->dilation_;
#if 0
	printf("kernel_shape_[0]:%d\n",this->kernel_shape_[0]);
	printf("kernel_shape_[1]:%d\n",this->kernel_shape_[1]);
	
	printf("stride_[0]:%d\n",this->stride_[0]);
	printf("stride_[1]:%d\n",this->stride_[1]);
	
	printf("pad_[0]:%d\n",this->pad_[0]);
	printf("pad_[1]:%d\n",this->pad_[1]);
	
	printf("dilation_[0]:%d\n",this->dilation_[0]);
#endif
	
	this->output_shape_size_ = 0;
	//printf("---num_spatial_axes_:%d\n",this->num_spatial_axes_);
	
	for (int i = 0; i < this->num_spatial_axes_; ++i) {
		// i + 1 to skip channel axis
		const int input_dim = input_shape(this,i + 1);
		const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
		const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        						/ stride_data[i] + 1;
		this->output_shape_[this->output_shape_size_++] = output_dim;
	}
}


