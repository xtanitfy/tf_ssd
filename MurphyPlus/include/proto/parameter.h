#ifndef __PARAMETER_H__
#define __PARAMETER_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "public.h"
#include "data_type.h"

#define PARSE_STR_NAME_SIZE 64 
#define false 0
#define true 1
typedef struct BlobShape_ BlobShape;
struct BlobShape_ {
	INT64 *dim;
	UINT32 dim_size;
};

typedef struct BlobProto_ BlobProto;
struct BlobProto_ {
	BlobShape shape;
	float *data;
	UINT32 data_size;
	float *diff;
	UINT32 diff_size;
	double *double_data;
	UINT32 double_data_size;
	double *double_diff;
	UINT32 double_diff_size;
	INT32 num;
	INT32 channels;
	INT32 height;
	INT32 width;
};

typedef struct BlobProtoVector_ BlobProtoVector;
struct BlobProtoVector_ {
	BlobProto *blobs;
	UINT32 blobs_size;
};

typedef struct LabelMapItem_ LabelMapItem;
struct LabelMapItem_ {
	char name[PARSE_STR_NAME_SIZE];
	INT32 label;
	char display_name[PARSE_STR_NAME_SIZE];
};

typedef struct LabelMap_ LabelMap;
struct LabelMap_ {
	LabelMapItem *item;
	UINT32 item_size;
};

typedef struct NormalizedBBox_ NormalizedBBox;
struct NormalizedBBox_ {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	INT32 label;
	BOOL difficult;
	float score;
	float size;
};

typedef struct Datum_ Datum;
struct Datum_ {
	INT32 channels;
	INT32 height;
	INT32 width;
	char data[PARSE_STR_NAME_SIZE];
	INT32 label;
	float *float_data;
	UINT32 float_data_size;
	BOOL encoded;
};

typedef struct Annotation_ Annotation;
struct Annotation_ {
	INT32 instance_id;
	NormalizedBBox bbox;
};

typedef struct AnnotationGroup_ AnnotationGroup;
struct AnnotationGroup_ {
	INT32 group_label;
	Annotation *annotation;
	UINT32 annotation_size;
};

typedef struct AnnotatedDatum_ AnnotatedDatum;
typedef enum {
	AnnotatedDatum_AnnotationType_BBOX = 0,
}AnnotatedDatum_AnnotationType;

struct AnnotatedDatum_ {
	Datum datum;
	AnnotatedDatum_AnnotationType type;
	AnnotationGroup *annotation_group;
	UINT32 annotation_group_size;
};

typedef enum {
	TRAIN = 0,
	TEST = 1,
}Phase;

typedef struct HDF5OutputParameter_ HDF5OutputParameter;
struct HDF5OutputParameter_ {
	char file_name[PARSE_STR_NAME_SIZE];
};

typedef struct FillerParameter_ FillerParameter;
typedef enum {
	FillerParameter_VarianceNorm_FAN_IN = 0,
	FillerParameter_VarianceNorm_FAN_OUT = 1,
	FillerParameter_VarianceNorm_AVERAGE = 2,
}FillerParameter_VarianceNorm;

struct FillerParameter_ {
	char type[PARSE_STR_NAME_SIZE];
	float value;
	float min;
	float max;
	float mean;
	float std;
	INT32 sparse;
	FillerParameter_VarianceNorm variance_norm;
};

typedef struct LossParameter_ LossParameter;
typedef enum {
	LossParameter_NormalizationMode_FULL = 0,
	LossParameter_NormalizationMode_VALID = 1,
	LossParameter_NormalizationMode_BATCH_SIZE = 2,
	LossParameter_NormalizationMode_NONE = 3,
}LossParameter_NormalizationMode;

struct LossParameter_ {
	INT32 ignore_label;
	LossParameter_NormalizationMode normalization;
	BOOL normalize;
};

typedef struct EmitConstraint_ EmitConstraint;
typedef enum {
	EmitConstraint_EmitType_CENTER = 0,
	EmitConstraint_EmitType_MIN_OVERLAP = 1,
}EmitConstraint_EmitType;

struct EmitConstraint_ {
	EmitConstraint_EmitType emit_type;
	float emit_overlap;
};

typedef struct SaltPepperParameter_ SaltPepperParameter;
struct SaltPepperParameter_ {
	float fraction;
	float *value;
	UINT32 value_size;
};

typedef struct ResizeParameter_ ResizeParameter;
typedef enum {
	ResizeParameter_Resize_mode_WARP = 1,
	ResizeParameter_Resize_mode_FIT_SMALL_SIZE = 2,
	ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD = 3,
}ResizeParameter_Resize_mode;

typedef enum {
	ResizeParameter_Pad_mode_CONSTANT = 1,
	ResizeParameter_Pad_mode_MIRRORED = 2,
	ResizeParameter_Pad_mode_REPEAT_NEAREST = 3,
}ResizeParameter_Pad_mode;

typedef enum {
	ResizeParameter_Interp_mode_LINEAR = 1,
	ResizeParameter_Interp_mode_AREA = 2,
	ResizeParameter_Interp_mode_NEAREST = 3,
	ResizeParameter_Interp_mode_CUBIC = 4,
	ResizeParameter_Interp_mode_LANCZOS4 = 5,
}ResizeParameter_Interp_mode;

struct ResizeParameter_ {
	float prob;
	ResizeParameter_Resize_mode resize_mode;
	UINT32 height;
	UINT32 width;
	ResizeParameter_Pad_mode pad_mode;
	float *pad_value;
	UINT32 pad_value_size;
	ResizeParameter_Interp_mode *interp_mode;
	UINT32 interp_mode_size;
};

typedef struct WindowDataParameter_ WindowDataParameter;
struct WindowDataParameter_ {
	char source[PARSE_STR_NAME_SIZE];
	float scale;
	char mean_file[PARSE_STR_NAME_SIZE];
	UINT32 batch_size;
	UINT32 crop_size;
	BOOL mirror;
	float fg_threshold;
	float bg_threshold;
	float fg_fraction;
	UINT32 context_pad;
	char crop_mode[PARSE_STR_NAME_SIZE];
	BOOL cache_images;
	char root_folder[PARSE_STR_NAME_SIZE];
};

typedef struct ThresholdParameter_ ThresholdParameter;
struct ThresholdParameter_ {
	float threshold;
};

typedef struct TanHParameter_ TanHParameter;
typedef enum {
	TanHParameter_Engine_DEFAULT = 0,
	TanHParameter_Engine_CAFFE = 1,
	TanHParameter_Engine_CUDNN = 2,
}TanHParameter_Engine;

struct TanHParameter_ {
	TanHParameter_Engine engine;
};

typedef struct SliceParameter_ SliceParameter;
struct SliceParameter_ {
	INT32 axis;
	UINT32 *slice_point;
	UINT32 slice_point_size;
	UINT32 slice_dim;
};

typedef struct SoftmaxParameter_ SoftmaxParameter;
typedef enum {
	SoftmaxParameter_Engine_DEFAULT = 0,
	SoftmaxParameter_Engine_CAFFE = 1,
	SoftmaxParameter_Engine_CUDNN = 2,
}SoftmaxParameter_Engine;

struct SoftmaxParameter_ {
	SoftmaxParameter_Engine engine;
	INT32 axis;
};

typedef struct SigmoidParameter_ SigmoidParameter;
typedef enum {
	SigmoidParameter_Engine_DEFAULT = 0,
	SigmoidParameter_Engine_CAFFE = 1,
	SigmoidParameter_Engine_CUDNN = 2,
}SigmoidParameter_Engine;

struct SigmoidParameter_ {
	SigmoidParameter_Engine engine;
};

typedef struct ReLUParameter_ ReLUParameter;
typedef enum {
	ReLUParameter_Engine_DEFAULT = 0,
	ReLUParameter_Engine_CAFFE = 1,
	ReLUParameter_Engine_CUDNN = 2,
}ReLUParameter_Engine;

struct ReLUParameter_ {
	float negative_slope;
	ReLUParameter_Engine engine;
};

typedef struct PowerParameter_ PowerParameter;
struct PowerParameter_ {
	float power;
	float scale;
	float shift;
};

typedef struct PoolingParameter_ PoolingParameter;
typedef enum {
	PoolingParameter_PoolMethod_MAX = 0,
	PoolingParameter_PoolMethod_AVE = 1,
	PoolingParameter_PoolMethod_STOCHASTIC = 2,
}PoolingParameter_PoolMethod;

typedef enum {
	PoolingParameter_Engine_DEFAULT = 0,
	PoolingParameter_Engine_CAFFE = 1,
	PoolingParameter_Engine_CUDNN = 2,
}PoolingParameter_Engine;

struct PoolingParameter_ {
	PoolingParameter_PoolMethod pool;
	UINT32 pad;
	UINT32 pad_h;
	UINT32 pad_w;
	UINT32 kernel_size;
	UINT32 kernel_h;
	UINT32 kernel_w;
	UINT32 stride;
	UINT32 stride_h;
	UINT32 stride_w;
	PoolingParameter_Engine engine;
	BOOL global_pooling;
};

typedef struct MVNParameter_ MVNParameter;
struct MVNParameter_ {
	BOOL normalize_variance;
	BOOL across_channels;
	float eps;
};

typedef struct MemoryDataParameter_ MemoryDataParameter;
struct MemoryDataParameter_ {
	UINT32 batch_size;
	UINT32 channels;
	UINT32 height;
	UINT32 width;
};

typedef struct LRNParameter_ LRNParameter;
typedef enum {
	LRNParameter_NormRegion_ACROSS_CHANNELS = 0,
	LRNParameter_NormRegion_WITHIN_CHANNEL = 1,
}LRNParameter_NormRegion;

typedef enum {
	LRNParameter_Engine_DEFAULT = 0,
	LRNParameter_Engine_CAFFE = 1,
	LRNParameter_Engine_CUDNN = 2,
}LRNParameter_Engine;

struct LRNParameter_ {
	UINT32 local_size;
	float alpha;
	float beta;
	LRNParameter_NormRegion norm_region;
	float k;
	LRNParameter_Engine engine;
};

typedef struct InfogainLossParameter_ InfogainLossParameter;
struct InfogainLossParameter_ {
	char source[PARSE_STR_NAME_SIZE];
};

typedef struct ImageDataParameter_ ImageDataParameter;
struct ImageDataParameter_ {
	char source[PARSE_STR_NAME_SIZE];
	UINT32 batch_size;
	UINT32 rand_skip;
	BOOL shuffle;
	UINT32 new_height;
	UINT32 new_width;
	BOOL is_color;
	float scale;
	char mean_file[PARSE_STR_NAME_SIZE];
	UINT32 crop_size;
	BOOL mirror;
	char root_folder[PARSE_STR_NAME_SIZE];
};

typedef struct HingeLossParameter_ HingeLossParameter;
typedef enum {
	HingeLossParameter_Norm_L1 = 1,
	HingeLossParameter_Norm_L2 = 2,
}HingeLossParameter_Norm;

struct HingeLossParameter_ {
	HingeLossParameter_Norm norm;
};

typedef struct HDF5DataParameter_ HDF5DataParameter;
struct HDF5DataParameter_ {
	char source[PARSE_STR_NAME_SIZE];
	UINT32 batch_size;
	BOOL shuffle;
};

typedef struct ExpParameter_ ExpParameter;
struct ExpParameter_ {
	float base;
	float scale;
	float shift;
};

typedef struct EltwiseParameter_ EltwiseParameter;
typedef enum {
	EltwiseParameter_EltwiseOp_PROD = 0,
	EltwiseParameter_EltwiseOp_SUM = 1,
	EltwiseParameter_EltwiseOp_MAX = 2,
}EltwiseParameter_EltwiseOp;

struct EltwiseParameter_ {
	EltwiseParameter_EltwiseOp operation;
	float *coeff;
	UINT32 coeff_size;
	BOOL stable_prod_grad;
};

typedef struct DropoutParameter_ DropoutParameter;
struct DropoutParameter_ {
	float dropout_ratio;
};

typedef struct DataParameter_ DataParameter;
typedef enum {
	DataParameter_DB_LEVELDB = 0,
	DataParameter_DB_LMDB = 1,
}DataParameter_DB;

struct DataParameter_ {
	char source[PARSE_STR_NAME_SIZE];
	UINT32 batch_size;
	UINT32 rand_skip;
	DataParameter_DB backend;
	float scale;
	char mean_file[PARSE_STR_NAME_SIZE];
	UINT32 crop_size;
	BOOL mirror;
	BOOL force_encoded_color;
	UINT32 prefetch;
};

typedef struct ContrastiveLossParameter_ ContrastiveLossParameter;
struct ContrastiveLossParameter_ {
	float margin;
	BOOL legacy_version;
};

typedef struct ConcatParameter_ ConcatParameter;
struct ConcatParameter_ {
	INT32 axis;
	UINT32 concat_dim;
};

typedef struct ArgMaxParameter_ ArgMaxParameter;
struct ArgMaxParameter_ {
	BOOL out_max_val;
	UINT32 top_k;
	INT32 axis;
};

typedef struct AccuracyParameter_ AccuracyParameter;
struct AccuracyParameter_ {
	UINT32 top_k;
	INT32 axis;
	INT32 ignore_label;
};

typedef struct NetStateRule_ NetStateRule;
struct NetStateRule_ {
	Phase phase;
	INT32 min_level;
	INT32 max_level;
	char (*stage)[PARSE_STR_NAME_SIZE];
	UINT32 stage_size;
	char (*not_stage)[PARSE_STR_NAME_SIZE];
	UINT32 not_stage_size;
};

typedef struct QuantParameter_ QuantParameter;
struct QuantParameter_ {
	UINT32 actives_quant_val;
	UINT32 weights_quant_val;
};

typedef struct VideoDataParameter_ VideoDataParameter;
typedef enum {
	VideoDataParameter_VideoType_WEBCAM = 0,
	VideoDataParameter_VideoType_VIDEO = 1,
}VideoDataParameter_VideoType;

struct VideoDataParameter_ {
	VideoDataParameter_VideoType video_type;
	INT32 device_id;
	char video_file[PARSE_STR_NAME_SIZE];
};

typedef struct TileParameter_ TileParameter;
struct TileParameter_ {
	INT32 axis;
	INT32 tiles;
};

typedef struct SPPParameter_ SPPParameter;
typedef enum {
	SPPParameter_PoolMethod_MAX = 0,
	SPPParameter_PoolMethod_AVE = 1,
	SPPParameter_PoolMethod_STOCHASTIC = 2,
}SPPParameter_PoolMethod;

typedef enum {
	SPPParameter_Engine_DEFAULT = 0,
	SPPParameter_Engine_CAFFE = 1,
	SPPParameter_Engine_CUDNN = 2,
}SPPParameter_Engine;

struct SPPParameter_ {
	UINT32 pyramid_height;
	SPPParameter_PoolMethod pool;
	SPPParameter_Engine engine;
};

typedef struct ReductionParameter_ ReductionParameter;
typedef enum {
	ReductionParameter_ReductionOp_SUM = 1,
	ReductionParameter_ReductionOp_ASUM = 2,
	ReductionParameter_ReductionOp_SUMSQ = 3,
	ReductionParameter_ReductionOp_MEAN = 4,
}ReductionParameter_ReductionOp;

struct ReductionParameter_ {
	ReductionParameter_ReductionOp operation;
	INT32 axis;
	float coeff;
};

typedef struct PythonParameter_ PythonParameter;
struct PythonParameter_ {
	char module[PARSE_STR_NAME_SIZE];
	char layer[PARSE_STR_NAME_SIZE];
	char param_str[PARSE_STR_NAME_SIZE];
	BOOL share_in_parallel;
};

typedef struct PriorBoxParameter_ PriorBoxParameter;
typedef enum {
	PriorBoxParameter_CodeType_CORNER = 1,
	PriorBoxParameter_CodeType_CENTER_SIZE = 2,
}PriorBoxParameter_CodeType;

struct PriorBoxParameter_ {
	float min_size;
	float max_size;
	float *aspect_ratio;
	UINT32 aspect_ratio_size;
	BOOL flip;
	BOOL clip;
	float *variance;
	UINT32 variance_size;
};

typedef struct PermuteParameter_ PermuteParameter;
struct PermuteParameter_ {
	UINT32 *order;
	UINT32 order_size;
};

typedef struct MultiBoxLossParameter_ MultiBoxLossParameter;
typedef enum {
	MultiBoxLossParameter_LocLossType_L2 = 0,
	MultiBoxLossParameter_LocLossType_SMOOTH_L1 = 1,
}MultiBoxLossParameter_LocLossType;

typedef enum {
	MultiBoxLossParameter_ConfLossType_SOFTMAX = 0,
	MultiBoxLossParameter_ConfLossType_LOGISTIC = 1,
}MultiBoxLossParameter_ConfLossType;

typedef enum {
	MultiBoxLossParameter_MatchType_BIPARTITE = 0,
	MultiBoxLossParameter_MatchType_PER_PREDICTION = 1,
}MultiBoxLossParameter_MatchType;

struct MultiBoxLossParameter_ {
	MultiBoxLossParameter_LocLossType loc_loss_type;
	MultiBoxLossParameter_ConfLossType conf_loss_type;
	float loc_weight;
	UINT32 num_classes;
	BOOL share_location;
	MultiBoxLossParameter_MatchType match_type;
	float overlap_threshold;
	BOOL use_prior_for_matching;
	UINT32 background_label_id;
	BOOL use_difficult_gt;
	BOOL do_neg_mining;
	float neg_pos_ratio;
	float neg_overlap;
	PriorBoxParameter_CodeType code_type;
	BOOL encode_variance_in_target;
	BOOL map_object_to_agnostic;
};

typedef struct LogParameter_ LogParameter;
struct LogParameter_ {
	float base;
	float scale;
	float shift;
};

typedef struct FlattenParameter_ FlattenParameter;
struct FlattenParameter_ {
	INT32 axis;
	INT32 end_axis;
};

typedef struct ELUParameter_ ELUParameter;
struct ELUParameter_ {
	float alpha;
};

typedef struct SaveOutputParameter_ SaveOutputParameter;
struct SaveOutputParameter_ {
	char output_directory[PARSE_STR_NAME_SIZE];
	char output_name_prefix[PARSE_STR_NAME_SIZE];
	char output_format[PARSE_STR_NAME_SIZE];
	char label_map_file[PARSE_STR_NAME_SIZE];
	char name_size_file[PARSE_STR_NAME_SIZE];
	UINT32 num_test_image;
};

typedef struct NonMaximumSuppressionParameter_ NonMaximumSuppressionParameter;
struct NonMaximumSuppressionParameter_ {
	float nms_threshold;
	INT32 top_k;
};

typedef struct DetectionEvaluateParameter_ DetectionEvaluateParameter;
struct DetectionEvaluateParameter_ {
	UINT32 num_classes;
	UINT32 background_label_id;
	float overlap_threshold;
	BOOL evaluate_difficult_gt;
	char name_size_file[PARSE_STR_NAME_SIZE];
};

typedef struct CropParameter_ CropParameter;
struct CropParameter_ {
	INT32 axis;
	UINT32 *offset;
	UINT32 offset_size;
};

typedef struct BatchNormParameter_ BatchNormParameter;
struct BatchNormParameter_ {
	BOOL use_global_stats;
	float moving_average_fraction;
	float eps;
};

typedef struct SampleConstraint_ SampleConstraint;
struct SampleConstraint_ {
	float min_jaccard_overlap;
	float max_jaccard_overlap;
	float min_sample_coverage;
	float max_sample_coverage;
	float min_object_coverage;
	float max_object_coverage;
};

typedef struct Sampler_ Sampler;
struct Sampler_ {
	float min_scale;
	float max_scale;
	float min_aspect_ratio;
	float max_aspect_ratio;
};

typedef struct ParamSpec_ ParamSpec;
typedef enum {
	ParamSpec_DimCheckMode_STRICT = 0,
	ParamSpec_DimCheckMode_PERMISSIVE = 1,
}ParamSpec_DimCheckMode;

struct ParamSpec_ {
	char name[PARSE_STR_NAME_SIZE];
	ParamSpec_DimCheckMode share_mode;
	float lr_mult;
	float decay_mult;
};

typedef struct NetState_ NetState;
struct NetState_ {
	Phase phase;
	INT32 level;
	char (*stage)[PARSE_STR_NAME_SIZE];
	UINT32 stage_size;
};

typedef struct ReshapeParameter_ ReshapeParameter;
struct ReshapeParameter_ {
	BlobShape shape;
	INT32 axis;
	INT32 num_axes;
};

typedef struct ParameterParameter_ ParameterParameter;
struct ParameterParameter_ {
	BlobShape shape;
};

typedef struct InputParameter_ InputParameter;
struct InputParameter_ {
	BlobShape *shape;
	UINT32 shape_size;
};

typedef struct ScaleParameter_ ScaleParameter;
struct ScaleParameter_ {
	INT32 axis;
	INT32 num_axes;
	FillerParameter filler;
	BOOL bias_term;
	FillerParameter bias_filler;
};

typedef struct RecurrentParameter_ RecurrentParameter;
struct RecurrentParameter_ {
	UINT32 num_output;
	FillerParameter weight_filler;
	FillerParameter bias_filler;
	BOOL debug_info;
	BOOL expose_hidden;
};

typedef struct PReLUParameter_ PReLUParameter;
struct PReLUParameter_ {
	FillerParameter filler;
	BOOL channel_shared;
};

typedef struct NormalizeParameter_ NormalizeParameter;
struct NormalizeParameter_ {
	BOOL across_spatial;
	FillerParameter scale_filler;
	BOOL channel_shared;
	float eps;
};

typedef struct InnerProductParameter_ InnerProductParameter;
struct InnerProductParameter_ {
	UINT32 num_output;
	BOOL bias_term;
	FillerParameter weight_filler;
	FillerParameter bias_filler;
	INT32 axis;
	BOOL transpose;
};

typedef struct EmbedParameter_ EmbedParameter;
struct EmbedParameter_ {
	UINT32 num_output;
	UINT32 input_dim;
	BOOL bias_term;
	FillerParameter weight_filler;
	FillerParameter bias_filler;
};

typedef struct DummyDataParameter_ DummyDataParameter;
struct DummyDataParameter_ {
	FillerParameter *data_filler;
	UINT32 data_filler_size;
	BlobShape *shape;
	UINT32 shape_size;
	UINT32 *num;
	UINT32 num_size;
	UINT32 *channels;
	UINT32 channels_size;
	UINT32 *height;
	UINT32 height_size;
	UINT32 *width;
	UINT32 width_size;
};

typedef struct ConvolutionParameter_ ConvolutionParameter;
typedef enum {
	ConvolutionParameter_Engine_DEFAULT = 0,
	ConvolutionParameter_Engine_CAFFE = 1,
	ConvolutionParameter_Engine_CUDNN = 2,
}ConvolutionParameter_Engine;

struct ConvolutionParameter_ {
	UINT32 num_output;
	BOOL bias_term;
	UINT32 *pad;
	UINT32 pad_size;
	UINT32 *kernel_size;
	UINT32 kernel_size_size;
	UINT32 *stride;
	UINT32 stride_size;
	UINT32 *dilation;
	UINT32 dilation_size;
	UINT32 pad_h;
	UINT32 pad_w;
	UINT32 kernel_h;
	UINT32 kernel_w;
	UINT32 stride_h;
	UINT32 stride_w;
	UINT32 group;
	FillerParameter weight_filler;
	FillerParameter bias_filler;
	ConvolutionParameter_Engine engine;
	INT32 axis;
	BOOL force_nd_im2col;
};

typedef struct BiasParameter_ BiasParameter;
struct BiasParameter_ {
	INT32 axis;
	INT32 num_axes;
	FillerParameter filler;
};

typedef struct NoiseParameter_ NoiseParameter;
struct NoiseParameter_ {
	float prob;
	BOOL hist_eq;
	BOOL inverse;
	BOOL decolorize;
	BOOL gauss_blur;
	float jpeg;
	BOOL posterize;
	BOOL erode;
	BOOL saltpepper;
	SaltPepperParameter saltpepper_param;
	BOOL clahe;
	BOOL convert_to_hsv;
	BOOL convert_to_lab;
};

typedef struct DetectionOutputParameter_ DetectionOutputParameter;
struct DetectionOutputParameter_ {
	UINT32 num_classes;
	BOOL share_location;
	INT32 background_label_id;
	NonMaximumSuppressionParameter nms_param;
	SaveOutputParameter save_output_param;
	PriorBoxParameter_CodeType code_type;
	BOOL variance_encoded_in_target;
	INT32 keep_top_k;
	float confidence_threshold;
	BOOL visualize;
	float visualize_threshold;
};

typedef struct BatchSampler_ BatchSampler;
struct BatchSampler_ {
	BOOL use_original_image;
	Sampler sampler;
	SampleConstraint sample_constraint;
	UINT32 max_sample;
	UINT32 max_trials;
};

typedef struct V0LayerParameter_ V0LayerParameter;
typedef enum {
	V0LayerParameter_PoolMethod_MAX = 0,
	V0LayerParameter_PoolMethod_AVE = 1,
	V0LayerParameter_PoolMethod_STOCHASTIC = 2,
}V0LayerParameter_PoolMethod;

struct V0LayerParameter_ {
	char name[PARSE_STR_NAME_SIZE];
	char type[PARSE_STR_NAME_SIZE];
	UINT32 num_output;
	BOOL biasterm;
	FillerParameter weight_filler;
	FillerParameter bias_filler;
	UINT32 pad;
	UINT32 kernelsize;
	UINT32 group;
	UINT32 stride;
	V0LayerParameter_PoolMethod pool;
	float dropout_ratio;
	UINT32 local_size;
	float alpha;
	float beta;
	float k;
	char source[PARSE_STR_NAME_SIZE];
	float scale;
	char meanfile[PARSE_STR_NAME_SIZE];
	UINT32 batchsize;
	UINT32 cropsize;
	BOOL mirror;
	BlobProto *blobs;
	UINT32 blobs_size;
	float *blobs_lr;
	UINT32 blobs_lr_size;
	float *weight_decay;
	UINT32 weight_decay_size;
	UINT32 rand_skip;
	float det_fg_threshold;
	float det_bg_threshold;
	float det_fg_fraction;
	UINT32 det_context_pad;
	char det_crop_mode[PARSE_STR_NAME_SIZE];
	INT32 new_num;
	INT32 new_channels;
	INT32 new_height;
	INT32 new_width;
	BOOL shuffle_images;
	UINT32 concat_dim;
	HDF5OutputParameter hdf5_output_param;
};

typedef struct TransformationParameter_ TransformationParameter;
struct TransformationParameter_ {
	float scale;
	BOOL mirror;
	UINT32 crop_size;
	UINT32 crop_h;
	UINT32 crop_w;
	char mean_file[PARSE_STR_NAME_SIZE];
	float *mean_value;
	UINT32 mean_value_size;
	BOOL force_color;
	BOOL force_gray;
	ResizeParameter resize_param;
	NoiseParameter noise_param;
	EmitConstraint emit_constraint;
};

typedef struct AnnotatedDataParameter_ AnnotatedDataParameter;
struct AnnotatedDataParameter_ {
	BatchSampler *batch_sampler;
	UINT32 batch_sampler_size;
	char label_map_file[PARSE_STR_NAME_SIZE];
};

typedef struct V1LayerParameter_ V1LayerParameter;
typedef enum {
	V1LayerParameter_LayerType_NONE = 0,
	V1LayerParameter_LayerType_ABSVAL = 35,
	V1LayerParameter_LayerType_ACCURACY = 1,
	V1LayerParameter_LayerType_ARGMAX = 30,
	V1LayerParameter_LayerType_BNLL = 2,
	V1LayerParameter_LayerType_CONCAT = 3,
	V1LayerParameter_LayerType_CONTRASTIVE_LOSS = 37,
	V1LayerParameter_LayerType_CONVOLUTION = 4,
	V1LayerParameter_LayerType_DATA = 5,
	V1LayerParameter_LayerType_DECONVOLUTION = 39,
	V1LayerParameter_LayerType_DROPOUT = 6,
	V1LayerParameter_LayerType_DUMMY_DATA = 32,
	V1LayerParameter_LayerType_EUCLIDEAN_LOSS = 7,
	V1LayerParameter_LayerType_ELTWISE = 25,
	V1LayerParameter_LayerType_EXP = 38,
	V1LayerParameter_LayerType_FLATTEN = 8,
	V1LayerParameter_LayerType_HDF5_DATA = 9,
	V1LayerParameter_LayerType_HDF5_OUTPUT = 10,
	V1LayerParameter_LayerType_HINGE_LOSS = 28,
	V1LayerParameter_LayerType_IM2COL = 11,
	V1LayerParameter_LayerType_IMAGE_DATA = 12,
	V1LayerParameter_LayerType_INFOGAIN_LOSS = 13,
	V1LayerParameter_LayerType_INNER_PRODUCT = 14,
	V1LayerParameter_LayerType_LRN = 15,
	V1LayerParameter_LayerType_MEMORY_DATA = 29,
	V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS = 16,
	V1LayerParameter_LayerType_MVN = 34,
	V1LayerParameter_LayerType_POOLING = 17,
	V1LayerParameter_LayerType_POWER = 26,
	V1LayerParameter_LayerType_RELU = 18,
	V1LayerParameter_LayerType_SIGMOID = 19,
	V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS = 27,
	V1LayerParameter_LayerType_SILENCE = 36,
	V1LayerParameter_LayerType_SOFTMAX = 20,
	V1LayerParameter_LayerType_SOFTMAX_LOSS = 21,
	V1LayerParameter_LayerType_SPLIT = 22,
	V1LayerParameter_LayerType_SLICE = 33,
	V1LayerParameter_LayerType_TANH = 23,
	V1LayerParameter_LayerType_WINDOW_DATA = 24,
	V1LayerParameter_LayerType_THRESHOLD = 31,
}V1LayerParameter_LayerType;

typedef enum {
	V1LayerParameter_DimCheckMode_STRICT = 0,
	V1LayerParameter_DimCheckMode_PERMISSIVE = 1,
}V1LayerParameter_DimCheckMode;

struct V1LayerParameter_ {
	char (*bottom)[PARSE_STR_NAME_SIZE];
	UINT32 bottom_size;
	char (*top)[PARSE_STR_NAME_SIZE];
	UINT32 top_size;
	char name[PARSE_STR_NAME_SIZE];
	NetStateRule *include;
	UINT32 include_size;
	NetStateRule *exclude;
	UINT32 exclude_size;
	V1LayerParameter_LayerType type;
	BlobProto *blobs;
	UINT32 blobs_size;
	char (*param)[PARSE_STR_NAME_SIZE];
	UINT32 param_size;
	V1LayerParameter_DimCheckMode *blob_share_mode;
	UINT32 blob_share_mode_size;
	float *blobs_lr;
	UINT32 blobs_lr_size;
	float *weight_decay;
	UINT32 weight_decay_size;
	float *loss_weight;
	UINT32 loss_weight_size;
	AccuracyParameter accuracy_param;
	ArgMaxParameter argmax_param;
	ConcatParameter concat_param;
	ContrastiveLossParameter contrastive_loss_param;
	ConvolutionParameter convolution_param;
	DataParameter data_param;
	DropoutParameter dropout_param;
	DummyDataParameter dummy_data_param;
	EltwiseParameter eltwise_param;
	ExpParameter exp_param;
	HDF5DataParameter hdf5_data_param;
	HDF5OutputParameter hdf5_output_param;
	HingeLossParameter hinge_loss_param;
	ImageDataParameter image_data_param;
	InfogainLossParameter infogain_loss_param;
	InnerProductParameter inner_product_param;
	LRNParameter lrn_param;
	MemoryDataParameter memory_data_param;
	MVNParameter mvn_param;
	PoolingParameter pooling_param;
	PowerParameter power_param;
	ReLUParameter relu_param;
	SigmoidParameter sigmoid_param;
	SoftmaxParameter softmax_param;
	SliceParameter slice_param;
	TanHParameter tanh_param;
	ThresholdParameter threshold_param;
	WindowDataParameter window_data_param;
	TransformationParameter transform_param;
	LossParameter loss_param;
	V0LayerParameter layer;
};

typedef struct LayerParameter_ LayerParameter;
struct LayerParameter_ {
	char name[PARSE_STR_NAME_SIZE];
	char type[PARSE_STR_NAME_SIZE];
	char (*bottom)[PARSE_STR_NAME_SIZE];
	UINT32 bottom_size;
	char (*top)[PARSE_STR_NAME_SIZE];
	UINT32 top_size;
	Phase phase;
	float *loss_weight;
	UINT32 loss_weight_size;
	ParamSpec *param;
	UINT32 param_size;
	BlobProto *blobs;
	UINT32 blobs_size;
	BOOL *propagate_down;
	UINT32 propagate_down_size;
	NetStateRule *include;
	UINT32 include_size;
	NetStateRule *exclude;
	UINT32 exclude_size;
	TransformationParameter transform_param;
	LossParameter loss_param;
	AccuracyParameter accuracy_param;
	AnnotatedDataParameter annotated_data_param;
	ArgMaxParameter argmax_param;
	BatchNormParameter batch_norm_param;
	BiasParameter bias_param;
	ConcatParameter concat_param;
	ContrastiveLossParameter contrastive_loss_param;
	ConvolutionParameter convolution_param;
	CropParameter crop_param;
	DataParameter data_param;
	DetectionEvaluateParameter detection_evaluate_param;
	DetectionOutputParameter detection_output_param;
	DropoutParameter dropout_param;
	DummyDataParameter dummy_data_param;
	EltwiseParameter eltwise_param;
	ELUParameter elu_param;
	EmbedParameter embed_param;
	ExpParameter exp_param;
	FlattenParameter flatten_param;
	HDF5DataParameter hdf5_data_param;
	HDF5OutputParameter hdf5_output_param;
	HingeLossParameter hinge_loss_param;
	ImageDataParameter image_data_param;
	InfogainLossParameter infogain_loss_param;
	InnerProductParameter inner_product_param;
	InputParameter input_param;
	LogParameter log_param;
	LRNParameter lrn_param;
	MemoryDataParameter memory_data_param;
	MultiBoxLossParameter multibox_loss_param;
	MVNParameter mvn_param;
	NormalizeParameter norm_param;
	ParameterParameter parameter_param;
	PermuteParameter permute_param;
	PoolingParameter pooling_param;
	PowerParameter power_param;
	PReLUParameter prelu_param;
	PriorBoxParameter prior_box_param;
	PythonParameter python_param;
	RecurrentParameter recurrent_param;
	ReductionParameter reduction_param;
	ReLUParameter relu_param;
	ReshapeParameter reshape_param;
	ScaleParameter scale_param;
	SigmoidParameter sigmoid_param;
	SoftmaxParameter softmax_param;
	SPPParameter spp_param;
	SliceParameter slice_param;
	TanHParameter tanh_param;
	ThresholdParameter threshold_param;
	TileParameter tile_param;
	VideoDataParameter video_data_param;
	WindowDataParameter window_data_param;
	QuantParameter quant_param;
};

typedef struct NetParameter_ NetParameter;
struct NetParameter_ {
	char name[PARSE_STR_NAME_SIZE];
	char (*input)[PARSE_STR_NAME_SIZE];
	UINT32 input_size;
	BlobShape *input_shape;
	UINT32 input_shape_size;
	INT32 *input_dim;
	UINT32 input_dim_size;
	BOOL force_backward;
	NetState state;
	BOOL debug_info;
	LayerParameter *layer;
	UINT32 layer_size;
	V1LayerParameter *layers;
	UINT32 layers_size;
};

typedef struct SolverState_ SolverState;
struct SolverState_ {
	INT32 iter;
	char learned_net[PARSE_STR_NAME_SIZE];
	BlobProto *history;
	UINT32 history_size;
	INT32 current_step;
};

#endif
