# 1. convert vggnet:
    1.1 $cd src/utils && mkdir caffe_model
    
    1.2 put the VGG_ILSVRC_16_layers_fc_reduced.caffemodel and VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt into caffemodel then run:
      $./ssdVgg2pkl.sh
      [vggnet download] (https://download.csdn.net/download/fireworkpark/10104507)

# 2.convert dataset:
    2.1 $cd data/VKITTI 
    
    2.2 Specify the path of the voc in voc2kitti.py then run:
      $pyhton voc2kitti.py
    
# 3.start train:
    3.1 ./scripts/train_ssd.sh

# 4.start eval:
    4.1 cd data/VKITTI && python create_test.py 
    4.2 ./scripts/eval_ssd.sh
    4.3 ./scripts/res_eval.sh (need spicify the train_model)
    
# 5.will do:
    5.1 trian too slow:
        Rewrite the function _match_bbox and _sparse_to_dense from data_layer.py by using c.