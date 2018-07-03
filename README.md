# 1. convert vggnet:
    1.1 $cd src/utils && mkdir caffe_model
    
    1.2 put the VGG_ILSVRC_16_layers_fc_reduced.caffemodel and VGG_ILSVRC_16_layers_fc_reduced_deploy.prototxt into caffemodel then run:
      $./ssdVgg2pkl.sh

# 2.convert dataset:
    2.1 $cd data/VKITTI 
    
    2.2 Specify the path of the voc in voc2kitti.py then run:
      $pyhton voc2kitti.py
    
# 3.start train:
    3.1 ./scripts/train_ssd.sh
