./ssd_detect \
--file_type=image \
--out_file=out.txt \
../../model/ssd/deploy.prototxt \
../../model/ssd/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel \
./test_images_path.txt