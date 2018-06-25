
make clean
rm -rf out/*

cd tools/model_codec

dos2unix *.prototxt
make clean
make codec

./codec \
../../src/proto/caffe.proto \
../../model/ssd/deploy.prototxt \
NetParameter 
if [ $? -ne 0 ];then
	echo "execute codec failed!"
	exit -1
fi

./codec \
../../src/proto/caffe.proto \
../../model/ssd/labelmap_voc.prototxt \
LabelMap
if [ $? -ne 0 ];then
	echo "execute codec failed!"
	exit -1
fi

make test_parameter
./test_parameter

make test_execute_parse
./test_execute_parse

make gen_weightsbin_file
./gen_weightsbin_file \
../../model/ssd/deploy.prototxt \
../../model/ssd/VGG_VOC0712_SSD_300x300_iter_60000.caffemodel 

mv parameter.h ../../include/proto
mv parse_NetParameter.c ../../src/proto
mv parse_LabelMap.c ../../src/proto
cd -



make all