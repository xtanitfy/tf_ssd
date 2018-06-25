CAFFE_ROOT_DIR=/home/samba/CNN/caffe_ssd

rm -rf out/*
make clean

cd tools/model_codec

dos2unix *.prototxt
make clean
make codec

./codec \
../../src/proto/caffe.proto \
../../model/mnist/deploy.prototxt \
NetParameter


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
../../model/mnist/deploy.prototxt \
../../model/mnist/lenet_iter_10000.caffemodel

mv parameter.h ../../include/proto
mv parse_NetParameter.c ../../src/proto
cd -


make 

