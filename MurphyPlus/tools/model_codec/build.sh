
if [ $# -ne 4 ];then
	echo "argv[1]:caffe proto file"
	echo "argv[2]:caffe modle prototxt file"
	echo "argv[3]:caffe proto message name"
	echo "argv[4]:caffe modle weights file"
	exit -1
fi

dos2unix *.prototxt
make clean
make codec

./codec $1 $2 $3
if [ $? -ne 0 ];then
	echo "execute codec failed!"
	exit -1
fi

make test_parameter
./test_parameter

make test_execute_parse
./test_execute_parse

make gen_weightsbin_file
./gen_weightsbin_file $2 $4  

PROJECT_DIR=../..
if [ $? -eq 0 ];then
	echo "generate weights binary file ok!"
#	mv weigts.bin $PROJECT_DIR/model
else
	echo "generate weights binary file failed!"
	exit -1
fi

mv parameter.h $PROJECT_DIR/include/proto
mv execute_parse.c $PROJECT_DIR/src/proto
