cd ./examples/classification
make clean
make classification 
cd -

cd tools/convert
./convert_img ../../model/mnist/2.bmp 1 28 28
./convert_mean ../../model/mnist/mean.binaryproto 1 28 28
if [ $? -ne 0 ];then
	echo "convert mnist failed"
	exit -1
fi
cd -

./examples/classification/classification \
model/mnist/weigts.bin \
model/mnist/mean.binaryproto \
model/mnist/label.txt \
model/mnist/2.bmp

chmod 777 out -R