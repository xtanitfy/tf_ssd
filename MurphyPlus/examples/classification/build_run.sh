cd ./examples/classification
make clean
make classification 
cd -

./examples/classification/classification \
model/mnist/weigts.bin \
model/mnist/mean.binaryproto \
model/mnist/label.txt \
model/mnist/2.bmp

chmod 777 out -R
