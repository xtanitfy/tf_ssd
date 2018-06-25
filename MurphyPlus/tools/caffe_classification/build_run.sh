dos2unix ../../model/mnist/label.txt
make  clean
make CnnDetector
./CnnDetector \
../../model/mnist/deploy.prototxt \
../../model/mnist/lenet_iter_10000.caffemodel \
../../model/mnist/mean.binaryproto \
../../model/mnist/label.txt \
../../model/mnist/2.bmp
