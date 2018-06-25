cd ./examples/detection
make clean
make  
cd -

cd tools/convert
./convert_img ../../model/ssd/test.jpg 3 300 300
if [ $? -ne 0 ];then
	echo "convert mnist failed"
	exit -1
fi
cd -

dos2unix model/ssd/mean.txt
dos2unix model/ssd/label.txt

./examples/detection/detect \
model/ssd/weigts.bin \
model/ssd/mean.txt \
model/ssd/label.txt \
model/ssd/test.jpg

chmod 777 out -R