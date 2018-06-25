rm -rf out/*

cd tools/model_codec
make clean
cd -

make clean

rm -rf ./model/weigts.bin
