CAFFE_LIB_DIR = /disk/caffe/caffe-ssd/build/lib
CAFFE_INCLUDE_DIR = /disk/caffe/caffe-ssd/include

PROJECT_DIR = ../..
include  $(PROJECT_DIR)/config.local

LIBS_DIR = -L$(PROJECT_DIR)/build/lib
LIBS_DIR += -L$(PROJECT_DIR)/build/lib/3rparty -L/usr/local/lib \
				-L/usr/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
				-L$(CAFFE_LIB_DIR) 
LIBS_DIR += -L$(PROJECT_DIR)/build/lib
LIBS_DIR += -L$(PROJECT_DIR)/build/lib/3rparty

INCLUDE_DIR = -I$(PROJECT_DIR)/include
INCLUDE_DIR += -I$(PROJECT_DIR)/include/platform
INCLUDE_DIR += -I$(PROJECT_DIR)/include/proto
INCLUDE_DIR += -I$(PROJECT_DIR)/include/3rparty
INCLUDE_DIR += -I$(PROJECT_DIR)/include/util -I/usr/include/python2.7 \
				-I/usr/lib/python2.7/dist-packages/numpy/core/include \
				-I/usr/local/include \
				-I/usr/include/hdf5/serial  \
				-I/usr/local/lib/python2.7/dist-packages/numpy/core/include \
				-I$(CAFFE_INCLUDE_DIR)

LIBS_FALGS = -lcblas -lblas -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lboost_regex -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb -lcaffe -lboost_thread -lstdc++ -lcblas -latlas -lblas -Wl,-rpath,\$ORIGIN/../../lib `pkg-config --cflags --libs opencv` -lpthread


FLAGS =   -DCPU_ONLY -DCAFFE_VERSION=1.0.0-rc3 -DNDEBUG -O2 -DUSE_OPENCV -DUSE_LEVELDB -DUSE_LMDB -DPARSE_PROTO_TREE

all:convert_img convert_mean

	
convert_img:convert_img.cpp $(PROJECT_DIR)/src/blob.c $(PROJECT_DIR)/src/util/math_functions.c
	g++  -o $@ $^ $(INCLUDE_DIR) $(LIBS_DIR) $(LIBS_FALGS)  $(FLAGS)
	
convert_mean:convert_mean.cpp
	g++  -o $@ $^ $(INCLUDE_DIR) $(LIBS_DIR) $(LIBS_FALGS)  $(FLAGS)

clean:
	rm -rf convert_img  convert_mean
	
	