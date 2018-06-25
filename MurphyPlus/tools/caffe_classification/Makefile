
CAFFE_LIB_DIR = /disk/caffe/caffe-ssd/build/lib
CAFFE_INCLUDE_DIR = /disk/caffe/caffe-ssd/include
all:    CnnDetector
#-fPIC
FLAGS +=   -DCPU_ONLY -DCAFFE_VERSION=1.0.0-rc3 -DNDEBUG -O2 -DUSE_OPENCV -DUSE_LEVELDB -DUSE_LMDB  -lpthread -DPARSE_PROTO_TREE

					
INCS_FLAGS = -I/usr/include/python2.7 \
			-I/usr/lib/python2.7/dist-packages/numpy/core/include \
			-I/usr/local/include \
			-I/usr/include/hdf5/serial  \
			-I/usr/local/lib/python2.7/dist-packages/numpy/core/include \
			-I$(CAFFE_INCLUDE_DIR) \
            $(PLATFORM_INCLUDE_FILE)
				

LIBS = -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lboost_regex -lm -lhdf5_hl -lhdf5 -lleveldb -lsnappy -llmdb -lcaffe -lboost_thread -lstdc++ -lcblas -latlas -lblas -Wl,-rpath,\$ORIGIN/../../lib `pkg-config --cflags --libs opencv` 
		
LIBRARY_DIRS = -L/usr/local/lib \
				-L/usr/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial \
				-L$(CAFFE_LIB_DIR)
                

CnnDetector:CnnDetector.cpp 
	g++ -o $@ $^ `pkg-config --cflags --libs opencv` $(PROTO_INCLUDE_DIR) $(INCS_FLAGS) $(FLAGS) $(LIBRARY_DIRS) $(LIBS)

#parse_prototxt:parse_prototxt.c	 ../common/stack.c ../common/muti_tree.c ../common/queue.c
#	gcc -o $@ $^ $(INCS_FLAGS)
	
clean:
	rm -rf CnnDetector 
 #parse_prototxt
