CAFFE_LIB_DIR = /disk/caffe/caffe-ssd/build/lib
CAFFE_INCLUDE_DIR = /disk/caffe/caffe-ssd/include
PROJECT_DIR = ../..

all:    codec test_parameter test_execute_parse gen_weightsbin_file
#-fPIC
FLAGS +=   -DCPU_ONLY -DCAFFE_VERSION=1.0.0-rc3 -DNDEBUG -O2 -DUSE_OPENCV -DUSE_LEVELDB -DUSE_LMDB  -lpthread -DPARSE_PROTO_TREE

PLATFORM_SRC_FILE = $(PROJECT_DIR)/src/platform/stack.c $(PROJECT_DIR)/src/platform/muti_tree.c $(PROJECT_DIR)/src/platform/queue.c
PLATFORM_INCLUDE_FILE = -I$(PROJECT_DIR)/include/platform  \
							-I$(PROJECT_DIR)/include
PROTO_INCLUDE_DIR = -I$(PROJECT_DIR)/include/proto
					
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
                

model_encode: model_encode.cpp 
	g++ -o $@ $^ `pkg-config --cflags --libs opencv` -w $(INCS_FLAGS) $(FLAGS) $(LIBRARY_DIRS) $(LIBS)
	
codec:codec.c  gen_headfile.c parse_prototxt.c $(PLATFORM_SRC_FILE)
	gcc -o $@ $^ $(INCS_FLAGS)  `pkg-config --cflags --libs opencv`  $(FLAGS) $(LIBRARY_DIRS) $(LIBS)

test_parameter:test_parameter.c 
	gcc -o $@ $^  $(PLATFORM_INCLUDE_FILE) -std=c99 

test_execute_parse:test_execute_parse.c parse_NetParameter.c
	gcc -o $@ $^ $(PLATFORM_INCLUDE_FILE) -std=c99

gen_weightsbin_file:gen_weightsbin_file.cpp $(PLATFORM_SRC_FILE)
	g++ -o $@ $^ `pkg-config --cflags --libs opencv` $(PROTO_INCLUDE_DIR) $(INCS_FLAGS) $(FLAGS) $(LIBRARY_DIRS) $(LIBS)

#parse_prototxt:parse_prototxt.c	 ../common/stack.c ../common/muti_tree.c ../common/queue.c
#	gcc -o $@ $^ $(INCS_FLAGS)
	
clean:
	rm -rf model_encode  test_parameter codec test_execute_parse gen_weightsbin_file
 #parse_prototxt
