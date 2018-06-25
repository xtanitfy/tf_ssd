PROJECT_DIR = ../..
include  $(PROJECT_DIR)/config.local

LIBS_DIR = -L$(PROJECT_DIR)/build/lib
LIBS_DIR += -L$(PROJECT_DIR)/build/lib/3rparty

INCLUDE_DIR = -I$(PROJECT_DIR)/include
INCLUDE_DIR += -I$(PROJECT_DIR)/include/platform
INCLUDE_DIR += -I$(PROJECT_DIR)/include/proto
#INCLUDE_DIR += -I$(PROJECT_DIR)/include/3rparty
INCLUDE_DIR += -I$(PROJECT_DIR)/include/util
INCLUDE_DIR += -I/usr/include
LIBS_FALGS = -lcommon -llayers -lplatform -lmyproto -lutil -lcblas -lblas

all:classification

	
classification:classification.c
	gcc -o $@ $^ $(INCLUDE_DIR) $(LIBS_DIR) $(LIBS_FALGS)  $(CFLAGS)  
	
clean:
	rm -rf classification 
	
	
	
	
