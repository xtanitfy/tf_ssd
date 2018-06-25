WTD_PROJECT = ../..
include  $(WTD_PROJECT)/config.local

LIB_PATH = $(WTD_PROJECT)/build/lib

#LOCAL_TARGET = math_functions

LIBS    =   $(LIB_PATH)/libplatform.a $(LIB_PATH)/3rparty/libcblas.a $(LIB_PATH)/3rparty/libblas.a

INCLUDE = -I./ -I$(WTD_PROJECT)/3rparty

TARGET = test 

CFLAGS = -lpthread  -lm -w -std=c99  

all:$(TARGET)

test: test.c math_functions.c
	$(CC) $^ -o $@    $(CFLAGS)  $(OBJS)  $(LIBS) $(INCLUDE) 

	
clean:
	rm -f *.o *.bak *.~ $(TARGET) test_log test_conv





