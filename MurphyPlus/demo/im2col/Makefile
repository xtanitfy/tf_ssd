WTD_PROJECT = ../..
include  $(WTD_PROJECT)/config.local

LIB_PATH = $(WTD_PROJECT)/build/lib

#LOCAL_TARGET = math_functions


INCLUDE = -I./ -I$(WTD_PROJECT)/3rparty 

TARGET = main 

CFLAGS = -lpthread  -lm -w -std=c99  

all:$(TARGET)

main: main.c demo_im2col.c
	$(CC) $^ -o $@    $(CFLAGS)  $(OBJS)  $(INCLUDE) 

	
clean:
	rm -f *.o *.bak *.~ $(TARGET) main 





