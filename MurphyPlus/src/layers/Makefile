WTD_PROJECT = ../..
include  $(WTD_PROJECT)/config.local

LIB_DIR=$(WTD_PROJECT)/build/lib

LIB_TARGET = liblayers.a
SRCS    = $(wildcard *.c)
OBJS    = $(SRCS:.c=.o)

INCLUDE = -I./ -I$(WTD_PROJECT)/include \
			-I$(WTD_PROJECT)/include/layers \
				-I$(WTD_PROJECT)/include/platform \
					-I$(WTD_PROJECT)/include/proto \
						-I$(WTD_PROJECT)/include/util
all: $(LIB_TARGET)
$(LIB_TARGET): $(OBJS)
	$(AR) -rc $(LIB_TARGET) $(OBJS)  
	mv $(LIB_TARGET) $(LIB_DIR)
	
%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS)  -c -fpic -Wall $< 
clean:
	rm -f *.o   *.bak  *~ $(LIB_DIR)/$(LIB_TARGET)