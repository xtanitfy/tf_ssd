WTD_PROJECT = ..
include  $(WTD_PROJECT)/config.local

LIB_TARGET = libcommon.a
SRCS    = $(wildcard *.c)
OBJS    = $(SRCS:.c=.o)

INCLUDE = -I./ -I$(WTD_PROJECT)/include  \
            -I$(WTD_PROJECT)/include/proto \
                -I$(WTD_PROJECT)/include/platform \
                    -I$(WTD_PROJECT)/include/util \
						-I$(WTD_PROJECT)/include/layers
 
all: other $(LIB_TARGET)

other:
	make -C platform all
	make -C proto all
	make -C util all
	make -C layers all
	
$(LIB_TARGET): $(OBJS)
	$(AR) -rc $(LIB_TARGET) $(OBJS) 
	mv $(LIB_TARGET) $(WTD_PROJECT)/build/lib/
	
%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS)  -c -fpic -Wall $< 
clean:
	make -C platform clean
	make -C proto clean
	make -C util clean
	make -C layers clean
	rm -f *.o   *.bak  *~ $(WTD_PROJECT)/build/lib/$(LIB_TARGET)
