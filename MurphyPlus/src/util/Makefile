WTD_PROJECT = ../..
include  $(WTD_PROJECT)/config.local

LIB_TARGET = libutil.a
SRCS    = $(wildcard *.c)
OBJS    = $(SRCS:.c=.o)

INCLUDE = -I./ -I$(WTD_PROJECT)/include  \
            -I$(WTD_PROJECT)/include/platform \
             -I$(WTD_PROJECT)/include/util \
			-I$(WTD_PROJECT)/include/proto \
			 -I/usr/include
			 

all: $(LIB_TARGET)
$(LIB_TARGET): $(OBJS)
	$(AR) -rc $(LIB_TARGET) $(OBJS) 
	mv $(LIB_TARGET) $(WTD_PROJECT)/build/lib/
	
%.o: %.c
	$(CC) $(INCLUDE) $(CFLAGS)  -c -fpic -Wall $< 
	
clean:
	rm -f *.o   *.bak  *~ $(WTD_PROJECT)/build/lib/$(LIB_TARGET)