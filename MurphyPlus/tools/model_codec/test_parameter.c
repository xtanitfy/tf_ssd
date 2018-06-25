#include "parameter.h"
#include <stdio.h>
#include "data_type.h"

#if 0
typedef struct NetState_ NetState;
struct NetState_{
	//Phase phase;
	int32 level;
	char (*stage)[64];
	uint32 stage_size;
};
#endif

//LayerParameter layer;

int main(int argc,char **argv)
{
	printf("sizeof(NetParameter):%ld\n",sizeof(NetParameter));
	printf("sizeof(LayerParameter):%ld\n",sizeof(LayerParameter));
	printf("test headfile ok!\n");	
	return 0;
}
