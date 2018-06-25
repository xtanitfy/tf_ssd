#include "util.h"

static struct timeval tv_begin, tv_end;
static unsigned long long gMsCnt = 0;

void CalTimeStart(void)
{
	gettimeofday(&tv_begin, NULL);
}

void CalTimeEnd(char *title)
{
	gettimeofday(&tv_end, NULL);

	unsigned long long tmp = (tv_end.tv_sec - tv_begin.tv_sec) * 1000 + \
		(tv_end.tv_usec - tv_begin.tv_usec)/1000;
	
	printf("[%s]: consume time:%ld ms\n",title,tmp);

	gMsCnt += tmp;
	
}

unsigned long long GetAllTimes(void)
{
	return gMsCnt;
}



