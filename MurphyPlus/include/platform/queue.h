#ifndef __QUEUE_H__
#define __QUEUE_H__

#include "public.h"
#include "data_type.h"

#define QUEUE_EMPTY -2
#define QUEUE_FULL -3

typedef struct
{
	int head;
	int tail;
	void *data;
	int nMax;
	int itemSize;
}QUEUE_s;


QUEUE_s *QUEUE_create(int itemSize,int nMax);
BOOL QUEUE_empty(QUEUE_s *pQ);
BOOL QUEUE_full(QUEUE_s *pQ);
int QUEUE_enqueue(QUEUE_s *pQ,void *item);
int QUEUE_dequeue(QUEUE_s *pQ,void *item);
int QUEUE_lastItem(QUEUE_s *pQ,void *item);
int QUEUE_reset(QUEUE_s *pQ);
	
#endif