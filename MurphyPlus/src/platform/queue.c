
#include "queue.h"
#include "public.h"


QUEUE_s *QUEUE_create(int itemSize,int nMax)
{
	QUEUE_s *pQ = (QUEUE_s *)malloc(sizeof(QUEUE_s));
	CHECK_EXPR_RET(pQ == NULL,NULL);

	pQ->data = (void *)malloc(itemSize * nMax);
	CHECK_EXPR_RET(pQ->data == NULL,NULL);
	pQ->itemSize = itemSize;
	pQ->nMax = nMax;
	pQ->head = 0;
	pQ->tail = 0;
	return pQ;
}


BOOL QUEUE_empty(QUEUE_s *pQ)
{
	CHECK_EXPR_RET(pQ == NULL,TRUE);

	if (pQ->head == pQ->tail ) {
		return TRUE;
	}

	return FALSE;
}

BOOL QUEUE_full(QUEUE_s *pQ)
{
	if (pQ->head == pQ->tail && pQ->tail == pQ->nMax) {
		return TRUE;
	}
	return FALSE;
}

int QUEUE_lastItem(QUEUE_s *pQ,void *item)
{
	int last = pQ->tail % pQ->nMax;
	last--;
	if (last < 0) {
		last = pQ->nMax - last;
	}
	memcpy(item,(char *)pQ->data + last*pQ->itemSize,pQ->itemSize);
	return 0;
}


int QUEUE_enqueue(QUEUE_s *pQ,void *item)
{
	CHECK_EXPR_RET(pQ == NULL,-1);

	if (QUEUE_full(pQ) == TRUE) {
		printf("Queue is full!\n");
		return QUEUE_FULL;
	}
	
	memcpy((char *)pQ->data + pQ->tail*pQ->itemSize,item,pQ->itemSize);
	pQ->tail++;
	pQ->tail %= pQ->nMax;
	
	return 0;
}

int QUEUE_dequeue(QUEUE_s *pQ,void *item)
{
	CHECK_EXPR_RET(pQ == NULL,-1);

	if (QUEUE_empty(pQ) == TRUE) {
		return QUEUE_EMPTY;
	}

	memcpy(item,(char *)pQ->data + pQ->head*pQ->itemSize,pQ->itemSize);
	pQ->head++;
	pQ->head %= pQ->nMax;
	
	return 0;
}


int QUEUE_reset(QUEUE_s *pQ)
{
	pQ->head = 0;
	pQ->tail = 0;
	return 0;
}


