

#include "stack.h"
#include "data_type.h"

void * STACK_init(int maxN,int itemLen)
{
	STACK *s;

	s = (STACK *)malloc(sizeof(STACK));
	CHECK_EXPR_RET(s == NULL, NULL);

	s->maxNum = maxN;
	s->itemLen = itemLen;
    s->data = (void *)malloc(s->maxNum * s->itemLen);
	CHECK_EXPR_RET(s->data == NULL, NULL);

    s->dataNum = 0;
	
	return s;
}

int STACK_empty(void *handle)
{
	STACK *s;
	s = (STACK *)handle;
	CHECK_EXPR_RET(handle == NULL, -1);

	if (s->dataNum == 0) {
		return TRUE;
	} else {
		return FALSE;
	}
}

int STACK_full(void *handle)
{
	STACK *s;
	
	if(handle == NULL){
		return -1;
	}
	s = (STACK *)handle;
	return (s->dataNum >= s->maxNum);
}

int STACK_peekLatest(void *handle,void *data)
{
	CHECK_EXPR_RET(handle == NULL, -1);
	CHECK_EXPR_RET(data == NULL, -1);
	CHECK_EXPR_RET(STACK_empty(handle), -1);

	STACK *s = (STACK *)handle;
	memcpy(data,(char *)s->data+(s->dataNum-1)*(s->itemLen),s->itemLen);
	
	return 0;
}

int STACK_push(void *handle,void *item)
{
	STACK *s;
	
	if(handle == NULL || item == NULL) {
		return -1;
	}
	s = (STACK *)handle;
	if(s->dataNum >= s->maxNum){
		return -1;
	}
	memcpy((char *)s->data+(s->dataNum)*(s->itemLen),(char *)item,s->itemLen);
	s->dataNum++;
	
	return 0;
}

int STACK_pop(void *handle,void *data)
{	
	STACK *s;

	CHECK_EXPR_RET(handle == NULL || data == NULL, -1);
	
	s = (STACK *)handle;
	CHECK_EXPR_RET(s->dataNum == 0, -1);
	s->dataNum--;
	memcpy(data,(char *)s->data+(s->dataNum)*(s->itemLen),s->itemLen);
	
	return 0;
}

int STACK_getStackDataArr(void *handle,void **data,int *len)
{
	STACK *s;
	
	if(handle == NULL || data == NULL || len == NULL){
		return -1;
	}
	if(STACK_empty(handle)){
		return -1;
	}
	s = (STACK *)handle;
	*data = s->data;
	*len = s->dataNum;
	
	return 0;
}

int STACK_destroy(void *handle)
{
	STACK *s = (STACK *)handle;
	
	if(handle == NULL)
		return -1;
	else
	{
		if(s->data != NULL)
			free(s->data);
		free(s);
	}
	return 0;
}
