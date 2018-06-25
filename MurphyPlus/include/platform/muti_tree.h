#ifndef __MUTI_TREE_H__
#define __MUTI_TREE_H__

#include "public.h"
#include "queue.h"
#include "data_type.h"
#include "dlist.h"

typedef struct MTREE_ITEM_ MTREE_ITEM_s;
#define PARSE_PROTO_TREE 1

#if defined(PARSE_PROTO_TREE)
#define MTREE_MAX_MUX 512
#define MTREE_MAX_END 4096*2
#define QUEUE_MAX_N MTREE_MAX_MUX*8
#else
#define MTREE_MAX_MUX 8
#define MTREE_MAX_END 512
#define QUEUE_MAX_N MTREE_MAX_MUX*8
#endif

struct MTREE_ITEM_
{
	int valCurr;

	MTREE_ITEM_s *next[MTREE_MAX_MUX];
	int nNext;
	int syncNext;
	
	MTREE_ITEM_s *prev[MTREE_MAX_MUX];
	int nPrev;
	int syncPrev;

	struct list_head list;
};


typedef struct
{
	int num;
	MTREE_ITEM_s *root; 
	QUEUE_s *pQAdd;
	QUEUE_s *pQForward;
	QUEUE_s *pQBackward;
	MTREE_ITEM_s *pEnd[MTREE_MAX_END];
	int nEnd;
	void *usr;
	MTREE_ITEM_s *rootInverse; 
	//int idEnd[MTREE_MAX_END];
	struct list_head head;
}MTREE_s;


MTREE_s *MTREE_create(void *usr);
int MTREE_addItem(MTREE_s *pTree,int prev,int val);
int MTREE_forward(MTREE_s *pTree,int (*callback)(void *usr,int idCurr,int *_idPrev,int _nIdPrev));
int MTREE_backward(MTREE_s *pTree,int (*callback)(void *usr,int idCurr,int *_idNext,int _nIdNext));
int MTREE_printByList(MTREE_s *pTree);
int MTREE_getEndIds(MTREE_s *pTree,int *endId,int *n);
int MTREE_backwardExt(MTREE_s *pTree,int (*callback)(void *usr,int idCurr,int *_idNext,int _nIdNext));

#endif
