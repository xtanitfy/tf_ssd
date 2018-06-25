#include "muti_tree.h"

static int MTREECalEndInfo(MTREE_s *pTree);
static int MTREEResetSyncNext(MTREE_s *pTree);
static int MTREEResetSyncPrev(MTREE_s *pTree);

MTREE_s *MTREE_create(void *usr)
{
	MTREE_s *pTree = (MTREE_s *)malloc(sizeof(MTREE_s));
	CHECK_EXPR_RET(pTree == NULL,NULL);
	memset(pTree,0,sizeof(MTREE_s));
	
	pTree->root = NULL;
	pTree->pQAdd = QUEUE_create(sizeof(MTREE_ITEM_s *),QUEUE_MAX_N);
	pTree->pQForward = QUEUE_create(sizeof(MTREE_ITEM_s *),QUEUE_MAX_N);
	pTree->pQBackward = QUEUE_create(sizeof(MTREE_ITEM_s *),QUEUE_MAX_N);
	pTree->num = 0;
	pTree->rootInverse = (MTREE_ITEM_s *)malloc(sizeof(MTREE_ITEM_s));
	CHECK_EXPR_RET(pTree->rootInverse == NULL,NULL);
	pTree->usr = usr;
	
	INIT_LIST_HEAD(&pTree->head);
	
	return pTree;
}

int MTREE_addItem(MTREE_s *pTree,int prev,int val)
{
	CHECK_EXPR_RET(pTree == NULL,-1);
	
	MTREE_ITEM_s *pItem = (MTREE_ITEM_s *)malloc(sizeof(MTREE_ITEM_s));
	memset(pItem,'\0',sizeof(MTREE_ITEM_s));
	pItem->valCurr = val;

	QUEUE_reset(pTree->pQAdd);
	MTREE_ITEM_s *pItemTmp = NULL;
	MTREE_ITEM_s *pRoot = pTree->root;
	if (pRoot == NULL) {
		pTree->root = pItem;
		list_add_tail(&pItem->list,&pTree->head);
	} else {
		if (pRoot->valCurr == prev) {
			pRoot->next[pRoot->nNext++] = pItem;
			pItem->prev[pItem->nPrev++] = pRoot;
			pRoot->syncNext++;
 			pItem->syncPrev++;
			list_add_tail(&pItem->list,&pTree->head);
			return 0;
		}
	
		for (int i = 0;i < pRoot->nNext;i++) {
			QUEUE_enqueue(pTree->pQAdd,&pRoot->next[i]);
		}

		BOOL isFindCurrVal = FALSE;
		MTREE_ITEM_s *currItem = NULL;
		/*先判断添加的val是否存在*/
		while(QUEUE_dequeue(pTree->pQAdd,&pItemTmp) != QUEUE_EMPTY) {
			if (pItemTmp->valCurr == val) {
				currItem = pItemTmp;
				isFindCurrVal = TRUE;
				break;
			}
			for (int i = 0;i < pItemTmp->nNext;i++) {
				QUEUE_enqueue(pTree->pQAdd,&pItemTmp->next[i]);
			}
		}
		
		QUEUE_reset(pTree->pQAdd);
		for (int i = 0;i < pRoot->nNext;i++) {
			QUEUE_enqueue(pTree->pQAdd,&pRoot->next[i]);
		}
		BOOL isFindPrevVal = FALSE;
		while(QUEUE_dequeue(pTree->pQAdd,&pItemTmp) != QUEUE_EMPTY) {
			if (pItemTmp->valCurr == prev) {
				isFindPrevVal = TRUE;
				/*如果已经发现有此值，就使用已经存在的节点*/
				if (isFindCurrVal == TRUE) {
					pItemTmp->next[pItemTmp->nNext++] = currItem;
					currItem->prev[currItem->nPrev++] = pItemTmp;
					pItemTmp->syncNext++;
					currItem->syncPrev++;
					break;
				} else {
				/*如果没有发现此值，就使用分配的节点*/
					pItemTmp->next[pItemTmp->nNext++] = pItem;
					pItem->prev[pItem->nPrev++] = pItemTmp;
					pItemTmp->syncNext++;
					pItem->syncPrev++;
					list_add_tail(&pItem->list,&pTree->head);
				}
				break;
			}				
			for (int i = 0;i < pItemTmp->nNext;i++) {
				QUEUE_enqueue(pTree->pQAdd,&pItemTmp->next[i]);
			}
		}
		if (isFindPrevVal == FALSE) {
			free(pItem);
			printf(" [muti tree not find its parents]!\n");
			getchar();
			//ERROR_DEBUG("AddItem failed:not find its parents!\n");
			return -1;
		}
	}
	return 0;
}

int MTREECalEndInfo(MTREE_s *pTree)
{
	CHECK_EXPR_RET(pTree == NULL,-1);

	MTREE_ITEM_s *pItem = NULL;
	pTree->nEnd = 0;
	int endCnt = 0;
	list_for_each_entry(pItem,&pTree->head,list,MTREE_ITEM_s) {
		if (pItem->nNext == 0) {
			endCnt++;
		}
	}

	printf("[MTREECalEndInfo]endCnt:%d\n",endCnt);
	CHECK_EXPR_RET(endCnt > MTREE_MAX_END,-1);
	
	list_for_each_entry(pItem,&pTree->head,list,MTREE_ITEM_s) {
		if (pItem->nNext == 0) {
			pTree->pEnd[pTree->nEnd++] = pItem;
		}
	}

	return 0;
}

int MTREEResetSyncNext(MTREE_s *pTree)
{
	CHECK_EXPR_RET(pTree == NULL,-1);

	MTREE_ITEM_s *pItem = NULL;
	list_for_each_entry(pItem,&pTree->head,list,MTREE_ITEM_s) {
		pItem->syncNext = pItem->nNext;
	}
	
	return 0;
}

int MTREEResetSyncPrev(MTREE_s *pTree)
{
	CHECK_EXPR_RET(pTree == NULL,-1);

	MTREE_ITEM_s *pItem = NULL;
	list_for_each_entry(pItem,&pTree->head,list,MTREE_ITEM_s) {
		pItem->syncPrev = pItem->nPrev;
	}
	
	return 0;
}



int MTREE_forward(MTREE_s *pTree,int (*callback)(void *usr,int idCurr,int *_idPrev,int _nIdPrev))
{
	MTREE_ITEM_s *pItemTmp = NULL;
	MTREE_ITEM_s *pRoot = pTree->root;
	int idPrev[MTREE_MAX_MUX];
	int nIdPrev = 0;

	//MTREEResetSyncNext(pTree);
	MTREEResetSyncPrev(pTree);
	
	QUEUE_reset(pTree->pQForward);

	CHECK_EXPR_RET(pRoot == NULL,-1);
	if (callback != NULL) {
		callback(pTree->usr,pRoot->valCurr,NULL,0);
	}
	
	//printf("curr:%d \n",pRoot->valCurr);
	for (int i = 0;i < pRoot->nNext;i++) {
		QUEUE_enqueue(pTree->pQForward,&pRoot->next[i]);
	}
	while(QUEUE_dequeue(pTree->pQForward,&pItemTmp) != QUEUE_EMPTY) {
		nIdPrev = 0;
		for (int i = 0;i < pItemTmp->nPrev;i++) {
			CHECK_EXPR_RET(pItemTmp->prev[i] == NULL,-1);
			idPrev[nIdPrev++] = pItemTmp->prev[i]->valCurr;
		}
		if (callback != NULL) {
			callback(pTree->usr,pItemTmp->valCurr,idPrev,nIdPrev);
		}
		for (int i = 0;i < pItemTmp->nNext;i++) {
			if (--pItemTmp->next[i]->syncPrev == 0) {
				QUEUE_enqueue(pTree->pQForward,&pItemTmp->next[i]);
			}
		}
	}
	return 0;
}

int MTREE_printByList(MTREE_s *pTree)
{
	CHECK_EXPR_RET(pTree == NULL,-1);

	MTREE_ITEM_s *pItem = NULL;
	pTree->nEnd = 0;

	printf("MTREE_printByList:\n");
	list_for_each_entry(pItem,&pTree->head,list,MTREE_ITEM_s) {
		printf("curr:%d\n",pItem->valCurr);
	}

	return 0;
}

int MTREE_getEndIds(MTREE_s *pTree,int *endId,int *n)
{
	CHECK_EXPR_RET(pTree == NULL,-1);
	CHECK_EXPR_RET(endId == NULL,-1);
	CHECK_EXPR_RET(n == NULL,-1);

	*n = 0;
	MTREE_ITEM_s *pItem = NULL;
	list_for_each_entry(pItem,&pTree->head,list,MTREE_ITEM_s) {
		if (pItem->nNext == 0) {
			endId[(*n)++] = pItem->valCurr;
		}
	}
	
	return 0;
}


int MTREE_backward(MTREE_s *pTree,int (*callback)(void *usr,int idCurr,int *_idNext,int _nIdNext))
{
	CHECK_EXPR_RET(pTree == NULL,-1);

	MTREE_ITEM_s *pItemTmp = NULL;
	int idNext[MTREE_MAX_MUX];
	int nIdNext = 0;

	/*Need only run once after adding all items!*/
	MTREECalEndInfo(pTree);
	CHECK_EXPR_RET(pTree->nEnd == 0,-1);

	MTREE_ITEM_s *pRoot = pTree->rootInverse;
	CHECK_EXPR_RET(pRoot == NULL,-1);

	printf("pTree->nEnd:%d\n",pTree->nEnd);

	pRoot->nNext = 0;
	for (int i = 0;i < pTree->nEnd;i++) {
		pRoot->next[pRoot->nNext++] = pTree->pEnd[i];
	}
	
	MTREEResetSyncNext(pTree);
	//MTREEResetSyncPrev(pTree);

	QUEUE_reset(pTree->pQBackward);
	for (int i = 0;i < pRoot->nNext;i++) {
		QUEUE_enqueue(pTree->pQBackward,&pRoot->next[i]);
	}
	while(QUEUE_dequeue(pTree->pQBackward,&pItemTmp) != QUEUE_EMPTY) {
		nIdNext = 0;
		for (int i = 0;i < pItemTmp->nNext;i++) {
			CHECK_EXPR_RET(pItemTmp->next[i] == NULL,-1);
			idNext[nIdNext++] = pItemTmp->next[i]->valCurr;
		}
		if (callback != NULL) {
			callback(pTree->usr,pItemTmp->valCurr,idNext,nIdNext);
		}
		for (int i = 0;i < pItemTmp->nPrev;i++) {
			if (--pItemTmp->prev[i]->syncNext == 0) {
				QUEUE_enqueue(pTree->pQBackward,&pItemTmp->prev[i]);
			}
		}
	}

	return 0;
}

int MTREE_backwardExt(MTREE_s *pTree,int (*callback)(void *usr,int idCurr,int *_idNext,int _nIdNext))
{
	CHECK_EXPR_RET(pTree == NULL,-1);

	MTREE_ITEM_s *pItemTmp = NULL;
	int idNext[MTREE_MAX_END];
	int nIdNext = 0;

	printf("pTree->nEnd:%d\n",pTree->nEnd);
	CHECK_EXPR_RET(pTree->nEnd > MTREE_MAX_END,-1);

	printf("1--------\n");
	/*Need only run once after adding all items!*/
	MTREECalEndInfo(pTree);
	//CHECK_EXPR_RET(pTree->nEnd == 0,-1);

	printf("2--------\n");
	MTREEResetSyncNext(pTree);
	//MTREEResetSyncPrev(pTree);
	QUEUE_reset(pTree->pQBackward);

	
	
	printf("3--------\n");
	for (int i = 0;i < pTree->nEnd;i++) {
		QUEUE_enqueue(pTree->pQBackward,&pTree->pEnd[i]);
	}
	while(QUEUE_dequeue(pTree->pQBackward,&pItemTmp) != QUEUE_EMPTY) {
		nIdNext = 0;
		for (int i = 0;i < pItemTmp->nNext;i++) {
			CHECK_EXPR_RET(pItemTmp->next[i] == NULL,-1);
			idNext[nIdNext++] = pItemTmp->next[i]->valCurr;
		}
		if (callback != NULL) {
			callback(pTree->usr,pItemTmp->valCurr,idNext,nIdNext);
		}
		for (int i = 0;i < pItemTmp->nPrev;i++) {
			if (--pItemTmp->prev[i]->syncNext == 0) {
				QUEUE_enqueue(pTree->pQBackward,&pItemTmp->prev[i]);
			}
		}
	}	
	return 0;
}

