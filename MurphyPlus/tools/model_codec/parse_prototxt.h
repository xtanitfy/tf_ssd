#ifndef __PARSE_PROTOTXT_FILE_H__
#define __PARSE_PROTOTXT_FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "public.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "data_type.h"
#include "dlist.h"
#include "codec.h"
#include "muti_tree.h"

#define PARSE_PROTO_ADD_ALL_DEFAULT_VALUES 

//for example: item1:
//			   parent1[parentsIdx:0 samePrentsTitleNum:3] 
//             parent2[parentsIdx:3,samePrentsTitleNum:4]
//parent1_type *pp1 = (parent1_type *)malloc(sizeof(samePrentsTitleNum));
//pp1[0].pp2[3].item1 = val;

#define EXECUTE_PARSE_PTOTOTXT_NAME "execute_parse.c"
#define PROTOTXT_FILE_NAME "deploy.prototxt"
#define PARSE_STR_NAME_SIZE  64

#define PARPTTREE_MAX_ITEMS  4096*10

typedef struct
{
	char key[STR_MAX_SIZE];
	char val[STR_MAX_SIZE];
	int idx;
	int sameItemsNum;
	
	struct list_head parentHead;

	MESSAGE_ITEM_t *pItem;
	MESSAGE_t *pRootMessage;
}SUB_ITEM_t;


typedef struct
{
	char parentTitle[STR_MAX_SIZE];
	int  parentsIdx;
	int samePrentsTitleNum;

	struct list_head list;

	MESSAGE_t *pMessage;
	MESSAGE_ITEM_t *pItem;
}PARENT_NODE_t;


typedef struct
{
	BOOL isTitle;
	
	char key[STR_MAX_SIZE];
	char val[STR_MAX_SIZE];
	int num;
	int index;
	int parentId;
	//char typeName[STR_MAX_SIZE];
}PAPRT_ITEM_t;

typedef int (*GetOneItemCallback_f)(SUB_ITEM_t *pItem);
typedef int (*GetOnelineCallback_f) (char *line,int len);
typedef int (*GetOneItemIdInfoCallback_f) (int id,int parentid);

//parseNetParameter prototxt file 

int PARPT_init(char *filename,GetOneItemCallback_f itemCallback,char *rootMessageName);
int PARPT_WriteFileStart(char *messageName);
int PARPT_WriteFileEnd();
int PARPT_WriteOneMessageStart(char *rootMessageName);
int PARPT_WriteOneMessageEnd();

void PARPT_getItemsInfo(MTREE_s **pParptTree,PAPRT_ITEM_t **pParptItems,int **parptItemN);
int PARPT_getLineNum();
int PARPT_buildTree();
int PARPT_run();
int PARPT_writeOneItem(SUB_ITEM_t *pItem);


#endif
