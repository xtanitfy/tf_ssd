#include "parse_prototxt.h"
#include "public.h"
#include "stack.h"


#define PARPT_STACK_MAX_SIZE  128
typedef struct
{
	char name[STR_MAX_SIZE];
	int cnt;
}VOTE_t;

static int PARPTReadFromFile(char *filename);
static int PARPTGetOneLine(char *start,char *line);
static int PARPTParseOneLine(char *line,int len);
static int PARPTParseBuf(GetOnelineCallback_f callback);
static BOOL PARPTFilterLine(char *line,int len);
static int PARPTGetLineNum(char *line,int len);
static int PARPTParseTitle(char *line,int len,PAPRT_ITEM_t *pItem);
static int PARPTSplitLine(char *line);
static int PARPTSwitchChar(char *buf,char old_c,char new_c);
static int PARPTAppendOneItem(PAPRT_ITEM_t *pItem);
static int PARPTOnGetOneItemIdInfo(int id,int parentId);
static int PARPTPrintSplitInfo();
static int PAPRTTreeCallback(void * usr, int idCurr, int * _idNext, int _nIdNext);
static int PAPARTGetAllRoutes();
static int PAPARTGetAllSubItems();
static int PARPTWriteFile(char *buf,int len);
static int PARPT_getStuName(char *messageName,char *stuName);
static BOOL PAPARTJudgeIsMalloced(char *buf,int len);
static int  PAPARTAddMallocFlag(char *buf,int len);
static int	PARPTWriteMallocInfo(SUB_ITEM_t *pSubItem);
static int	PARPTWriteDefaultVal(char *buf,int *plen,PARENT_NODE_t *pParentNode);
static int  PAPARTAddGiveDefaultValFlag(char *buf,int len);
static BOOL PAPARTJudgeIsGiveDefaultVal(char *buf,int len);
static int	PARPTAddRootStr(char *buf,int *pLen,SUB_ITEM_t *pSubItem);
static int PARPTSplitLineExt(char *line,int len,char splitChar);
static int PARPTFilterChar(char *buf,char c);

static char *pFileBuffer = NULL;
static int gFileBufferLen = 0;
static void *gStack;
static MTREE_s *pTree = NULL;


static GetOneItemCallback_f pItemCallback = NULL;
static GetOneItemIdInfoCallback_f pGetOneItemIdInfoCallback = NULL;
static PAPRT_ITEM_t *pItems = NULL;
static BOOL *gIdflags = NULL;
static int gItemsId = 0;
static int gLineNum = 0;
static void *pStack = NULL; 

#define PAPRT_ONE_LINE_MAX_SPLITS 16
static char gLineSplit[PAPRT_ONE_LINE_MAX_SPLITS][STR_MAX_SIZE];
static int  gLineSplitLen = 0;

static SUB_ITEM_t *pSubItem = NULL;
static int gSubItemNum = 0;
static FILE *pExecuteParseFile = NULL;

static char (*gHaveMallocedStr)[256];
static int gHaveMallocedStrLen = 0;

static char (*gHaveGiveDefaultVal)[256];
static int gHaveGiveDefaultValLen = 0;

static VOTE_t *pVote = NULL;
static int voteNum = 0;

static char gRootMessageName[STR_MAX_SIZE];

extern TypeMapStr_t gTypeMapStr[];
extern MESSAGE_t *getMessageNodeByName(char *messageName);

#if 0
int main(int argc,char **argv)
{
	PARPT_init(PROTOTXT_FILE_NAME,NULL);
	PARPT_run();

	return 0;
}
#endif

void PARPT_getItemsInfo(MTREE_s **pParptTree,PAPRT_ITEM_t **pParptItems,int **parptItemN)
{
	*pParptTree = pTree;
	*pParptItems = pItems;
	*parptItemN = &gItemsId;
}

int PARPT_init(char *filename,GetOneItemCallback_f itemCallback,char *rootMessageName)
{
	int ret = -1;
	
	ret = PARPTReadFromFile(filename);
	CHECK_EXPR_RET(ret < 0, -1);

	gStack = STACK_init(PARPT_STACK_MAX_SIZE, sizeof(int));
	CHECK_EXPR_RET(gStack == NULL, -1);

	pItemCallback = itemCallback;
	gLineNum = 0; 

	pTree = MTREE_create(NULL);
	CHECK_EXPR_RET(pTree == NULL, -1);

	MTREE_addItem(pTree,-2,-1);
	pVote = (VOTE_t *)malloc(sizeof(VOTE_t) * MTREE_MAX_MUX);
	voteNum = 0;

	strcpy(gRootMessageName,rootMessageName);
	
	return 0;
}

int PARPT_writeCheckStringlen(char *buf,int len,char *str)
{
	snprintf(buf,len,"CHECK_EXPR_RET(strlen(%s) > PARSE_STR_NAME_SIZE - 1,-1);\n",
					str);
	PARPTWriteFile("\t",1);
	PARPTWriteFile(buf,strlen(buf));

	return 0;
}


int PARPT_writeContent(SUB_ITEM_t *pSubItem)
{
	char buf[1024];
	char tmp[256];
	char tmp1[256];
	int len1 = 0;
	int len = 0;
	char rootName[STR_MAX_SIZE];
	MESSAGE_t *pMessage = NULL;
	MESSAGE_ITEM_t *pItem = NULL;

	PARPTAddRootStr(buf, &len, pSubItem);
	
	PARENT_NODE_t *pParentNode = NULL;
	list_for_each_entry(pParentNode, &pSubItem->parentHead, list, PARENT_NODE_t) {
		pItem = pParentNode->pItem;

		//先不malloc
		strcpy(&buf[len],pParentNode->parentTitle);
		len += strlen(pParentNode->parentTitle);
		if (pItem->attr == ATTR_repeated) {
			snprintf(tmp,sizeof(tmp),"[%d]",pParentNode->parentsIdx);
			strcpy(&buf[len], tmp);
			len += strlen(tmp);
			
		}
		buf[len++] = '.';	
	}

	//先这样写 element的size跟malloc写一块
	pItem = pSubItem->pItem;
	if (pItem->attr == ATTR_repeated) {

		if (pItem->isEnumModifed == TRUE) {
			char tmp1[256];
			strncpy(tmp1,buf,len);
			tmp1[len] = '\0';
			
			snprintf(tmp,sizeof(tmp),"%s%s[%d] = %s_%s;\n",
				tmp1,pSubItem->key,pSubItem->idx,pItem->messageTypeName,pSubItem->val);
			PARPTWriteFile("\t",1);
			PARPTWriteFile(tmp,strlen(tmp));
		} else {
		
			if (pItem->type == TYPE_string || pItem->type == TYPE_bytes) {
				char tmp1[256];
				strncpy(tmp1,buf,len);
				tmp1[len] = '\0';

				PARPT_writeCheckStringlen(tmp,sizeof(tmp),pSubItem->val);
				
				snprintf(tmp,sizeof(tmp),"strcpy(%s%s[%d],%s);\n",
								tmp1,pSubItem->key,pSubItem->idx,pSubItem->val);
				PARPTWriteFile("\t",1);
				PARPTWriteFile(tmp,strlen(tmp));
			} else {
				snprintf(tmp,sizeof(tmp),"%s[%d] = %s;\n",
								pSubItem->key,pSubItem->idx,pSubItem->val);
				strcpy(&buf[len], tmp);
				len += strlen(tmp);
				PARPTWriteFile("\t",1);
				PARPTWriteFile(buf,len);
			}
		}
		
	} else {
		if (pItem->isEnumModifed == TRUE) {
			char tmp1[256];
			strncpy(tmp1,buf,len);
			tmp1[len] = '\0';
			snprintf(tmp,sizeof(tmp),"%s%s = %s_%s;\n",
					tmp1,pSubItem->key,pItem->messageTypeName,pSubItem->val);
			PARPTWriteFile("\t",1);
			PARPTWriteFile(tmp,strlen(tmp));
		} else {
			if (pItem->type == TYPE_string || pItem->type == TYPE_bytes) {
				char tmp1[256];
				strncpy(tmp1,buf,len);
				tmp1[len] = '\0';

				PARPT_writeCheckStringlen(tmp,sizeof(tmp),pSubItem->val);
				snprintf(tmp,sizeof(tmp),"strcpy(%s%s,%s);\n",
								tmp1,pSubItem->key,pSubItem->val);
				PARPTWriteFile("\t",1);
				PARPTWriteFile(tmp,strlen(tmp));
				
			} else {
				snprintf(tmp,sizeof(tmp),"%s = %s;\n",pSubItem->key,pSubItem->val);
				strcpy(&buf[len], tmp);
				len += strlen(tmp);
				PARPTWriteFile("\t",1);
				PARPTWriteFile(buf,len);
			}
			
		}
		
	}
	
	return 0;
}

int PARPT_getStuName(char *messageName,char *stuName)
{
	CHECK_EXPR_RET(messageName == NULL, -1);
	CHECK_EXPR_RET(stuName == NULL, -1);
	
	strcpy(stuName,messageName);

	stuName[0] -= 'A' - 'a';

	return 0;
}

int	PARPTAddRootStr(char *buf,int *pLen,SUB_ITEM_t *pSubItem)
{
	char rootName[STR_MAX_SIZE];
	PARPT_getStuName(pSubItem->pRootMessage->name, rootName);

	int len = *pLen;
	strcpy(&buf[len],rootName);
	len += strlen(rootName);
	buf[len++] = '-';
	buf[len++] = '>';
	*pLen = len;
	
	return  0;
}

static int printBuffer(char *buf,int len,int index)
{
	char tmp[1024];

	CHECK_EXPR_RET(len > 1024, -1);

	memcpy(tmp,buf,len);
	tmp[len] = '\0';
	
	printf("[%d] buffer:%s len:%d\n",index,tmp,len);
	
	return 0;
}

int	PARPTWriteMallocInfo(SUB_ITEM_t *pSubItem)
{

	char buf[1024];
	char tmp[256];
	char tmp1[256];
	int len1 = 0;
	int len = 0;
	char rootName[STR_MAX_SIZE];
	MESSAGE_t *pMessage = NULL;
	MESSAGE_ITEM_t *pItem = NULL;

	memset(buf,'\0',sizeof(buf));
#if 0	
	PARPT_getStuName(pSubItem->pRootMessage->name, rootName);
	strcpy(&buf[len],rootName);
	len += strlen(rootName);
	buf[len++] = '.';
#else
	PARPTAddRootStr(buf,&len,pSubItem);
#endif
	
	int len_tmp;
	PARENT_NODE_t *pParentNode = NULL;
	list_for_each_entry(pParentNode, &pSubItem->parentHead, list, PARENT_NODE_t) {
		pItem = pParentNode->pItem;

		//write title
		strcpy(&buf[len],pParentNode->parentTitle);
		len += strlen(pParentNode->parentTitle);

		//printBuffer(buf,len,0);
		
		//start malloc
		if (pItem->attr == ATTR_repeated && pParentNode->parentsIdx == 0) {
			if (PAPARTJudgeIsMalloced(buf,len) == FALSE) { 
				char tmp1[256];
				memcpy(tmp1,buf,len);
				tmp1[len] = '\0';

				char tmp2[256];
				memcpy(tmp2,buf,len);
				tmp2[len] = '\0';

				len_tmp = len;
				
				PAPARTAddMallocFlag(tmp1,len);
				
				pItem = pParentNode->pItem;
				CHECK_EXPR_RET(pItem == NULL, -1);			

				char tmp3[256];
				strcpy(tmp3,tmp1);
				
				snprintf(tmp,sizeof(tmp)," = (%s *)malloc(sizeof(%s)*%d);\n",
					pItem->messageTypeName,pItem->messageTypeName,pParentNode->samePrentsTitleNum);
				memcpy(&tmp1[len],tmp,strlen(tmp));
				len += strlen(tmp);
				//add 20170423
				snprintf(tmp,sizeof(tmp),"\tmemset(%s,'\\0',sizeof(%s) * %d);\n",
							tmp3,pItem->messageTypeName,pParentNode->samePrentsTitleNum);
				memcpy(&tmp1[len],tmp,strlen(tmp));
				len += strlen(tmp);
				//add end

				PARPTWriteFile("\t",1);
				PARPTWriteFile(tmp1,len);

				snprintf(tmp,sizeof(tmp),"%s_size = %d;\n",
						tmp2,pParentNode->samePrentsTitleNum);
				PARPTWriteFile("\t",1);
				PARPTWriteFile(tmp,strlen(tmp));
				#if defined(PARSE_PROTO_ADD_ALL_DEFAULT_VALUES)
				
				#else
					PARPTWriteDefaultVal(buf,&len,pParentNode);
				#endif
				len = len_tmp;
			}
		} else {
		#if !defined(PARSE_PROTO_ADD_ALL_DEFAULT_VALUES)
			PARPTWriteDefaultVal(buf,&len,pParentNode);
		#endif
		}
		if (pParentNode->samePrentsTitleNum > 1 || pItem->attr == ATTR_repeated) {
			snprintf(tmp,sizeof(tmp),"[%d].",pParentNode->parentsIdx);
			memcpy(&buf[len],tmp,strlen(tmp));
			len += strlen(tmp);
		} else {
			buf[len++] = '.';
		}
	}
	
	//printBuffer(buf,len,1);
	pItem = pSubItem->pItem;
	if (pItem->attr == ATTR_repeated && pSubItem->idx == 0) {
		
		//write title
		memcpy(&buf[len],pSubItem->key,strlen(pSubItem->key));
		len += strlen(pSubItem->key);
		
		if (PAPARTJudgeIsMalloced(buf,len) == FALSE) { 
			
			PAPARTAddMallocFlag(buf,len);
			
			char *typeStr = gTypeMapStr[pItem->type].w_str;

			char tmp1[256];
			memcpy(tmp1,buf,len);
			tmp1[len] = '\0';

			char tmp2[256];
			
			if (pItem->type == TYPE_string || pItem->type == TYPE_bytes) {
				snprintf(tmp,sizeof(tmp)," = (char (*)[PARSE_STR_NAME_SIZE])malloc(PARSE_STR_NAME_SIZE*%d);\n",
										pSubItem->sameItemsNum);
				snprintf(tmp2,sizeof(tmp2),"\tmemset(%s,'\\0',PARSE_STR_NAME_SIZE * %d);\n",tmp1,pSubItem->sameItemsNum);
			} else {
				if (pItem->isEnumModifed == TRUE) {
					snprintf(tmp,sizeof(tmp)," = (%s *)malloc(sizeof(%s)*%d);\n",
							pItem->messageTypeName,pItem->messageTypeName,pSubItem->sameItemsNum);
					
					snprintf(tmp2,sizeof(tmp2),"\tmemset(%s,'\\0',sizeof(%s)*%d);\n",
									tmp1,pItem->messageTypeName,pSubItem->sameItemsNum);
				} else {
					snprintf(tmp,sizeof(tmp)," = (%s *)malloc(sizeof(%s)*%d);\n",
								typeStr,typeStr,pSubItem->sameItemsNum);
					snprintf(tmp2,sizeof(tmp2),"\tmemset(%s,'\\0',sizeof(%s)*%d);\n",
									tmp1,typeStr,pSubItem->sameItemsNum);
				}
			}
			
			strcpy(&buf[len],tmp);
			len += strlen(tmp);

			PARPTWriteFile("\t",1);
			PARPTWriteFile(buf,len);
			PARPTWriteFile(tmp2,strlen(tmp2));

			snprintf(tmp,sizeof(tmp),"%s_size = %d;\n",
						tmp1,pSubItem->sameItemsNum);

			PARPTWriteFile("\t",1);
			PARPTWriteFile(tmp,strlen(tmp));
			
		}
	}

	return 0;
}

int	PARPTWriteDefaultVal(char *buf,int *plen,PARENT_NODE_t *pParentNode)
{
	CHECK_EXPR_RET(buf == NULL, -1);
	CHECK_EXPR_RET(pSubItem == NULL, -1);
	CHECK_EXPR_RET(plen == NULL, -1);

	if (PAPARTJudgeIsGiveDefaultVal(buf,*plen) == TRUE) {
		return 0;
	}
	
	int len = *plen;
	
	char tmp[256];
	char bufPrefix[256];
	memcpy(bufPrefix,buf,len);
	bufPrefix[len] = '\0';

	MESSAGE_ITEM_t *pItem = pParentNode->pItem;
	MESSAGE_t *pMessage = getMessageNodeByName(pItem->messageTypeName);
	CHECK_EXPR_RET(pMessage == NULL, -1);

	//printf("pMessage->messageTypeName:%s\n",pMessage->name);
	BOOL isHaveDefault = FALSE;
	for (int i = 0;i < pMessage->itemN;i++) {
		pItem = &pMessage->pItems[i];
		if (pItem->isHaveDefaultVal == TRUE) {
			isHaveDefault = TRUE;
		}	
	}

	if (isHaveDefault == TRUE) {
		char *str = "\n\t//Give default values begin:\n";
		PARPTWriteFile(str,strlen(str));
		if (pParentNode->samePrentsTitleNum > 1) {
			snprintf(tmp,sizeof(tmp),"for (int i = 0;i < %d;i++) {\n",pParentNode->samePrentsTitleNum);
			PARPTWriteFile("\t",1);
			PARPTWriteFile(tmp, strlen(tmp));
			for (int i = 0;i < pMessage->itemN;i++) {
				pItem = &pMessage->pItems[i];
				if (pItem->isHaveDefaultVal == TRUE) {
					isHaveDefault = TRUE;
					
					if (pItem->isEnumModifed == TRUE) {
						snprintf(tmp,sizeof(tmp),"%s[i].%s = %s_%s;//default value\n",
									bufPrefix,pItem->keystr,pItem->messageTypeName,pItem->defaultVal);
					} else {
						if (pItem->type == TYPE_string || pItem->type == TYPE_bytes) {
							
							snprintf(tmp,sizeof(tmp),"strcpy(%s[i].%s,%s);//default value\n",
											bufPrefix,pItem->keystr,pItem->defaultVal);
						} else {
							snprintf(tmp,sizeof(tmp),"%s[i].%s = %s;//default value\n",
										bufPrefix,pItem->keystr,pItem->defaultVal);

						}
						
					}
					PARPTWriteFile("\t",1);
					PARPTWriteFile("\t",1);
					PARPTWriteFile(tmp, strlen(tmp));
				}	
			}
			PARPTWriteFile("\t",1);
			PARPTWriteFile("}\n",2);
		} else {
			
			for (int i = 0;i < pMessage->itemN;i++) {
				pItem = &pMessage->pItems[i];
				if (pItem->isHaveDefaultVal == TRUE) {
					isHaveDefault = TRUE;
					if (pItem->isEnumModifed == TRUE) {
						snprintf(tmp,sizeof(tmp),"%s.%s = %s_%s;//default value\n",
								bufPrefix,pItem->keystr,pItem->messageTypeName,pItem->defaultVal);
					} else {
						if (pItem->type == TYPE_string || pItem->type == TYPE_bytes) {
							
							snprintf(tmp,sizeof(tmp),"strcpy(%s.%s,%s);//default value\n",
											bufPrefix,pItem->keystr,pItem->defaultVal);
						}  else {
							snprintf(tmp,sizeof(tmp),"%s.%s = %s;//default value\n",bufPrefix,pItem->keystr,pItem->defaultVal);	
						}
						
					}
					PARPTWriteFile("\t",1);
					PARPTWriteFile(tmp, strlen(tmp));
				}	
			}
		}
		str = "\t//Give default values end\n\n";
		PARPTWriteFile(str,strlen(str));
	}
	
	PAPARTAddGiveDefaultValFlag(buf,len);

	*plen = len;
	return 0;
}


int PARPT_writeOneItem(SUB_ITEM_t *pSubItem)
{
	MESSAGE_t *pRootMessage = pSubItem->pRootMessage;
	CHECK_EXPR_RET(pRootMessage == NULL, -1);

	PARPTWriteMallocInfo(pSubItem);
	
	PARPT_writeContent(pSubItem);
	
	return 0;
}

static BOOL JudgeIsMalloced(char *buf,int len)
{
	BOOL isMalloced = FALSE;

	for (int i = 0;i < gSubItemNum;i++) {
		if (strncmp(buf,gHaveMallocedStr[i],len) == 0) {
			isMalloced = TRUE;
			break;
		}
	}
	return isMalloced;
}


static int PAPARTAddGiveDefaultValFlag(char *buf,int len)
{
	gHaveGiveDefaultVal[gHaveGiveDefaultValLen][len] = '\0';
	strncpy(gHaveGiveDefaultVal[gHaveGiveDefaultValLen++],buf,len);

	return 0;
}

static int PAPARTAddMallocFlag(char *buf,int len)
{
	gHaveMallocedStr[gHaveMallocedStrLen][len] = '\0';
	strncpy(gHaveMallocedStr[gHaveMallocedStrLen++],buf,len);
	return 0;
}

static BOOL PAPARTJudgeIsGiveDefaultVal(char *buf,int len)
{
	BOOL isGiveded = FALSE;
	char tmp[len+1];
	
	strncpy(tmp,buf,len);
	tmp[len] = '\0';
	for (int i = 0;i < gHaveGiveDefaultValLen;i++) {
		if (strcmp(tmp,gHaveGiveDefaultVal[i]) == 0) {
			isGiveded = TRUE;
			break;
		}
	}
	return isGiveded;
}


static BOOL PAPARTJudgeIsMalloced(char *buf,int len)
{
	BOOL isMalloced = FALSE;
	char tmp[len+1];
	
	strncpy(tmp,buf,len);
	tmp[len] = '\0';

	for (int i = 0;i < gHaveMallocedStrLen;i++) {
		if (strcmp(tmp,gHaveMallocedStr[i]) == 0) {
			isMalloced = TRUE;
			break;
		}
	}
	
	return isMalloced;
}

static int PAPARTGetAllSubItems()
{	
	gHaveMallocedStr = (char (*)[256])malloc(256 * sizeof(char) * gSubItemNum);
	CHECK_EXPR_RET(gHaveMallocedStr == NULL, -1);

	gHaveGiveDefaultVal = (char (*)[256])malloc(256 * sizeof(char) * gSubItemNum);
	CHECK_EXPR_RET(gHaveGiveDefaultVal == NULL, -1);
	
	for (int i = 0;i < gSubItemNum;i++) {
		if (pItemCallback != NULL) {
			pItemCallback(&pSubItem[i]);
		}
	}
	
	return 0;
}


static int PAPARTGetAllRoutes()
{
	int endIds[MTREE_MAX_END];
	int n;

	MTREE_getEndIds(pTree, endIds, &n);
	
	pSubItem = (SUB_ITEM_t *)malloc(sizeof(SUB_ITEM_t)*n);
	CHECK_EXPR_RET(pSubItem == NULL, -1);
	gSubItemNum = n;

	int id = -1;
	int parentId = -1;
	PAPRT_ITEM_t *pItem = NULL;
	for (int i = 0;i < n;i++) {
		pItem = &pItems[endIds[i]];
		strcpy(pSubItem[i].key,pItem->key);
		strcpy(pSubItem[i].val,pItem->val);
		
		pSubItem[i].sameItemsNum = pItem->num;
		pSubItem[i].idx = pItem->index;

		INIT_LIST_HEAD(&pSubItem[i].parentHead);
		
		parentId = pItem->parentId;
		int cnt = n;
		while(parentId != -1 && --cnt) {
			
			pItem = &pItems[parentId];
			
			PARENT_NODE_t *pParentNode = (PARENT_NODE_t *)malloc(sizeof(PARENT_NODE_t));
			CHECK_EXPR_RET(pParentNode == NULL, -1);

			pParentNode->parentsIdx = pItem->index;
			pParentNode->samePrentsTitleNum = pItem->num;
			strcpy(pParentNode->parentTitle,pItem->key);

			_list_add(&pParentNode->list, &pSubItem[i].parentHead);
			parentId = pItem->parentId;
		}

		CHECK_EXPR_RET(cnt == 0, -1);
	}
	
	return 0;
}

static int PAPRTTreeCallback(void * usr, int idCurr, int * _idNext, int _nIdNext)
{
	CHECK_EXPR_RET(idCurr > gItemsId, -1);
	if ( idCurr == -2) {
		return 0;
	}

	for (int i = 0;i < _nIdNext;i++) {
		pItems[_idNext[i]].parentId = idCurr;
	}
	
	int voteId,id;

	voteNum = 0;
	
	id = _idNext[0];
	strcpy(pVote[voteNum].name,pItems[id].key);
	pVote[voteNum].cnt = 1;
	pItems[id].index = 0;
	voteNum++;

	BOOL isExist = FALSE;
	
	for (int i = 1; i < _nIdNext;i++) {
		isExist = FALSE;
		for (int j = 0;j < voteNum;j++) {
			id = _idNext[i];
			if (strcmp(pVote[j].name,pItems[id].key) == 0) {
				pItems[id].index = pVote[j].cnt;
				pVote[j].cnt++;
				isExist = TRUE;
			}
		}
		
		if (isExist == FALSE) {
			id = _idNext[i];
			strcpy(pVote[voteNum].name,pItems[id].key);
			pVote[voteNum].cnt = 1;
			pItems[id].index = 0;
			voteNum++;
		}
	}

	for (int i = 0; i < _nIdNext;i++) {
		for (int j = 0;j < voteNum;j++) {
			id = _idNext[i];
			if (strcmp(pVote[j].name,pItems[id].key) == 0) {
				pItems[id].num = pVote[j].cnt;
			}
		}
	}
	return 0;
}


int PARPT_getLineNum()
{
	return gLineNum;
}


int PARPT_buildTree()
{
	PARPTParseBuf(PARPTGetLineNum);

	pItems = (PAPRT_ITEM_t *)malloc(sizeof(PAPRT_ITEM_t) * PARPTTREE_MAX_ITEMS);
	CHECK_EXPR_RET(pItems == NULL, -1);

//	gIdflags = (BOOL *)malloc(sizeof(BOOL) * gLineNum);
//	CHECK_EXPR_RET(gIdflags == NULL, -1);
	
	printf("gLineNum:%d\n",gLineNum);
	
	pGetOneItemIdInfoCallback = PARPTOnGetOneItemIdInfo;
	PARPTParseBuf(PARPTParseOneLine);
	return 0;
}

int PARPT_run()
{
	printf("[PARPT_run]  0\n");
	MTREE_backwardExt(pTree, PAPRTTreeCallback);
	//MTREE_backwardExt(pTree, NULL);

	printf("[PARPT_run]  1\n");
	PAPARTGetAllRoutes();

	printf("[PARPT_run]  2\n");
	//PARPT_WriteFileStart();
	PAPARTGetAllSubItems();
	//PARPT_WriteFileEnd();
	return 0;
}


static int PARPTOnGetOneItemIdInfo(int id,int parentId)
{
	int ret = MTREE_addItem(pTree, parentId, id);
	CHECK_EXPR_RET(ret < 0, -1);

	
	return 0;
}


int PARPTReadFromFile(char *filename)
{
	printf("[PARPTReadFromFile:%s]\n",filename);
	int fd = open(filename,O_RDWR,0666);
	CHECK_EXPR_RET(fd < 0,-1);

	int fileLen = lseek(fd,0,SEEK_END);
	
	pFileBuffer = (char *)malloc(sizeof(char) * fileLen + 2);
	
	lseek(fd,0,SEEK_SET);
	
	int nread = 0;
	nread = read(fd,pFileBuffer,fileLen);
	
	CHECK_EXPR_RET(nread != fileLen,-1);
	gFileBufferLen = nread;
	pFileBuffer[gFileBufferLen++] = '\n';
	pFileBuffer[gFileBufferLen++] = '\0';

	close(fd);
	return 0;
}



static int PARPTGetLineNum(char *line,int len)
{
	gLineNum++;
	return 0;
}

static int PARPTPrintSplitInfo()
{
	for (int i = 0;i < gLineSplitLen;i++) {
		printf("[%s] ",gLineSplit[i]);
	}
	printf("\n");
	
	return 0;
}


static int PARPTSplitLine(char *line)
{
	int i = 0;
	char prev = ' ';
	gLineSplitLen = 0;
	int cnt = 0;
	
	while(line[i] != '\0') {
		if (line[i] != ' ') {
			gLineSplit[gLineSplitLen][cnt++] = line[i]; 
			
		} else if ((line[i] == ' ' || line[i] == ':') && prev != ' ') {
			gLineSplit[gLineSplitLen++][cnt] = '\0';
			cnt = 0;
		} 
		prev = line[i];
		i++;
	}
	
	if (line[i] == '\0' && prev != ' ') {
		gLineSplit[gLineSplitLen++][cnt] = '\0';
		cnt = 0;
	}

	//printf("gLineSplitLen:%d\n",gLineSplitLen);
	//PARPTPrintSplitInfo();
	return 0;
}


static int PARPTSplitLineExt(char *line,int len,char splitChar)
{
	char *prev = line;
	char *curr = line;
	BOOL bStop = FALSE;
	
	gLineSplitLen = 0;
	do {
		curr = strchr(prev,splitChar);
		if (curr == NULL) {
			curr = line + len;
			bStop = TRUE;
		}
		int itemLen = curr - prev;
		memcpy(gLineSplit[gLineSplitLen],prev,itemLen);
		gLineSplit[gLineSplitLen++][itemLen] = '\0';
		curr++;
		prev = curr;
	
	}while(bStop == FALSE);

	for (int i = 0;i < gLineSplitLen;i++) {
		PARPTFilterChar(gLineSplit[i], ' ');
	}

	return 0;
}


static int PARPTSwitchChar(char *buf,char old_c,char new_c)
{
	int i = 0;
	while(buf[i] != '\0') {
		if (buf[i] == old_c) {
			buf[i] = new_c;
		}
		i++;
	}
	return 0;
}

static int PARPTAppendOneItem(PAPRT_ITEM_t *pItem)
{
	memcpy(&pItems[gItemsId],pItem,sizeof(PAPRT_ITEM_t));
	gItemsId++;
	
	return 0;
}

static int PARPTParseKeyVal(char *line,int len,PAPRT_ITEM_t *pItem)
{
	CHECK_EXPR_RET(pItem == NULL, -1);
	CHECK_EXPR_RET(line == NULL, -1);
#if 0
	int ret = PARPTSplitLine(line);
	CHECK_EXPR_RET(ret < 0, -1);
	CHECK_EXPR_RET(strstr(gLineSplit[0],":") == NULL, -1);
	PARPTSwitchChar(gLineSplit[0],':','\0');
#else
	int ret = PARPTSplitLineExt(line,len,':');
	CHECK_EXPR_RET(ret < 0, -1);
#endif

	pItem->isTitle = FALSE;
	strcpy(pItem->key,gLineSplit[0]);
	strcpy(pItem->val,gLineSplit[1]);
//	printf("[PARPTParseKeyVal]line:%s\n",line);
//	printf("[PARPTParseKeyVal]key:%s val:%s\n",pItem->key,pItem->val);
//	getchar();
	return 0;
}


static int PARPTParseTitle(char *line,int len,PAPRT_ITEM_t *pItem)
{
	CHECK_EXPR_RET(pItem == NULL, -1);
	CHECK_EXPR_RET(line == NULL, -1);
	
	int ret = PARPTSplitLine(line);
	CHECK_EXPR_RET(ret < 0, -1);

	pItem->isTitle = TRUE;
	strcpy(pItem->key,gLineSplit[0]);

	return 0;
}

static int PARPTParseOneLine(char *line,int len)
{
	int ret = -1;
	PAPRT_ITEM_t item;
	int parentId;

	//printf("line:%s\n",line);
	if (strstr(line,"{") == NULL && \
			strstr(line,"}") == NULL && \
				STACK_empty(gStack) == TRUE) {
		parentId = -1;
		ret = PARPTParseKeyVal(line,len,&item);
		CHECK_EXPR_RET(ret < 0,-1);

		memcpy(&pItems[gItemsId],&item,sizeof(PAPRT_ITEM_t));
		if (pGetOneItemIdInfoCallback != NULL) {
			pGetOneItemIdInfoCallback(gItemsId,parentId);
		}
		gItemsId++;
		
	}  else if (strstr(line,"{") != NULL) {
		if (STACK_empty(gStack) == TRUE) {
			parentId = -1;
		}
		ret = PARPTParseTitle(line,len,&item);
		CHECK_EXPR_RET(ret < 0,-1);
		
		if (STACK_empty(gStack) == FALSE) {
			ret = STACK_peekLatest(gStack,&parentId);
			CHECK_EXPR_RET(ret < 0,-1);
		}

		STACK_push(gStack, &gItemsId);
	
		memcpy(&pItems[gItemsId],&item,sizeof(item));
		if (pGetOneItemIdInfoCallback != NULL) {
			pGetOneItemIdInfoCallback(gItemsId,parentId);
		}
		gItemsId++;
		
	} else if (strstr(line,"}") != NULL) {
		int id;
		STACK_pop(gStack, &id);
		
	} else {
		ret = STACK_peekLatest(gStack,&parentId);
		CHECK_EXPR_RET(ret < 0,-1);
		
		ret = PARPTParseKeyVal(line,len,&item);
		CHECK_EXPR_RET(ret < 0,-1);
		
		memcpy(&pItems[gItemsId],&item,sizeof(PAPRT_ITEM_t));
		if (pGetOneItemIdInfoCallback != NULL) {
			pGetOneItemIdInfoCallback(gItemsId,parentId);
		}
		gItemsId++;
	}
	
	return 0;
}


int PARPTParseBuf(GetOnelineCallback_f callback)
{
	int i = 0;
	char line[1024];
	char *ptr = pFileBuffer;
	int n = gFileBufferLen - 1;

	BOOL bPass = FALSE;
	int len = 0;
	while(n > 0){
		PARPTGetOneLine(ptr,line);
		len = strlen(line);
		ptr += len+1;
		n -= len+1;
		bPass = PARPTFilterLine(line,len);
		if (bPass == TRUE) {
			if (callback != NULL) {
				callback(line,len);
			}
			
		}
	}
	return 0;
}

static int PARPTFilterChar(char *buf,char c)
{
	char tmp[1024];

	int j = 0;
	for (int i = 0;i < strlen(buf);i++) {
		if (buf[i] != c) {
			tmp[j++] = buf[i];
		}
 	}

	int k = 0;
	for (k = 0;k < j;k++) {
		buf[k] = tmp[k];
	}
	buf[k] = '\0';
	
	return 0;
}

BOOL PARPTFilterLine(char *line,int len)
{
	char tmp[1024];

	if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
		return FALSE;
	}

	int i = 0;
	while((line[i] == '\t' || line[i] == ' ') && line[i] != '\0') {
		i++;
	}

	if (line[i] == '\0') {
		return FALSE;
	}

	int cnt = 0;
	for (int k = i;k < len;k++) {
		tmp[cnt++] = line[k];
	}

	for (int k = 0;k < cnt;k++) {
		line[k] = tmp[k];
	}
	line[cnt] = '\0'; 
	
	return TRUE;
}

int PARPTGetOneLine(char *start,char *line)
{
	int i = 0;
	while(start[i] != '\n') {
		line[i] = start[i];
		i++;
	}
	line[i] = '\0';
	
	return i;
}

int PARPT_WriteOneMessageStart(char *rootMessageName)
{
	char buf[256];
	char buf1[256];
	PARPT_getStuName(rootMessageName,buf1);
	snprintf(buf,sizeof(buf),"int parse%s(%s * %s)\n",rootMessageName,rootMessageName,buf1);
	PARPTWriteFile(buf,strlen(buf));
	char *str = "{\n";
	PARPTWriteFile(str,strlen(str));

	snprintf(buf,sizeof(buf),"\tmemset(%s,'\\0',sizeof(%s));\n",
					buf1,rootMessageName);
	PARPTWriteFile(buf,strlen(buf));
}


int PARPT_WriteOneMessageEnd()
{
	char *str = "\treturn 0;\n";
	PARPTWriteFile(str,strlen(str));
	
	str = "\n}\n";
	PARPTWriteFile(str,strlen(str));

	return 0;
}

int PARPT_WriteFileStart(char *messageName)
{
	char fileName[256];
	snprintf(fileName,sizeof(fileName),"parse_%s.c",messageName);
	pExecuteParseFile = fopen(fileName,"w+");
	
	CHECK_EXPR_RET(pExecuteParseFile == NULL, -1);

	char *str = "#include \"parameter.h\"\n";
	PARPTWriteFile(str,strlen(str));
	
	return 0;
}

int PARPT_WriteFileEnd()
{
	CHECK_EXPR_RET(pExecuteParseFile == NULL, -1);
	fclose(pExecuteParseFile);
	
	return 0;
}

int PARPTWriteFile(char *buf,int len)
{
	int nwrite = fwrite(buf,len,1,pExecuteParseFile);
	CHECK_EXPR_RET(nwrite != 1, -1);

	return 0;
}


