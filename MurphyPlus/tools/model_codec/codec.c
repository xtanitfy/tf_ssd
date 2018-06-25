
//#define USE_CAFFE_TEST 

//#include <iostream>
#include "codec.h"
#include "public.h"
#include "stack.h"
#include "muti_tree.h"
#include "parse_prototxt.h"

#if defined(USE_CAFFE_TEST)
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"
#include "caffe/util/bbox_util.hpp"
#include "caffe/net.hpp"
#endif

#include "gen_headfile.h"

#if defined(USE_CAFFE_TEST)
using namespace caffe;
#endif


//using namespace std;

#define ONE_MESSAGE_MAX_ITEM 256
#define ENUM_MAX_ITEMS 1024
#define LINE_MAX_SIZE ENUM_MAX_ITEMS
#define ONE_LINE_MAX_SPLITS  ENUM_MAX_ITEMS
#define ROOT_MESSAGE_TYPE_NAME "NetParameter"
#define ROUTE_MAX_DEPTH 32 
//#define MODEL_FILE "model/goolenet_deploy.prototxt"
#define MODEL_PROTOTXT_FILE "model/ssd300_deploy.prototxt"
#define CAFFE_PROTO_FILE "../../src/proto/caffe.proto"
#define WEIGHTS_FILE "VGG_VOC0712_SSD_300x300_iter_60000.caffemodel"
#define STACK_SIZE 4096
#define ELEMENT(i,x) (netParam.layer(i).x()) 

static int parseBuf(char  *buf);
static int getLineBuf(char *start,char *line);
static int readFromFile(char *protofile);
static int printSplitInfo();
static int filterChar(char *buf,char c);
static int parseEnum(BOOL isIncludeINmessage);
static int printMessageInfo();
static int printMessageItemInfo();
static int parseEnumOneLine(char *enumStr,MESSAGE_t *pItem);
static int filterAnnotation(char *buf);
static int parseLine(char *line,int len);
static int parseEnumInMessage(char *enumStr,MESSAGE_ITEM_t *pItem);
static int getItemIdByName(char *messageName,char *name);
static int getAllItemsByMessageName(char *messageName,int (*handle_id)(int id,int parentId));
static void *isMessageTypeisEnum(int id,ENUM_TYPE_t *enumType);
static int parseBuf(char *buf);
static int addIds();
static int addIdMap();
static int writeHeadFile();
static int switchChar(char *buf,char old_c,char new_c);
static int addItemsToMutiTree(int id,int parentId);
static int writeOneMessageToHeadfile(void * usr, int idCurr, int * _idNext, int _nIdNext);
static int generateRoute(int id,int parentId);
static int initMessageWritten();
static int generateHeadFile();
static int generateHeadFileByRootMessage(MTREE_s **pTree,char *rootMessage);
static int generateFileParsePrototxt(char *protoTxtFile,char *messageName);
static int OnGetSubItem(SUB_ITEM_t *pItem);
static int GetChildIds(void * usr, int idCurr, int * _idNext, int _nIdNext);
static int generateWeightsBinFile();
static int AddAllItemsHaveDefaultValue();
MESSAGE_t *getMessageNodeByName(char *messageName);
static int getRootMessage();

TypeMapStr_t gTypeMapStr[] = {
	{TYPE_int32,"int32","INT32"},
	{TYPE_int64,"int64","INT64"},
	{TYPE_float,"float","float"},
	{TYPE_double,"double","double"},
	{TYPE_message,"message","message"},
	{TYPE_bool,"bool","BOOL"},
	{TYPE_string,"string","char *"},
	{TYPE_uint32,"uint32","UINT32"},
	{TYPE_bytes,"bytes","char *"}
};

AttrMapStr_t gAttrMapStr[] = {
	{ATTR_optional,"optional"},
	{ATTR_repeated,"repeated"},
	{ATTR_enum,"enum"}
};
	
static struct list_head gMessageList;
static int gItemCnt = 0; 
	
static struct list_head gLayerList;

static MESSAGE_t *pCurrMessage = NULL;
static MESSAGE_t *pRootMessage = NULL;

static char *pBuffer = NULL;
static int bufferLen = 0;

static char gLineSplit[ONE_LINE_MAX_SPLITS][STR_MAX_SIZE];
static int  gLineSplitLen = 0;

static char gLineArr[ONE_MESSAGE_MAX_ITEM][LINE_MAX_SIZE];
static int  gLineArrLen = 0;

static char gEnumItemBuf[ENUM_MAX_ITEMS][LINE_MAX_SIZE];
static int  gEnumItemN = 0;

static void *pStack;
static char *messageStr = "message";
static char *enumStr = "enum";

static ID_MAP_t *gIdMap = NULL;
static BOOL *gIdFlags = NULL;

static MTREE_s *pTree = NULL;
static MTREE_s *pTree1 = NULL;
static MTREE_s *pTreeTmp = NULL;

static int gAllMessageNum = 0;

static struct list_head gAllIdRoutes; 

static int gMessageWrittenN;
static char (*gMessageWritten)[STR_MAX_SIZE];

static int *gSecondRootIds;
static int gSecondRootIdsNum = 0;

static char (*gAllRootMessageName)[STR_MAX_SIZE];
static int gAllRootMessageNameSize = 0;
#define PARPT_TREE_MAX_DEPTHS 8
typedef struct
{
	int arr[PARPT_TREE_MAX_DEPTHS];
	int len;
}INT_ARRAY_t;

static MTREE_s **pTrees = NULL;
static MTREE_s *pParptTree = NULL;

static PAPRT_ITEM_t *pParptItems = NULL;
static int *pIdParent;
static int (*pIdsChild)[MESSAGE_MAX_ITEM];
static int *idChildN = NULL;
static char **parptIdmessageType;
static int *parptItemsN = 0;

static int gParptTreeSecondRoots[4096];
static int gParptTreeSecondRootsN = 0;


static INT_ARRAY_t *pParptIdsArr;
//static int (*pParptIdsArr)[PARPT_TREE_MAX_DEPTHS];
static int pParptIdsArrN = 0;
static int pParptLineNum = 0;

int main(int argc,char **argv)
{
	int ret = -1;
	printf("sizeof(MESSAGE_t):%ld\n",sizeof(MESSAGE_t));

	if (argc != 4) {
		printf("argv[1]:caffe protofile\n");
		printf("argv[2]:model prototxt file\n");
		printf("argv[3]:proto message name\n");
		return -1;
	}

	//getParseStat();

	pStack = STACK_init(STACK_SIZE,sizeof(char *));
    CHECK_EXPR_RET(pStack == NULL, -1);
	
	ret = readFromFile(argv[1]);
	CHECK_EXPR_RET(ret < 0, -1);
	
	INIT_LIST_HEAD(&gMessageList);
	INIT_LIST_HEAD(&gLayerList);
	
	ret = parseBuf(pBuffer);
	CHECK_EXPR_RET(ret < 0, -1);
	
	ret = addIds();
	CHECK_EXPR_RET(ret < 0, -1);

	initMessageWritten();
	
	ret = addIdMap();
	CHECK_EXPR_RET(ret < 0, -1);
	
	ret = printMessageInfo();
	CHECK_EXPR_RET(ret < 0, -1);
	
	
#if defined(USE_CAFFE_TEST)
	Net<float> caffe_net(MODEL_FILE, caffe::TEST,0,NULL);
    caffe_net.CopyTrainedLayersFrom(WEIGHTS_FILE);
    
    float iter_loss;
    caffe_net.Forward(&iter_loss);
#endif

	ret = generateHeadFile();
	CHECK_EXPR_RET(ret < 0, -1);

	PARPT_WriteFileStart(argv[3]);
	ret = generateFileParsePrototxt(argv[2],argv[3]);
	CHECK_EXPR_RET(ret < 0, -1);
	printf("parseNetParameter %s finish!\n",argv[2]);
	PARPT_WriteFileEnd();
	STACK_destroy(pStack);
	printf("codec finish!\n");

	//getFflushStat();
	
	return 0;
}

int getRootMessage()
{
	MESSAGE_t *p = NULL;
	BOOL isExist = FALSE;
	MESSAGE_t *p1 = NULL;
	int allMessageNum = 0;
	
	list_for_each_entry(p,&gMessageList,list,MESSAGE_t) {
		if (p->isEnum == FALSE) {
			allMessageNum++;
		}
	}
	gAllRootMessageName = (char (*)[STR_MAX_SIZE])malloc(sizeof(char)*STR_MAX_SIZE*allMessageNum);
	CHECK_EXPR_RET(gAllRootMessageName == NULL,-1);
	
	list_for_each_entry(p,&gMessageList,list,MESSAGE_t) {
		isExist = FALSE;
		if (p->isEnum == FALSE) {
			list_for_each_entry(p1,&gMessageList,list,MESSAGE_t) {
				if (strcmp(p1->name,p->name) == 0) {
					continue;
				}
				
				isExist = FALSE;
				for (int i= 0;i < p1->itemN;i++) {
					MESSAGE_ITEM_t *pItem = &p1->pItems[i];
					if (pItem->type == TYPE_message) {
						if (strcmp(pItem->messageTypeName,p->name) == 0) {
							isExist = TRUE;
							break;
						}
					}
				}
				if (isExist == TRUE) {
					break;
				}
			}
			if (isExist == FALSE) {
				strcpy(gAllRootMessageName[gAllRootMessageNameSize++],p->name);
			}
		}
	}

	return 0;
}

int generateHeadFile()
{
	int ret = GenHead_init();
	CHECK_EXPR_RET(ret < 0, -1);
	GenHead_writeHeadfileStart();
	
	getRootMessage();
	for (int i = 0;i < gAllRootMessageNameSize;i++) {
		printf("rootmessage:[%d][%s]\n",i,gAllRootMessageName[i]);
	}
	pTrees = (MTREE_s **)malloc(sizeof(MTREE_s *) * gAllRootMessageNameSize);
	CHECK_EXPR_RET(pTrees == NULL,-1);
	
	for (int i = 0;i < gAllRootMessageNameSize;i++) {
		printf("generate %s\n",gAllRootMessageName[i]);
		if (strcmp(gAllRootMessageName[i],"SolverParameter") == 0) {
			strcpy(gAllRootMessageName[i],"NetParameter");
		}
		ret = generateHeadFileByRootMessage(&pTrees[i],gAllRootMessageName[i]);
		CHECK_EXPR_RET(ret < 0, -1);
	}

	
#if 0
	ret = generateHeadFileByRootMessage(&pTree,ROOT_MESSAGE_TYPE_NAME);
	CHECK_EXPR_RET(ret < 0, -1);
	ret = generateHeadFileByRootMessage(&pTree1,"LabelMap");
	CHECK_EXPR_RET(ret < 0, -1);
#endif
	GenHead_writeHeadfileEnd();
	printf("finish generateHeadFile!\n");



	return 0;
}

int GetChildIds(void * usr, int idCurr, int * _idNext, int _nIdNext)
{
	if (idCurr == -1) {
		gSecondRootIds = (int *)malloc(sizeof(int) * _nIdNext);
		CHECK_EXPR_RET(gSecondRootIds == NULL, -1);
	
		gSecondRootIdsNum = 0;
		for (int i = 0;i < _nIdNext;i++) {
			gSecondRootIds[gSecondRootIdsNum++] = _idNext[i];
		}
		return 0;
	}

	for (int i = 0;i < _nIdNext;i++) {
		gIdMap[idCurr].childId[i] = _idNext[i];
	}
	gIdMap[idCurr].childIdNum = _nIdNext;

	return 0;
}


int getRootMessageTreeId(char *messageName)
{
	for (int i = 0;i < gAllRootMessageNameSize;i++) {
		if (strcmp(gAllRootMessageName[i],messageName) == 0) {
			return i;
		}
	}

	return -1;
}

int generateFileParsePrototxt(char *protoTxtFile,char *pRootMessageName)
{
#if 0//defined(USE_CAFFE_TEST)
	NetParameter netParam;
	ReadNetParamsFromTextFileOrDie(MODEL_FILE,&netParam);
	//cout<< "netParam.name():" << netParam.name() <<  endl;
	//cout<< "layer_size:" << netParam.layer_size() <<  endl;

	ReadNetParamsFromBinaryFileOrDie(WEIGHTS_FILE, &netParam);

	LayerParameter layerParameter;
	for (int i = 0;i < netParam.layer_size();i++) {
		//cout<< "	[" << netParam.layer(i).type() << "]" << endl;
		//cout<< "-============netParam.blobs_size():" << netParam.layer(i).blobs_size() << endl;
		char *str = "PriorBox";
		layerParameter = netParam.layer(i);
		if (strcmp(netParam.layer(i).type().c_str(),str) == 0) {
			//cout<< str << endl;
			//cout<< layerParameter.prior_box_param().aspect_ratio_size() << endl;
			getchar();
		}
	}
#endif

	PARPT_WriteOneMessageStart(pRootMessageName);

	//use muti_tree forward
	pRootMessage = getMessageNodeByName(pRootMessageName);
	CHECK_EXPR_RET(pRootMessage == NULL, -1);
	int ret = -1;
#if 1
	int id = getRootMessageTreeId(pRootMessageName);
	CHECK_EXPR_RET(id < 0,-1);
	pTree = pTrees[id];
#else
	//pTree = (MTREE_s *)malloc(sizeof(MTREE_s));
	//CHECK_EXPR_RET(pTree == NULL,-1);
	ret = generateHeadFileByRootMessage(&pTree,pRootMessageName);
	CHECK_EXPR_RET(ret < 0, -1);

#endif
	ret = MTREE_backwardExt(pTree, GetChildIds);
	CHECK_EXPR_RET(ret < 0, -1);

	ret = PARPT_init(protoTxtFile,OnGetSubItem,pRootMessageName);
	CHECK_EXPR_RET(ret < 0, -1);

	PARPT_buildTree();

#if defined(PARSE_PROTO_ADD_ALL_DEFAULT_VALUES)//add all default values
	pParptLineNum = PARPT_getLineNum();
	pParptIdsArr = (INT_ARRAY_t *)malloc(sizeof(INT_ARRAY_t) * PARPTTREE_MAX_ITEMS);
	CHECK_EXPR_RET(pParptIdsArr == NULL,-1);
	pIdParent = (int *)malloc(sizeof(int) * PARPTTREE_MAX_ITEMS);
	CHECK_EXPR_RET(pIdParent == NULL,-1);
	pIdsChild = (int (*)[MESSAGE_MAX_ITEM])malloc(sizeof(int) * PARPTTREE_MAX_ITEMS * MESSAGE_MAX_ITEM);
	CHECK_EXPR_RET(pIdsChild == NULL,-1); 
	idChildN = (int *)malloc(sizeof(int) * PARPTTREE_MAX_ITEMS);
	memset(idChildN,'\0',PARPTTREE_MAX_ITEMS);
	
	int *pParptIdsArrN = 0;
	PARPT_getItemsInfo(&pParptTree,&pParptItems,&parptItemsN);

	parptIdmessageType = (char **)malloc(sizeof(char *) * PARPTTREE_MAX_ITEMS);
	CHECK_EXPR_RET(parptIdmessageType == NULL,-1);
	for (int i = 0;i < pParptLineNum;i++) {
		parptIdmessageType[i] = NULL;
	}
	
	AddAllItemsHaveDefaultValue();
#endif
	PARPT_run();	

	PARPT_WriteOneMessageEnd();
	return 0;
}

static int inverseArr(int *arr,int len)
{
	int tmp = 0;
	for (int i = 0;i < len/2;i++) {
		tmp = arr[i];
		arr[i] = arr[len-i-1];
		arr[len-i-1] = tmp;
	}
	
}

static int AddItemsDefaultsRoutes(int id)
{
	//get allroutes 
	INT_ARRAY_t arrTmp;
	int cnt = 0;

	arrTmp.len = 0;
	while(id != -1) {
		arrTmp.arr[arrTmp.len++] = id;
		id = pIdParent[id];
	}	
	inverseArr(arrTmp.arr,arrTmp.len);
	for (int i = 0;i < arrTmp.len;i++) {
		pParptIdsArr[pParptIdsArrN].arr[i] = arrTmp.arr[i];
	}
	pParptIdsArr[pParptIdsArrN].len = arrTmp.len;
	pParptIdsArrN++;

	return 0;
}


static int AddParentIds(void * usr,int idCurr,int *_idPrev,int _nIdPrev)
{
	if (_nIdPrev > 0) {
		pIdParent[idCurr] = *_idPrev;
		printf("%d:%d\n",idCurr,*_idPrev);
	}
	return 0;	
}

static int AddChildIds(void * usr,int idCurr,int *_idPrev,int _nIdPrev)
{
	if (idCurr == -1) {
		for (int i = 0;i < _nIdPrev;i++) {
			//if (pParptItems[_idPrev[i]].isTitle == TRUE) {
				gParptTreeSecondRoots[gParptTreeSecondRootsN++] = _idPrev[i];
				
			//}
		}
		if (_nIdPrev > 0) {
			printf("curr:%d children:",idCurr);
			for (int i = 0;i < _nIdPrev;i++) {
				printf("%d ",_idPrev[i]);
			}
			printf("\n");
		} else {
			printf("curr:%d no children\n",idCurr);
		}

		return 0;
	}

	for (int i = 0;i < _nIdPrev;i++) {
		//if (pParptItems[_idPrev[i]].isTitle == TRUE) {
			pIdsChild[idCurr][i] = _idPrev[i];
		//}
	}
	idChildN[idCurr] = _nIdPrev;
	if (idChildN[idCurr] > 0) {
		printf("curr:%d children:",idCurr);
		for (int i = 0;i < idChildN[idCurr];i++) {
			printf("%d ",pIdsChild[idCurr][i]);
		}
		printf("\n");
 
	} else {
		printf("curr:%d no children\n",idCurr);
	}
	
	return 0;	
}


static int PARPTTreeCallback(void * usr,int idCurr,int *_idPrev,int _nIdPrev)
{
	if (pParptItems[idCurr].isTitle == FALSE) {
		return 0;
	}
	if (_nIdPrev >= 1) {
		AddItemsDefaultsRoutes(idCurr);
	} 
	return 0;
}

static int ParptGetMessageType(void * usr,int idCurr,int *_idPrev,int _nIdPrev)
{
	if (idCurr == -1) {
		return 0;
	}

	if (pIdParent[idCurr] == -1) {
		return 0;
	}

	if (pParptItems[idCurr].isTitle == FALSE) {
		return 0;
	}
	
	if (parptIdmessageType[idCurr] == NULL) {
		char *parentMessageType = parptIdmessageType[pIdParent[idCurr]];
		CHECK_EXPR_RET(parentMessageType == NULL,-1);

		MESSAGE_t *pMessage = getMessageNodeByName(parentMessageType);
		CHECK_EXPR_RET(pMessage == NULL,-1);

		BOOL isExist = FALSE;

		PAPRT_ITEM_t *pParptItem = &pParptItems[idCurr];
		isExist = FALSE;
		int j = 0;
		for (;j < pMessage->itemN;j++) {
			if (strcmp(pParptItem->key,pMessage->pItems[j].keystr) == 0) {
				isExist = TRUE;
				break;
			}
		}
		CHECK_EXPR_RET(isExist == FALSE,-1);
		MESSAGE_ITEM_t *pSubItem = &pMessage->pItems[j];
		if (pSubItem->type == TYPE_message) {
			parptIdmessageType[idCurr] = pSubItem->messageTypeName;
		//	printf("id:%d messageType:%s\n",idCurr,parptIdmessageType[idCurr]);
		} else {
			parptIdmessageType[idCurr] = NULL;
		//	printf("id:%d no messageType\n",idCurr);
		}
	}

	return 0;
}


static int PARPTGetAllItemsMessageType()
{
	PAPRT_ITEM_t *pParptItem = NULL;
	BOOL isExist = FALSE;
	MESSAGE_ITEM_t *pSubItem = NULL;
	CHECK_EXPR_RET(pRootMessage == NULL,-1);
	printf("gParptTreeSecondRootsN:%d\n",gParptTreeSecondRootsN);
	
	for (int i = 0;i < gParptTreeSecondRootsN;i++) {
		int id = gParptTreeSecondRoots[i];
		pParptItem = &pParptItems[id];
		isExist = FALSE;
		int j = 0;
		for (;j < pRootMessage->itemN;j++) {
			if (strcmp(pParptItem->key,pRootMessage->pItems[j].keystr) == 0) {
				isExist = TRUE;
				break;
			}
		}
		CHECK_EXPR_RET(isExist == FALSE,-1);
		pSubItem = &pRootMessage->pItems[j];
		if (pSubItem->type == TYPE_message) {
			parptIdmessageType[id] = pSubItem->messageTypeName;
		//	printf("id:%d messageType:%s\n",id,parptIdmessageType[id]);
		} else {
			parptIdmessageType[id] = NULL;
		//	printf("id:%d no messageType\n",id);
		}
	}

	MTREE_forward(pParptTree,ParptGetMessageType);

	return 0;
}

static int ParptGetOneId()
{	
	int id;
	//printf("*parptItemsN:%d\n",*parptItemsN);
	CHECK_EXPR_RET((*parptItemsN) >= PARPTTREE_MAX_ITEMS,-1);
	id = *parptItemsN;
	(*parptItemsN)++;
	return id;
}


static int ParptAddItemDefault(MESSAGE_ITEM_t *pMessageItem,int parentId,int currId)
{
	PAPRT_ITEM_t parptItem;
	
	if (pMessageItem->type == TYPE_message) {
		parptItem.isTitle = TRUE;
	} else {
		parptItem.isTitle = FALSE;
	}
	parptItem.index = 0;
	parptItem.num = 1;

	//printf("add [parentId:%d currId:%d] [keystr:%s]\n",parentId,
	//				currId,pMessageItem->keystr);

	strcpy(parptItem.key,pMessageItem->keystr);
	
	if (pMessageItem->isHaveDefaultVal == TRUE) {
		strcpy(parptItem.val,pMessageItem->defaultVal);
	}
	parptItem.parentId = parentId;

	//printf("*parptItemsN:%d pParptLineNum:%d\n",*parptItemsN,pParptLineNum);
	
	pParptItems[currId] = parptItem;
	
	return MTREE_addItem(pParptTree,parentId,currId);

}



static  BOOL judgeIsMessageHaveDefaultValue(MESSAGE_t *pMessage)
{
#define STACK_MAX_SIZE 512

#if 0
	isMessageHaveDefaultValue = FALSE;
	printf("judgeIsMessageHaveDefaultValue :pMessage->name:%s\n",pMessage->name);
	getAllItemsByMessageName(pMessage->name,judgeDefaultCallback);

	return isMessageHaveDefaultValue;
#endif
	
	CHECK_EXPR_RET(pMessage == NULL, -1);
	int id  = 0;
	void *pStackItemsId = STACK_init(STACK_MAX_SIZE,sizeof(int));
	CHECK_EXPR_RET(pStackItemsId == NULL, -1);

	for (int i = 0;i < pMessage->itemN;i++) {
		id = pMessage->pItems[i].id;
		STACK_push(pStackItemsId, &id);
	}
	
	MESSAGE_ITEM_t *pItem = NULL;
	BOOL isHave = FALSE;
	while(STACK_empty(pStackItemsId) == FALSE) {
		STACK_pop(pStackItemsId,&id);
		
		pItem = gIdMap[id].pItem;
		if (pItem->type == TYPE_message) {
		//	printf("pItem->messagetype:%s\n",pItem->messageTypeName);
		}
		
		
		if (pItem->isHaveDefaultVal == TRUE) {
			isHave = TRUE;
			break;
		}
		
		MESSAGE_t *pSubmessage = getMessageNodeByName(pItem->messageTypeName);
		if (pSubmessage == NULL) {
			continue;
		}
		for (int i = 0;i < pSubmessage->itemN;i++) {
			id = pSubmessage->pItems[i].id;
			STACK_push(pStackItemsId, &id);
		}
	}

	STACK_destroy(pStackItemsId);

	return isHave;
}

static BOOL JudgeIsParptTreeItemInMessage(MESSAGE_t *pMessage,int parptTreeId)
{
	BOOL isExist = FALSE;
	PAPRT_ITEM_t *parptItem = &pParptItems[parptTreeId];;
	for (int i = 0;i < pMessage->itemN;i++) {
		if (strcmp(pMessage->pItems[i].keystr,parptItem->key) == 0) {
			isExist = TRUE;
			break;
		}
	}
	return isExist;
}

static BOOL JudgeItemsInParptTree(MESSAGE_ITEM_t *pSubItem,int parptTreeParentId)
{
	BOOL isExist = FALSE;
	PAPRT_ITEM_t *parptItem = NULL;

	for (int i = 0;i < idChildN[parptTreeParentId];i++) {
		parptItem = &pParptItems[pIdsChild[parptTreeParentId][i]];
		/*
		if (strstr(pSubItem->keystr,"param") != NULL &&\
				strstr(parptItem->key,"param") != NULL) {
			printf("pSubItem->keystr:<%s>\n",pSubItem->keystr);
			printf("parptItem->key:<%s>\n",parptItem->key);
			getchar();
		}
		*/
		if (strcmp(pSubItem->keystr,parptItem->key) == 0) {
			isExist = TRUE;
			break;
		}
	}
	
	return isExist;
}

static BOOL ParptFilterLayerParameter(MESSAGE_t *pMessage,MESSAGE_ITEM_t *pItem,int parentId)
{
	BOOL isNeedFilter = FALSE;
	char buf[128];

	if (strcmp(pMessage->name,"LayerParameter") == 0) {
		if (pItem->type == TYPE_message) {
			BOOL isExist = FALSE;
			PAPRT_ITEM_t *parptItem = NULL;
			
			for (int i = 0;i < idChildN[parentId];i++) {
				parptItem = &pParptItems[pIdsChild[parentId][i]];
				if (strcmp(parptItem->key,"type") == 0){
					isExist = TRUE; 
					break;
				}
			}
			if (isExist == TRUE && strstr(pItem->messageTypeName,"Parameter") != NULL) {
				snprintf(buf,sizeof(buf),"%sParameter",parptItem->val);
				//printf("buf:%s\n",buf);
				//printf("pItem->messageTypeName:%s\n",pItem->messageTypeName);
				filterChar(buf,'\"');
				if (strcmp(buf,pItem->messageTypeName) != 0){
					//printf("Not equals!\n");
					return TRUE;
				} else {
					//printf("equals!\n");
				}
			}
		}
	}
	return FALSE;
}

static int AddItemsDefaultToParptTree()
{
#define DEFAULT_TREE_STACK_SIZE 512
typedef struct
{
	int type;//0:is select pMessageItem 1:select 
	MESSAGE_ITEM_t *pMessageItem;
	int currId;
	int parentId;
}STACK_ITEM_t;

	void *pDefaultStack = STACK_init(DEFAULT_TREE_STACK_SIZE,sizeof(STACK_ITEM_t));

	//get all parttree item's message type by pIdParent  
	PARPTGetAllItemsMessageType();
	int ret = -1;
	PAPRT_ITEM_t *parptItem = NULL;
	MESSAGE_t *pMessage = NULL;
	int pItemId;
	STACK_ITEM_t stackItem;
	BOOL isExist = FALSE;
	MESSAGE_ITEM_t *pItem = NULL;

#if 1
	for (int i = 0;i < pRootMessage->itemN;i++) {
		//first: judge is exist in parpttree
		pItem = &pRootMessage->pItems[i];
		
		isExist = FALSE;
		for (int j = 0;j < gParptTreeSecondRootsN;j++) {
			parptItem = &pParptItems[gParptTreeSecondRoots[j]];
			if (strcmp(pItem->keystr,parptItem->key) == 0) {
				isExist = TRUE;
				break;
			}
		}
		if (pItem->type == TYPE_message) {
			//printf("%s existInParptTree:%d\n",pItem->messageTypeName,isExist);
		}
		
		//second:if no,then add the default value to parpttree
		if (isExist == FALSE && pItem->isHaveDefaultVal == TRUE) {
			ret = ParptAddItemDefault(pItem,-1,ParptGetOneId());
			CHECK_EXPR_RET(ret < 0,-1);
			continue;
		}

		//add the message that include default values to parpttree and push the subitems to stack
		if (isExist == FALSE) {
			if (pItem->type == TYPE_message) {
				//if is messagetype,then push subitems to stack
				//judge is have default value
				pMessage = getMessageNodeByName(pItem->messageTypeName);
				CHECK_EXPR_RET(pMessage == NULL,-1);
				BOOL isHave = judgeIsMessageHaveDefaultValue(pMessage);
				printf("%s isHaveDefaultValue:%d\n",pMessage->name,isHave);
				if (isHave == FALSE) {
					continue;
				}
				//if yes;then add it and push subotems to stack;
				//ParptAddItemDefault(pItem,-1,&pItemId);
				STACK_ITEM_t stackItem;
				stackItem.type = 0;
				stackItem.pMessageItem = pItem;
				stackItem.currId = ParptGetOneId();
				stackItem.parentId = -1;
				STACK_push(pDefaultStack,&stackItem);
			}
			
		} 
	}
#endif
	
#if 1
	int parentId = 0;
	for (int j = 0;j < gParptTreeSecondRootsN;j++) {
		STACK_ITEM_t stackItem;
		stackItem.type = 1;
		stackItem.currId = gParptTreeSecondRoots[j];
		stackItem.parentId = -1;
		STACK_push(pDefaultStack,&stackItem);
		//printf("-----stackItem.parentId:%d stackItem.currId:%d [%s]\n",
			//	stackItem.parentId,stackItem.currId,pParptItems[stackItem.currId].key);
		
	}
//	getchar();
	int currId = 0;
	//return 0;
#endif
	while(STACK_empty(pDefaultStack) == FALSE) {
		STACK_pop(pDefaultStack,&stackItem);
		if (stackItem.type == 0) {
			#if 1
			pItem = stackItem.pMessageItem;
		
			if (pItem->isHaveDefaultVal == TRUE) {
				//isExist = JudgeItemsInParptTree(pItem,stackItem.parentId);
				//if (isExist == FALSE) {
					ret = ParptAddItemDefault(pItem,stackItem.parentId,stackItem.currId);
					CHECK_EXPR_RET(ret < 0,-1);
				//}
				
				continue;
			}
			if (pItem->type != TYPE_message) {
				continue;
			}
			
			if (strstr(pItem->messageTypeName,"_") != NULL) {
				continue;
			}
			//printf("----pItem->messageTypeName:%s\n",pItem->messageTypeName);
			
			pMessage = getMessageNodeByName(pItem->messageTypeName);
			CHECK_EXPR_RET(pMessage == NULL,-1);
			BOOL isHave = judgeIsMessageHaveDefaultValue(pMessage);
			if (isHave == FALSE) {
				continue;
			}
			ret = ParptAddItemDefault(pItem,stackItem.parentId,stackItem.currId);
			CHECK_EXPR_RET(ret < 0,-1);

			int parentId = stackItem.currId;
			pMessage = getMessageNodeByName(pItem->messageTypeName);
			//parptIdmessageType[pItemId] = pItem->messageTypeName;
			for (int j = 0;j < pMessage->itemN;j++) {
				STACK_ITEM_t stackItem;
				stackItem.type = 0;
				stackItem.pMessageItem = &pMessage->pItems[j];
				stackItem.parentId = parentId;
				stackItem.currId = ParptGetOneId();				
				STACK_push(pDefaultStack,&stackItem);
			}
			#endif
		} else {
			#if 1
			MESSAGE_t *pSubMessage = NULL;
			currId = stackItem.currId;

			if (parptIdmessageType[currId] == NULL) {
				continue;
			}
			
			pSubMessage = getMessageNodeByName(parptIdmessageType[currId]);
			CHECK_EXPR_RET(pSubMessage == NULL,-1);
			
			for (int i = 0;i < pSubMessage->itemN;i++) {
				pItem = &pSubMessage->pItems[i];
				//filter other layerparameter
				BOOL isNeedFilter = ParptFilterLayerParameter(pSubMessage,pItem,currId);
				if (isNeedFilter == TRUE) {
					continue;
				}
				
				isExist = JudgeItemsInParptTree(pItem,currId);
				if (isExist == FALSE) {
					if (strcmp(pItem->keystr,"param") == 0) {
						printf("type:%d\n",pItem->type);
						for (int i = 0;i < idChildN[currId];i++) {
							parptItem = &pParptItems[pIdsChild[currId][i]];
							printf("parptItem->key:<%s>\n",parptItem->key);
						}
						//getchar();
					}
					
					if (pItem->isHaveDefaultVal == TRUE) {
						ret = ParptAddItemDefault(pItem,stackItem.currId,ParptGetOneId());
						CHECK_EXPR_RET(ret < 0,-1);
					} else {
						if (pItem->type != TYPE_message) {
							continue;
						}
						if (strstr(pItem->messageTypeName,"_") != NULL) {
							continue;
						}
						pMessage = getMessageNodeByName(pItem->messageTypeName);
						CHECK_EXPR_RET(pMessage == NULL,-1);
						BOOL isHave = judgeIsMessageHaveDefaultValue(pMessage);
						if (isHave == FALSE) {
							//printf("%s:have no default value!\n",pMessage->name);
							continue;
						}
	
						int id = ParptGetOneId();
						ret = ParptAddItemDefault(pItem,stackItem.currId,id);
						CHECK_EXPR_RET(ret < 0,-1);
		
						parptIdmessageType[id] = pMessage->name;
						//int parentId = stackItem.currId;
						for (int j = 0;j < pMessage->itemN;j++) {
							STACK_ITEM_t stackItem;
							stackItem.type = 0;
							stackItem.pMessageItem = &pMessage->pItems[j];
							stackItem.parentId= id;
							stackItem.currId = ParptGetOneId();
							parptIdmessageType[currId] = pMessage->pItems[j].messageTypeName;
							//parptIdmessageType[pItemId] = pItem->messageTypeName;
							STACK_push(pDefaultStack,&stackItem);
						}
						//printf("5------------\n");
					}
				}
			}

			for (int k = 0;k < idChildN[currId];k++) {
				STACK_ITEM_t stackItem;
				int id = pIdsChild[currId][k];
				stackItem.type = 1;
				stackItem.currId= id;
				stackItem.parentId = currId;
				STACK_push(pDefaultStack,&stackItem);
			}
			#endif
		}
	}
	STACK_destroy(pDefaultStack);

	printf("AddItemsDefaultToParptTree finish!\n");

	return 0;
}

static int AddAllItemsHaveDefaultValue()
{
	MTREE_forward(pParptTree,AddParentIds);
	printf("pParptItems[0].key:%s\n",pParptItems[0].key);
	
	MTREE_backwardExt(pParptTree,AddChildIds);

	AddItemsDefaultToParptTree();
	
	return 0;
}


static int OnGetSubItem(SUB_ITEM_t *pItem)
{
	PARENT_NODE_t *pParentNode = NULL;
#if 0
	printf("---[%s:%s %d %d]",pItem->key,pItem->val,pItem->idx,pItem->sameItemsNum);
	list_for_each_entry(pParentNode, &pItem->parentHead, list, PARENT_NODE_t) {
		printf("{%s %d %d} ",pParentNode->parentTitle,
			pParentNode->samePrentsTitleNum,pParentNode->parentsIdx);
	}
	printf("\n");
	
#endif

	CHECK_EXPR_RET(pItem == NULL, -1);

	int depth = 0;
	BOOL isExist = FALSE;
	if (list_empty(&pItem->parentHead)) {
		pItem->pRootMessage =  pRootMessage;
		isExist = FALSE;
		for (int i = 0;i < gSecondRootIdsNum;i++) {
			int id = gSecondRootIds[i];
			if (strcmp(gIdMap[id].pItem->keystr,pItem->key) == 0) {
				isExist = TRUE;			
				pItem->pItem = gIdMap[id].pItem;
				break;
			}
		}
		CHECK_EXPR_RET(isExist  == FALSE, -1);
		PARPT_writeOneItem(pItem);
		return 0;
	}
	
	//gRootIds
	pItem->pRootMessage =  pRootMessage;
	
	int prevId = -1;
	MESSAGE_ITEM_t *pItemMessage = NULL;
	
	list_for_each_entry(pParentNode, &pItem->parentHead, list, PARENT_NODE_t) {
		if (depth == 0) {
			isExist = FALSE;
			for (int i = 0;i < gSecondRootIdsNum;i++) {
				int id = gSecondRootIds[i];
				pItemMessage = gIdMap[id].pItem;
				if (pItemMessage->type == TYPE_message) {
				//	printf("pItemMessage->keystr:%s\n",pItemMessage->keystr);
					if (strcmp(pParentNode->parentTitle,pItemMessage->keystr) == 0) {
						isExist = TRUE;			
						pParentNode->pMessage = gIdMap[id].pMessage;
						pParentNode->pItem = gIdMap[id].pItem;
						prevId = id;
						break;
					}
					//printf("gIdMap[%d].pItem->messageTypeName:%s\n",id,gIdMap[id].pItem->keystr);
				}
			}
			
			//printf("pParentNode->parentTitle:%s\n",pParentNode->parentTitle);
			CHECK_EXPR_RET(isExist  == FALSE, -1);
		} else {
			
			isExist = FALSE;
			for (int i = 0;i < gIdMap[prevId].childIdNum;i++) {
				int id = gIdMap[prevId].childId[i];
				pItemMessage = gIdMap[id].pItem;
				if (strcmp(pParentNode->parentTitle,pItemMessage->keystr) == 0) {
					isExist = TRUE;		
					pParentNode->pMessage = gIdMap[id].pMessage;
					pParentNode->pItem = gIdMap[id].pItem;
					prevId = id;
					break;
				}
			}
			CHECK_EXPR_RET(isExist  == FALSE, -1);
			
		}
		depth++;
	}

	isExist = FALSE;
	for (int i = 0;i < gIdMap[prevId].childIdNum;i++) { 
		int id = gIdMap[prevId].childId[i];	
		if (strcmp(pItem->key,gIdMap[id].pItem->keystr) == 0) {
			isExist = TRUE;			
			pItem->pItem = gIdMap[id].pItem;
			break;
		}		
	}
	CHECK_EXPR_RET(isExist  == FALSE, -1);
	PARPT_writeOneItem(pItem);
	
	return 0;
}


int generateHeadFileByRootMessage(MTREE_s **ppTree,char *rootMessage)
{
	int ret = -1;
	
	MTREE_s *pTree = MTREE_create(NULL);
	*ppTree = pTree;
	//There exist a bug that need the follow step.
	MTREE_addItem(pTree, -2, -1);
	pTreeTmp = pTree;
	ret = getAllItemsByMessageName(rootMessage,addItemsToMutiTree);
	CHECK_EXPR_RET(ret < 0, -1);

	//MTREE_printByList(pTree);

	//writeHeadFile();
	
	ret = MTREE_backwardExt(pTree, writeOneMessageToHeadfile);
	CHECK_EXPR_RET(ret < 0, -1);

	//write the root message
	MESSAGE_t *pMessage = getMessageNodeByName(rootMessage);
	CHECK_EXPR_RET(pMessage == NULL, -1);
	GenHead_writeMessageTypedef(pMessage);
	GenHead_writeOneMessage(pMessage);
	
	
	
	return 0;
}

int initMessageWritten()
{
	gMessageWritten =  (char (*)[STR_MAX_SIZE])malloc(sizeof(char)*STR_MAX_SIZE*gAllMessageNum);
	CHECK_EXPR_RET(gMessageWritten == NULL, -1);
	gMessageWrittenN = 0;
	return 0;
}

int resetIdFlags()
{
	for (int i = 0;i < gItemCnt;i++) {
		gIdFlags[i] = FALSE;
	}

	return 0;
}


int generateRoute(int id,int parentId)	
{
	MESSAGE_t *pMessage = NULL;
	MESSAGE_ITEM_t *pItem = NULL;
	int tmpid = -1;
	int tmpParentId = -1;
	
	int idRoute[ROUTE_MAX_DEPTH];
	int idRouteCnt = 0;
	idRoute[idRouteCnt++] = id;
	int cnt = ROUTE_MAX_DEPTH;

	if (gIdMap[id].pItem->type == TYPE_message) {
		return -1;	
	}

	do {
		tmpParentId = gIdMap[tmpid].pItem->id;
		idRoute[idRouteCnt++] = tmpParentId;
		tmpid = tmpParentId;
	}while (tmpid != 0 && cnt--);

	CHECK_EXPR_RET(tmpid != 0, -1);
	
	ID_ROUTE_NODE_t *pIdRouteNode = (ID_ROUTE_NODE_t *)malloc(sizeof(ID_ROUTE_NODE_t));
	CHECK_EXPR_RET(pIdRouteNode == NULL, -1);

	pIdRouteNode->idroute = (int *)malloc(sizeof(int) * idRouteCnt);
	CHECK_EXPR_RET(pIdRouteNode->idroute == NULL, -1);

	int *pIdArr = pIdRouteNode->idroute;
	for (int i = idRouteCnt-1;i>=0;i--) {
		pIdArr[i] = idRoute[idRouteCnt-1-i];
	}

	list_add_tail(&pIdRouteNode->list, &gAllIdRoutes);

	return 0;	
}

int writeOneMessageToHeadfile(void * usr, int idCurr, int * _idNext, int _nIdNext)
{
	MESSAGE_t *pMessage = NULL;
	MESSAGE_ITEM_t *pItem = NULL;

	BOOL isExist = FALSE;
	if (idCurr != -1 && idCurr != -2) {
		pItem =  gIdMap[idCurr].pItem;
		pMessage = getMessageNodeByName(pItem->messageTypeName);
		if (pItem->type == TYPE_message && pMessage != NULL) {

			for (int i = 0;i < gMessageWrittenN;i++) {
				if (strcmp(gMessageWritten[i],pMessage->name) == 0) {
					isExist = TRUE;
				}
			}
			if (isExist == FALSE) {
				strcpy(gMessageWritten[gMessageWrittenN++],pMessage->name);
				GenHead_writeMessageTypedef(pMessage);
				GenHead_writeOneMessage(pMessage);
			}			
		}
	}

	return 0;
}

int addItemsToMutiTree(int id,int parentId)
{
	CHECK_EXPR_RET(pTreeTmp == NULL,-1);
	//cout<< "----id:" << id << " parentId:" << parentId << endl; 
	//getchar();
	MTREE_addItem(pTreeTmp, parentId, id);
	
	return 0;
}

int writeHeadFile()
{
	GenHead_writeHeadfileStart();
	MESSAGE_t *p = NULL;
	
	list_for_each_entry(p,&gMessageList,list,MESSAGE_t) { 
		GenHead_writeMessageTypedef(p);
	}
	
	list_for_each_entry(p,&gMessageList,list,MESSAGE_t) { 
		GenHead_writeOneMessage(p);
	}
	GenHead_writeHeadfileEnd();
	
	return 0;
}

int getItemIdByName(char *messageName,char *name)
{
	MESSAGE_t *p = NULL;
	list_for_each_entry(p,&gMessageList,list,MESSAGE_t) {
		if (p->isEnum == TRUE) {
			if (strcmp(p->enumName,name) == 0 && strcmp(messageName,"enum") == 0) {
				 return p->enumId; 
			}
		} else {
			if  (strcmp(p->name,messageName) == 0) {
				for (int i= 0;i < p->itemN;i++) {
					if (strcmp(p->pItems[i].keystr,name) == 0) {
						return p->pItems[i].id;
					}
				}
			}
		}
	}
	return -1; 
}

int addIdMap()
{
	CHECK_EXPR_RET(gItemCnt <= 0,-1);
	gIdMap = (ID_MAP_t *)malloc(sizeof(ID_MAP_t) * gItemCnt);
	
	
	MESSAGE_t *p = NULL;
	int id = 0;
	list_for_each_entry(p,&gMessageList,list,MESSAGE_t) {
		if (p->isEnum == TRUE) {
			gIdMap[id].pMessage = p;
			gIdMap[id].pItem = NULL;
			id++;
		} else {
			for (int i= 0;i < p->itemN;i++) {
				gIdMap[id].pMessage = p;
				gIdMap[id].pItem = &p->pItems[i];
				id++;
			}
		}
	}
	return 0;
}

int addIds()
{
	MESSAGE_t *p = NULL;
	int id = 0;
	list_for_each_entry(p,&gMessageList,list,MESSAGE_t) {
		if (p->isEnum == TRUE) {
			p->enumId = id++;
		} else {
			for (int i= 0;i < p->itemN;i++) {
				p->pItems[i].id = id++;
			}
		}
		gAllMessageNum++;
	}
	gItemCnt = id;
	
	return 0;
}

MESSAGE_t *getMessageNodeByName(char *messageName)
{
	MESSAGE_t *p = NULL;
	list_for_each_entry(p,&gMessageList,list,MESSAGE_t) {
		if (p->isEnum == TRUE) {
			if (strcmp(messageName,p->enumName) == 0) {
				return p;
			}
		} else {
			if (strcmp(messageName,p->name) == 0) {
				return p;
			}
		}
	}
	return NULL;
}

//判断message中的类型是不是一个enum

void *isMessageTypeisEnum(int id,ENUM_TYPE_t *enumType)
{
	CHECK_EXPR_RET(enumType == NULL, NULL);

	MESSAGE_t *pMessage = NULL;
	MESSAGE_ITEM_t *pItem = NULL;

	pMessage = gIdMap[id].pMessage;
	pItem = gIdMap[id].pItem;

	//如果有点就表示这个enum的定义是复合定义的，详见
	if (strstr(pItem->messageTypeName,".") != NULL) {
		*enumType = ENUM_TYPE_LOCAL_MUTI; 
		return pMessage;
	}
	
	if (pMessage->isEnum == TRUE) {
		*enumType = ENUM_TYPE_GLOBAL;
		return pMessage;
	}
	
	for (int i = 0;i < pMessage->itemN;i++) {
		if (pMessage->pItems[i].attr == ATTR_enum) {
			if (strcmp(pItem->messageTypeName,pMessage->pItems[i].enumTitle) == 0) {
				*enumType = ENUM_TYPE_LOCAL;
				return &pMessage->pItems[i];
			}
		}
	}

	*enumType = ENUM_TYPE_NONE;
	return pMessage;
}
 
int getAllItemsByMessageName(char *messageName,int (*handle_id)(int id,int parentId))	
{
#define STACK1_MAX_SIZE 256

	CHECK_EXPR_RET(messageName == NULL, -1);
	MESSAGE_t *pMessage = getMessageNodeByName(messageName);
	CHECK_EXPR_RET(pMessage == NULL, -1);
	
	//cout<< "sizeof(int) : " << sizeof(int) << endl;
	void *pStackItemsId = STACK_init(STACK1_MAX_SIZE,sizeof(ID_PAIR_t));
	CHECK_EXPR_RET(pStackItemsId == NULL, -1);

	ID_PAIR_t idPair;
	for (int i = 0;i < pMessage->itemN;i++) {
		idPair.id = pMessage->pItems[i].id;
		idPair.parentId = -1;
		if (handle_id != NULL) {
			handle_id(idPair.id,idPair.parentId);
		}
		STACK_push(pStackItemsId, &idPair);
	}
	
	MESSAGE_ITEM_t *pItem = NULL;
	
	while(STACK_empty(pStackItemsId) == FALSE) {
		STACK_pop(pStackItemsId,&idPair);
		
		pItem = gIdMap[idPair.id].pItem;

		if (pItem->type == TYPE_message) {
			ENUM_TYPE_t enumType;
			isMessageTypeisEnum(idPair.id, &enumType);
			if (enumType != ENUM_TYPE_NONE) {
				continue;
			}
		
			MESSAGE_t *pSubmessage = getMessageNodeByName(pItem->messageTypeName);
			
			//printf("pItem->messageTypeName:%s\n",pItem->messageTypeName);
			//printf("pItem->defaultVal:%s\n",pItem->defaultVal);
			//cout<< "pItem->messageTypeName:" << pItem->messageTypeName << endl;
			CHECK_EXPR_RET(pSubmessage == NULL, -1);
			if (pSubmessage->isEnum == FALSE) {
				for (int i = 0;i < pSubmessage->itemN;i++) {
					idPair.id = pSubmessage->pItems[i].id;
					idPair.parentId = pItem->id;
					if (handle_id != NULL) {
						handle_id(idPair.id,idPair.parentId);
					}
					STACK_push(pStackItemsId, &idPair);
				}
			}
		}
	}
	
	return 0;
}

int printMessageItemInfo(MESSAGE_ITEM_t *pMessageItem)
{	
	printf("[%d] attr:%s ",pMessageItem->id,gAttrMapStr[pMessageItem->attr].str);
	if (pMessageItem->attr == ATTR_enum) {
		printf(" enumTitle:%s\n",pMessageItem->enumTitle);
		
		for (int i = 0;i < pMessageItem->enumItemArrN;i++) {
			printf("   [%s %d]\n",pMessageItem->enumItemArr[i],pMessageItem->enumId[i]);
		}
	} else {
		if (pMessageItem->type == TYPE_message) {
			printf(" type:%s ",pMessageItem->messageTypeName);
		} else {
			printf(" type:%s ",gTypeMapStr[pMessageItem->type].str);
		}
		
		printf(" keystr:%s ",pMessageItem->keystr);
		if (pMessageItem->isHaveDefaultVal) {
			printf(" defaultVal:%s ",pMessageItem->defaultVal);
		}
		printf(" isPacked:%d",pMessageItem->isPacked);
	}

	return 0;
}


int printMessageInfo()
{
	MESSAGE_t *p = NULL;
	list_for_each_entry(p,&gMessageList,list,MESSAGE_t) {
		if (p->isEnum == TRUE) {
			printf("enum [%s]\n",p->enumName);
			for (int i = 0;i < p->enumItemArrN;i++) {
				printf("[%s] = [%d]\n",p->enumItemArr[i],p->enumVal[i]);
			}
		} else {
			printf("message [%s]:\n",p->name);
			for (int i= 0;i < p->itemN;i++) {
				printMessageItemInfo(&p->pItems[i]);	
				printf("\n");
			}
			printf("\n");
		}
	}
	printf("\n");

	return 0;
}

int printSplitInfo()
{
	for (int i = 0;i < gLineSplitLen;i++) {
		printf("%s ",gLineSplit[i]);
	}
	printf("\n");
	
	return 0;
}

int splitLine(char *line)
{
	int i = 0;
	char prev = ' ';

	memset(gLineSplit[gLineSplitLen],'\0',STR_MAX_SIZE);
	gLineSplitLen = 0;
	int cnt = 0;

	if (line[0] == '}') {
		return 0;
	}
	
	while(line[i] != '\0') {
		if (line[i] != ' ' && line[i] != ';') {
			gLineSplit[gLineSplitLen][cnt++] = line[i]; 
			
		} else if ((line[i] == ' ' && prev != ' ') || \
					line[i] == ';') {
			gLineSplit[gLineSplitLen++][cnt] = '\0';
			cnt = 0;
		} 
		prev = line[i];
		i++;
	}

	//printSplitInfo();

	return 0;
}

int readFromFile(char *protofile)
{
	int fd = open(protofile,O_RDWR,0666);
	CHECK_EXPR_RET(fd < 0,-1);

	int fileLen = lseek(fd,0,SEEK_END);

	pBuffer = (char *)malloc(sizeof(char) * fileLen + 2);
	
	lseek(fd,0,SEEK_SET);
	
	int nread = 0;
	nread = read(fd,pBuffer,fileLen);
	
	CHECK_EXPR_RET(nread != fileLen,-1);
	bufferLen = nread;
	pBuffer[bufferLen++] = '\n';
	pBuffer[bufferLen++] = '\0';
	
	return 0;
}


int parseMessagHead(char *line)
{
#if 0
	int i = 0;
	while(line[i] != '\0') {
		if (line[i] == '{') {
			line[i] = '\0';
		}
		i++;
	}
#endif
	switchChar(line, '{', '\0');

	splitLine(line);
	
	pCurrMessage = (MESSAGE_t *)malloc(sizeof(MESSAGE_t));
	CHECK_EXPR_RET(pCurrMessage == NULL,-1);
	memset(pCurrMessage,'\0',sizeof(MESSAGE_t));
	
	strcpy(pCurrMessage->name,gLineSplit[1]);
	printf("----------->Message:%s\n",pCurrMessage->name);
	list_add_tail(&pCurrMessage->list,&gMessageList);
	
	return 0;
}

int parseEnumInMessage(char *enumStr,MESSAGE_ITEM_t *pItem)
{
	pItem->attr = ATTR_enum;
	splitLine(enumStr);

	printf("[%s]\n",enumStr);
	if (strstr(enumStr,"LayerType") != NULL) {
		//cout<< "-------------" << enumStr << endl;
		//getchar();
	}
	strcpy(pItem->enumTitle,gLineSplit[1]);
	
	int enumItemN = (gLineSplitLen-2)/3;
	pItem->enumItemArrN = enumItemN;
	
	pItem->enumItemArr = (char (*)[STR_MAX_SIZE])malloc(STR_MAX_SIZE*enumItemN);
	CHECK_EXPR_RET(pItem->enumItemArr == NULL, -1);
	
	pItem->enumId = (int *)malloc(sizeof(int)*enumItemN);
	CHECK_EXPR_RET(pItem->enumId == NULL, -1);

	int m = 0;
	m += 2;

	for (int i = 0;i < enumItemN;i++) {
		strcpy(pItem->enumItemArr[i],gLineSplit[m]);
		pItem->enumId[i] = atoi(gLineSplit[m+2]);
		printf("[%s %d]\n",pItem->enumItemArr[i],pItem->enumId[i]);
		m += 3;
	}
	
	return 0;
}

int switchChar(char *buf,char old_c,char new_c)
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


int parseEnum(BOOL isIncludeINmessage)
{
	if (isIncludeINmessage == FALSE) {
		MESSAGE_t *pMessage = (MESSAGE_t *)malloc(sizeof(MESSAGE_t));
		CHECK_EXPR_RET(pMessage == NULL, -1);
		pMessage->isEnum = TRUE;

		splitLine(gEnumItemBuf[0]);
		strcpy(pMessage->enumName,gLineSplit[1]);

		pMessage->enumItemArr = (char (*)[STR_MAX_SIZE])malloc(STR_MAX_SIZE * (gEnumItemN-1));
		CHECK_EXPR_RET(pMessage->enumItemArr == NULL, -1);

		pMessage->enumVal = (int *)malloc(sizeof(int) * (gEnumItemN-1));
		CHECK_EXPR_RET(pMessage->enumVal == NULL, -1);

		pMessage->enumItemArrN = 0;		
		for (int i = 1;i < gEnumItemN;i++) {
			switchChar(gEnumItemBuf[i], ';', ' ');
			splitLine(gEnumItemBuf[i]);
			strcpy(pMessage->enumItemArr[pMessage->enumItemArrN],gLineSplit[0]);
			pMessage->enumVal[pMessage->enumItemArrN] = atoi(gLineSplit[2]);
			pMessage->enumItemArrN++;
		}
		list_add_tail(&pMessage->list, &gMessageList);
			
	} else {
		int len = 0;
		
		for (int i = 0;i < gEnumItemN;i++) {
			strcpy(&gLineArr[gLineArrLen][len],gEnumItemBuf[i]);
			len += strlen(gEnumItemBuf[i]);
		}
		switchChar(gLineArr[gLineArrLen],';',' ');
		if (strstr(gLineArr[gLineArrLen],"LayerType") != NULL) {
			//cout<< "===========" << gLineArr[gLineArrLen] << endl;
			//getchar();
		}
		
		
		gLineArrLen++;
	}
	return 0;
}


int parseMessageBody()
{
	MESSAGE_ITEM_t *pItems = (MESSAGE_ITEM_t *)malloc(sizeof(MESSAGE_ITEM_t)*gLineArrLen);
	CHECK_EXPR_RET(pItems == NULL,-1);
	
	pCurrMessage->pItems = pItems;
	pCurrMessage->itemN = gLineArrLen;

	for (int i = 0;i < gLineArrLen;i++) {
		
		if (strstr(gLineArr[i],"enum") != NULL) {
			parseEnumInMessage(gLineArr[i], &pCurrMessage->pItems[i]);
			continue;
		}
	
		splitLine(gLineArr[i]);

		//attributtion
		BOOL checkAttrOK = FALSE;
		for (int j = 0;j < DIM_OF(gAttrMapStr);j++) {
			if (strcmp(gLineSplit[0],gAttrMapStr[j].str) == 0) {
				pItems[i].attr = gAttrMapStr[j].type;
				checkAttrOK = TRUE;
				break;
			}
		}
		CHECK_EXPR_RET(checkAttrOK == FALSE,-1);
		
		//type
		BOOL isMeesageType = TRUE;
		for (int j = 0;j < DIM_OF(gTypeMapStr);j++) {
			if (strcmp(gLineSplit[1],gTypeMapStr[j].str) == 0) {
				pItems[i].type = gTypeMapStr[j].type;
				isMeesageType = FALSE;
				break;
			}
		}

		if (isMeesageType == TRUE) {
			pItems[i].type = TYPE_message;
			strcpy(pItems[i].messageTypeName,gLineSplit[1]);
		}

		//key string
		strcpy(pItems[i].keystr,gLineSplit[2]);
		pItems[i].id = atoi(gLineSplit[4]);

		if (gLineSplitLen <= 5) {
			continue;
		}
		
		if (strstr(gLineSplit[5],"packed") != NULL) {
			if (strncmp(gLineSplit[7],"true",strlen("true")) == 0) {
				pCurrMessage->pItems[i].isPacked = TRUE;
			} else {
				pCurrMessage->pItems[i].isPacked = FALSE;
			}
		} else {
			pCurrMessage->pItems[i].isPacked = FALSE;
		}
		
		if(strstr(gLineSplit[5],"default") != NULL) {
			pCurrMessage->pItems[i].isHaveDefaultVal = TRUE;
			strcpy(pCurrMessage->pItems[i].defaultVal,gLineSplit[7]);
			filterChar(pCurrMessage->pItems[i].defaultVal,']');
			switchChar(pCurrMessage->pItems[i].defaultVal, '\'', '\"');
			//printf("defaultVal:%s\n",pCurrMessage->pItems[i].defaultVal);
			
		} else {
			pCurrMessage->pItems[i].isHaveDefaultVal = FALSE;
		}
	}
	
	return 0;
}

int filterAnnotation(char *buf)
{
	for (int i = 0;i < strlen(buf);i++) {
		if (buf[i] == '{') {
			buf[i] = '\0';
			break;
		}
 	}
	return 0;
}

int filterChar(char *buf,char c)
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

int parseLine(char *line,int len)
{
    BOOL bstart = FALSE;
	int cnt = 0;
	char buf[1024];
	
	int i = 0;
	for (;i < len;i++) {
		if (line[i] != ' ' && line[i] != '\t' && line[i] != '\n') {
			break;
		}
	}
	
	for (int j = i;j < len;j++) {
		buf[cnt++] = line[j];
	}
	buf[cnt] = '\0';
	if (cnt == 0) {
		return 0;
	}
	if (buf[0] == '/') {
		return 0;
	}

	
	char *str = NULL; 
	int ret = 0;
    if (strstr(buf,messageStr) != NULL) {
		ret = STACK_push(pStack, &messageStr);
		CHECK_EXPR_RET(ret < 0, -1);
		parseMessagHead(buf);
		
	} else if(strstr(buf,enumStr) != NULL) {
		ret = STACK_push(pStack, &enumStr);
		CHECK_EXPR_RET(ret < 0, -1);
		filterAnnotation(buf);
		filterChar(buf, '{');
		strcpy(gEnumItemBuf[gEnumItemN++],buf);
		
	} else if(buf[0] == '}') {
		ret = STACK_pop(pStack,&str);
		CHECK_EXPR_RET(ret < 0, -1);
		
		if (strcmp(str,messageStr) == 0) {
			parseMessageBody();
			gLineArrLen = 0;

		} else if (strcmp(str,enumStr) == 0) {
			if (STACK_empty(pStack)) {
				parseEnum(FALSE);
			} else {
				parseEnum(TRUE);
			}
			gEnumItemN = 0;
		}
		
	} else {
		ret = STACK_peekLatest(pStack,&str);
		CHECK_EXPR_RET(ret < 0, -1);
		if (strcmp(str,messageStr) == 0) {
			strcpy(gLineArr[gLineArrLen++],buf);
			
		} else if (strcmp(str,enumStr) == 0) {
			strcpy(gEnumItemBuf[gEnumItemN++],buf);
		}
	}
    
    return 0;
}

int parseBuf(char *buf)
{
	int i = 0;
	char line[1024];
	char *ptr = buf;
	int n = strlen(buf);
	while(n > 0){
		getLineBuf(ptr,line);
		if (strstr(line,"syntax") == NULL && strstr(line,"package") == NULL) {
			parseLine(line,strlen(line));
		}
		ptr += strlen(line)+1;
		n -= strlen(line)+1;
	}
	return 0;
}

int getLineBuf(char *start,char *line)
{
	int i = 0;
	while(start[i] != '\n') {
		line[i] = start[i];
		i++;
	}
	line[i] = '\0';
	
	return i;
}


