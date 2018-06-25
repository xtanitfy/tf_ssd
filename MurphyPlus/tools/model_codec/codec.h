#ifndef __PARSE_H__
#define __PARSE_H__

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




#define ENUM_MAX_IMTEM 12
#define STR_MAX_SIZE 128
#define MESSAGE_MAX_ITEM 128
#define MESSAGE_MAX_ENUM 4



typedef enum
{
	ATTR_optional,//optional
	ATTR_repeated,//repeated
	ATTR_enum,
	ATTR_NUM
}ATTR_t;

typedef enum
{
	TYPE_int32,
	TYPE_int64,
	TYPE_float,
	TYPE_double,
	TYPE_message,
	TYPE_bool,
	TYPE_string,
	TYPE_uint32,
	TYPE_bytes,
	TYPE_CNT
}TYPE_t;

typedef struct
{
	TYPE_t type;
	char *str;
	char *w_str;
}TypeMapStr_t;


typedef struct
{
	ATTR_t type;
	char *str;
}AttrMapStr_t;


typedef struct
{
	ATTR_t attr;
	TYPE_t type;
	char messageTypeName[STR_MAX_SIZE];
	char keystr[STR_MAX_SIZE];
	int id;
	BOOL isHaveDefaultVal;
	char defaultVal[STR_MAX_SIZE];
	BOOL isPacked;

	char enumTitle[STR_MAX_SIZE];
	char (*enumItemArr)[STR_MAX_SIZE];//if attr is ATTR_enum
	int *enumId;
	int enumItemArrN;

	/*the information of enum modified:*/
	BOOL isEnumModifed;
	//char newEnumNamePrefix[STR_MAX_SIZE];
}MESSAGE_ITEM_t;


typedef struct
{
	BOOL isEnum;
	char enumName[STR_MAX_SIZE];
	char (*enumItemArr)[STR_MAX_SIZE];//if isEnum is true
	int *enumVal;
	int enumItemArrN;
	int enumId; // if isEnum == TRUE,then id is not null
	
	MESSAGE_ITEM_t *pItems;
	int itemN;
	char name[STR_MAX_SIZE];
	struct list_head list;
}MESSAGE_t;


typedef struct {
	MESSAGE_t *pMessage;
	MESSAGE_ITEM_t *pItem;
	int childId[MESSAGE_MAX_ITEM];
	int childIdNum;
}ID_MAP_t;

typedef enum
{
	ENUM_TYPE_GLOBAL,
	ENUM_TYPE_LOCAL,
	ENUM_TYPE_LOCAL_MUTI,//这里的enum定义是级联定义
	ENUM_TYPE_NONE
}ENUM_TYPE_t;

typedef struct
{
	int id;
	int parentId;
}ID_PAIR_t;

typedef struct
{
	int *idroute;
	int len;
	struct list_head list;
}ID_ROUTE_NODE_t;



#endif
