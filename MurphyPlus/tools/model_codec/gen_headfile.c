#include "gen_headfile.h"
//#include <iostream>

//using namespace std;

FILE *pParamHeadFile = NULL;

static int GenHeadMicroHead();
static int GenHeadMicroTail();
static int GenHeadWriteFile(char *buf,int len);
static int GenHeadWriteLocalEnum(MESSAGE_ITEM_t *pItem);
static int GenHeadWriteGlobalEnum(MESSAGE_t *pItem);
static int GenHeadWriteStruct(MESSAGE_t *pMessage,MESSAGE_ITEM_t *pItem,BOOL bRepeat);
static int GenHeadWriteBaseDataStu(MESSAGE_ITEM_t *pItem,BOOL bRepeat);
static int GenHeadLocalEnumType(MESSAGE_t *pMessage);
static int GenHeadSwitchChar(char *buf,char old_c,char new_c);

extern TypeMapStr_t gTypeMapStr[];
extern AttrMapStr_t gAttrMapStr[];

int GenHead_init()
{
	pParamHeadFile = fopen(PARAMTER_FILE_NAME,"w+");
	CHECK_EXPR_RET(pParamHeadFile == NULL, -1);

	
	return 0;
}

int GenHead_writeHeadfileStart()
{
	GenHeadMicroHead();
}

int GenHeadSwitchChar(char *buf,char old_c,char new_c)
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

int GenHead_writeHeadfileEnd()
{
	GenHeadMicroTail();
	fclose(pParamHeadFile);
}

int GenHeadWriteBaseDataStu(MESSAGE_ITEM_t *pItem,BOOL bRepeat)
{
	char buf[256];

	CHECK_EXPR_RET(pItem->type < TYPE_int32, -1);
	CHECK_EXPR_RET(pItem->type >= TYPE_CNT, -1);

	if (bRepeat == TRUE) {
		if (pItem->type == TYPE_string || pItem->type == TYPE_bytes) {
			snprintf(buf,sizeof(buf),"\tchar (*%s)[PARSE_STR_NAME_SIZE];\n",pItem->keystr);
		} else {
			snprintf(buf,sizeof(buf),"\t%s *%s;\n",
				gTypeMapStr[pItem->type].w_str,pItem->keystr);
		}
		GenHeadWriteFile(buf,strlen(buf));
		
		snprintf(buf,sizeof(buf),"\tUINT32 %s_size;\n",pItem->keystr);
		GenHeadWriteFile(buf,strlen(buf));
		//GenHeadWriteFile("\n",1);
		
	} else {
		if (pItem->type == TYPE_string || pItem->type == TYPE_bytes) {
			snprintf(buf,sizeof(buf),"\tchar %s[PARSE_STR_NAME_SIZE];\n",pItem->keystr);
			GenHeadWriteFile(buf,strlen(buf));
		
			//snprintf(buf,sizeof(buf),"\tuint32 %s_size;\n",pItem->keystr);
			//GenHeadWriteFile(buf,strlen(buf));
			//GenHeadWriteFile("\n",1);
		} else {
			snprintf(buf,sizeof(buf),"\t%s %s;\n",
				gTypeMapStr[pItem->type].w_str,pItem->keystr);
			GenHeadWriteFile(buf,strlen(buf));
			//GenHeadWriteFile("\n",1);
		}
	}
	return 0;
}


int GenHeadWriteStruct(MESSAGE_t *pMessage,MESSAGE_ITEM_t *pItem,BOOL bRepeat)
{
	MESSAGE_ITEM_t *p1;
	char buf[STR_MAX_SIZE];
	char buf1[STR_MAX_SIZE];
	#if 1
	//如果是enum的值就重新定义枚举类型
	for (int i = 0;i < pMessage->itemN;i++) {
		p1 = &pMessage->pItems[i];
		if (p1->attr == ATTR_enum) {
			if (strcmp(p1->enumTitle,pItem->messageTypeName) == 0) {
				snprintf(buf,sizeof(buf),"%s_%s",pMessage->name,pItem->messageTypeName);
				strcpy(pItem->messageTypeName,buf);
				printf("<%s>\n",pItem->messageTypeName);
				pItem->isEnumModifed = TRUE;
				#if 0
				snprintf(buf1,sizeof(buf1),"%s_%s",buf,pItem->defaultVal);
				strcpy(pItem->defaultVal,buf1);
				//getchar();
				#endif
			}
		}
	}
	#endif
	
	if (bRepeat == TRUE) {
		snprintf(buf,sizeof(buf),"\t%s *%s;\n",
			pItem->messageTypeName,pItem->keystr);
		GenHeadSwitchChar(buf,'.','_');
		GenHeadWriteFile(buf,strlen(buf));
	
		snprintf(buf,sizeof(buf),"\tUINT32 %s_size;\n",pItem->keystr);
		GenHeadSwitchChar(buf,'.','_');
		GenHeadWriteFile(buf,strlen(buf));
		//GenHeadWriteFile("\n",1);
	} else {
		snprintf(buf,sizeof(buf),"\t%s %s;\n",
			pItem->messageTypeName,pItem->keystr);
		
		if (strstr(buf,".") != NULL) {
			GenHeadSwitchChar(pItem->messageTypeName,'.','_');
			pItem->isEnumModifed = TRUE;
		}
		
		GenHeadSwitchChar(buf,'.','_');
		GenHeadWriteFile(buf,strlen(buf));
		//GenHeadWriteFile("\n",1);
	}
	
	return 0;
}


int GenHeadWriteGlobalEnum(MESSAGE_t *pMessage)
{
	CHECK_EXPR_RET(pMessage == NULL, -1);
	CHECK_EXPR_RET(pMessage->isEnum == FALSE, -1);

	char buf[256];
	snprintf(buf,sizeof(buf),"typedef enum {\n");
	GenHeadWriteFile(buf,strlen(buf));
	
	for (int i = 0;i < pMessage->enumItemArrN;i++) {
		snprintf(buf,sizeof(buf),"\t%s = %d,\n",
				pMessage->enumItemArr[i],pMessage->enumVal[i]);
		GenHeadWriteFile(buf,strlen(buf));
		
	}
	
	snprintf(buf,sizeof(buf),"}%s;\n",pMessage->enumName);
	GenHeadWriteFile(buf,strlen(buf));
	GenHeadWriteFile("\n",1);	

	return 0;
}

int GenHeadLocalEnumType(MESSAGE_t *pMessage)
{
	char buf[256];

	MESSAGE_ITEM_t *pItem;
	for (int i = 0;i < pMessage->itemN;i++) {
		pItem = &pMessage->pItems[i];
		if (pItem->attr == ATTR_enum) {	
			snprintf(buf,sizeof(buf),"typedef enum {\n");
			GenHeadWriteFile(buf,strlen(buf));
		
			for (int i = 0;i < pItem->enumItemArrN;i++) {
				snprintf(buf,sizeof(buf),"\t%s_%s_%s = %d,\n",
						pMessage->name,pItem->enumTitle,pItem->enumItemArr[i],pItem->enumId[i]);
				GenHeadWriteFile(buf,strlen(buf));
			}
			snprintf(buf,sizeof(buf),"}%s_%s;\n",pMessage->name,pItem->enumTitle);
			GenHeadWriteFile(buf,strlen(buf));
			GenHeadWriteFile("\n",1);
		}
	}
	return 0;
}

int GenHeadWriteLocalEnum(MESSAGE_ITEM_t *pItem)
{
	CHECK_EXPR_RET(pItem == NULL, -1);
	
	char *str2 = "\ttypedef enum {\n";
	GenHeadWriteFile(str2,strlen(str2));

	char buf[256];
	for (int i = 0;i < pItem->enumItemArrN;i++) {
		//snprintf(buf,sizeof(buf),"\t\t%s = %d,\n",
		//		pItem->enumItemArr[i],pItem->enumId[i]);
		snprintf(buf,sizeof(buf),"\t\t%s = %d,\n",
				pItem->enumItemArr[i],pItem->enumId[i]);
		GenHeadWriteFile(buf,strlen(buf));
		
	}
	snprintf(buf,sizeof(buf),"\t}%s;\n",pItem->enumTitle);
	GenHeadWriteFile(buf,strlen(buf));
	GenHeadWriteFile("\n",1);	
}

int GenHead_writeMessageTypedef(MESSAGE_t *pMessage)
{
	char buf[256];
	if (pMessage->isEnum == TRUE) {
		//snprintf(buf,sizeof(buf),"typedef enum %s_ %s;\n",pMessage->enumName,pMessage->enumName);
		//GenHeadWriteFile(buf,strlen(buf));
	} else {
		snprintf(buf,sizeof(buf),"typedef struct %s_ %s;\n",pMessage->name,pMessage->name);
		GenHeadWriteFile(buf,strlen(buf));
	}
	
	return 0;
}

int GenHead_writeOneMessage(MESSAGE_t *pMessage)
{
	CHECK_EXPR_RET(pMessage == NULL, -1);
	
	if (pMessage->isEnum == TRUE)
	{
		GenHeadWriteGlobalEnum(pMessage);
		return 0;
	}
	GenHeadLocalEnumType(pMessage);

	char buf[256];
	snprintf(buf,sizeof(buf),"struct %s_ {\n",pMessage->name);
	GenHeadWriteFile(buf,strlen(buf));
	
	
	MESSAGE_ITEM_t *pItem = NULL;
	for (int i = 0;i < pMessage->itemN;i++) {

		pItem = &pMessage->pItems[i];

		if (pItem->attr == ATTR_enum) {
			//GenHeadWriteLocalEnum(pItem);

		} else if(pItem->attr == ATTR_repeated) {
			if (pItem->type == TYPE_message) {
				GenHeadWriteStruct(pMessage,pItem,TRUE);
			} else {
				GenHeadWriteBaseDataStu(pItem,TRUE);
			}
			
		} else {
			if (pItem->type == TYPE_message) {
				GenHeadWriteStruct(pMessage,pItem,FALSE);
			} else {
				GenHeadWriteBaseDataStu(pItem,FALSE);
			}
		}		
	}

	GenHeadWriteFile("};\n",3);
	GenHeadWriteFile("\n",1);

	return 0;
}


int GenHeadMicroHead()
{
	char *str = "#ifndef __PARAMETER_H__\n";
	GenHeadWriteFile(str,strlen(str));
	
	str = "#define __PARAMETER_H__\n\n";
	GenHeadWriteFile(str,strlen(str));

	str = "#include <stdio.h>\n";
	GenHeadWriteFile(str,strlen(str));

	str = "#include <stdlib.h>\n";
	GenHeadWriteFile(str,strlen(str));
		
	str = "#include <string.h>\n";
	GenHeadWriteFile(str,strlen(str));

	str = "#include \"public.h\"\n";
	GenHeadWriteFile(str,strlen(str));
		
	str = "#include \"data_type.h\"\n\n";
	GenHeadWriteFile(str,strlen(str));

	str = "#define PARSE_STR_NAME_SIZE 64 \n";
	GenHeadWriteFile(str,strlen(str));

	str = "#define false 0\n";
	GenHeadWriteFile(str,strlen(str));

	str = "#define true 1\n";
	GenHeadWriteFile(str,strlen(str));

	//str = "typedef unsigned char bool;\n";
	//GenHeadWriteFile(str,strlen(str));
	
	return 0;
}


int GenHeadMicroTail()
{
	char *str = "#endif\n";
	GenHeadWriteFile(str,strlen(str));
	return 0;
}

int GenHeadWriteFile(char *buf,int len)
{
	int nwrite = fwrite(buf,len,1,pParamHeadFile);
	CHECK_EXPR_RET(nwrite != 1, -1);
}




