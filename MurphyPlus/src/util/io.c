#include "io.h"

int IO_createdDirectory(char *dirName)
{
	char cmd[128];

	snprintf(cmd,sizeof(cmd),"mkdir %s",dirName);
	system(cmd);
	return 0;
}

BOOL IO_MapLabelToName(LabelMap *map, BOOL strict_check,
    							MAP_LABEL_NAME_t **label_to_name,int *label_to_name_size) 
{
	MAP_LABEL_NAME_t *pLabelName = (MAP_LABEL_NAME_t *)malloc(sizeof(MAP_LABEL_NAME_t)*map->item_size);
	BOOL isExist = FALSE;
	for (int i = 0;i < map->item_size;i++) {
		if (strict_check == TRUE) {
			isExist = FALSE;
			for (int j = 0;j < i;j++) {
				if (pLabelName[j].label == map->item[i].label) {
					isExist = TRUE;
					break;
				}
			}
			if (isExist == TRUE) {
				free(pLabelName);
				return FALSE;
			}
		}
		pLabelName[i].label = map->item[i].label;
		strcpy(pLabelName[i].name,map->item[i].name);
	}
	*label_to_name = pLabelName;
	*label_to_name_size = map->item_size;
	
	return TRUE;
}

BOOL IO_MapLabelToDisplayName(LabelMap *map, BOOL strict_check,
    							MAP_LABEL_DISPLAYNAME_t **label_to_display_name,int *size) 
{
	MAP_LABEL_DISPLAYNAME_t *pLabelName = (MAP_LABEL_DISPLAYNAME_t *)malloc(
									sizeof(MAP_LABEL_DISPLAYNAME_t)*map->item_size);
	BOOL isExist = FALSE;
	for (int i = 0;i < map->item_size;i++) {
		if (strict_check == TRUE) {
			isExist = FALSE;
			for (int j = 0;j < i;j++) {
				if (pLabelName[j].label == map->item[i].label) {
					isExist = TRUE;
					break;
				}
			}
			if (isExist == TRUE) {
				free(pLabelName);
				return FALSE;
			}
		}
		pLabelName[i].label = map->item[i].label;
		strcpy(pLabelName[i].name,map->item[i].name);
	}
	*label_to_display_name = pLabelName;
	*size = map->item_size;
	
	return TRUE;
}


BOOL IO_readLineFromFile(FILE *fp,char *line,int *len)
{
	int ch;
	int nread = -1;
	int cnt = 0;
	BOOL isEnd = FALSE;
	do {
		nread = fread(&ch,1,1,fp);
		if (nread <= 0) {
			isEnd = TRUE;
			break;
		}
		if (ch == '\n') {
			break;
		} else {
			line[cnt++] = ch;	
		}
	}while(nread > 0);

	line[cnt] = '\0';
	*len = cnt;
	
	return isEnd;
}


