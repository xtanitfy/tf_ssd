#ifndef __GEN_HEADFILE_H__
#define __GEN_HEADFILE_H__

#include "public.h"
#include "codec.h"

#define PARAMTER_FILE_NAME "parameter.h"

int GenHead_init();
int GenHead_writeHeadfileStart();
int GenHead_writeMessageTypedef(MESSAGE_t *pMessage);
int GenHead_writeOneMessage(MESSAGE_t *pMessage);
int GenHead_writeHeadfileEnd();

#endif