#ifndef __UTIL_H__
#define __UTIL_H__
#include <sys/time.h>
#include "public.h"
void CalTimeStart(void);
void CalTimeEnd(char *title);
unsigned long long GetAllTimes(void);

#endif