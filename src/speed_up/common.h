#ifndef __COMMON_H__
#define __COMMON_H__

#define ASSERT(x) \
    if (!(x)) { \
        printf("[%s %s %d] error\n",__FILE__,__FUNCTION__,__LINE__); \
        exit(-1); \
    }

#define VOS_MAX(x,y) ((x) > (y)?(x):(y))
#define VOS_MIN(x,y) ((x) > (y)?(y):(x))

#endif