#include <stdio.h>

int main(int argc,char **argv)
{
    if (argc != 2) {
        printf("argv[1]:need test binary filename\n");
        return -1;
    }
    
    FILE *fp = fopen(argv[1],"rb");
    if (fp == NULL) {
        printf("fopen failed!\n");
        return 1;
    }
    
    double tmp;
    while((fread(&tmp,sizeof(tmp),1,fp) == 1)) {
        printf("read:%.10f\n",tmp);
    }    
    fclose(fp);
    
    return 0;
}
