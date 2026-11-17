
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"
/* Extract the matcher from server.c first:
 * awk "/^\/\* ---- Stop word matcher/{p=1} /^\/\* GPU decode path selector/{p=0} p" server.c > tests/matcher.inc
 * gcc -fsanitize=address,undefined -I. tests/test_stop_matcher.c cJSON.c -lm */
#include "matcher.inc"
/* reference: full text, truncated at earliest stop word occurrence */
static int ref(const char*full,char**w,int n,char*out){
  int best=-1; for(int i=0;i<n;i++){char*p=strstr(full,w[i]); if(p&&(best<0||p-full<best))best=p-full;}
  if(best<0){strcpy(out,full);return 0;} memcpy(out,full,best);out[best]=0;return 1;}
int main(int argc,char**argv){
  srand(argc>1?atoi(argv[1]):1); int fails=0;
  const char*alpha="abc<>/ ";
  for(int it=0;it<200000;it++){
    int nw=1+rand()%2; char*w[2]; char wb[2][8];
    for(int i=0;i<nw;i++){int l=1+rand()%4;for(int j=0;j<l;j++)wb[i][j]=alpha[rand()%7];wb[i][l]=0;w[i]=wb[i];}
    cJSON*arr=cJSON_CreateArray(); for(int i=0;i<nw;i++)cJSON_AddItemToArray(arr,cJSON_CreateString(w[i]));
    stop_matcher_t sm; stop_matcher_init(&sm,arr);
    char full[256]="",sent[512]="",exp[256]; int np=1+rand()%8; int stopped=0;
    for(int k=0;k<np&&!stopped;k++){
      char piece[16]; int l=rand()%5; for(int j=0;j<l;j++)piece[j]=alpha[rand()%7]; piece[l]=0;
      strcat(full,piece);
      stop_action_t a; const char*s=stop_matcher_feed(&sm,piece,1,&a); strcat(sent,s);
      if(a==STOP_FULL)stopped=1;
    }
    if(!stopped){ /* end of generation: flush pending */
      strcat(sent,stop_matcher_flush(&sm));
    }
    int r=ref(full,w,nw,exp);
    if(strcmp(sent,exp)){ if(fails<8)printf("FAIL stop=[%s%s%s] full=\"%s\" sent=\"%s\" exp=\"%s\"\n",w[0],nw>1?"|":"",nw>1?w[1]:"",full,sent,exp); fails++;}
    stop_matcher_free(&sm); cJSON_Delete(arr);
  }
  printf("fails=%d\n",fails); return fails!=0;
}
