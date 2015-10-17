#include <iostream>
#include <stdio.h>
#include <stdlib.h>


void sumS(int *A,int N ,int *r){
  int value=0;
	for(int i=0;i<N;i++)
		value+=A[i];
  *r=value;
}

void fillVec(int *A,int N,int a){
	for(int  i = 0; i <N; i++ )
        A[i] = a;
}

void printVec(int *A,int N){
	for(int i = 0; i <N; i++)
        printf("%d ",A[i]);

	printf("\n");
}

int main(){
  int N = 1000000;
  int s;
  int bytes=(N)*sizeof(int);
  int *A=(int*)malloc(bytes);
  int *R=(int*)malloc(bytes);
  fillVec(A,N,1);
  fillVec(R,N,0);


//Secuencial
  clock_t start = clock();
  sumS(A,N,&s);
  clock_t end= clock();
  double elapsed_seconds=end-start;
  printf("Tiempo Secuencial: %lf\n", (elapsed_seconds / CLOCKS_PER_SEC));



  free(A);
  free(R);

  return 0;
}
