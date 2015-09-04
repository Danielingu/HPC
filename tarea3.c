#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#include <math.h>
#define v 512
#define blockSize 1024






__global__ void vecAdd(int *A, int *B, int *C,int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < n)
    C[i] = A[i] + B[i];
}


int vectorAdd(int *A, int *B, int *C, int n){
  int dimGrid = 0;
  int size = n*sizeof(int);
  int *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, size);
  cudaMalloc((void**)&d_B, size);
  cudaMalloc((void**)&d_C, size);
  dimGrid = (int)ceil((float)v / blockSize);
  printf("%d\n", dimGrid);
  clock_t t;
    t = clock();
  cudaMemcpy(d_A,A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B,B, size, cudaMemcpyHostToDevice);
  vecAdd<<< dimGrid , blockSize >>>(d_A ,d_B, d_C, n);
  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
  printf( "Número de segundos transcurridos desde el comienzo del programa: %.8f s\n", (clock()-t)/(double)CLOCKS_PER_SEC );
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}

int main (int argc, char *argv[]){
  int *A,*B,*C,n;
  A = (int*)malloc(v*sizeof(int));
	B = (int*)malloc(v*sizeof(int));
	C = (int*)malloc(v*sizeof(int));
  n = v;
  
  for(int i = 0; i < n; i++){
    A[i] = i+1;
    B[i] = i+1;    
  }  
  vectorAdd(A,B,C,n);
  free(A);
  free(B);
  free(C);
  return 0;
}
