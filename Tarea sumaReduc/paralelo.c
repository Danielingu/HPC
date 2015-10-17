#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define BLCK_SIZE 32

__global__ void reduce(int *g_idata, int *g_odata,int num_vec) {
	__shared__ int sharedBlock[BLCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  	if(i<num_vec)
    	sharedBlock[tid] = g_idata[i];
 	else
      sharedBlock[tid]=0;
    __syncthreads();
    for(unsigned int s=blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s){
            sharedBlock[tid] += sharedBlock[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sharedBlock[0];
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
  int N=1000000;

  int bytes=(N)*sizeof(int);
  int *A=(int*)malloc(bytes);
  int *R=(int*)malloc(bytes);


  fillVec(A,N,1);
  fillVec(R,N,0);




//Paralelo
  int *d_A=(int*)malloc(bytes);
  int *d_R=(int*)malloc(bytes);
  cudaMalloc((void**)&d_A,bytes);
  cudaMalloc((void**)&d_R,bytes);

  clock_t start2 = clock();
  cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_R, R, bytes, cudaMemcpyHostToDevice);
  float blocksize=BLCK_SIZE;

  int i=N;
  while(i>1){
  	dim3 dimBlock(BLCK_SIZE,1,1);
    int grid=ceil(i/blocksize);
    dim3 dimGrid(grid,1,1);

 		reduce<<<dimGrid,dimBlock>>>(d_A,d_R,i);
		cudaDeviceSynchronize();
  	cudaMemcpy(d_A, d_R, bytes, cudaMemcpyDeviceToDevice);
    i=ceil(i/blocksize);
  }
  cudaMemcpy(R, d_R, bytes, cudaMemcpyDeviceToHost);

  clock_t end2= clock();
  double elapsed_seconds2=end2-start2;
  printf("Paralelo %lf\n", (elapsed_seconds2 / CLOCKS_PER_SEC));



  cudaFree(d_A);
  cudaFree(d_R);
  return 0;
}
