#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define sizex 4
#define sizey 4
#define blockSize 1024



__global__ void maxAdd(int *A, int *B, int *C){
	
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		int j = blockIdx.y * blockDim.y + threadIdx.y; 
  
		if( i < sizex){
			if(j < sizey){
        
				C[i*sizey+j]=A[i*sizey+j] + B[i*sizey+j];
        printf("  %d  ", C[i*sizey+j]);	
			}
     
		}
}

int matrixAdd(int *A, int *B, int *C){
	
	int dimGrid = 0;
	int size;
	size = sizex * sizey;
	int sizexy = size*sizeof(int);
	int *d_A, *d_B, *d_C;
	
	cudaMalloc((void**)&d_A, sizexy);
	cudaMalloc((void**)&d_B, sizexy);
	cudaMalloc((void**)&d_C, sizexy);
  
	dimGrid = (int)ceil((float)size / blockSize);
	printf("%d\n", dimGrid);
	
	clock_t t;
  t = clock();
	
	cudaMemcpy(d_A,A, sizexy, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B, sizexy, cudaMemcpyHostToDevice);
	
	maxAdd<<< dimGrid , blockSize >>>(d_A ,d_B, d_C);
	
	cudaMemcpy(C, d_C, sizexy, cudaMemcpyDeviceToHost);

	printf( "Número de segundos transcurridos desde el comienzo del programa: %.8f s\n", (clock()-t)/(double)CLOCKS_PER_SEC );
		
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
  return 0;
}




int main (int argc, char *argv[]){
		
	int *a, *b, *c;
	int i, size;
	
	size = sizex * sizey;	
	
	a = (int*)malloc(size*sizeof(int));
	b = (int*)malloc(size*sizeof(int));
	c = (int*)malloc(size*sizeof(int));
	
	for(i = 0; i < size ; i++){
	
			a[i] = 2;
			b[i] = 2;		
	}
	
  matrixAdd(a, b, c);	
  free(a);
  free(b);
  free(c);
  
return 0;
}
