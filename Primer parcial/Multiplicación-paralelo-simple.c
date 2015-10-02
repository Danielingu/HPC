#include <stdio.h>
#include <stdlib.h>
#include <time.h>



//////////////////////////////////////////////////////
//Funcion que  multiplica palarelamente las matrices//
//////////////////////////////////////////////////////

__global__ void matMultParallel(int* A, int* B, int* C,  int n, int m, int o){
	int row= blockIdx.y*blockDim.y + threadIdx.y;
	int col= blockIdx.x*blockDim.x + threadIdx.x;

	if ((row < n) && (col < o)){
		int cont = 0;		
		for(int i = 0 ; i < m ; i++ ){
			cont += A[row * m + i] * B[i * o + col];
		}
		C[row * o + col] = cont ;
	}	
}

int matrixMult( int *A, int *B, int *C,int n, int m ,int o, int flag){  
	int sizeA = n*m*sizeof(int);
	int sizeB = m*o*sizeof(int);
	int sizeC = n*o*sizeof(int);
  
	int *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A,	sizeA);
	cudaMalloc((void **)&d_B, 	sizeB);
	cudaMalloc((void **)&d_C,	sizeC);
	
	clock_t t;
	t=clock();
	//Copio los datos al device
	cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
  
	dim3 dimBlock(32.0,32.0); //mayor cantidad de hilos por bloque
	dim3 dimGrid(ceil((float)n/dimBlock.x),ceil((float)n/dimBlock.y));
	// Ejecuto el Kernel (del dispositivo)
	
	matMultParallel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C,n,m,o);
	cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
	printf("Multiplicacion paralela sin tiling\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

void FillMatrix(int *matrix,int n,int m, int r){
	for(int i=0; i < n*m; i++){
		matrix[i]=rand()% r;
	}
}

int main(){
	int n=5;
	int m=10;
	int o=5;
    
	int *A=(int *) malloc(n*m*sizeof(int));	
	int *B=(int *) malloc(m*o*sizeof(int));	
	int *C=(int *) malloc(n*o*sizeof(int));

	FillMatrix(A, n, m,10);
	FillMatrix(B, m, o,10);
	printf("==============Tiempos==============\n");	
	matrixMult(A,B,C,n,m,o,0);	
	free(A);
	free(B);
	free(C);
	return 0;

}



