#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 32

//////////////////////////////////////////////////
//Funcion que  multiplica las matrices con tiled//
//////////////////////////////////////////////////

__global__ void matMultParallelTiled(int* A, int* B, int* C, int n, int m, int o){
  
	__shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float cont = 0;
	for (int i = 0; i < (m + TILE_WIDTH-1)/(TILE_WIDTH); ++i){
		if (i * TILE_WIDTH + tx < m && row < n){
			Mds[ty][tx] = A[row * m + i*TILE_WIDTH + tx];
		}else{
			Mds[ty][tx] = 0;
		}
		if (i*TILE_WIDTH + ty < m && col < o){
			Nds[ty][tx] = B[(i * TILE_WIDTH + ty) * o + col];
		}else{
			Nds[ty][tx] =0;
		}
		__syncthreads();
		for(int i = 0; i < TILE_WIDTH; ++i){
			cont += Mds[ty][i] * Nds[i][tx];
		}
		__syncthreads();
	}
	if (row < n && col < o){
		C[row * o + col] = cont;
	}
}


///////////////////////////////////
//Funcion que tiene codigo cuda c//
///////////////////////////////////

int matrixMult( int *A, int *B, int *C,int n, int m ,int o){
  
	int sizeA=n*m*sizeof(int);
	int sizeB=m*o*sizeof(int);
	int sizeC=n*o*sizeof(int);
  
	int *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A,	sizeA);
	cudaMalloc((void **)&d_B, sizeB);
	cudaMalloc((void **)&d_C,	sizeC);
	
	clock_t t;
	t=clock();
	//Copio los datos al device
	cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
  
	dim3 dimBlock(32.0,32.0); //mayor cantidad de hilos por bloque
	dim3 dimGrid(ceil((float)n/dimBlock.x),ceil((float)n/dimBlock.y));
	// Ejecuto el Kernel (del dispositivo)
	
	matMultParallelTiled<<<dimGrid,dimBlock>>>(d_A, d_B, d_C,n,m,o);
  cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
	printf("Multiplicacion paralela con tiling\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
	
	
	
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

/////////////////////////////////////
//Funcion que  imprime las matrices//
/////////////////////////////////////

void PrintMatrix(int *matrix, int n, int m){
	printf("\n");
	for(int i=0; i< n; i++){
		for(int j=0 ; j< m;j++){
			printf("%i ",matrix[i*m+j]);
		}
		printf("\n");
	}
}

//////////////////////////////////////////////////////////
//Funcion que  llena las matrices con valores aleatorios//
//////////////////////////////////////////////////////////

void FillMatrix(int *matrix,int n,int m, int r){
	for(int i=0; i < n*m; i++){
		matrix[i]=rand()% r;
	}
}

//////////////////////////
//Funcion main principal//
//////////////////////////

int main(){
	int n=512;
	int m=1024;
	int o=512;
    
	int *A=(int *) malloc(n*m*sizeof(int));
	
	int *B=(int *) malloc(m*o*sizeof(int));
	
	int *C=(int *) malloc(n*o*sizeof(int));

	FillMatrix(A, n, m,10);
	FillMatrix(B, m, o,10);
	printf("==============Tiempos==============\n");	
	matrixMult(A,B,C,n,m,o);	
	
	
	free(A);
	free(B);
	free(C);	
	
	return 0;

}