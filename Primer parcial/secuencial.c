#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 



////////////////////////////////////////////////////////////
//Funcion que  multiplica las matrices de forma secuencial//
////////////////////////////////////////////////////////////

void multMatrixsequential(int *A, int *B, int *C, int n, int m, int o){
	for (int i=0; i<n; i++){
		for (int j=0; j<o; j++){
			int sum=0;
			for (int k=0; k<m; k++){
				sum += A[m*i+k]*B[o*k+j];
			}
			C[o*i+j] = sum;
		}
	}
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
	int n=128;
	int m=256;
	int o=128;
    
	int *A=(int *) malloc(n*m*sizeof(int));	
	int *B=(int *) malloc(m*o*sizeof(int));	
	int *C=(int *) malloc(n*o*sizeof(int));

	FillMatrix(A, n, m,10);	
	printf("==============Tiempos==============\n");
	clock_t t;
	t=clock();
	multMatrixsequential(A,B,C,n,m,o);
	printf("Multiplicacion secuencial\t\t\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);	
	free(A);
	free(B);
	free(C);	
	
	return 0;
