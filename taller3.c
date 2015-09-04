#include <stdio.h>
#include <malloc.h>
#include <time.h> 
#define size 10000000

int main (int argc, char *argv[]){
	
	int *a, *b, *c;
	int i;	
	a = (int*)malloc(size*sizeof(int));
	b = (int*)malloc(size*sizeof(int));
	c = (int*)malloc(size*sizeof(int));
	
	for(i = 0; i < size ; i++){		
		a[i] = i+1;	
		b[i] = i+1;
	}
	
	clock_t t;
	t = clock();
	for(i = 0; i < size ; i++){
		c[i] = a[i] + b[i];				
	}	
	printf( "Número de segundos transcurridos desde el comienzo del programa: %.8f s\n", (clock()-t)/(double)CLOCKS_PER_SEC );
	
		
  return 0;

}
