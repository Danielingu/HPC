#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>

#define sizex 4
#define sizey 4

int main (int argc, char *argv[]){
		
	int *a, *b, *c;
	int i, j, size;
	
	size = sizex * sizey;	
	
	a = (int*)malloc(size*sizeof(int));
	b = (int*)malloc(size*sizeof(int));
	c = (int*)malloc(size*sizeof(int));
	
	for(i = 0; i < size ; i++){
	
      a[i] = 2;
			b[i] = 2;		
		}
	
	
	clock_t t;
	t = clock();
	
	
	for(i = 0; i < sizex ; i++){
		for(j = 0; j < sizey ; j++){
				c[i*sizey+j]=a[i*sizey+j] + b[i*sizey+j];		
      	printf("  %d  ", c[i*sizey+j]);
		}
    printf("\n");
	}
	
	printf( "Número de segundos transcurridos desde el comienzo del programa: %.8f s\n", (clock()-t)/(double)CLOCKS_PER_SEC );
	
  free(a);
  free(b);
  free(c);
return 0;

}
