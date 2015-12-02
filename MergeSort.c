#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <time.h>
#define SIZE 600000

#define min(a, b) (a < b ? a : b)

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.

//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}


void mergesortgpu(long* data, int size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
  	
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays
   cudaMalloc((void **)&D_data,size*sizeof(long));
   cudaMalloc((void **) &D_swp, size * sizeof(long));
    
    // Copy from our input list into the first array
   cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice);
   
    //
    // Copy the thread / block info to the GPU as well
    //
   cudaMalloc((void**) &D_threads, sizeof(dim3));
   cudaMalloc((void**) &D_blocks, sizeof(dim3));

   
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice) ;

  	

    long* A =(long*)malloc( size*sizeof(long));

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted

    //

                    clock_t paralelo;
                    paralelo= clock();
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(D_data, D_swp, size, width, slices, D_threads, D_blocks);

        
    }

    //
    // Get the list back from the GPU
    //
    cudaMemcpy(A,D_swp, size * sizeof(long), cudaMemcpyDeviceToHost);
   
    printf("tiempo en secuencial t: %.8f\n", (clock()-paralelo)/(double)CLOCKS_PER_SEC);
    
    // Free the GPU memory
    cudaFree(A);
       
}

/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
void merge(int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;

    /* create temp arrays */
    int L[n1], R[n2];

    /* Copy data to temp arrays L[] and R[] */
    for(i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for(j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

/* l is for left index and r is right index of the sub-array
  of arr to be sorted */
void mergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        int m = l+(r-l)/2; //Same as (l+r)/2, but avoids overflow for large l and h
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);
        merge(arr, l, m, r);
    }
}


/* UITLITY FUNCTIONS */
/* Function to print an array */
void printArray(int A[], int size)
{
    int i;
    for (i=0; i < size; i++)
        printf("%d ", A[i]);
    printf("\n");
}

/* Driver program to test above functions */
int main()
{

	dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

   
     int size= SIZE;
  


   
    long* data;
    data = (long*)malloc( size*sizeof(long) );
    for(int j=0;j<size;j++){
        data[j]=rand() % 100 ;
    }
    
    

 	mergesortgpu(data, size, threadsPerBlock, blocksPerGrid);

    int i;
    int arr[SIZE];
    for (i = 0; i < SIZE; i++){
        arr[i] = rand()%100;
    }
    int arr_size = sizeof(arr)/sizeof(arr[0]);

    //printf("Given array is \n");
  
    //printArray(arr, arr_size);
    clock_t t;
	t=clock();
    mergeSort(arr, 0, arr_size - 1);
    printf("Tiempo paralelo\t: %.8f\n",(clock()-t)/(double)CLOCKS_PER_SEC);
    //printf("\nSorted array is \n");
    //printArray(arr, arr_size);
    return 0;
}
