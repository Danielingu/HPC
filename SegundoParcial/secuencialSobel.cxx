#include <iostream>
#include <highgui.h>
#include <cv.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

__device__ unsigned char conv(int v){
  if(v>255)
    return 255;
  else if(v<0)
    return 0;
    
  return v;
}

__global__ void KernelConvolutionBasic(unsigned char *Img_in, char *M,unsigned char *Img_out,int Mask_Width,int rowImg,int colImg){
  
  unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
  unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

  int N_start_point_i = row - (Mask_Width/2);
  int N_start_point_j = col - (Mask_Width/2);

    int Pvalue=0;
    for (int ii= 0;ii<Mask_Width;ii++) {
      for (int jj= 0;jj<Mask_Width;jj++) {
        if ((N_start_point_i+ii >= 0 && N_start_point_i + ii < colImg)&& (N_start_point_j+jj >= 0 && N_start_point_j + jj < rowImg)) {
          Pvalue+=Img_in[(N_start_point_i+ii)*rowImg+(N_start_point_j+jj)]*M[ii*Mask_Width+jj];
        }

      }
  }
    Img_out[row*rowImg+col]=conv(Pvalue);
}

int main(){

//Constantes usadas en la función Sobel
  int scale = 1;
  int delta = 0;
  int ddepth = CV_8UC1;
  
//Reloj
	clock_t secuencial;
	clock_t paralelo;
 
  
//Imagenes	
  Mat imagen;
  Mat gradiente_x;
  Mat gradiente_y;
  Mat imagenGris;
  
//Leer imagen y separar memoria
  imagen = imread("inputs/img1.jpg");
  Size s = imagen.size();
  int row=s.width;
  int col=s.height;
  char M[9] = {-1,0,1,-2,0,2,-1,0,1};
  imagenGris.create(col,row,CV_8UC1);
  
//Separo memoria para las imagenes en el host
  int sizeM= sizeof(unsigned char)*9;
  int size = sizeof(unsigned char)*row*col;
  unsigned char *img=(unsigned char*)malloc(size);
  unsigned char *img_out=(unsigned char*)malloc(size);

  img=imagen.data;

//Secuencial 

  secuencial = clock();
  Sobel( imagen, gradiente_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  printf("Tiempo secuencial: %.8f\n", (clock()-secuencial)/(double)CLOCKS_PER_SEC);
  imwrite("./outputs/1088302627.png",gradiente_x);

//Paralelo

  float blocksize=32;
  dim3 dimBlock((int)blocksize,(int)blocksize,1);
  dim3 dimGrid(ceil(row/blocksize),ceil(col/blocksize),1);
  unsigned char *d_img;
  unsigned char *d_img_out;
  char *d_M;
  cudaMalloc((void**)&d_img,size);
  cudaMalloc((void**)&d_img_out,size);
  cudaMalloc((void**)&d_M,sizeM);

  paralelo = clock();
  cudaMemcpy(d_M,M,sizeM,cudaMemcpyHostToDevice);
  cudaMemcpy(d_img,img,size, cudaMemcpyHostToDevice);
  KernelConvolutionBasic<<<dimGrid,dimBlock>>>(d_img,d_M,d_img_out,3,row,col);
  cudaDeviceSynchronize();
  cudaMemcpy(img_out,d_img_out,size,cudaMemcpyDeviceToHost);
  printf("Tiempo paralelo: %.8f\n", (clock()-paralelo)/(double)CLOCKS_PER_SEC);

  imagenGris.data = img_out;
  //imwrite("./outputs/1088302627.png",imagenGris);

  cudaFree(d_img);
  cudaFree(d_img_out);
  cudaFree(d_M);
  
  return 0; 
}
