#include <iostream>
#include <highgui.h>
#include <cv.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath> 

#define MASK_size 3


using namespace std;
using namespace cv;

__constant__ char MASKx[MASK_size * MASK_size];
__constant__ char MASKy[MASK_size * MASK_size];

__device__ unsigned char clamp(int v){
  if(v>255)
    return 255;
  else if(v<0)
    return 0;
    
  return v;
}

__global__ void KernelConvolutionBasic1(unsigned char *Img_in,unsigned char *Img_out,unsigned int Mask_Width,int rowImg,int colImg){
  
  unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
  unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

  int N_start_point_i = row - (Mask_Width/2);
  int N_start_point_j = col - (Mask_Width/2);

    int Pvalue=0;
    for (int i= 0;i<Mask_Width;i++) {
      for (int j= 0;j<Mask_Width;j++) {
        if ((N_start_point_i+i >= 0 && N_start_point_i + i < colImg)&& (N_start_point_j+j >= 0 && N_start_point_j + j < rowImg)) {
          Pvalue+=Img_in[(N_start_point_i+i)*rowImg+(N_start_point_j+j)]*MASKx[i*Mask_Width+j];
        }

      }
  }
    Img_out[row*rowImg+col]=clamp(Pvalue);
}

__global__ void KernelConvolutionBasic2(unsigned char *Img_in,unsigned char *Img_out,unsigned int Mask_Width,int rowImg,int colImg){
  
  unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
  unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

  int N_start_point_i = row - (Mask_Width/2);
  int N_start_point_j = col - (Mask_Width/2);

    int Pvalue=0;
    for (int i= 0;i<Mask_Width;i++) {
      for (int j= 0;j<Mask_Width;j++) {
        if ((N_start_point_i+i >= 0 && N_start_point_i + i < colImg)&& (N_start_point_j+j >= 0 && N_start_point_j + j < rowImg)) {
          Pvalue+=Img_in[(N_start_point_i+i)*rowImg+(N_start_point_j+j)]*MASKy[i*Mask_Width+j];
        }

      }
  }
    Img_out[row*rowImg+col]=clamp(Pvalue);
}

__global__ void KernelNormalConvolution(unsigned char *Img_inx, unsigned char *Img_iny, unsigned char *Img_out, int rowImg,int colImg){
  
  unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
  unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;
  
	int cont = 0;
  
   if ((col < rowImg) && (row < colImg)){
			cont = sqrt((float)((Img_inx[row * rowImg + col] * Img_inx[row * rowImg + col]) + (Img_iny[row * rowImg + col] * Img_iny[row * rowImg + col]))); 
	}
	Img_out[row*rowImg+col]=clamp(cont);
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
  Mat grad;
  Mat gradiente_x;
  Mat gradiente_y;
  Mat imagenGris;
  Mat abs_grad_x, abs_grad_y;
  
//Leer imagen y separar memoria
  imagen = imread("inputs/img1.jpg", 0);
  Size s = imagen.size();
  int row=s.width;
  int col=s.height;
  char Mx[9] = {-1,0,1,-2,0,2,-1,0,1};
  char My[9] = {-1,-2,-1,0,0,0,1,2,1};
  imagenGris.create(col,row,CV_8UC1);
  
  int sizeMx= sizeof(char)*(MASK_size*MASK_size);
  int sizeMy= sizeof(char)*(MASK_size*MASK_size);
  int size = sizeof(unsigned char)*row*col;
  unsigned char *img=(unsigned char*)malloc(size);
  unsigned char *img_out=(unsigned char*)malloc(size);
  unsigned char *img_out_final=(unsigned char*)malloc(size);
  img=imagen.data;

//Secuencial 
  for(int i =1; i<21; i++){
    
    secuencial = clock();
    Sobel( imagen, gradiente_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( gradiente_x, abs_grad_x );
    
    Sobel( imagen, gradiente_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( gradiente_y, abs_grad_y );
    
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    
    printf("%.8f\n", (clock()-secuencial)/(double)CLOCKS_PER_SEC);
    
   	//imwrite("./outputs/1088302627.png",grad);
    
  }
    printf("\n");
//Paralelo
  for(int j =1; j<21; j++){
    
    float blocksize=32;
    dim3 dimBlock((int)blocksize,(int)blocksize,1);
    dim3 dimGrid(ceil(row/blocksize),ceil(col/blocksize),1);
    
    
    unsigned char *d_img;
    unsigned char *d_img_outx;
    unsigned char *d_img_outy;
    unsigned char *d_img_final;
     
    //char *d_Mx;
    //char *d_My;
    
    cudaMalloc((void**)&d_img,size);
    cudaMalloc((void**)&d_img_final,size);
    
    cudaMalloc((void**)&d_img_outx,size);
   // cudaMalloc((void**)&d_Mx,sizeMx);
    
    cudaMalloc((void**)&d_img_outy,size);
    //cudaMalloc((void**)&d_My,sizeMy);

    paralelo = clock();
    
    cudaMemcpyToSymbol(MASKx,Mx,sizeMx);
    cudaMemcpyToSymbol(MASKy,My,sizeMy);
    
    //cudaMemcpy(d_Mx,Mx,sizeMx,cudaMemcpyHostToDevice);
    //cudaMemcpy(d_My,My,sizeMy,cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_img,img,size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_final,img,size, cudaMemcpyHostToDevice);

  	KernelConvolutionBasic1<<<dimGrid,dimBlock>>>(d_img,d_img_outx,MASK_size,row,col);
    KernelConvolutionBasic2<<<dimGrid,dimBlock>>>(d_img,d_img_outy,MASK_size,row,col);
    
    KernelNormalConvolution<<<dimGrid,dimBlock>>>(d_img_outx, d_img_outy,d_img_final,row,col);
    
    
    cudaDeviceSynchronize();
    
    
    
    cudaMemcpy(img_out_final,d_img_final,size,cudaMemcpyDeviceToHost);
  
    printf("%.8f\n", (clock()-paralelo)/(double)CLOCKS_PER_SEC);

    imagenGris.data = img_out_final;
    imwrite("./outputs/1088302627.png",imagenGris);

    cudaFree(d_img);
    cudaFree(d_img_final);
    cudaFree(d_img_outx);
    cudaFree(d_img_outy);
    //cudaFree(d_Mx);
   // cudaFree(d_My);
  }  
  return 0; 
}
