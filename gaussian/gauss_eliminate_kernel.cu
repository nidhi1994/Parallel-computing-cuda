 /* Device code. */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel_divide(float *U, int i_gpu, int n)
{
     int x_dim=blockDim.x;
     int tid_x=threadIdx.x;
     int tid_y=threadIdx.y;
     int y_dim=blockDim.y;
     int i = i_gpu + (blockIdx.y * (x_dim * y_dim)) + ((tid_y * x_dim) + tid_x);
     int step_size = (x_dim * y_dim * gridDim.y);
     
    i++;
    while ( i < n)
    {
        U[(i_gpu *n) + i] /= U[(i_gpu * n) + i_gpu];
         i =i+ step_size;
    }
}

__global__ void gauss_eliminate_kernel_eliminate(float *U, int i_gpu, int n)
{
    __shared__ float partials[BLOCK_SIZE][BLOCK_SIZE];
    int tid_x=threadIdx.x;
    int tid_y=threadIdx.y;
    //int tid_z=threadIdx.z;
    int row_step =  (gridDim.y*blockDim.y);
    int col_step = blockDim.x ;
    int row_start = ((i_gpu+(blockIdx.y * blockDim.y)) + tid_y)  ;
    int col_start = (tid_x+(blockIdx.x * blockDim.x)) ;
    
     U[i_gpu * n + i_gpu] = 1;
    

    float temp;
    int i, j;
    row_start++;
    for (i = row_start; i < n; i = i + row_step)
    {
        temp = U[i_gpu+(i * n)];
     //   __syncthreads();
          j=col_start;
          if(j<n)
          {
             a:partials[threadIdx.x][threadIdx.y] = U[i_gpu * n + j];
                    U[i * n + j] -= __fmul_rn(temp, partials[threadIdx.x][threadIdx.y]);
                    j=j+col_step;
                if(j<n)
                 goto a;   

          }      
        
    }
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
