#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

// includes, kernels
#include "trap_kernel.cu"


#define LEFT_ENDPOINT 10
#define RIGHT_ENDPOINT 1005
#define NUM_TRAPEZOIDS 100000000

double compute_on_device(float, float, int, float);
extern "C" double compute_gold(float, float, int, float);
float ts1,ts2;

int 
main(void) 
{
    int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);

    struct timeval start, stop;
    gettimeofday(&start, NULL);
	       double reference = compute_gold(a, b, n, h);
    gettimeofday(&stop, NULL);
    ts1 = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
    printf("CPU Execution Time = %fs. \n", ts1 );

    printf("Reference solution computed on the CPU = %f \n", reference);

	/* Write this function to complete the trapezoidal on the GPU. */
	double gpu_result = compute_on_device(a, b, n, h);
	printf("Solution computed on the GPU = %f \n", gpu_result);
} 


/* Complete this function to perform the trapezoidal rule on the GPU. */
double 
compute_on_device(float a, float b, int n, float h)
{

    int Block_num = ceil((GRID/BLOCK));
    size_t fs = sizeof(float);
   //size_t is = sizeof(int);
    double sum;
    // one for each block
    float *partial_result = (float *)malloc(GRID/BLOCK * fs);
    float *Result_fromGPU;
    cudaMalloc((void**)&Result_fromGPU, Block_num * fs);


    dim3 thread_block(BLOCK, 1, 1);
    dim3 grid(Block_num, 1);

    struct timeval start, stop;
    gettimeofday(&start, NULL);
            trap_kernel <<< grid, thread_block >>> (a, b, n, h, Result_fromGPU);
            cudaThreadSynchronize();
    gettimeofday(&stop, NULL);

    cudaMemcpy(partial_result, Result_fromGPU, Block_num *fs, cudaMemcpyDeviceToHost);

    ts2 = (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000);
    printf("GPU Execution time = %fs. \n", ts2);
    printf("SPEEDUP = %fs. \n", ts1/ts2);

    sum = ((b) + 1)/sqrt(pow(b, 2) + (b + 1)) + ((a) + 1)/sqrt(pow(a, 2) + (a + 1))/2;

    int i = 0;
    while(i < Block_num)
    {
        sum = sum + partial_result[i];
        i++;
    }

    cudaFree(Result_fromGPU);
    free(partial_result);

    return (h*(sum));
}
