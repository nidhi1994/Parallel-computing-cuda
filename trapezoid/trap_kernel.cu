/* Write GPU kernels to compete the functionality of estimating the integral via the trapezoidal rule. */ 

#define BLOCK 64
#define GRID 16384

__global__ void trap_kernel(float a, float b, int n, float h, float *Result_fromGPU)
{
    int tx = threadIdx.x;
    int bd = blockDim.x;
    int bi = blockIdx.x;

    int i;
    int j =1;
    float x,y;
    double partial_sum = 0.00;

     __shared__ float sum[BLOCK];

     // stride 1 grid at time in loop
    for (i = (bi* bd) + (tx + 1); i < n; i = i+ GRID)
    {
        x = (i * h);
        y = a + x;
        partial_sum  = partial_sum + (y + 1)/sqrt(pow(y, 2) + y + 1);
    }


    sum[tx] = partial_sum;

    // Reduce code from here
    int half;
    if(BLOCK % 2 ==0)
    {
        half = BLOCK/2;
    }
    else
    {
        half = ((BLOCK+1)/2);
    }

    while (half > 0) 
    {
    
        if (tx < half)
        {
            sum[tx] = sum[tx] + sum[tx+ j];
        }

        half = half/2;
    }


    if (tx)
        Result_fromGPU[bi] = sum[0];
       // atomicAdd(Result_fromGPU, sum[0]); -- doesnt work :( Phatom bug)
}
