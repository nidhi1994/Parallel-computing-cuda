
#ifndef _2DCONVOLUTION_KERNEL_H_
#define _2DCONVOLUTION_KERNEL_H_

#include <stdio.h>
#include "2Dconvolution.h"

__global__ void ConvolutionKernel(Matrix M,Matrix N, Matrix P)
{

			int hN=N.height,wN=N.width;
			int i=blockIdx.y*blockDim.y+threadIdx.y;
			int j=blockIdx.x*blockDim.x+threadIdx.x;	
			double sum = 0;
			// check the start and end values of m and n to prevent overrunning the 
			//  matrix edges
			unsigned int mbegin = (i < 2)? 2 - i : 0;
			unsigned int mend = (i > (hN - 3))?
									hN - i + 2 : 5;
			unsigned int nbegin = (j < 2)? 2 - j : 0;
			unsigned int nend = (j > (wN - 3))?
									(wN-j) + 2 : 5;
			// overlay M over N centered at element (i,j).  For each 
			//  overlapping element, multiply the two and accumulate
			for(unsigned int m = mbegin; m < mend; ++m) {
				for(unsigned int n = nbegin; n < nend; n++) {
					sum +=M.elements[m * 5 + n] * 
							N.elements[wN*(i + m - 2) + (j+n - 2)];
				}
			}
			// store the result
			P.elements[i*wN + j] = (float)sum;        	
}


#endif // #ifndef _2DCONVOLUTION_KERNEL_H_
