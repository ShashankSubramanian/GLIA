//cuda helpers
#ifndef _UTILSCUDA_H
#define _UTILSCUDA_H

#include <complex>
#include "cuda.h"
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"

void computeWeierstrassFilterCuda (double *f, double *s, double sigma, int *isize, int *istart, int *n);
void hadamardComplexProductCuda (cuDoubleComplex *y, cuDoubleComplex *x, double *alph, int *sz);


#endif