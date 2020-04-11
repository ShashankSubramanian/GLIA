#ifndef _SPECTRALOPERATORS_H
#define _SPECTRALOPERATORS_H

#include "Parameters.h"

class SpectralOperators {
 public:
  SpectralOperators(int fft_mode = ACCFFT) { fft_mode_ = fft_mode; }

  int fft_mode_;
  fft_plan *plan_;
  int n_[3];
  int isize_[3];
  int osize_[3];
  int istart_[3];
  int ostart_[3];

#ifdef CUDA
  cufftHandle plan_r2c_;
  cufftHandle plan_c2r_;
#endif
  int64_t alloc_max_;

  ComplexType *x_hat_, *wx_hat_;
  ScalarType *d1_ptr_;

  void setup(int *n, int *isize, int *istart, int *osize, int *ostart, MPI_Comm c_comm);
  void executeFFTR2C(ScalarType *f, ComplexType *f_hat);
  void executeFFTC2R(ComplexType *f_hat, ScalarType *f);

  PetscErrorCode computeGradient(Vec grad_x, Vec grad_y, Vec grad_z, Vec x, std::bitset<3> *pXYZ, double *timers);
  PetscErrorCode computeDivergence(Vec div, Vec dx, Vec dy, Vec dz, double *timers);

  PetscErrorCode weierstrassSmoother(Vec Wc, Vec c, std::shared_ptr<Parameters> params, ScalarType sigma);
  int weierstrassSmoother(ScalarType *Wc, ScalarType *c, std::shared_ptr<Parameters> params, ScalarType sigma);

  ~SpectralOperators();
};

#ifdef CUDA
void initSpecOpsCudaConstants(int *n, int *istart, int *ostart);
void multiplyXWaveNumberCuda(CudaComplexType *w_f, CudaComplexType *f, int *sz);
void multiplyYWaveNumberCuda(CudaComplexType *w_f, CudaComplexType *f, int *sz);
void multiplyZWaveNumberCuda(CudaComplexType *w_f, CudaComplexType *f, int *sz);
void computeWeierstrassFilterCuda(ScalarType *f, ScalarType *sum, ScalarType sigma, int *sz);
#endif

#endif
