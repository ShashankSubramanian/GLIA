#include "SpectralOperators.h"

int64_t get_local_sizes_single(int *n, int *isize, int *istart, int *osize, int *ostart, MPI_Comm c_comm) {
  isize[0] = n[0];
  isize[1] = n[1];
  isize[2] = n[2];
  osize[0] = n[0];
  osize[1] = n[1];
  osize[2] = n[2]/2 + 1;
  istart[0] = 0;
  istart[1] = 0;
  istart[2] = 0;
  ostart[0] = 0;
  ostart[1] = 0;
  ostart[2] = 0;
  return osize[0]*osize[1]*osize[2]*2*sizeof(ScalarType);
}

void SpectralOperators::setup(int *n, int *isize, int *istart, int *osize, int *ostart, MPI_Comm c_comm) {
#ifdef CUDA
  // only single gpu here; multigpu deprecated now. Use accfft_gpu if needed.
  alloc_max_ = get_local_sizes_single(n, isize, istart, osize, ostart, c_comm);
#else
  alloc_max_ = fft_local_size_dft_r2c(n, isize, istart, osize, ostart, c_comm);
#endif
  memcpy(n_, n, 3 * sizeof(int));
  memcpy(isize_, isize, 3 * sizeof(int));
  memcpy(osize_, osize, 3 * sizeof(int));
  memcpy(istart_, istart, 3 * sizeof(int));
  memcpy(ostart_, ostart, 3 * sizeof(int));

#ifdef CUDA
  cufftResult cufft_status;

  cudaMalloc((void **)&x_hat_, alloc_max_);
  cudaMalloc((void **)&wx_hat_, alloc_max_);
  cudaMalloc((void **)&d1_ptr_, alloc_max_);

//  plan_ = fft_plan_dft_3d_r2c(n, d1_ptr_, (ScalarType *)x_hat_, c_comm, ACCFFT_MEASURE);
  plan_ = new fft_plan(alloc_max_); //no accfft; only single GPU support for now. multigpu is deprecated
  if (fft_mode_ == CUFFT) {
#ifdef SINGLE
    cufft_status = cufftPlan3d(&plan_r2c_, n[0], n[1], n[2], CUFFT_R2C);
    cufftCheckError(cufft_status);
    cufft_status = cufftPlan3d(&plan_c2r_, n[0], n[1], n[2], CUFFT_C2R);
    cufftCheckError(cufft_status);
#else
    cufft_status = cufftPlan3d(&plan_r2c_, n[0], n[1], n[2], CUFFT_D2Z);
    cufftCheckError(cufft_status);
    cufft_status = cufftPlan3d(&plan_c2r_, n[0], n[1], n[2], CUFFT_Z2D);
    cufftCheckError(cufft_status);
#endif
  }

  // define constants for the gpu
  initCudaConstants(isize, osize, istart, ostart, n);
  // spec-ops
  initSpecOpsCudaConstants(n, istart, ostart, isize, osize);

#else
  d1_ptr_ = (ScalarType *)accfft_alloc(alloc_max_);
  x_hat_ = (ComplexType *)accfft_alloc(alloc_max_);
  wx_hat_ = (ComplexType *)accfft_alloc(alloc_max_);
  plan_ = fft_plan_dft_3d_r2c(n, d1_ptr_, (ScalarType *)x_hat_, c_comm, ACCFFT_MEASURE);
#endif
}

void SpectralOperators::executeFFTR2C(ScalarType *f, ComplexType *f_hat) {
#ifdef CUDA
  cufftResult cufft_status;
  if (fft_mode_ == ACCFFT) {
    TU_assert(false, "ACCFFT is switched off for CUDA (only single GPU support)");
//    fft_execute_r2c(plan_, f, f_hat);
  } else {
    cufft_status = cufftExecuteR2C(plan_r2c_, (CufftScalarType *)f, (CufftComplexType *)f_hat);
    cufftCheckError(cufft_status);
    cudaDeviceSynchronize();
  }
#else
  fft_execute_r2c(plan_, f, f_hat);
#endif
}

void SpectralOperators::executeFFTC2R(ComplexType *f_hat, ScalarType *f) {
#ifdef CUDA
  cufftResult cufft_status;
  if (fft_mode_ == ACCFFT) {
    TU_assert(false, "ACCFFT is switched off for CUDA (only single GPU support)");
//    fft_execute_c2r(plan_, f_hat, f);
  } else {
    cufft_status = cufftExecuteC2R(plan_c2r_, (CufftComplexType *)f_hat, (CufftScalarType *)f);
    cufftCheckError(cufft_status);
    cudaDeviceSynchronize();
  }
#else
  fft_execute_c2r(plan_, f_hat, f);
#endif
}

PetscErrorCode SpectralOperators::computeGradient(Vec grad_x, Vec grad_y, Vec grad_z, Vec x, std::bitset<3> *pXYZ, double *timers) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ScalarType *grad_x_ptr, *grad_y_ptr, *grad_z_ptr, *x_ptr;
#ifdef CUDA
  //ierr = VecCUDAGetArrayReadWrite(grad_x, &grad_x_ptr); CHKERRQ(ierr);
  //ierr = VecCUDAGetArrayReadWrite(grad_y, &grad_y_ptr); CHKERRQ(ierr);
  //ierr = VecCUDAGetArrayReadWrite(grad_z, &grad_z_ptr); CHKERRQ(ierr);
  //ierr = VecCUDAGetArrayReadWrite(x, &x_ptr); CHKERRQ(ierr);
  ierr = VecCUDAGetArray(grad_x, &grad_x_ptr); CHKERRQ(ierr);
  ierr = VecCUDAGetArray(grad_y, &grad_y_ptr); CHKERRQ(ierr);
  ierr = VecCUDAGetArray(grad_z, &grad_z_ptr); CHKERRQ(ierr);
  ierr = VecCUDAGetArray(x, &x_ptr); CHKERRQ(ierr);

  if (fft_mode_ == ACCFFT) {
    TU_assert(false, "ACCFFT is switched off for CUDA (only single GPU support)");
//    accfftGrad(grad_x_ptr, grad_y_ptr, grad_z_ptr, x_ptr, plan_, pXYZ, timers);
  } else {
    cufftResult cufft_status;

    // compute forward transform
    cufft_status = cufftExecuteR2C(plan_r2c_, (CufftScalarType *)x_ptr, (CufftComplexType *)x_hat_);
    cufftCheckError(cufft_status);
    cudaDeviceSynchronize();

    // set the bits
    std::bitset<3> XYZ;
    if (pXYZ != NULL) {
      XYZ = *pXYZ;
    } else {
      XYZ[0] = 1;
      XYZ[1] = 1;
      XYZ[2] = 1;
    }

    if (XYZ[0]) {
      // compute x gradient
      multiplyXWaveNumberCuda((CudaComplexType *)wx_hat_, (CudaComplexType *)x_hat_, osize_);
      // backwards transform
      cufft_status = cufftExecuteC2R(plan_c2r_, (CufftComplexType *)wx_hat_, (CufftScalarType *)grad_x_ptr);
      cufftCheckError(cufft_status);
      cudaDeviceSynchronize();
    }

    if (XYZ[1]) {
      // compute y gradient
      multiplyYWaveNumberCuda((CudaComplexType *)wx_hat_, (CudaComplexType *)x_hat_, osize_);
      // backwards transform
      cufft_status = cufftExecuteC2R(plan_c2r_, (CufftComplexType *)wx_hat_, (CufftScalarType *)grad_y_ptr);
      cufftCheckError(cufft_status);
      cudaDeviceSynchronize();
    }

    if (XYZ[2]) {
      // compute z gradient
      multiplyZWaveNumberCuda((CudaComplexType *)wx_hat_, (CudaComplexType *)x_hat_, osize_);
      // backwards transform
      cufft_status = cufftExecuteC2R(plan_c2r_, (CufftComplexType *)wx_hat_, (CufftScalarType *)grad_z_ptr);
      cufftCheckError(cufft_status);
      cudaDeviceSynchronize();
    }
  }

  //ierr = VecCUDARestoreArrayReadWrite(grad_x, &grad_x_ptr); CHKERRQ(ierr);
  //ierr = VecCUDARestoreArrayReadWrite(grad_y, &grad_y_ptr); CHKERRQ(ierr);
  //ierr = VecCUDARestoreArrayReadWrite(grad_z, &grad_z_ptr); CHKERRQ(ierr);
  //ierr = VecCUDARestoreArrayReadWrite(x, &x_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(grad_x, &grad_x_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(grad_y, &grad_y_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(grad_z, &grad_z_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(x, &x_ptr); CHKERRQ(ierr);
#else
  ierr = VecGetArray(grad_x, &grad_x_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(grad_y, &grad_y_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(grad_z, &grad_z_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);

  accfftGrad(grad_x_ptr, grad_y_ptr, grad_z_ptr, x_ptr, plan_, pXYZ, timers);

  ierr = VecRestoreArray(grad_x, &grad_x_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(grad_y, &grad_y_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(grad_z, &grad_z_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
#endif
  PetscFunctionReturn(ierr);
}

PetscErrorCode SpectralOperators::computeDivergence(Vec div, Vec dx, Vec dy, Vec dz, double *timers) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ScalarType *div_ptr, *dx_ptr, *dy_ptr, *dz_ptr;
#ifdef CUDA
  ierr = VecCUDAGetArray(div, &div_ptr); CHKERRQ(ierr);
  ierr = VecCUDAGetArray(dx, &dx_ptr); CHKERRQ(ierr);
  ierr = VecCUDAGetArray(dy, &dy_ptr); CHKERRQ(ierr);
  ierr = VecCUDAGetArray(dz, &dz_ptr); CHKERRQ(ierr);

  if (fft_mode_ == ACCFFT) {
    TU_assert(false, "ACCFFT is switched off for CUDA (only single GPU support)");
//    accfftDiv(div_ptr, dx_ptr, dy_ptr, dz_ptr, plan_, timers);
  } else {
    cufftResult cufft_status;
    // cublas for axpy
    cublasStatus_t status;
    cublasHandle_t handle;
    // cublas for vec scale
    PetscCUBLASGetHandle(&handle);
    ScalarType alp = 1.;

    // compute forward transform for dx
    cufft_status = cufftExecuteR2C(plan_r2c_, (CufftScalarType *)dx_ptr, (CufftComplexType *)x_hat_);
    cufftCheckError(cufft_status);
    cudaDeviceSynchronize();

    multiplyXWaveNumberCuda((CudaComplexType *)wx_hat_, (CudaComplexType *)x_hat_, osize_);
    // backwards transform
    cufft_status = cufftExecuteC2R(plan_c2r_, (CufftComplexType *)wx_hat_, (CufftScalarType *)div_ptr);
    cufftCheckError(cufft_status);
    cudaDeviceSynchronize();

    // compute forward transform for dz
    cufft_status = cufftExecuteR2C(plan_r2c_, (CufftScalarType *)dz_ptr, (CufftComplexType *)x_hat_);
    cufftCheckError(cufft_status);
    cudaDeviceSynchronize();

    multiplyZWaveNumberCuda((CudaComplexType *)wx_hat_, (CudaComplexType *)x_hat_, osize_);
    // backwards transform
    cufft_status = cufftExecuteC2R(plan_c2r_, (CufftComplexType *)wx_hat_, (CufftScalarType *)d1_ptr_);
    cufftCheckError(cufft_status);
    cudaDeviceSynchronize();

    status = cublasAXPY(handle, isize_[0] * isize_[1] * isize_[2], &alp, d1_ptr_, 1, div_ptr, 1);
    cublasCheckError(status);
    cudaDeviceSynchronize();

    // compute forward transform for dy
    cufft_status = cufftExecuteR2C(plan_r2c_, (CufftScalarType *)dy_ptr, (CufftComplexType *)x_hat_);
    cufftCheckError(cufft_status);
    cudaDeviceSynchronize();

    multiplyYWaveNumberCuda((CudaComplexType *)wx_hat_, (CudaComplexType *)x_hat_, osize_);
    // backwards transform
    cufft_status = cufftExecuteC2R(plan_c2r_, (CufftComplexType *)wx_hat_, (CufftScalarType *)d1_ptr_);
    cufftCheckError(cufft_status);
    cudaDeviceSynchronize();

    status = cublasAXPY(handle, isize_[0] * isize_[1] * isize_[2], &alp, d1_ptr_, 1, div_ptr, 1);
    cublasCheckError(status);
    cudaDeviceSynchronize();
  }

  //ierr = VecCUDARestoreArrayReadWrite(div, &div_ptr); CHKERRQ(ierr);
  //ierr = VecCUDARestoreArrayReadWrite(dx, &dx_ptr); CHKERRQ(ierr);
  //ierr = VecCUDARestoreArrayReadWrite(dy, &dy_ptr); CHKERRQ(ierr);
  //ierr = VecCUDARestoreArrayReadWrite(dz, &dz_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(div, &div_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(dx, &dx_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(dy, &dy_ptr); CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(dz, &dz_ptr); CHKERRQ(ierr);
#else
  ierr = VecGetArray(div, &div_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(dx, &dx_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(dy, &dy_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(dz, &dz_ptr); CHKERRQ(ierr);

  accfftDiv(div_ptr, dx_ptr, dy_ptr, dz_ptr, plan_, timers);

  ierr = VecRestoreArray(div, &div_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(dx, &dx_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(dy, &dy_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(dz, &dz_ptr); CHKERRQ(ierr);
#endif
  PetscFunctionReturn(ierr);
}

// apply weierstrass smoother
PetscErrorCode SpectralOperators::weierstrassSmoother(Vec wc, Vec c, std::shared_ptr<Parameters> params, ScalarType sigma) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("spectral-smoother");
  std::array<double, 7> t = {0};

  double self_exec_time = -MPI_Wtime();

  if (sigma == 0) {
    PetscFunctionReturn(0);
  }

  ScalarType *wc_ptr, *c_ptr;

  ierr = vecGetArray(wc, &wc_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(c, &c_ptr); CHKERRQ(ierr);

  ierr = weierstrassSmoother(wc_ptr, c_ptr, params, sigma);

  ierr = vecRestoreArray(wc, &wc_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(c, &c_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();

  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();

  PetscFunctionReturn(ierr);
}

int SpectralOperators::weierstrassSmoother(ScalarType *Wc, ScalarType *c, std::shared_ptr<Parameters> params, ScalarType sigma) {
  MPI_Comm c_comm = params->grid_->c_comm_;
  int nprocs, procid;
  MPI_Comm_rank(c_comm, &procid);
  MPI_Comm_size(c_comm, &nprocs);

  int *N = params->grid_->n_;
  int *istart, *isize, *osize, *ostart;
  istart = params->grid_->istart_;
  ostart = params->grid_->ostart_;
  isize = params->grid_->isize_;
  osize = params->grid_->osize_;

  const int Nx = params->grid_->n_[0], Ny = params->grid_->n_[1], Nz = params->grid_->n_[2];
  const ScalarType pi = M_PI, twopi = 2.0 * pi, factor = 1.0 / (Nx * Ny * Nz);
  const ScalarType hx = twopi / Nx, hy = twopi / Ny, hz = twopi / Nz;
  fft_plan *plan = params->grid_->plan_;

  ScalarType sum_f_local = 0., sum_f = 0;
#ifdef CUDA
  // user define cuda call
  computeWeierstrassFilterCuda(d1_ptr_, &sum_f_local, sigma, isize);
#else
  ScalarType X, Y, Z, Xp, Yp, Zp;
  int64_t ptr;
  for (int i = 0; i < isize[0]; i++)
    for (int j = 0; j < isize[1]; j++)
      for (int k = 0; k < isize[2]; k++) {
        X = (istart[0] + i) * hx;
        Xp = X - twopi;
        Y = (istart[1] + j) * hy;
        Yp = Y - twopi;
        Z = (istart[2] + k) * hz;
        Zp = Z - twopi;
        ptr = i * isize[1] * isize[2] + j * isize[2] + k;
        d1_ptr_[ptr] = std::exp((-X * X - Y * Y - Z * Z) / sigma / sigma / 2.0) + std::exp((-Xp * Xp - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

        d1_ptr_[ptr] += std::exp((-Xp * Xp - Y * Y - Z * Z) / sigma / sigma / 2.0) + std::exp((-X * X - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

        d1_ptr_[ptr] += std::exp((-X * X - Y * Y - Zp * Zp) / sigma / sigma / 2.0) + std::exp((-Xp * Xp - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

        d1_ptr_[ptr] += std::exp((-Xp * Xp - Y * Y - Zp * Zp) / sigma / sigma / 2.0) + std::exp((-X * X - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

        if (d1_ptr_[ptr] != d1_ptr_[ptr]) d1_ptr_[ptr] = 0.;  // To avoid Nan
        sum_f_local += d1_ptr_[ptr];
      }
#endif

  MPI_Allreduce(&sum_f_local, &sum_f, 1, MPIType, MPI_SUM, MPI_COMM_WORLD);
  ScalarType normalize_factor = 1. / (sum_f * hx * hy * hz);

#ifdef CUDA
  cublasStatus_t status;
  cublasHandle_t handle;
  // cublas for vec scale
  PetscCUBLASGetHandle(&handle);
  status = cublasScale(handle, isize[0] * isize[1] * isize[2], &normalize_factor, d1_ptr_, 1);
  cublasCheckError(status);
#else
  for (int i = 0; i < isize[0] * isize[1] * isize[2]; i++) d1_ptr_[i] = d1_ptr_[i] * normalize_factor;
#endif

  /* Forward transform */
  executeFFTR2C(d1_ptr_, x_hat_);
  executeFFTR2C(c, wx_hat_);

// Perform the Hadamard Transform f_hat=f_hat.*c_hat
#ifdef CUDA
  ScalarType alp = factor * hx * hy * hz;
  hadamardComplexProductCuda((CudaComplexType *)x_hat_, (CudaComplexType *)wx_hat_, osize);
#ifdef SINGLE
  status = cublasCsscal(handle, osize[0] * osize[1] * osize[2], &alp, (CudaComplexType *)x_hat_, 1);
#else
  status = cublasZdscal(handle, osize[0] * osize[1] * osize[2], &alp, (CudaComplexType *)x_hat_, 1);
#endif
  cublasCheckError(status);
#else
  std::complex<ScalarType> *cf_hat = (std::complex<ScalarType> *)(ScalarType *)x_hat_;
  std::complex<ScalarType> *cc_hat = (std::complex<ScalarType> *)(ScalarType *)wx_hat_;
  for (int i = 0; i < osize[0] * osize[1] * osize[2]; i++) cf_hat[i] *= (cc_hat[i] * factor * hx * hy * hz);
#endif

  /* Backward transform */
  executeFFTC2R(x_hat_, Wc);

  return 0;
}

SpectralOperators::~SpectralOperators() {
  fft_free(x_hat_);
  fft_free(d1_ptr_);
  fft_free(wx_hat_);

#ifdef CUDA
  cufftDestroy(plan_r2c_);
  cufftDestroy(plan_c2r_);
  if (plan_ != nullptr) delete plan_;
#else
  accfft_destroy_plan(plan_);
  accfft_cleanup();
#endif
}

void accfftCreateComm(MPI_Comm in_comm, int *c_dims, MPI_Comm *c_comm) {
  int nprocs, procid;
  MPI_Comm_rank(in_comm, &procid);
  MPI_Comm_size(in_comm, &nprocs);
  std::stringstream ss;

  if (c_dims[0] * c_dims[1] != nprocs) {
    c_dims[0] = 0;
    c_dims[1] = 0;
    MPI_Dims_create(nprocs, 2, c_dims);
  }

  ss << " Creating distributed communication grid with dim: " << c_dims[0] << " x " << c_dims[1];
  tuMSGstd(ss.str()); ss.str(""); ss.clear();

  /* Create Cartesian Communicator */
  int period[2], reorder;
  int coord[2];
  period[0] = 0;
  period[1] = 0;
  reorder = 1;

  MPI_Cart_create(in_comm, 2, c_dims, period, reorder, c_comm);
}

// initialization routines are part of spectral operators because accfft controls the memory distribution
PetscErrorCode initializeGrid(int n, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  int N[3] = {n, n, n};
  int isize[3];
  int istart[3];
  int osize[3];
  int ostart[3];
  int c_dims[2] = {0};
  MPI_Comm c_comm;

#ifdef CUDA
#else
  accfft_init();
#endif
  accfftCreateComm(MPI_COMM_WORLD, c_dims, &c_comm);
  spec_ops->setup(N, isize, istart, osize, ostart, c_comm);
  int64_t alloc_max = spec_ops->alloc_max_;
  fft_plan *plan = spec_ops->plan_;
  params->createGrid(N, isize, osize, istart, ostart, plan, c_comm, c_dims);
  PetscFunctionReturn(ierr);
}

