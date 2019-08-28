#include "SpectralOperators.h"

void SpectralOperators::setup (int *n, int *isize, int *istart, int *osize, int *ostart, MPI_Comm c_comm) {
    alloc_max_ = fft_local_size_dft_r2c (n, isize, istart, osize, ostart, c_comm);
    isize_ = isize;
    istart_ = istart;
    osize_ = osize;
    ostart_ = ostart;
    n_ = n;

    #ifdef CUDA
        cufftResult cufft_status;

        cudaMalloc ((void**) &x_hat_, alloc_max_);
        cudaMalloc ((void**) &wx_hat_, alloc_max_);
        cudaMalloc ((void**) &d1_ptr_, alloc_max_);

        cudaMalloc ((void**) &c_hat_, alloc_max_);

        plan_ = fft_plan_dft_3d_r2c (n, d1_ptr_, (ScalarType*) x_hat_, c_comm, ACCFFT_MEASURE);
        if (fft_mode_ == CUFFT) {
            #ifdef SINGLE
                cufft_status = cufftPlan3d (&plan_r2c_, n[0], n[1], n[2], CUFFT_R2C);   cufftCheckError (cufft_status);
                cufft_status = cufftPlan3d (&plan_c2r_, n[0], n[1], n[2], CUFFT_C2R);   cufftCheckError (cufft_status);
            #else
                cufft_status = cufftPlan3d (&plan_r2c_, n[0], n[1], n[2], CUFFT_D2Z);   cufftCheckError (cufft_status);
                cufft_status = cufftPlan3d (&plan_c2r_, n[0], n[1], n[2], CUFFT_Z2D);   cufftCheckError (cufft_status);
            #endif
        }

        // define constants for the gpu
        initCudaConstants (isize, osize, istart, ostart, n);
    #else
        d1_ptr_ = (ScalarType*) accfft_alloc (alloc_max_);
        x_hat_ = (ComplexType*) accfft_alloc (alloc_max_);
        wx_hat_ = (ComplexType*) accfft_alloc (alloc_max_);

        c_hat_ = (ComplexType*) accfft_alloc (alloc_max_);

        plan_ = fft_plan_dft_3d_r2c (n, d1_ptr_, (ScalarType*) x_hat_, c_comm, ACCFFT_MEASURE);        
    #endif
}

void SpectralOperators::executeFFTR2C (ScalarType *f, ComplexType *f_hat) {
    #ifdef CUDA
        cufftResult cufft_status;
        if (fft_mode_ == ACCFFT)
            fft_execute_r2c (plan_, f, f_hat);
        else {
            cufft_status = cufftExecuteR2C (plan_r2c_, (CufftScalarType*) f, (CufftComplexType*) f_hat);
            cufftCheckError (cufft_status);
            cudaDeviceSynchronize ();
        }
    #else
        fft_execute_r2c (plan_, f, f_hat);
    #endif
}

void SpectralOperators::executeFFTC2R (ComplexType *f_hat, ScalarType *f) {
    #ifdef CUDA
        cufftResult cufft_status;
        if (fft_mode_ == ACCFFT)
            fft_execute_c2r (plan_, f_hat, f);
        else {
            cufft_status = cufftExecuteC2R (plan_c2r_, (CufftComplexType*) f_hat, (CufftScalarType*) f);
            cufftCheckError (cufft_status);
            cudaDeviceSynchronize ();
        }
    #else
        fft_execute_c2r (plan_, f_hat, f);
    #endif
}

PetscErrorCode SpectralOperators::computeGradient (Vec grad_x, Vec grad_y, Vec grad_z, Vec x, std::bitset<3> *pXYZ, double *timers) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ScalarType *grad_x_ptr, *grad_y_ptr, *grad_z_ptr, *x_ptr;
    #ifdef CUDA
        ierr = VecCUDAGetArrayReadWrite (grad_x, &grad_x_ptr);
        ierr = VecCUDAGetArrayReadWrite (grad_y, &grad_y_ptr);
        ierr = VecCUDAGetArrayReadWrite (grad_z, &grad_z_ptr);
        ierr = VecCUDAGetArrayReadWrite (x, &x_ptr);

        if (fft_mode_ == ACCFFT)
            accfftGrad (grad_x_ptr, grad_y_ptr, grad_z_ptr, x_ptr, plan_, pXYZ, timers);
        else {
            cufftResult cufft_status;

            // compute forward transform
            cufft_status = cufftExecuteR2C (plan_r2c_, (CufftScalarType*) x_ptr, (CufftComplexType*) x_hat_);
            cufftCheckError (cufft_status);
            cudaDeviceSynchronize ();

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
                multiplyXWaveNumberCuda ((CudaComplexType*) wx_hat_, (CudaComplexType*) x_hat_, osize_);
                // backwards transform
                cufft_status = cufftExecuteC2R (plan_c2r_, (CufftComplexType*) wx_hat_, (CufftScalarType*) grad_x_ptr);
                cufftCheckError (cufft_status);
                cudaDeviceSynchronize ();
            }

            if (XYZ[1]) {
                // compute y gradient
                multiplyYWaveNumberCuda ((CudaComplexType*) wx_hat_, (CudaComplexType*) x_hat_, osize_);
                // backwards transform
                cufft_status = cufftExecuteC2R (plan_c2r_, (CufftComplexType*) wx_hat_, (CufftScalarType*) grad_y_ptr);
                cufftCheckError (cufft_status);
                cudaDeviceSynchronize ();
            }

            if (XYZ[2]) {
                // compute z gradient
                multiplyZWaveNumberCuda ((CudaComplexType*) wx_hat_, (CudaComplexType*) x_hat_, osize_);
                // backwards transform
                cufft_status = cufftExecuteC2R (plan_c2r_, (CufftComplexType*) wx_hat_, (CufftScalarType*) grad_z_ptr);
                cufftCheckError (cufft_status);
                cudaDeviceSynchronize ();
            }
        }

        ierr = VecCUDARestoreArrayReadWrite (grad_x, &grad_x_ptr);
        ierr = VecCUDARestoreArrayReadWrite (grad_y, &grad_y_ptr);
        ierr = VecCUDARestoreArrayReadWrite (grad_z, &grad_z_ptr);
        ierr = VecCUDARestoreArrayReadWrite (x, &x_ptr);
    #else
        ierr = VecGetArray (grad_x, &grad_x_ptr);
        ierr = VecGetArray (grad_y, &grad_y_ptr);
        ierr = VecGetArray (grad_z, &grad_z_ptr);
        ierr = VecGetArray (x, &x_ptr);

        accfftGrad (grad_x_ptr, grad_y_ptr, grad_z_ptr, x_ptr, plan_, pXYZ, timers);

        ierr = VecRestoreArray (grad_x, &grad_x_ptr);
        ierr = VecRestoreArray (grad_y, &grad_y_ptr);
        ierr = VecRestoreArray (grad_z, &grad_z_ptr);
        ierr = VecRestoreArray (x, &x_ptr);
    #endif
    PetscFunctionReturn (0);
}

PetscErrorCode SpectralOperators::computeDivergence (Vec div, Vec dx, Vec dy, Vec dz, double *timers) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ScalarType *div_ptr, *dx_ptr, *dy_ptr, *dz_ptr;
    #ifdef CUDA
        ierr = VecCUDAGetArrayReadWrite (div, &div_ptr);
        ierr = VecCUDAGetArrayReadWrite (dx, &dx_ptr);
        ierr = VecCUDAGetArrayReadWrite (dy, &dy_ptr);
        ierr = VecCUDAGetArrayReadWrite (dz, &dz_ptr);

        if (fft_mode_ == ACCFFT) {
            accfftDiv (div_ptr, dx_ptr, dy_ptr, dz_ptr, plan_, timers);
        } else {
            cufftResult cufft_status;
            // cublas for axpy
            cublasStatus_t status;
            cublasHandle_t handle;
            // cublas for vec scale
            PetscCUBLASGetHandle (&handle);
            ScalarType alp = 1.;

            // compute forward transform for dx
            cufft_status = cufftExecuteR2C (plan_r2c_, (CufftScalarType*) dx_ptr, (CufftComplexType*) x_hat_);
            cufftCheckError (cufft_status);
            cudaDeviceSynchronize ();

            multiplyXWaveNumberCuda ((CudaComplexType*) wx_hat_, (CudaComplexType*) x_hat_, osize_);
            // backwards transform
            cufft_status = cufftExecuteC2R (plan_c2r_, (CufftComplexType*) wx_hat_, (CufftScalarType*) div_ptr);
            cufftCheckError (cufft_status);
            cudaDeviceSynchronize ();

             // compute forward transform for dz
            cufft_status = cufftExecuteR2C (plan_r2c_, (CufftScalarType*) dz_ptr, (CufftComplexType*) x_hat_);
            cufftCheckError (cufft_status);
            cudaDeviceSynchronize ();

            multiplyZWaveNumberCuda ((CudaComplexType*) wx_hat_, (CudaComplexType*) x_hat_, osize_);
            // backwards transform
            cufft_status = cufftExecuteC2R (plan_c2r_, (CufftComplexType*) wx_hat_, (CufftScalarType*) d1_ptr_);
            cufftCheckError (cufft_status);
            cudaDeviceSynchronize ();

            status = cublasAXPY (handle, isize_[0] * isize_[1] * isize_[2], &alp, d1_ptr_, 1, div_ptr, 1);
            cublasCheckError (status);
            cudaDeviceSynchronize ();

            // compute forward transform for dy
            cufft_status = cufftExecuteR2C (plan_r2c_, (CufftScalarType*) dy_ptr, (CufftComplexType*) x_hat_);
            cufftCheckError (cufft_status);
            cudaDeviceSynchronize ();

            multiplyYWaveNumberCuda ((CudaComplexType*) wx_hat_, (CudaComplexType*) x_hat_, osize_);
            // backwards transform
            cufft_status = cufftExecuteC2R (plan_c2r_, (CufftComplexType*) wx_hat_, (CufftScalarType*) d1_ptr_);
            cufftCheckError (cufft_status);
            cudaDeviceSynchronize ();

            status = cublasAXPY (handle, isize_[0] * isize_[1] * isize_[2], &alp, d1_ptr_, 1, div_ptr, 1);
            cublasCheckError (status);
            cudaDeviceSynchronize ();
        }

        ierr = VecCUDARestoreArrayReadWrite (div, &div_ptr);
        ierr = VecCUDARestoreArrayReadWrite (dx, &dx_ptr);
        ierr = VecCUDARestoreArrayReadWrite (dy, &dy_ptr);
        ierr = VecCUDARestoreArrayReadWrite (dz, &dz_ptr);
    #else
        ierr = VecGetArray (div, &div_ptr);
        ierr = VecGetArray (dx, &dx_ptr);
        ierr = VecGetArray (dy, &dy_ptr);
        ierr = VecGetArray (dz, &dz_ptr);

        accfftDiv (div_ptr, dx_ptr, dy_ptr, dz_ptr, plan_, timers);

        ierr = VecRestoreArray (div, &div_ptr);
        ierr = VecRestoreArray (dx, &dx_ptr);
        ierr = VecRestoreArray (dy, &dy_ptr);
        ierr = VecRestoreArray (dz, &dz_ptr);
    #endif
    PetscFunctionReturn (0);
}

// apply weierstrass smoother
PetscErrorCode SpectralOperators::weierstrassSmoother (Vec wc, Vec c, std::shared_ptr<NMisc> n_misc, ScalarType sigma) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("spectral-smoother");
    std::array<double, 7> t = {0};

    double self_exec_time = -MPI_Wtime ();

    ScalarType *wc_ptr, *c_ptr;

    ierr = vecGetArray (wc, &wc_ptr);
    ierr = vecGetArray (c, &c_ptr);

    ierr = weierstrassSmoother (wc_ptr, c_ptr, n_misc, sigma);

    ierr = vecRestoreArray (wc, &wc_ptr);
    ierr = vecRestoreArray (c, &c_ptr);

    self_exec_time += MPI_Wtime ();

    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();

    PetscFunctionReturn (0);
}

int SpectralOperators::weierstrassSmoother (ScalarType* Wc, ScalarType *c, std::shared_ptr<NMisc> n_misc, ScalarType sigma) {
    MPI_Comm c_comm = n_misc->c_comm_;
    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);

    int *N = n_misc->n_;
    int *istart, *isize, *osize, *ostart;
    istart = n_misc->istart_;
    ostart = n_misc->ostart_;
    isize = n_misc->isize_;
    osize = n_misc->osize_;
    
    const int Nx = n_misc->n_[0], Ny = n_misc->n_[1], Nz = n_misc->n_[2];
    const ScalarType pi = M_PI, twopi = 2.0 * pi, factor = 1.0 / (Nx * Ny * Nz);
    const ScalarType hx = twopi / Nx, hy = twopi / Ny, hz = twopi / Nz;
    fft_plan *plan = n_misc->plan_;

    ScalarType sum_f_local = 0., sum_f = 0;
    #ifdef CUDA
        // user define cuda call
        computeWeierstrassFilterCuda (d1_ptr_, &sum_f_local, sigma, isize);
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
                    d1_ptr_[ptr] = std::exp((-X * X - Y * Y - Z * Z) / sigma / sigma / 2.0)
                            + std::exp((-Xp * Xp - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

                    d1_ptr_[ptr] += std::exp((-Xp * Xp - Y * Y - Z * Z) / sigma / sigma / 2.0)
                            + std::exp((-X * X - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

                    d1_ptr_[ptr] += std::exp((-X * X - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
                            + std::exp((-Xp * Xp - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

                    d1_ptr_[ptr] += std::exp((-Xp * Xp - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
                            + std::exp((-X * X - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

                    if (d1_ptr_[ptr] != d1_ptr_[ptr])
                        d1_ptr_[ptr] = 0.; // To avoid Nan
                    sum_f_local += d1_ptr_[ptr];
                }
    #endif
    

    MPI_Allreduce(&sum_f_local, &sum_f, 1, MPIType, MPI_SUM, MPI_COMM_WORLD);
    ScalarType normalize_factor = 1. / (sum_f * hx * hy * hz);

    #ifdef CUDA
        cublasStatus_t status;
        cublasHandle_t handle;
        // cublas for vec scale
        PetscCUBLASGetHandle (&handle);
        status = cublasScale (handle, isize[0] * isize[1] * isize[2], &normalize_factor, d1_ptr_, 1);
        cublasCheckError (status);
    #else
        for (int i = 0; i < isize[0] * isize[1] * isize[2]; i++)
            d1_ptr_[i] = d1_ptr_[i] * normalize_factor;
    #endif

    /* Forward transform */
    executeFFTR2C (d1_ptr_, x_hat_);
    executeFFTR2C (c, c_hat_);    

    // Perform the Hadamard Transform f_hat=f_hat.*c_hat
    #ifdef CUDA
        ScalarType alp = factor * hx * hy * hz;
        hadamardComplexProductCuda ((CudaComplexType*) x_hat_, (CudaComplexType*) c_hat_, osize);
        #ifdef SINGLE
        status = cublasCsscal (handle, osize[0] * osize[1] * osize[2], &alp, (CudaComplexType*) x_hat_, 1);
        #else
        status = cublasZdscal (handle, osize[0] * osize[1] * osize[2], &alp, (CudaComplexType*) x_hat_, 1);
        #endif
        cublasCheckError (status);
    #else   
        std::complex<ScalarType>* cf_hat = (std::complex<ScalarType>*) (ScalarType*) x_hat_;
        std::complex<ScalarType>* cc_hat = (std::complex<ScalarType>*) (ScalarType*) c_hat_;
        for (int i = 0; i < osize[0] * osize[1] * osize[2]; i++)
            cf_hat[i] *= (cc_hat[i] * factor * hx * hy * hz);
    #endif

    /* Backward transform */
    executeFFTC2R (x_hat_, Wc);

    return 0;
}

SpectralOperators::~SpectralOperators () {
    accfft_destroy_plan (plan_);

    fft_free (x_hat_);
    fft_free (d1_ptr_);
    fft_free (wx_hat_);
    fft_free (c_hat_);

    #ifdef CUDA
        cufftDestroy (plan_r2c_);
        cufftDestroy (plan_c2r_);
    #endif

	accfft_cleanup ();
}