#include "DiffCoef.h"

DiffCoef::DiffCoef (std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops) : spec_ops_ (spec_ops),
  k_scale_(1E-2)
, k_gm_wm_ratio_(1.0 / 5.0)
, k_glm_wm_ratio_(3.0 / 5.0)
, smooth_flag_(0)
, filter_avg_ (0.0) {
    PetscErrorCode ierr;
    ierr = VecCreate (PETSC_COMM_WORLD, &kxx_);
    ierr = VecSetSizes (kxx_, n_misc->n_local_, n_misc->n_global_);
    ierr = setupVec (kxx_);

    ierr = VecDuplicate (kxx_, &kxy_);
    ierr = VecDuplicate (kxx_, &kxz_);
    ierr = VecDuplicate (kxx_, &kyy_);
    ierr = VecDuplicate (kxx_, &kyz_);
    ierr = VecDuplicate (kxx_, &kzz_);

    // create 8 work vectors (will be pointed to tumor work vectors, thus no memory handling here)
    temp_ = new Vec[7];
    #ifdef CUDA 
      cudaMalloc ((void**) &temp_accfft_, n_misc->accfft_alloc_max_);
      cudaMalloc ((void**) &work_cuda_, 7 * sizeof(ScalarType));
    #else 
      temp_accfft_ = (ScalarType *) accfft_alloc (n_misc->accfft_alloc_max_);
    #endif

    ierr = VecSet (kxx_ , 0);
    ierr = VecSet (kxy_ , 0);
    ierr = VecSet (kxz_ , 0);
    ierr = VecSet (kyy_ , 0);
    ierr = VecSet (kyz_ , 0);
    ierr = VecSet (kzz_ , 0);

    kxx_avg_ = kxy_avg_ = kxz_avg_ = kyy_avg_ = kyz_avg_ = kzz_avg_ = 0.0;
}

PetscErrorCode DiffCoef::setWorkVecs(Vec * workvecs) {
  PetscErrorCode ierr;
  for (int i = 0; i < 7; ++i){
    temp_[i] = workvecs[i];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DiffCoef::updateIsotropicCoefficients (ScalarType k1, ScalarType k2, ScalarType k3, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  /*       k_1 = dk_dm_wm  = k_scale * 1;                     WM              */
  /*       k_2 = dk_dm_gm  = k_scale * k_gm_wm_ratio_;        GM              */
  /*       k_3 = dk_dm_glm = k_scale * k_glm_wm_ratio_;       GLM             */
  // compute new ratios
  k_scale_        = k1;
  k_gm_wm_ratio_  = (n_misc->nk_ == 1) ? n_misc->k_gm_wm_ratio_ : k2 / k1;    // if we want to invert for just one parameter (a.k.a diffusivity in WM), then
                                                                              // provide user with option to control the diffusivity in others from n_misc
  k_glm_wm_ratio_ = (n_misc->nk_ == 1) ? n_misc->k_glm_wm_ratio_ : k3 / k1;   // glm is always zero. TODO:  take it out in new iterations of the solver
  // and set the values
  setValues (k_scale_, k_gm_wm_ratio_, k_glm_wm_ratio_, mat_prop, n_misc);
  PetscFunctionReturn (0);
}

PetscErrorCode DiffCoef::setValues (ScalarType k_scale, ScalarType k_gm_wm_ratio, ScalarType k_glm_wm_ratio, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    k_scale_ = k_scale;
    k_gm_wm_ratio_  = k_gm_wm_ratio;
    k_glm_wm_ratio_ = k_glm_wm_ratio;
    n_misc->k_gm_wm_ratio_  = k_gm_wm_ratio_;    // update values in n_misc
    n_misc->k_glm_wm_ratio_ = k_glm_wm_ratio_;
    n_misc->k_              = k_scale_;

    ScalarType dk_dm_gm  = k_scale * k_gm_wm_ratio_;        //GM
    ScalarType dk_dm_wm  = k_scale;                         //WM
    ScalarType dk_dm_glm = k_scale * k_glm_wm_ratio_;       //GLM
    // if ratios <= 0, only diffuse in white matter
    dk_dm_gm   = (dk_dm_gm <= 0)  ? 0.0 : dk_dm_gm;
    dk_dm_glm  = (dk_dm_glm <= 0) ? 0.0 : dk_dm_glm;


    if (n_misc->testcase_ != BRAIN && n_misc->testcase_ != BRAINNEARMF && n_misc->testcase_ != BRAINFARMF) {
        ScalarType *kxx_ptr, *kyy_ptr, *kzz_ptr;
        ierr = VecGetArray (kxx_, &kxx_ptr);                     CHKERRQ (ierr);
        ierr = VecGetArray (kyy_, &kyy_ptr);                     CHKERRQ (ierr);
        ierr = VecGetArray (kzz_, &kzz_ptr);                     CHKERRQ (ierr);
        int64_t X, Y, Z, index;
        ScalarType amp;
        if (n_misc->testcase_ == CONSTCOEF)
            amp = 0.0;
        else if (n_misc->testcase_ == SINECOEF)
            amp = std::min ((ScalarType)1.0, k_scale_);

        ScalarType freq = 4.0;
        for (int x = 0; x < n_misc->isize_[0]; x++) {
            for (int y = 0; y < n_misc->isize_[1]; y++) {
                for (int z = 0; z < n_misc->isize_[2]; z++) {
                    X = n_misc->istart_[0] + x;
                    Y = n_misc->istart_[1] + y;
                    Z = n_misc->istart_[2] + z;

                    index = x * n_misc->isize_[1] * n_misc->isize_[2] + y * n_misc->isize_[2] + z;

                    kxx_ptr[index] = k_scale + amp * sin (freq * 2.0 * M_PI / n_misc->n_[0] * X)
                                                   * sin (freq * 2.0 * M_PI / n_misc->n_[1] * Y)
                                                   * sin (freq * 2.0 * M_PI / n_misc->n_[2] * Z);

                    // kxx_ptr[index] = 1E-2 + 0.5 * sin (2.0 * M_PI / n_misc->n_[0] * X) * cos (2.0 * M_PI / n_misc->n_[1] * Y)
                    //                             + 0.5 + 1E-3 * (1 - 0.5 * sin (2.0 * M_PI / n_misc->n_[0] * X) * cos (2.0 * M_PI / n_misc->n_[1] * Y)
                    //                                     + 0.5);

                    kyy_ptr[index] = kxx_ptr[index];
                    kzz_ptr[index] = kxx_ptr[index];
                }
            }
        }
        ierr = VecRestoreArray (kxx_, &kxx_ptr);                 CHKERRQ (ierr);
        ierr = VecRestoreArray (kyy_, &kyy_ptr);                 CHKERRQ (ierr);
        ierr = VecRestoreArray (kzz_, &kzz_ptr);                 CHKERRQ (ierr);
    }
    else {
        ierr = VecSet  (kxx_, 0.0);                              CHKERRQ (ierr);
        ierr = VecAXPY (kxx_, dk_dm_gm, mat_prop->gm_);          CHKERRQ (ierr);
        ierr = VecAXPY (kxx_, dk_dm_wm, mat_prop->wm_);          CHKERRQ (ierr);
        ierr = VecAXPY (kxx_, dk_dm_glm, mat_prop->glm_);        CHKERRQ (ierr);

        ierr = VecCopy (kxx_, kyy_);                             CHKERRQ (ierr);
        ierr = VecCopy (kxx_, kzz_);                             CHKERRQ (ierr);
    }

    if (n_misc->writeOutput_) {
        dataOut (kxx_, n_misc, "kxx.nc");
    }

    //Average diff coeff values for preconditioner for diffusion solve
    ierr = VecSum (kxx_, &kxx_avg_);                             CHKERRQ (ierr);
    ierr = VecSum (kxy_, &kxy_avg_);                             CHKERRQ (ierr);
    ierr = VecSum (kxz_, &kxz_avg_);                             CHKERRQ (ierr);
    ierr = VecSum (kyy_, &kyy_avg_);                             CHKERRQ (ierr);
    ierr = VecSum (kyz_, &kyz_avg_);                             CHKERRQ (ierr);
    ierr = VecSum (kzz_, &kzz_avg_);                             CHKERRQ (ierr);
    ierr = VecSum (mat_prop->filter_, &filter_avg_);             CHKERRQ (ierr);

    kxx_avg_ *= 1.0 / filter_avg_;
    kxy_avg_ *= 1.0 / filter_avg_;
    kxz_avg_ *= 1.0 / filter_avg_;
    kyy_avg_ *= 1.0 / filter_avg_;
    kyz_avg_ *= 1.0 / filter_avg_;
    kzz_avg_ *= 1.0 / filter_avg_;

    #ifdef CUDA
        cudaMemcpy (&work_cuda_[1], &kxx_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
        cudaMemcpy (&work_cuda_[2], &kxy_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
        cudaMemcpy (&work_cuda_[3], &kxz_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
        cudaMemcpy (&work_cuda_[4], &kyz_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
        cudaMemcpy (&work_cuda_[5], &kyy_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
        cudaMemcpy (&work_cuda_[6], &kzz_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
    #endif

    if (smooth_flag_) {
        ierr = this->smooth (n_misc); CHKERRQ (ierr);
    }

    PetscFunctionReturn(0);
}

PetscErrorCode DiffCoef::smooth (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    ScalarType sigma = 2.0 * M_PI / n_misc->n_[0];
    

    ierr = spec_ops_->weierstrassSmoother (kxx_, kxx_, n_misc, sigma);
    ierr = spec_ops_->weierstrassSmoother (kxy_, kxy_, n_misc, sigma);
    ierr = spec_ops_->weierstrassSmoother (kxz_, kxz_, n_misc, sigma);
    ierr = spec_ops_->weierstrassSmoother (kyy_, kyy_, n_misc, sigma);
    ierr = spec_ops_->weierstrassSmoother (kyz_, kyz_, n_misc, sigma);
    ierr = spec_ops_->weierstrassSmoother (kzz_, kzz_, n_misc, sigma);


    PetscFunctionReturn(0);
}

PetscErrorCode DiffCoef::applyK (Vec x, Vec y, Vec z) {
    PetscErrorCode ierr = 0;
    Event e ("tumor-diff-coeff-apply-K");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    for (int i = 1; i < 4; i++) {
        ierr = VecSet (temp_[i] , 0);                            CHKERRQ (ierr);
    }

    //X
    ierr = VecPointwiseMult (temp_[0], kxx_, x);                 CHKERRQ (ierr);
    ierr = VecAXPY (temp_[1], 1.0, temp_[0]);                    CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kxy_, y);                 CHKERRQ (ierr);
    ierr = VecAXPY (temp_[1], 1.0, temp_[0]);                    CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kxz_, z);                 CHKERRQ (ierr);
    ierr = VecAXPY (temp_[1], 1.0, temp_[0]);                    CHKERRQ (ierr);

    //Y
    ierr = VecPointwiseMult (temp_[0], kxy_, x);                 CHKERRQ (ierr);
    ierr = VecAXPY (temp_[2], 1.0, temp_[0]);                    CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kyy_, y);                 CHKERRQ (ierr);
    ierr = VecAXPY (temp_[2], 1.0, temp_[0]);                    CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kyz_, z);                 CHKERRQ (ierr);
    ierr = VecAXPY (temp_[2], 1.0, temp_[0]);                    CHKERRQ (ierr);

    //Z
    ierr = VecPointwiseMult (temp_[0], kxz_, x);                 CHKERRQ (ierr);
    ierr = VecAXPY (temp_[3], 1.0, temp_[0]);                    CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kyz_, y);                 CHKERRQ (ierr);
    ierr = VecAXPY (temp_[3], 1.0, temp_[0]);                    CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[0], kzz_, z);                 CHKERRQ (ierr);
    ierr = VecAXPY (temp_[3], 1.0, temp_[0]);                    CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();

    PetscFunctionReturn(0);
}

PetscErrorCode DiffCoef::applyD (Vec dc, Vec c) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-diff-coeff-apply-D");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    std::bitset<3> XYZ;
    XYZ[0] = 1;
    XYZ[1] = 1;
    XYZ[2] = 1;

    ierr = spec_ops_->computeGradient (temp_[4], temp_[5], temp_[6], c, &XYZ, t.data());
    ierr = applyK (temp_[4], temp_[5], temp_[6]);
    ierr = spec_ops_->computeDivergence (dc, temp_[1], temp_[2], temp_[3], t.data());

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn(0);
}

// TODO: only correct for isotropic diffusion
// TODO: assumes that geometry map has ordered components (WM, GM ,CSF, GLM)^T
PetscErrorCode DiffCoef::compute_dKdm_gradc_gradp(Vec x1, Vec x2, Vec x3, Vec x4, Vec c, Vec p, fft_plan *plan) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tumor-diff-coeff-apply-dKdm-gradc-gradp");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();

  std::bitset<3> XYZ; XYZ[0] = 1; XYZ[1] = 1; XYZ[2] = 1;
  // clear work fields
  for (int i = 1; i < 7; i++) {
      ierr = VecSet (temp_[i] , 0);                              CHKERRQ (ierr);
  }
  // compute gradient of state variable c(t)
  spec_ops_->computeGradient (temp_[1], temp_[2], temp_[3], c, &XYZ, t.data());
  // compute gradient of adjoint variable p(t)
  spec_ops_->computeGradient (temp_[4], temp_[5], temp_[6], p, &XYZ, t.data());
  // scalar product (grad c)^T grad \alpha
  ierr = VecPointwiseMult (temp_[0], temp_[1], temp_[4]);        CHKERRQ (ierr);  // c_x * \alpha_x
  ierr = VecPointwiseMult (temp_[1], temp_[2], temp_[5]);        CHKERRQ (ierr);  // c_y * \alpha_y
  ierr = VecAXPY (temp_[0], 1.0,  temp_[1]);                     CHKERRQ (ierr);
  ierr = VecPointwiseMult (temp_[1], temp_[3], temp_[6]);        CHKERRQ (ierr);  // c_z * \alpha_z
  ierr = VecAXPY (temp_[0], 1.0,  temp_[1]);                     CHKERRQ (ierr);

  ScalarType dk_dm_gm  = k_scale_ * k_gm_wm_ratio_;        //GM
  ScalarType dk_dm_wm  = k_scale_;                         //WM
  ScalarType dk_dm_glm = k_scale_ * k_glm_wm_ratio_;       //GLM
  // if ratios <= 0, only diffuse in white matter
  dk_dm_gm   = (dk_dm_gm <= 0)  ? 0.0 : dk_dm_gm;
  dk_dm_glm  = (dk_dm_glm <= 0) ? 0.0 : dk_dm_glm;
  // compute dK/dm * (grad c)^T grad \alpha ..
  // assumes that geometry map has ordered components (WM, GM ,CSF, GLM)^T
  if(x1 != nullptr) {                                                    // WM
    ierr = VecSet (x1 , 0.0);                                    CHKERRQ (ierr);
    ierr = VecAXPY (x1, dk_dm_wm, temp_[0]);                     CHKERRQ (ierr);
  }
  if(x2 != nullptr) {                                                    // GM
    ierr = VecSet (x2 , 0.0);                                    CHKERRQ (ierr);
    ierr = VecAXPY (x2, dk_dm_gm, temp_[0]);                     CHKERRQ (ierr);
  }
  if(x3 != nullptr) {                                                    // CSF
    ierr = VecSet (x3 , 0.0);                                    CHKERRQ (ierr);
    //ierr = VecAXPY (x3, 0, temp_[0]);                     CHKERRQ (ierr);
  }
  if(x4 != nullptr) {                                                    // GLM
    ierr = VecSet (x4 , 0.0);                                    CHKERRQ (ierr);
    ierr = VecAXPY (x4, dk_dm_glm, temp_[0]);                    CHKERRQ (ierr);
  }

  self_exec_time += MPI_Wtime();
  //accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings (t); e.stop ();
  PetscFunctionReturn(0);
}

DiffCoef::~DiffCoef () {
    PetscFunctionBegin;
    PetscErrorCode ierr;
    ierr = VecDestroy (&kxx_);
    ierr = VecDestroy (&kxy_);
    ierr = VecDestroy (&kxz_);
    ierr = VecDestroy (&kyy_);
    ierr = VecDestroy (&kyz_);
    ierr = VecDestroy (&kzz_);
    delete [] temp_;
    fft_free (temp_accfft_);
    #ifdef CUDA 
      fft_free (work_cuda_);
    #endif

}
