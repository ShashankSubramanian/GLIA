#include "DiffCoef.h"

DiffCoef::DiffCoef(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops)
    : spec_ops_(spec_ops), k_scale_(1E-2), k_gm_wm_ratio_(1.0 / 5.0), k_glm_wm_ratio_(3.0 / 5.0), smooth_flag_(0), filter_avg_(0.0) {
  PetscErrorCode ierr;
  ierr = VecCreate(PETSC_COMM_WORLD, &kxx_);
  ierr = VecSetSizes(kxx_, params->grid_->nl_, params->grid_->ng_);
  ierr = setupVec(kxx_);

  ierr = VecDuplicate(kxx_, &kxy_);
  ierr = VecDuplicate(kxx_, &kxz_);
  ierr = VecDuplicate(kxx_, &kyy_);
  ierr = VecDuplicate(kxx_, &kyz_);
  ierr = VecDuplicate(kxx_, &kzz_);

  // create 8 work vectors (will be pointed to tumor work vectors, thus no memory handling here)
  temp_ = new Vec[8];
#ifdef CUDA
  initDiffCoefCudaConstants(params->grid_->n_, params->grid_->ostart_);
  cudaMalloc((void **)&temp_accfft_, params->grid_->accfft_alloc_max_);
  cudaMalloc((void **)&work_cuda_, 7 * sizeof(ScalarType));
#else
  temp_accfft_ = (ScalarType *)accfft_alloc(params->grid_->accfft_alloc_max_);
#endif

  ierr = VecSet(kxx_, 0);
  ierr = VecSet(kxy_, 0);
  ierr = VecSet(kxz_, 0);
  ierr = VecSet(kyy_, 0);
  ierr = VecSet(kyz_, 0);
  ierr = VecSet(kzz_, 0);

  kxx_avg_ = kxy_avg_ = kxz_avg_ = kyy_avg_ = kyz_avg_ = kzz_avg_ = 0.0;
}

PetscErrorCode DiffCoef::setWorkVecs(Vec *workvecs) {
  PetscErrorCode ierr;
  for (int i = 0; i < 8; ++i) {
    temp_[i] = workvecs[i];
  }
  PetscFunctionReturn(ierr);
}

PetscErrorCode DiffCoef::setSecondaryCoefficients(ScalarType k1, ScalarType k2, ScalarType k3, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  /* temp_[7] holds \sum m_i \times \k_tilde_i */
  ierr = VecCopy(mat_prop->wm_, temp_[7]); CHKERRQ(ierr);
  ierr = VecScale(temp_[7], k1); CHKERRQ(ierr);
  k2 = (params->tu_->nk_ == 1) ? params->tu_->k_gm_wm_ratio_ * k1 : k2;
  k3 = (params->tu_->nk_ == 1) ? params->tu_->k_glm_wm_ratio_ * k1 : k3;

  ierr = VecAXPY(temp_[7], k2, mat_prop->gm_); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[7], k3, mat_prop->csf_); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode DiffCoef::updateIsotropicCoefficients(ScalarType k1, ScalarType k2, ScalarType k3, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  /*       k_1 = dk_dm_wm  = k_scale * 1;                     WM              */
  /*       k_2 = dk_dm_gm  = k_scale * k_gm_wm_ratio_;        GM              */
  /*       k_3 = dk_dm_glm = k_scale * k_glm_wm_ratio_;       GLM             */
  // compute new ratios
  k_scale_ = k1;
  k_gm_wm_ratio_ = (params->tu_->nk_ == 1) ? params->tu_->k_gm_wm_ratio_ : k2 / k1;    // if we want to invert for just one parameter (a.k.a diffusivity in WM), then
                                                                                       // provide user with option to control the diffusivity in others from params
  k_glm_wm_ratio_ = (params->tu_->nk_ == 1) ? params->tu_->k_glm_wm_ratio_ : k3 / k1;  // glm is always zero. TODO:  take it out in new iterations of the solver
  // and set the values
  ierr = setValues(k_scale_, params->tu_->kf_, k_gm_wm_ratio_, k_glm_wm_ratio_, mat_prop, params);
  PetscFunctionReturn(ierr);
}

PetscErrorCode DiffCoef::setValues(ScalarType k_scale, ScalarType kf_scale, ScalarType k_gm_wm_ratio, ScalarType k_glm_wm_ratio, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  k_scale_ = k_scale;
  k_gm_wm_ratio_ = k_gm_wm_ratio;
  k_glm_wm_ratio_ = k_glm_wm_ratio;
  params->tu_->k_gm_wm_ratio_ = k_gm_wm_ratio_;  // update values in params
  params->tu_->k_glm_wm_ratio_ = k_glm_wm_ratio_;
  params->tu_->k_ = k_scale_;

  ScalarType dk_dm_gm = k_scale * k_gm_wm_ratio_;    // GM
  ScalarType dk_dm_wm = k_scale;                     // WM
  ScalarType dk_dm_glm = k_scale * k_glm_wm_ratio_;  // GLM
  // if ratios <= 0, only diffuse in white matter
  dk_dm_gm = (dk_dm_gm <= 0) ? 0.0 : dk_dm_gm;
  dk_dm_glm = (dk_dm_glm <= 0) ? 0.0 : dk_dm_glm;

  ierr = VecSet(kxx_, 0.0); CHKERRQ(ierr);
  ierr = VecSet(kxy_, 0.0); CHKERRQ(ierr);
  ierr = VecSet(kxz_, 0.0); CHKERRQ(ierr);
  ierr = VecSet(kyz_, 0.0); CHKERRQ(ierr);

  ierr = VecAXPY(kxx_, dk_dm_gm, mat_prop->gm_); CHKERRQ(ierr);
  ierr = VecAXPY(kxx_, dk_dm_wm, mat_prop->wm_); CHKERRQ(ierr);
  ierr = VecAXPY(kxx_, dk_dm_glm, mat_prop->csf_); CHKERRQ(ierr);
  
  ierr = VecCopy(kxx_, kyy_); CHKERRQ(ierr);
  ierr = VecCopy(kxx_, kzz_); CHKERRQ(ierr);
  ScalarType alpha = params->tu_->kf_ - params->tu_->k_;
  ierr = VecAXPY(kxx_, alpha, mat_prop->kfxx_);
  ierr = VecAXPY(kxy_, alpha, mat_prop->kfxy_);
  ierr = VecAXPY(kxz_, alpha, mat_prop->kfxz_);
  ierr = VecAXPY(kyy_, alpha, mat_prop->kfyy_);
  ierr = VecAXPY(kyz_, alpha, mat_prop->kfyz_);
  ierr = VecAXPY(kzz_, alpha, mat_prop->kfzz_);
  ierr = dataOut(kyz_, params, "kyz.nc"); CHKERRQ(ierr);    

  // Average diff coeff values for preconditioner for diffusion solve
  ierr = VecSum(kxx_, &kxx_avg_); CHKERRQ(ierr);
  ierr = VecSum(kxy_, &kxy_avg_); CHKERRQ(ierr);
  ierr = VecSum(kxz_, &kxz_avg_); CHKERRQ(ierr);
  ierr = VecSum(kyy_, &kyy_avg_); CHKERRQ(ierr);
  ierr = VecSum(kyz_, &kyz_avg_); CHKERRQ(ierr);
  ierr = VecSum(kzz_, &kzz_avg_); CHKERRQ(ierr);
  ierr = VecSum(mat_prop->filter_, &filter_avg_); CHKERRQ(ierr);

  kxx_avg_ *= 1.0 / filter_avg_;
  kxy_avg_ *= 1.0 / filter_avg_;
  kxz_avg_ *= 1.0 / filter_avg_;
  kyy_avg_ *= 1.0 / filter_avg_;
  kyz_avg_ *= 1.0 / filter_avg_;
  kzz_avg_ *= 1.0 / filter_avg_;

#ifdef CUDA
  cudaMemcpy(&work_cuda_[1], &kxx_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[2], &kxy_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[3], &kxz_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[4], &kyz_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[5], &kyy_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[6], &kzz_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
#endif

  if (smooth_flag_) {
    ierr = this->smooth(params); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode DiffCoef::setValuesSinusoidal(std::shared_ptr<Parameters> params, ScalarType scale) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *k_ptr;
  ierr = VecGetArray(kxx_, &k_ptr); CHKERRQ (ierr);
  ScalarType freq = 4;
  k_scale_ = scale;
  int64_t X, Y, Z, index;
  for (int x = 0; x < params->grid_->isize_[0]; x++) {
    for (int y = 0; y < params->grid_->isize_[1]; y++) {
      for (int z = 0; z < params->grid_->isize_[2]; z++) {
        X = params->grid_->istart_[0] + x;
        Y = params->grid_->istart_[1] + y;
        Z = params->grid_->istart_[2] + z;

        index = x * params->grid_->isize_[1] * params->grid_->isize_[2] + y * params->grid_->isize_[2] + z;

        k_ptr[index] = scale * (0.5 + 0.5 * sin (freq * 2.0 * M_PI / params->grid_->n_[0] * X)
                                  * sin (freq * 2.0 * M_PI / params->grid_->n_[1] * Y)
                                  * sin (freq * 2.0 * M_PI / params->grid_->n_[2] * Z));
      }
    }
  }
  ierr = VecRestoreArray(kxx_, &k_ptr); CHKERRQ (ierr);

  ierr = VecCopy(kxx_, kyy_); CHKERRQ(ierr);
  ierr = VecCopy(kxx_, kzz_); CHKERRQ(ierr);

  // Average diff coeff values for preconditioner for diffusion solve
  ierr = VecSum(kxx_, &kxx_avg_); CHKERRQ(ierr);
  ierr = VecSum(kxy_, &kxy_avg_); CHKERRQ(ierr);
  ierr = VecSum(kxz_, &kxz_avg_); CHKERRQ(ierr);
  ierr = VecSum(kyy_, &kyy_avg_); CHKERRQ(ierr);
  ierr = VecSum(kyz_, &kyz_avg_); CHKERRQ(ierr);
  ierr = VecSum(kzz_, &kzz_avg_); CHKERRQ(ierr);
  filter_avg_ = params->grid_->ng_;

  kxx_avg_ *= 1.0 / filter_avg_;
  kxy_avg_ *= 1.0 / filter_avg_;
  kxz_avg_ *= 1.0 / filter_avg_;
  kyy_avg_ *= 1.0 / filter_avg_;
  kyz_avg_ *= 1.0 / filter_avg_;
  kzz_avg_ *= 1.0 / filter_avg_;

#ifdef CUDA
  cudaMemcpy(&work_cuda_[1], &kxx_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[2], &kxy_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[3], &kxz_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[4], &kyz_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[5], &kyy_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(&work_cuda_[6], &kzz_avg_, sizeof(ScalarType), cudaMemcpyHostToDevice);
#endif

  PetscFunctionReturn(ierr);
}

PetscErrorCode DiffCoef::smooth(std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ScalarType sigma = 2.0 * M_PI / params->grid_->n_[0];

  ierr = spec_ops_->weierstrassSmoother(kxx_, kxx_, params, sigma);
  ierr = spec_ops_->weierstrassSmoother(kxy_, kxy_, params, sigma);
  ierr = spec_ops_->weierstrassSmoother(kxz_, kxz_, params, sigma);
  ierr = spec_ops_->weierstrassSmoother(kyy_, kyy_, params, sigma);
  ierr = spec_ops_->weierstrassSmoother(kyz_, kyz_, params, sigma);
  ierr = spec_ops_->weierstrassSmoother(kzz_, kzz_, params, sigma);

  PetscFunctionReturn(ierr);
}

PetscErrorCode DiffCoef::applyK(Vec x, Vec y, Vec z) {
  PetscErrorCode ierr = 0;
  Event e("tumor-diff-coeff-apply-K");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  for (int i = 1; i < 4; i++) {
    ierr = VecSet(temp_[i], 0); CHKERRQ(ierr);
  }

  // X
  ierr = VecPointwiseMult(temp_[0], kxx_, x); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[1], 1.0, temp_[0]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[0], kxy_, y); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[1], 1.0, temp_[0]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[0], kxz_, z); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[1], 1.0, temp_[0]); CHKERRQ(ierr);

  // Y
  ierr = VecPointwiseMult(temp_[0], kxy_, x); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[2], 1.0, temp_[0]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[0], kyy_, y); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[2], 1.0, temp_[0]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[0], kyz_, z); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[2], 1.0, temp_[0]); CHKERRQ(ierr);

  // Z
  ierr = VecPointwiseMult(temp_[0], kxz_, x); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[3], 1.0, temp_[0]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[0], kyz_, y); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[3], 1.0, temp_[0]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[0], kzz_, z); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[3], 1.0, temp_[0]); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();

  PetscFunctionReturn(ierr);
}

PetscErrorCode DiffCoef::applyD(Vec dc, Vec c) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-diff-coeff-apply-D");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;

  ierr = spec_ops_->computeGradient(temp_[4], temp_[5], temp_[6], c, &XYZ, t.data());
  ierr = applyK(temp_[4], temp_[5], temp_[6]);
  ierr = spec_ops_->computeDivergence(dc, temp_[1], temp_[2], temp_[3], t.data());

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode DiffCoef::applyDWithSecondaryCoeffs(Vec dc, Vec c) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-diff-coeff-apply-D");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;

  /* NOTE: temp_[7] is unused anywhere else within solveState - this is important else
     we have to reset temp_[7] everytime */

  spec_ops_->computeGradient(temp_[4], temp_[5], temp_[6], c, &XYZ, t.data());
  ierr = VecPointwiseMult(temp_[1], temp_[7], temp_[4]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[2], temp_[7], temp_[5]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[3], temp_[7], temp_[6]); CHKERRQ(ierr);
  spec_ops_->computeDivergence(dc, temp_[1], temp_[2], temp_[3], t.data());

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// TODO: only correct for isotropic diffusion
// TODO: assumes that geometry map has ordered components (WM, GM ,CSF, GLM)^T
PetscErrorCode DiffCoef::compute_dKdm_gradc_gradp(Vec x1, Vec x2, Vec x3, Vec x4, Vec c, Vec p, fft_plan *plan) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-diff-coeff-apply-dKdm-gradc-gradp");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  // clear work fields
  for (int i = 1; i < 7; i++) {
    ierr = VecSet(temp_[i], 0); CHKERRQ(ierr);
  }
  // compute gradient of state variable c(t)
  spec_ops_->computeGradient(temp_[1], temp_[2], temp_[3], c, &XYZ, t.data());
  // compute gradient of adjoint variable p(t)
  spec_ops_->computeGradient(temp_[4], temp_[5], temp_[6], p, &XYZ, t.data());
  // scalar product (grad c)^T grad \alpha
  ierr = VecPointwiseMult(temp_[0], temp_[1], temp_[4]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[1], temp_[2], temp_[5]); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[0], 1.0, temp_[1]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp_[1], temp_[3], temp_[6]); CHKERRQ(ierr);
  ierr = VecAXPY(temp_[0], 1.0, temp_[1]); CHKERRQ(ierr);

  ScalarType dk_dm_gm = k_scale_ * k_gm_wm_ratio_;    // GM
  ScalarType dk_dm_wm = k_scale_;                     // WM
  ScalarType dk_dm_glm = k_scale_ * k_glm_wm_ratio_;  // GLM
  // if ratios <= 0, only diffuse in white matter
  dk_dm_gm = (dk_dm_gm <= 0) ? 0.0 : dk_dm_gm;
  dk_dm_glm = (dk_dm_glm <= 0) ? 0.0 : dk_dm_glm;
  // compute dK/dm * (grad c)^T grad \alpha ..
  // assumes that geometry map has ordered components (WM, GM ,CSF, GLM)^T
  if (x1 != nullptr) {  // WM
    ierr = VecSet(x1, 0.0); CHKERRQ(ierr);
    ierr = VecAXPY(x1, dk_dm_wm, temp_[0]); CHKERRQ(ierr);
  }
  if (x2 != nullptr) {  // GM
    ierr = VecSet(x2, 0.0); CHKERRQ(ierr);
    ierr = VecAXPY(x2, dk_dm_gm, temp_[0]); CHKERRQ(ierr);
  }
  if (x3 != nullptr) {  // CSF
    ierr = VecSet(x3, 0.0); CHKERRQ(ierr);
    // ierr = VecAXPY (x3, 0, temp_[0]);                     CHKERRQ (ierr);
  }
  if (x4 != nullptr) {  // GLM
    ierr = VecSet(x4, 0.0); CHKERRQ(ierr);
    ierr = VecAXPY(x4, dk_dm_glm, temp_[0]); CHKERRQ(ierr);
  }

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

DiffCoef::~DiffCoef() {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ierr = VecDestroy(&kxx_);
  ierr = VecDestroy(&kxy_);
  ierr = VecDestroy(&kxz_);
  ierr = VecDestroy(&kyy_);
  ierr = VecDestroy(&kyz_);
  ierr = VecDestroy(&kzz_);
  delete[] temp_;
  fft_free(temp_accfft_);
#ifdef CUDA
  fft_free(work_cuda_);
#endif
}
