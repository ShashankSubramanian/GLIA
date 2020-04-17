#include <ElasticitySolver.h>
#include <petsc/private/vecimpl.h>

#include "petsc/private/kspimpl.h"

ElasticitySolver::ElasticitySolver(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops) : ctx_() {
  PetscErrorCode ierr = 0;
  ctx_ = std::make_shared<CtxElasticity>(params, tumor, spec_ops);

#ifdef CUDA
  cudaMalloc((void **)&ctx_->fx_hat_, params->grid_->accfft_alloc_max_);
  cudaMalloc((void **)&ctx_->fy_hat_, params->grid_->accfft_alloc_max_);
  cudaMalloc((void **)&ctx_->fz_hat_, params->grid_->accfft_alloc_max_);
  cudaMalloc((void **)&ctx_->ux_hat_, params->grid_->accfft_alloc_max_);
  cudaMalloc((void **)&ctx_->uy_hat_, params->grid_->accfft_alloc_max_);
  cudaMalloc((void **)&ctx_->uz_hat_, params->grid_->accfft_alloc_max_);
  initElasticityCudaConstants(params->grid_->n_, params->grid_->ostart_);
#else
  ctx_->fx_hat_ = (ComplexType *)accfft_alloc(params->grid_->accfft_alloc_max_);
  ctx_->fy_hat_ = (ComplexType *)accfft_alloc(params->grid_->accfft_alloc_max_);
  ctx_->fz_hat_ = (ComplexType *)accfft_alloc(params->grid_->accfft_alloc_max_);
  ctx_->ux_hat_ = (ComplexType *)accfft_alloc(params->grid_->accfft_alloc_max_);
  ctx_->uy_hat_ = (ComplexType *)accfft_alloc(params->grid_->accfft_alloc_max_);
  ctx_->uz_hat_ = (ComplexType *)accfft_alloc(params->grid_->accfft_alloc_max_);
#endif

  // compute average coefficients
  ctx_->mu_avg_ = (ctx_->computeMu(params->tu_->E_healthy_, params->tu_->nu_healthy_) + ctx_->computeMu(params->tu_->E_bg_, params->tu_->nu_bg_) + ctx_->computeMu(params->tu_->E_csf_, params->tu_->nu_csf_) +
                   ctx_->computeMu(params->tu_->E_tumor_, params->tu_->nu_tumor_)) /
                  4;
  ctx_->lam_avg_ = (ctx_->computeLam(params->tu_->E_healthy_, params->tu_->nu_healthy_) + ctx_->computeLam(params->tu_->E_bg_, params->tu_->nu_bg_) + ctx_->computeLam(params->tu_->E_csf_, params->tu_->nu_csf_) +
                    ctx_->computeLam(params->tu_->E_tumor_, params->tu_->nu_tumor_)) /
                   4;
  ctx_->screen_avg_ = (params->tu_->screen_low_ + params->tu_->screen_high_) / 2;

  int factor = 3;  // vector equations

  ierr = VecCreate(PETSC_COMM_WORLD, &rhs_);
  ierr = VecSetSizes(rhs_, factor * params->grid_->nl_, factor * params->grid_->ng_);
  ierr = setupVec(rhs_);
  ierr = VecSet(rhs_, 0);

  ierr = VecDuplicate(rhs_, &ctx_->disp_);

  ierr = MatCreateShell(PETSC_COMM_WORLD, factor * params->grid_->nl_, factor * params->grid_->nl_, factor * params->grid_->ng_, factor * params->grid_->ng_, ctx_.get(), &A_);
  ierr = MatShellSetOperation(A_, MATOP_MULT, (void (*)(void))operatorVariableCoefficients);
#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 10)
  ierr = MatShellSetOperation(A_, MATOP_CREATE_VECS, (void (*)(void))operatorCreateVecsElas);
#endif

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp_);
  ierr = KSPSetOperators(ksp_, A_, A_);
  ierr = KSPSetTolerances(ksp_, 1E-3, PETSC_DEFAULT, PETSC_DEFAULT, 100);
  ierr = KSPSetType(ksp_, KSPCG);
  // ierr = KSPMonitorSet(ksp_, elasticitySolverKSPMonitor, ctx_.get(), 0);
  ierr = KSPSetInitialGuessNonzero(ksp_, PETSC_TRUE);
  ierr = KSPSetOptionsPrefix(ksp_, "tumor_elasticity_");
  ierr = KSPSetFromOptions(ksp_);
  ierr = KSPSetUp(ksp_);

  ierr = KSPGetPC(ksp_, &pc_);
  ierr = PCSetType(pc_, PCSHELL);
  ierr = PCShellSetApply(pc_, operatorConstantCoefficients);
  ierr = PCShellSetContext(pc_, ctx_.get());
  ierr = KSPSetFromOptions(ksp_);
  ierr = KSPSetUp(ksp_);
}

PetscErrorCode elasticitySolverKSPMonitor(KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  Vec x;
  int maxit;
  ScalarType divtol, abstol, reltol;
  ierr = KSPBuildSolution(ksp, NULL, &x); CHKERRQ(ierr);
  ierr = KSPGetTolerances(ksp, &reltol, &abstol, &divtol, &maxit); CHKERRQ(ierr);
  CtxElasticity *ctx = reinterpret_cast<CtxElasticity *>(ptr);  // get user context

  std::stringstream s;
  if (its == 0) {
    s << std::setw(3) << " KSP:"
      << " computing solution of elasticity system (tol=" << std::scientific << std::setprecision(5) << reltol << ")";
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }
  s << std::setw(3) << " KSP:" << std::setw(15) << " " << std::setfill('0') << std::setw(3) << its << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();

  // int ksp_itr;
  // ierr = KSPGetIterationNumber (ksp, &ksp_itr);                                 CHKERRQ (ierr);
  // ScalarType e_max, e_min;
  // if (ksp_itr % 10 == 0 || ksp_itr == maxit) {
  //   ierr = KSPComputeExtremeSingularValues (ksp, &e_max, &e_min);       CHKERRQ (ierr);
  //   s << "Condition number of matrix is: " << e_max / e_min << " | largest singular values is: " << e_max << ", smallest singular values is: " << e_min << std::endl; CHKERRQ(ierr);
  //   s.str (""); s.clear ();
  // }
  PetscFunctionReturn(ierr);
}

PetscErrorCode operatorCreateVecsElas(Mat A, Vec *left, Vec *right) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  CtxElasticity *ctx;
  ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);
  std::shared_ptr<Parameters> params = ctx->params_;

  if (right) {
    ierr = VecDuplicate(ctx->disp_, right); CHKERRQ(ierr);
  }
  if (left) {
    ierr = VecDuplicate(ctx->disp_, left); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode operatorConstantCoefficients(PC pc, Vec x, Vec y) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  Event e("tumor-elasticity-prec");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  CtxElasticity *ctx;
  ierr = PCShellGetContext(pc, (void **)&ctx); CHKERRQ(ierr);

  int lock_state, lock_state_y;
  ierr = VecLockGet(x, &lock_state); CHKERRQ(ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }
  ierr = VecLockGet(y, &lock_state_y); CHKERRQ(ierr);
  if (lock_state_y != 0) {
    y->lock = 0;
  }

  std::shared_ptr<Parameters> params = ctx->params_;
  std::shared_ptr<Tumor> tumor = ctx->tumor_;
  std::shared_ptr<VecField> force = ctx->force_;
  std::shared_ptr<VecField> displacement = ctx->displacement_;

  ierr = force->setIndividualComponents(x); CHKERRQ(ierr);
  ierr = displacement->setIndividualComponents(y); CHKERRQ(ierr);

  // FFT of each component
  ScalarType *fx_ptr, *fy_ptr, *fz_ptr;
  ScalarType *ux_ptr, *uy_ptr, *uz_ptr;

  ierr = force->getComponentArrays(fx_ptr, fy_ptr, fz_ptr);
  ierr = displacement->getComponentArrays(ux_ptr, uy_ptr, uz_ptr);

  ComplexType *fx_hat = ctx->fx_hat_;
  ComplexType *fy_hat = ctx->fy_hat_;
  ComplexType *fz_hat = ctx->fz_hat_;
  ComplexType *ux_hat = ctx->ux_hat_;
  ComplexType *uy_hat = ctx->uy_hat_;
  ComplexType *uz_hat = ctx->uz_hat_;

  ctx->spec_ops_->executeFFTR2C(fx_ptr, fx_hat);
  ctx->spec_ops_->executeFFTR2C(fy_ptr, fy_hat);
  ctx->spec_ops_->executeFFTR2C(fz_ptr, fz_hat);

#ifdef CUDA
  precFactorElasticityCuda((CudaComplexType *)ux_hat, (CudaComplexType *)uy_hat, (CudaComplexType *)uz_hat, (CudaComplexType *)fx_hat, (CudaComplexType *)fy_hat, (CudaComplexType *)fz_hat,
                           ctx->lam_avg_, ctx->mu_avg_, ctx->screen_avg_, params->grid_->osize_);
#else
  ScalarType s1, s2, s1_square, s3, scale;

  int64_t wx, wy, wz;
  ScalarType wTw, wTf_real, wTf_imag;
  int64_t x_global, y_global, z_global;
  int64_t ptr;

  ScalarType factor = 1.0 / (params->grid_->n_[0] * params->grid_->n_[1] * params->grid_->n_[2]);

  s2 = ctx->lam_avg_ + ctx->mu_avg_;

  for (int i = 0; i < params->grid_->osize_[0]; i++) {
    for (int j = 0; j < params->grid_->osize_[1]; j++) {
      for (int k = 0; k < params->grid_->osize_[2]; k++) {
        ptr = i * params->grid_->osize_[1] * params->grid_->osize_[2] + j * params->grid_->osize_[2] + k;

        x_global = i + params->grid_->ostart_[0];
        y_global = j + params->grid_->ostart_[1];
        z_global = k + params->grid_->ostart_[2];

        wx = x_global;
        if (x_global > params->grid_->n_[0] / 2)  // symmetric frequencies
          wx -= params->grid_->n_[0];
        if (x_global == params->grid_->n_[0] / 2)  // nyquist frequency
          wx = 0;

        wy = y_global;
        if (y_global > params->grid_->n_[1] / 2)  // symmetric frequencies
          wy -= params->grid_->n_[1];
        if (y_global == params->grid_->n_[1] / 2)  // nyquist frequency
          wy = 0;

        wz = z_global;
        if (z_global > params->grid_->n_[2] / 2)  // symmetric frequencies
          wz -= params->grid_->n_[2];
        if (z_global == params->grid_->n_[2] / 2)  // nyquist frequency
          wz = 0;

        wTw = -1.0 * (wx * wx + wy * wy + wz * wz);

        s1 = -ctx->screen_avg_ + ctx->mu_avg_ * wTw;
        s1_square = s1 * s1;
        s3 = 1.0 / (1.0 + (wTw * s2) / s1);

        wTf_real = wx * fx_hat[ptr][0] + wy * fy_hat[ptr][0] + wz * fz_hat[ptr][0];
        wTf_imag = wx * fx_hat[ptr][1] + wy * fy_hat[ptr][1] + wz * fz_hat[ptr][1];

        // real part
        scale = -1.0 * wx * wTf_real;
        ux_hat[ptr][0] = factor * (fx_hat[ptr][0] * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale);
        // imaginary part
        scale = -1.0 * wx * wTf_imag;
        ux_hat[ptr][1] = factor * (fx_hat[ptr][1] * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale);

        // real part
        scale = -1.0 * wy * wTf_real;
        uy_hat[ptr][0] = factor * (fy_hat[ptr][0] * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale);
        // imaginary part
        scale = -1.0 * wy * wTf_imag;
        uy_hat[ptr][1] = factor * (fy_hat[ptr][1] * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale);

        // real part
        scale = -1.0 * wz * wTf_real;
        uz_hat[ptr][0] = factor * (fz_hat[ptr][0] * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale);
        // imaginary part
        scale = -1.0 * wz * wTf_imag;
        uz_hat[ptr][1] = factor * (fz_hat[ptr][1] * (1.0 / s1) - (1.0 / s1_square) * s2 * s3 * scale);
      }
    }
  }
#endif

  ctx->spec_ops_->executeFFTC2R(ux_hat, ux_ptr);
  ctx->spec_ops_->executeFFTC2R(uy_hat, uy_ptr);
  ctx->spec_ops_->executeFFTC2R(uz_hat, uz_ptr);

  ierr = force->restoreComponentArrays(fx_ptr, fy_ptr, fz_ptr);
  ierr = displacement->restoreComponentArrays(ux_ptr, uy_ptr, uz_ptr);

  ierr = displacement->getIndividualComponents(y);  // get the individual components of u and set it to y (o/p)

  if (lock_state != 0) {
    x->lock = lock_state;
  }
  if (lock_state_y != 0) {
    y->lock = lock_state;
  }

  self_exec_time += MPI_Wtime();
  accumulateTimers(ctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// Defines Lu
PetscErrorCode operatorVariableCoefficients(Mat A, Vec x, Vec y) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  Event e("tumor-elasticity-matvec");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  CtxElasticity *ctx;
  ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);

  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;

  int lock_state;
  ierr = VecLockGet(x, &lock_state); CHKERRQ(ierr);
  if (lock_state != 0) {
    x->lock = 0;
  }

  std::shared_ptr<Parameters> params = ctx->params_;
  std::shared_ptr<Tumor> tumor = ctx->tumor_;
  std::shared_ptr<VecField> force = ctx->force_;
  std::shared_ptr<VecField> displacement = ctx->displacement_;

  ierr = displacement->setIndividualComponents(x); CHKERRQ(ierr);

  // second term: grad(lambda * div(u)) :  stored in work[1],[2],[3]
  ctx->spec_ops_->computeDivergence(tumor->work_[0], displacement->x_, displacement->y_, displacement->z_, t.data());
  ierr = VecPointwiseMult(tumor->work_[0], ctx->lam_, tumor->work_[0]); CHKERRQ(ierr);
  ctx->spec_ops_->computeGradient(tumor->work_[1], tumor->work_[2], tumor->work_[3], tumor->work_[0], &XYZ, t.data());

  // first term: div (mu .* (gradu + graduT))
  ctx->spec_ops_->computeGradient(tumor->work_[4], tumor->work_[5], tumor->work_[6], displacement->x_, &XYZ, t.data());
  XYZ[1] = 0;  // work_[8] is re-used
  ctx->spec_ops_->computeGradient(tumor->work_[7], tumor->work_[8], tumor->work_[9], displacement->y_, &XYZ, t.data());
  XYZ[1] = 1;  // reset back
  XYZ[2] = 0;
  ctx->spec_ops_->computeGradient(tumor->work_[10], tumor->work_[11], tumor->work_[0], displacement->z_, &XYZ, t.data());
  XYZ[2] = 1;

  ierr = VecScale(tumor->work_[4], 2.); CHKERRQ(ierr);
  ierr = VecWAXPY(tumor->work_[8], 1.0, tumor->work_[5], tumor->work_[7]); CHKERRQ(ierr);
  ierr = VecWAXPY(tumor->work_[0], 1.0, tumor->work_[6], tumor->work_[10]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[4], ctx->mu_, tumor->work_[4]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[8], ctx->mu_, tumor->work_[8]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[0], ctx->mu_, tumor->work_[0]); CHKERRQ(ierr);

  ctx->spec_ops_->computeDivergence(force->x_, tumor->work_[4], tumor->work_[8], tumor->work_[0], t.data());
  ierr = VecAXPY(force->x_, 1.0, tumor->work_[1]); CHKERRQ(ierr);

  XYZ[0] = 0;
  XYZ[1] = 1;
  XYZ[2] = 0;
  ctx->spec_ops_->computeGradient(tumor->work_[7], tumor->work_[8], tumor->work_[9], displacement->y_, &XYZ, t.data());
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  ierr = VecWAXPY(tumor->work_[4], 1.0, tumor->work_[7], tumor->work_[5]); CHKERRQ(ierr);
  ierr = VecScale(tumor->work_[8], 2.); CHKERRQ(ierr);
  ierr = VecWAXPY(tumor->work_[0], 1.0, tumor->work_[9], tumor->work_[11]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[4], ctx->mu_, tumor->work_[4]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[8], ctx->mu_, tumor->work_[8]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[0], ctx->mu_, tumor->work_[0]); CHKERRQ(ierr);

  ctx->spec_ops_->computeDivergence(force->y_, tumor->work_[4], tumor->work_[8], tumor->work_[0], t.data());
  ierr = VecAXPY(force->y_, 1.0, tumor->work_[2]); CHKERRQ(ierr);

  XYZ[0] = 0;
  XYZ[1] = 0;
  XYZ[2] = 1;
  ctx->spec_ops_->computeGradient(tumor->work_[10], tumor->work_[11], tumor->work_[0], displacement->z_, &XYZ, t.data());
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;
  ierr = VecWAXPY(tumor->work_[4], 1.0, tumor->work_[10], tumor->work_[6]); CHKERRQ(ierr);
  ierr = VecWAXPY(tumor->work_[8], 1.0, tumor->work_[11], tumor->work_[9]); CHKERRQ(ierr);
  ierr = VecScale(tumor->work_[0], 2.); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[4], ctx->mu_, tumor->work_[4]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[8], ctx->mu_, tumor->work_[8]); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[0], ctx->mu_, tumor->work_[0]); CHKERRQ(ierr);

  ctx->spec_ops_->computeDivergence(force->z_, tumor->work_[4], tumor->work_[8], tumor->work_[0], t.data());
  ierr = VecAXPY(force->z_, 1.0, tumor->work_[3]); CHKERRQ(ierr);

  // screening term
  ierr = VecPointwiseMult(tumor->work_[4], ctx->screen_, displacement->x_); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[8], ctx->screen_, displacement->y_); CHKERRQ(ierr);
  ierr = VecPointwiseMult(tumor->work_[0], ctx->screen_, displacement->z_); CHKERRQ(ierr);

  ierr = VecAXPY(force->x_, -1.0, tumor->work_[4]); CHKERRQ(ierr);
  ierr = VecAXPY(force->y_, -1.0, tumor->work_[8]); CHKERRQ(ierr);
  ierr = VecAXPY(force->z_, -1.0, tumor->work_[0]); CHKERRQ(ierr);

  ierr = force->getIndividualComponents(y);

  if (lock_state != 0) {
    x->lock = lock_state;
  }

  self_exec_time += MPI_Wtime();
  accumulateTimers(ctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode VariableLinearElasticitySolver::computeMaterialProperties() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  CtxElasticity *ctx;
  ierr = MatShellGetContext(A_, &ctx); CHKERRQ(ierr);

  ScalarType mu_bg, mu_healthy, mu_tumor, mu_csf;
  ScalarType lam_bg, lam_healthy, lam_tumor, lam_csf;
  std::shared_ptr<Parameters> params = ctx->params_;
  std::shared_ptr<Tumor> tumor = ctx->tumor_;

  mu_bg = ctx_->computeMu(params->tu_->E_bg_, params->tu_->nu_bg_);
  mu_healthy = ctx_->computeMu(params->tu_->E_healthy_, params->tu_->nu_healthy_);
  mu_tumor = ctx_->computeMu(params->tu_->E_tumor_, params->tu_->nu_tumor_);
  mu_csf = ctx_->computeMu(params->tu_->E_csf_, params->tu_->nu_csf_);

  lam_bg = ctx_->computeLam(params->tu_->E_bg_, params->tu_->nu_bg_);
  lam_healthy = ctx_->computeLam(params->tu_->E_healthy_, params->tu_->nu_healthy_);
  lam_tumor = ctx_->computeLam(params->tu_->E_tumor_, params->tu_->nu_tumor_);
  lam_csf = ctx_->computeLam(params->tu_->E_csf_, params->tu_->nu_csf_);

  ScalarType c_threshold = 0.005;
  // Compute material properties vectors
  ierr = VecSet(ctx->mu_, 0.); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->mu_, mu_bg, tumor->mat_prop_->bg_); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->mu_, mu_healthy, tumor->mat_prop_->wm_); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->mu_, mu_healthy, tumor->mat_prop_->gm_); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->mu_, mu_csf, tumor->mat_prop_->vt_); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->mu_, mu_csf, tumor->mat_prop_->csf_); CHKERRQ(ierr);

  ierr = VecSet(ctx->lam_, 0.); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->lam_, lam_bg, tumor->mat_prop_->bg_); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->lam_, lam_healthy, tumor->mat_prop_->wm_); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->lam_, lam_healthy, tumor->mat_prop_->gm_); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->lam_, lam_csf, tumor->mat_prop_->vt_); CHKERRQ(ierr);
  ierr = VecAXPY(ctx->lam_, lam_csf, tumor->mat_prop_->csf_); CHKERRQ(ierr);

  // set the tumor
  ScalarType *bg_ptr, *wm_ptr, *gm_ptr, *c_ptr, *mu_ptr, *screen_ptr, *lam_ptr;
  ierr = vecGetArray(ctx->screen_, &screen_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(ctx->mu_, &mu_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(ctx->lam_, &lam_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor->c_t_, &c_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor->mat_prop_->bg_, &bg_ptr); CHKERRQ(ierr);

#ifdef CUDA
  computeTumorLameCuda(mu_ptr, lam_ptr, c_ptr, mu_tumor, lam_tumor, params->grid_->nl_);
  computeScreeningCuda(screen_ptr, c_ptr, bg_ptr, params->tu_->screen_low_, params->tu_->screen_high_, params->grid_->nl_);
#else
  for (int i = 0; i < params->grid_->nl_; i++) {
    mu_ptr[i] = (mu_ptr[i] > 0) ? mu_ptr[i] : 0;
    lam_ptr[i] = (lam_ptr[i] > 0) ? lam_ptr[i] : 0;

    mu_ptr[i] += (c_ptr[i] > 0) ? (mu_tumor * c_ptr[i]) : 0;
    lam_ptr[i] += (c_ptr[i] > 0) ? (lam_tumor * c_ptr[i]) : 0;

    screen_ptr[i] = (c_ptr[i] >= c_threshold) ? params->tu_->screen_low_ : params->tu_->screen_high_;
    if (bg_ptr[i] > 0.95) screen_ptr[i] = 1E6;  // screen out the background completely to ensure no movement
  }
#endif

  ierr = vecRestoreArray(ctx->screen_, &screen_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(ctx->mu_, &mu_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(ctx->lam_, &lam_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor->c_t_, &c_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor->mat_prop_->bg_, &bg_ptr); CHKERRQ(ierr);

  // average the material properties for use in preconditioner
  ierr = VecSum(ctx->mu_, &ctx->mu_avg_); CHKERRQ(ierr);
  ierr = VecSum(ctx->lam_, &ctx->lam_avg_); CHKERRQ(ierr);
  ierr = VecSum(ctx->screen_, &ctx->screen_avg_); CHKERRQ(ierr);

  ctx->mu_avg_ /= params->grid_->ng_;
  ctx->lam_avg_ /= params->grid_->ng_;
  ctx->screen_avg_ /= params->grid_->ng_;

  PetscFunctionReturn(ierr);
}

PetscErrorCode VariableLinearElasticitySolver::smoothMaterialProperties() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  CtxElasticity *ctx;
  ierr = MatShellGetContext(A_, &ctx); CHKERRQ(ierr);
  std::shared_ptr<Parameters> params = ctx->params_;

  ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / params->grid_->n_[0];
  ierr = ctx->spec_ops_->weierstrassSmoother(ctx->mu_, ctx->mu_, params, sigma_smooth); CHKERRQ(ierr);
  ierr = ctx->spec_ops_->weierstrassSmoother(ctx->lam_, ctx->lam_, params, sigma_smooth); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode VariableLinearElasticitySolver::solve(std::shared_ptr<VecField> displacement, std::shared_ptr<VecField> rhs) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  Event e("tumor-elasticity-solve");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  CtxElasticity *ctx;
  ierr = MatShellGetContext(A_, &ctx); CHKERRQ(ierr);
  std::shared_ptr<Parameters> params = ctx->params_;
  std::stringstream s;

  bool flag_smooth_force_disp = false;
  bool flag_smooth_mat_props = false;
  // smooth the force
  ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / params->grid_->n_[0];

  if (flag_smooth_force_disp) {
    ierr = ctx->spec_ops_->weierstrassSmoother(rhs->x_, rhs->x_, params, sigma_smooth); CHKERRQ(ierr);
    ierr = ctx->spec_ops_->weierstrassSmoother(rhs->y_, rhs->y_, params, sigma_smooth); CHKERRQ(ierr);
    ierr = ctx->spec_ops_->weierstrassSmoother(rhs->z_, rhs->z_, params, sigma_smooth); CHKERRQ(ierr);
  }

  ierr = rhs->getIndividualComponents(rhs_); CHKERRQ(ierr);
  ierr = displacement->getIndividualComponents(ctx->disp_);  // get the three disp components in disp to use as IC

  ierr = computeMaterialProperties(); CHKERRQ(ierr);
  if (flag_smooth_mat_props) {
    // clip and smooth material properties
    ierr = smoothMaterialProperties(); CHKERRQ(ierr);
  }

  // KSP solve
  ierr = KSPSolve(ksp_, rhs_, ctx->disp_); CHKERRQ(ierr);

  ierr = displacement->setIndividualComponents(ctx->disp_); CHKERRQ(ierr);

  int itr = 0;
  if (params->tu_->verbosity_ > 1) {
    ierr = KSPGetIterationNumber(ksp_, &itr); CHKERRQ(ierr);
    ScalarType res_norm;
    ierr = KSPGetResidualNorm(ksp_, &res_norm); CHKERRQ(ierr);
    s << "[Elasticity solver] Conjugate gradients convergence - iterations: " << itr << "    residual: " << res_norm;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  // smooth the displacement
  if (flag_smooth_force_disp) {
    ierr = ctx->spec_ops_->weierstrassSmoother(displacement->x_, displacement->x_, params, sigma_smooth); CHKERRQ(ierr);
    ierr = ctx->spec_ops_->weierstrassSmoother(displacement->y_, displacement->y_, params, sigma_smooth); CHKERRQ(ierr);
    ierr = ctx->spec_ops_->weierstrassSmoother(displacement->z_, displacement->z_, params, sigma_smooth); CHKERRQ(ierr);
  }

  self_exec_time += MPI_Wtime();
  accumulateTimers(ctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings(t);
  e.stop();

  PetscFunctionReturn(ierr);
}

ElasticitySolver::~ElasticitySolver() {
  PetscErrorCode ierr = 0;
  ierr = MatDestroy(&A_);
  ierr = KSPDestroy(&ksp_);
  ierr = VecDestroy(&rhs_);
}