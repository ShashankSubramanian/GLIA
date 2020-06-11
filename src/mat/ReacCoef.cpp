#include "ReacCoef.h"

ReacCoef::ReacCoef(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : spec_ops_(spec_ops) {
  PetscErrorCode ierr;
  ierr = VecCreate(PETSC_COMM_WORLD, &rho_vec_);
  ierr = VecSetSizes(rho_vec_, params->grid_->nl_, params->grid_->ng_);
  ierr = setupVec(rho_vec_);
  ierr = VecSet(rho_vec_, 0);

  smooth_flag_ = 0;
}

PetscErrorCode ReacCoef::setValues(ScalarType rho_scale, ScalarType r_gm_wm_ratio, ScalarType r_glm_wm_ratio, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  rho_scale_ = rho_scale;
  r_gm_wm_ratio_ = r_gm_wm_ratio;
  r_glm_wm_ratio_ = r_glm_wm_ratio;
  ScalarType dr_dm_gm = rho_scale * r_gm_wm_ratio;    // GM
  ScalarType dr_dm_wm = rho_scale;                    // WM
  ScalarType dr_dm_glm = rho_scale * r_glm_wm_ratio;  // GLM

  params->tu_->r_gm_wm_ratio_ = r_gm_wm_ratio_;  // update values in params
  params->tu_->r_glm_wm_ratio_ = r_glm_wm_ratio_;
  params->tu_->rho_ = rho_scale_;

  dr_dm_gm = (dr_dm_gm <= 0) ? 0.0 : dr_dm_gm;
  dr_dm_glm = (dr_dm_glm <= 0) ? 0.0 : dr_dm_glm;

  ierr = VecSet(rho_vec_, 0.0); CHKERRQ(ierr);
  ierr = VecAXPY(rho_vec_, dr_dm_gm, mat_prop->gm_); CHKERRQ(ierr);
  ierr = VecAXPY(rho_vec_, dr_dm_wm, mat_prop->wm_); CHKERRQ(ierr);
  ierr = VecAXPY(rho_vec_, dr_dm_glm, mat_prop->csf_); CHKERRQ(ierr);
  ierr = dataOut(rho_vec_, params, "rho.nc"); 
  if (smooth_flag_) this->smooth(params);

  PetscFunctionReturn(ierr);
}

PetscErrorCode ReacCoef::updateIsotropicCoefficients(ScalarType rho_1, ScalarType rho_2, ScalarType rho_3, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  // compute new ratios
  rho_scale_ = rho_1;
  r_gm_wm_ratio_ = (params->tu_->nr_ == 1) ? params->tu_->r_gm_wm_ratio_ : rho_2 / rho_1;
  r_glm_wm_ratio_ = (params->tu_->nr_ == 1) ? params->tu_->r_glm_wm_ratio_ : rho_3 / rho_1;
  // and set the values
  ierr = setValues(rho_scale_, r_gm_wm_ratio_, r_glm_wm_ratio_, mat_prop, params);
  PetscFunctionReturn(ierr);
}

PetscErrorCode ReacCoef::smooth(std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  ScalarType sigma = 2.0 * M_PI / params->grid_->n_[0];

  ierr = spec_ops_->weierstrassSmoother(rho_vec_, rho_vec_, params, sigma);

  PetscFunctionReturn(ierr);
}

PetscErrorCode ReacCoef::applydRdm(Vec x1, Vec x2, Vec x3, Vec x4, Vec input) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // Event e ("tumor-reac-coeff-apply-dRdm");
  // std::array<double, 7> t = {0};
  // double self_exec_time = -MPI_Wtime ();

  ScalarType dr_dm_gm = rho_scale_ * r_gm_wm_ratio_;    // GM
  ScalarType dr_dm_wm = rho_scale_;                     // WM
  ScalarType dr_dm_glm = rho_scale_ * r_glm_wm_ratio_;  // GLM
  // if ratios <= 0, only diffuse in white matter
  dr_dm_gm = (dr_dm_gm <= 0) ? 0.0 : dr_dm_gm;
  dr_dm_glm = (dr_dm_glm <= 0) ? 0.0 : dr_dm_glm;
  // compute dKdm
  // compute dK/dm * (grad c)^T grad \alpha ..
  // assumes that geometry map has ordered components (WM, GM ,CSF, GLM)^T
  if (x1 != nullptr) {
    ierr = VecAXPY(x1, dr_dm_wm, input); CHKERRQ(ierr);
  }  // WM
  if (x2 != nullptr) {
    ierr = VecAXPY(x2, dr_dm_gm, input); CHKERRQ(ierr);
  }  // GM
  // if(x3 != nullptr) {ierr = VecAXPY (x3, 0, input);             CHKERRQ (ierr);} // CSF
  if (x4 != nullptr) {
    ierr = VecAXPY(x4, dr_dm_glm, input); CHKERRQ(ierr);
  }  // GLM

  // self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time); e.addTimings (t); e.stop ();
  PetscFunctionReturn(ierr);
}

ReacCoef::~ReacCoef() {
  PetscErrorCode ierr;
  ierr = VecDestroy(&rho_vec_);
}
