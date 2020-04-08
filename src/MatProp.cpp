#include "MatProp.h"

MatProp::MatProp(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : spec_ops_(spec_ops) {
  PetscErrorCode ierr;
  ierr = VecCreate(PETSC_COMM_WORLD, &gm_);
  ierr = VecSetSizes(gm_, params->grid_->nl_, params->grid_->ng_);
  ierr = setupVec(gm_);

  ierr = VecDuplicate(gm_, &wm_);
  ierr = VecDuplicate(gm_, &csf_);
  ierr = VecDuplicate(gm_, &glm_);
  ierr = VecDuplicate(gm_, &bg_);
  ierr = VecDuplicate(gm_, &filter_);

  mri_ = nullptr;

  params_ = params;
}

PetscErrorCode MatProp::setValues(std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = VecSet(gm_, 0); CHKERRQ(ierr);
  ierr = VecSet(wm_, 1); CHKERRQ(ierr);
  ierr = VecSet(csf_, 0); CHKERRQ(ierr);
  ierr = VecSet(glm_, 0); CHKERRQ(ierr);
  ierr = VecSet(bg_, 0); CHKERRQ(ierr);
  ierr = VecSet(filter_, 1); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::clipHealthyTissues() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // clip at zero
  ScalarType *gm_ptr, *wm_ptr, *csf_ptr, *glm_ptr;
  ierr = vecGetArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(csf_, &csf_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(glm_, &glm_ptr); CHKERRQ(ierr);

#ifdef CUDA
  clipVectorCuda(gm_ptr, params_->grid_->nl_);
  clipVectorCuda(wm_ptr, params_->grid_->nl_);
  clipVectorCuda(glm_ptr, params_->grid_->nl_);
  clipVectorCuda(csf_ptr, params_->grid_->nl_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    gm_ptr[i] = (gm_ptr[i] <= 0.) ? 0. : gm_ptr[i];
    wm_ptr[i] = (wm_ptr[i] <= 0.) ? 0. : wm_ptr[i];
    csf_ptr[i] = (csf_ptr[i] <= 0.) ? 0. : csf_ptr[i];
    glm_ptr[i] = (glm_ptr[i] <= 0.) ? 0. : glm_ptr[i];
  }
#endif

  ierr = vecRestoreArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(csf_, &csf_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(glm_, &glm_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::setAtlas(Vec gm, Vec wm, Vec glm, Vec csf, Vec bg) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  gm_0_ = gm;
  wm_0_ = wm;
  csf_0_ = csf;
  glm_0_ = glm;

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::resetValues() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = VecCopy(gm_0_, gm_); CHKERRQ(ierr);
  ierr = VecCopy(wm_0_, wm_); CHKERRQ(ierr);
  ierr = VecCopy(csf_0_, csf_); CHKERRQ(ierr);
  ierr = VecCopy(glm_0_, glm_); CHKERRQ(ierr);

  // Set bg prob as 1 - sum
  ierr = VecWAXPY(bg_, 1., gm_, wm_); CHKERRQ(ierr);
  ierr = VecAXPY(bg_, 1., csf_); CHKERRQ(ierr);
  ierr = VecAXPY(bg_, 1., glm_); CHKERRQ(ierr);
  ierr = VecShift(bg_, -1.0); CHKERRQ(ierr);
  ierr = VecScale(bg_, -1.0); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::setValuesCustom(Vec gm, Vec wm, Vec glm, Vec csf, Vec bg, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  if (wm != nullptr) {
    ierr = VecCopy(wm, wm_); CHKERRQ(ierr);
  } else {
    ierr = VecSet(wm_, 0.0); CHKERRQ(ierr);
  }
  if (gm != nullptr) {
    ierr = VecCopy(gm, gm_); CHKERRQ(ierr);
  } else {
    ierr = VecSet(gm_, 0.0); CHKERRQ(ierr);
  }
  if (csf != nullptr) {
    ierr = VecCopy(csf, csf_); CHKERRQ(ierr);
  } else {
    ierr = VecSet(csf_, 0.0); CHKERRQ(ierr);
  }
  if (glm != nullptr) {
    ierr = VecCopy(glm, glm_); CHKERRQ(ierr);
  } else {
    ierr = VecSet(glm_, 0.0); CHKERRQ(ierr);
  }
  if (bg != nullptr) {
    ierr = VecCopy(bg, bg_); CHKERRQ(ierr);
  } else {
    ierr = VecSet(bg_, 0.0); CHKERRQ(ierr);
  }

  // clip tissues to ensure positivty of coefficients
  ierr = clipHealthyTissues(); CHKERRQ(ierr);
  // Set bg prob as 1 - sum
  ierr = VecWAXPY(bg_, 1., gm_, wm_); CHKERRQ(ierr);
  ierr = VecAXPY(bg_, 1., csf_); CHKERRQ(ierr);
  ierr = VecAXPY(bg_, 1., glm_); CHKERRQ(ierr);
  ierr = VecShift(bg_, -1.0); CHKERRQ(ierr);
  ierr = VecScale(bg_, -1.0); CHKERRQ(ierr);

  ScalarType *gm_ptr, *wm_ptr, *csf_ptr, *glm_ptr, *filter_ptr, *bg_ptr;
  ierr = VecGetArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(csf_, &csf_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(glm_, &glm_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(filter_, &filter_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(bg_, &bg_ptr); CHKERRQ(ierr);

  for (int i = 0; i < params->grid_->nl_; i++) {
    if ((wm_ptr[i] > 0.1 || gm_ptr[i] > 0.1) && csf_ptr[i] < 0.8)
      filter_ptr[i] = 1.0;
    else
      filter_ptr[i] = 0.0;
  }

  if (params->write_output_) {
    dataOut(gm_ptr, params, "gm.nc");
    dataOut(wm_ptr, params, "wm.nc");
    dataOut(csf_ptr, params, "csf.nc");
    dataOut(glm_ptr, params, "glm.nc");
    dataOut(bg_ptr, params, "bg.nc");
  }

  ierr = VecRestoreArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(csf_, &csf_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(glm_, &glm_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(filter_, &filter_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(bg_, &bg_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::filterTumor(Vec c) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *c_ptr, *wm_ptr, *gm_ptr;
  ierr = VecGetArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(c, &c_ptr); CHKERRQ(ierr);

  for (int i = 0; i < params_->grid_->nl_; i++) {
    wm_ptr[i] *= (1. - c_ptr[i]);
    gm_ptr[i] *= (1. - c_ptr[i]);
  }

  ierr = VecRestoreArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(c, &c_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::filterBackgroundAndSmooth(Vec in) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = VecShift(bg_, -1.0); CHKERRQ(ierr);
  ierr = VecScale(bg_, -1.0); CHKERRQ(ierr);
  ierr = VecPointwiseMult(in, in, bg_); CHKERRQ(ierr);
  ierr = VecScale(bg_, -1.0); CHKERRQ(ierr);
  ierr = VecShift(bg_, 1.0); CHKERRQ(ierr);

  ScalarType sigma_smooth = 1. * params_->tu_->smoothing_factor_ * 2 * M_PI / params_->grid_->n_[0];
  ierr = spec_ops_->weierstrassSmoother(in, in, params_, sigma_smooth);

  PetscFunctionReturn(ierr);
}

MatProp::~MatProp() {
  PetscErrorCode ierr;
  ierr = VecDestroy(&gm_);
  ierr = VecDestroy(&wm_);
  ierr = VecDestroy(&csf_);
  ierr = VecDestroy(&glm_);
  ierr = VecDestroy(&bg_);
  ierr = VecDestroy(&filter_);

  if (mri_ != nullptr) {
    ierr = VecDestroy(&mri_);
  }
}
