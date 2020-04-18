#include "MatProp.h"

MatProp::MatProp(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : spec_ops_(spec_ops) {
  PetscErrorCode ierr;
  ierr = VecCreate(PETSC_COMM_WORLD, &gm_);
  ierr = VecSetSizes(gm_, params->grid_->nl_, params->grid_->ng_);
  ierr = setupVec(gm_);

  ierr = VecDuplicate(gm_, &wm_);
  ierr = VecDuplicate(gm_, &vt_);
  ierr = VecDuplicate(gm_, &csf_);
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
  ierr = VecSet(vt_, 0); CHKERRQ(ierr);
  ierr = VecSet(csf_, 0); CHKERRQ(ierr);
  ierr = VecSet(bg_, 0); CHKERRQ(ierr);
  ierr = VecSet(filter_, 1); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::setValuesSinusoidal(std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = VecSet(vt_, 0); CHKERRQ(ierr);
  ierr = VecSet(csf_, 0); CHKERRQ(ierr);
  ierr = VecSet(bg_, 0); CHKERRQ(ierr);
  ierr = VecSet(filter_, 1); CHKERRQ(ierr);

  ScalarType *wm_ptr, *gm_ptr;
  ierr = VecGetArray(wm_, &wm_ptr); CHKERRQ (ierr);
  ierr = VecGetArray(gm_, &gm_ptr); CHKERRQ (ierr);
  ScalarType freq = 4;
  int64_t X, Y, Z, index;
  for (int x = 0; x < params->grid_->isize_[0]; x++) {
    for (int y = 0; y < params->grid_->isize_[1]; y++) {
      for (int z = 0; z < params->grid_->isize_[2]; z++) {
        X = params->grid_->istart_[0] + x;
        Y = params->grid_->istart_[1] + y;
        Z = params->grid_->istart_[2] + z;

        index = x * params->grid_->isize_[1] * params->grid_->isize_[2] + y * params->grid_->isize_[2] + z;

        wm_ptr[index] = 0.5 + 0.5 * sin (freq * 2.0 * M_PI / params->grid_->n_[0] * X)
                                  * sin (freq * 2.0 * M_PI / params->grid_->n_[1] * Y)
                                  * sin (freq * 2.0 * M_PI / params->grid_->n_[2] * Z);
        gm_ptr[index] = 1 - wm_ptr[index];
      }
    }
  }
  ierr = VecRestoreArray(wm_, &wm_ptr); CHKERRQ (ierr);
  ierr = VecRestoreArray(gm_, &gm_ptr); CHKERRQ (ierr);

  PetscFunctionReturn(ierr);
}


PetscErrorCode MatProp::clipHealthyTissues() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // clip at zero
  ScalarType *gm_ptr, *wm_ptr, *vt_ptr, *csf_ptr;
  ierr = vecGetArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(csf_, &csf_ptr); CHKERRQ(ierr);

#ifdef CUDA
  clipVectorCuda(gm_ptr, params_->grid_->nl_);
  clipVectorCuda(wm_ptr, params_->grid_->nl_);
  clipVectorCuda(csf_ptr, params_->grid_->nl_);
  clipVectorCuda(vt_ptr, params_->grid_->nl_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    gm_ptr[i] = (gm_ptr[i] <= 0.) ? 0. : gm_ptr[i];
    wm_ptr[i] = (wm_ptr[i] <= 0.) ? 0. : wm_ptr[i];
    vt_ptr[i] = (vt_ptr[i] <= 0.) ? 0. : vt_ptr[i];
    csf_ptr[i] = (csf_ptr[i] <= 0.) ? 0. : csf_ptr[i];
  }
#endif

  ierr = vecRestoreArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(csf_, &csf_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::setAtlas(Vec gm, Vec wm, Vec csf, Vec vt, Vec bg) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  gm_0_ = gm;
  wm_0_ = wm;
  vt_0_ = vt;
  csf_0_ = csf;

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::resetValues() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = VecCopy(gm_0_, gm_); CHKERRQ(ierr);
  ierr = VecCopy(wm_0_, wm_); CHKERRQ(ierr);
  ierr = VecCopy(vt_0_, vt_); CHKERRQ(ierr);
  ierr = VecCopy(csf_0_, csf_); CHKERRQ(ierr);

  // Set bg prob as 1 - sum
  ierr = VecWAXPY(bg_, 1., gm_, wm_); CHKERRQ(ierr);
  ierr = VecAXPY(bg_, 1., vt_); CHKERRQ(ierr);
  ierr = VecAXPY(bg_, 1., csf_); CHKERRQ(ierr);
  ierr = VecShift(bg_, -1.0); CHKERRQ(ierr);
  ierr = VecScale(bg_, -1.0); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode MatProp::setValuesCustom(Vec gm, Vec wm, Vec csf, Vec vt, Vec bg, std::shared_ptr<Parameters> params) {
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
  if (vt != nullptr) {
    ierr = VecCopy(vt, vt_); CHKERRQ(ierr);
  } else {
    ierr = VecSet(vt_, 0.0); CHKERRQ(ierr);
  }
  if (csf != nullptr) {
    ierr = VecCopy(csf, csf_); CHKERRQ(ierr);
  } else {
    ierr = VecSet(csf_, 0.0); CHKERRQ(ierr);
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
  ierr = VecAXPY(bg_, 1., vt_); CHKERRQ(ierr);
  ierr = VecAXPY(bg_, 1., csf_); CHKERRQ(ierr);
  ierr = VecShift(bg_, -1.0); CHKERRQ(ierr);
  ierr = VecScale(bg_, -1.0); CHKERRQ(ierr);

  ScalarType *gm_ptr, *wm_ptr, *vt_ptr, *csf_ptr, *filter_ptr, *bg_ptr;
  ierr = VecGetArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(csf_, &csf_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(filter_, &filter_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(bg_, &bg_ptr); CHKERRQ(ierr);

  for (int i = 0; i < params->grid_->nl_; i++) {
    if ((wm_ptr[i] > 0.1 || gm_ptr[i] > 0.1) && vt_ptr[i] < 0.8)
      filter_ptr[i] = 1.0;
    else
      filter_ptr[i] = 0.0;
  }

  if (params->tu_->write_output_) {
    dataOut(gm_ptr, params, "gm.nc");
    dataOut(wm_ptr, params, "wm.nc");
    dataOut(vt_ptr, params, "vt.nc");
    dataOut(csf_ptr, params, "csf.nc");
    dataOut(bg_ptr, params, "bg.nc");
  }

  ierr = VecRestoreArray(gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(csf_, &csf_ptr); CHKERRQ(ierr);
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
  ierr = VecDestroy(&vt_);
  ierr = VecDestroy(&csf_);
  ierr = VecDestroy(&bg_);
  ierr = VecDestroy(&filter_);
}
