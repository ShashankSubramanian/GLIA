#include "Tumor.h"

Tumor::Tumor(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : params_(params), spec_ops_(spec_ops), p_(nullptr) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  k_ = std::make_shared<DiffCoef>(params, spec_ops);
  rho_ = std::make_shared<ReacCoef>(params, spec_ops);
  obs_ = std::make_shared<Obs>(params);

  ierr = VecCreate(PETSC_COMM_WORLD, &c_t_);
  ierr = VecSetSizes(c_t_, params->grid_->nl_, params->grid_->ng_);
  ierr = setupVec(c_t_);
  ierr = VecDuplicate(c_t_, &c_0_);
  ierr = VecDuplicate(c_t_, &p_0_);
  ierr = VecDuplicate(c_0_, &p_t_);

  // allocating memory for work vectors
  work_ = new Vec[12];
  for (int i = 0; i < 12; i++) {
    ierr = VecDuplicate(c_t_, &work_[i]);
    ierr = VecSet(work_[i], 0);
  }
  // setting work vecs for diffusion coefficient (first 8)
  k_->setWorkVecs(work_);

  ierr = VecSet(c_t_, 0);
  ierr = VecSet(c_0_, 0);
  ierr = VecSet(p_0_, 0);
  ierr = VecSet(p_t_, 0);

  ierr = VecDuplicate(c_t_, &seg_);
  ierr = VecSet(seg_, 0);

  if(params_->tu_->adv_velocity_set_ || params_->tu_->model_ == 4 || params_->tu_->model_ == 5) {
    velocity_ = std::make_shared<VecField>(params->grid_->nl_, params->grid_->ng_);
    work_field_ = std::make_shared<VecField>(params->grid_->nl_, params->grid_->ng_);
  }
  if (params->tu_->model_ == 4 || params_->tu_->model_ == 5) {  // mass effect model -- allocate space for more variables
    force_ = std::make_shared<VecField>(params->grid_->nl_, params->grid_->ng_);
    displacement_ = std::make_shared<VecField>(params->grid_->nl_, params->grid_->ng_);
  }

  if (params_->tu_->model_ == 5) {
    std::vector<Vec> c(params->tu_->num_species_);
    for (int i = 0; i < c.size(); i++) {
      ierr = VecDuplicate(c_t_, &c[i]);
      ierr = VecSet(c_t_, 0.);
    }
    // Insert the different species
    species_.insert(std::pair<std::string, Vec>("proliferative", c[0]));
    species_.insert(std::pair<std::string, Vec>("infiltrative", c[1]));
    species_.insert(std::pair<std::string, Vec>("necrotic", c[2]));
    species_.insert(std::pair<std::string, Vec>("oxygen", c[3]));
    species_.insert(std::pair<std::string, Vec>("edema", c[4]));
  }
}

PetscErrorCode Tumor::initialize(Vec p, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Phi> phi, std::shared_ptr<MatProp> mat_prop) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  if (mat_prop == nullptr) {
    mat_prop_ = std::make_shared<MatProp>(params, spec_ops);
    ierr = mat_prop_->setValues(params); CHKERRQ(ierr);
  } else
    mat_prop_ = mat_prop;
  ierr = k_->setValues(params->tu_->k_, params->tu_->k_gm_wm_ratio_, params->tu_->k_glm_wm_ratio_, mat_prop_, params); CHKERRQ(ierr);
  ierr = rho_->setValues(params->tu_->rho_, params->tu_->r_gm_wm_ratio_, params->tu_->r_glm_wm_ratio_, mat_prop_, params); CHKERRQ(ierr);
  ierr = VecDuplicate(p, &p_); CHKERRQ(ierr);
  ierr = VecCopy(p, p_); CHKERRQ(ierr);

  if (phi == nullptr) {
    phi_ = std::make_shared<Phi>(params, spec_ops);
    ierr = phi_->setGaussians(params->tu_->user_cm_, params->tu_->phi_sigma_, params->tu_->phi_spacing_factor_, params->tu_->np_); CHKERRQ(ierr);
    ierr = phi_->setValues(mat_prop_); CHKERRQ(ierr);
  } else
    phi_ = phi;

  PetscFunctionReturn(ierr);
}

PetscErrorCode Tumor::setParams(Vec p, std::shared_ptr<Parameters> params, bool npchanged) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  if (npchanged) {
    // re-create p vectors
    if (p_ != nullptr) {
      ierr = VecDestroy(&p_); CHKERRQ(ierr);
    }
    ierr = VecDuplicate(p, &p_); CHKERRQ(ierr);
  }
  ierr = VecCopy(p, p_); CHKERRQ(ierr);
  // set new values
  ierr = k_->setValues(params->tu_->k_, params->tu_->k_gm_wm_ratio_, params->tu_->k_glm_wm_ratio_, mat_prop_, params); CHKERRQ(ierr);
  ierr = rho_->setValues(params->tu_->rho_, params->tu_->r_gm_wm_ratio_, params->tu_->r_glm_wm_ratio_, mat_prop_, params); CHKERRQ(ierr);
  ierr = phi_->setValues(mat_prop_); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode Tumor::setSinusoidalCoefficients(std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = mat_prop_->setValuesSinusoidal(params); CHKERRQ(ierr);
  ierr = k_->setValues(params->tu_->k_, params->tu_->k_gm_wm_ratio_, params->tu_->k_glm_wm_ratio_, mat_prop_, params); CHKERRQ(ierr);
  ierr = rho_->setValues(params->tu_->rho_, params->tu_->r_gm_wm_ratio_, params->tu_->r_glm_wm_ratio_, mat_prop_, params); CHKERRQ(ierr);
  ierr = phi_->setValues(mat_prop_); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode Tumor::computeForce(Vec c1) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  Event e("tumor-compute-force");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  std::bitset<3> XYZ;
  XYZ[0] = 1;
  XYZ[1] = 1;
  XYZ[2] = 1;

  ScalarType *c_ptr, *fx_ptr, *fy_ptr, *fz_ptr;
  ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / params_->grid_->n_[0];  // smooth because c might be too sharp at vt boundaries

  ierr = VecCopy(c1, work_[0]); CHKERRQ(ierr);
  ierr = spec_ops_->weierstrassSmoother(work_[0], work_[0], params_, sigma_smooth); CHKERRQ(ierr);
  spec_ops_->computeGradient(force_->x_, force_->y_, force_->z_, work_[0], &XYZ, t.data());

  // scale force by constant
  ierr = force_->scale(params_->tu_->forcing_factor_); CHKERRQ(ierr);

  if (params_->tu_->use_tanh_scaling_) {
    ierr = force_->getComponentArrays(fx_ptr, fy_ptr, fz_ptr); CHKERRQ(ierr);
    ierr = vecGetArray(work_[0], &c_ptr); CHKERRQ(ierr);
#ifdef CUDA
    nonlinearForceScalingCuda(c_ptr, fx_ptr, fy_ptr, fz_ptr, params_->grid_->nl_);
#else
    for (int i = 0; i < params_->grid_->nl_; i++) {
      fx_ptr[i] *= tanh(c_ptr[i]);
      fy_ptr[i] *= tanh(c_ptr[i]);
      fz_ptr[i] *= tanh(c_ptr[i]);
    }
#endif
    ierr = vecRestoreArray(work_[0], &c_ptr); CHKERRQ(ierr);
    ierr = force_->restoreComponentArrays(fx_ptr, fy_ptr, fz_ptr); CHKERRQ(ierr);
  }

  self_exec_time += MPI_Wtime();
  accumulateTimers(params_->tu_->timers_, t, self_exec_time);
  e.addTimings(t);
  e.stop();

  PetscFunctionReturn(ierr);
}

PetscErrorCode Tumor::computeEdema() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *i_ptr, *ed_ptr;
  ierr = VecGetArray(species_["infiltrative"], &i_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(species_["edema"], &ed_ptr); CHKERRQ(ierr);

  for (int i = 0; i < params_->grid_->nl_; i++) ed_ptr[i] = (i_ptr[i] > params_->tu_->invasive_threshold_) ? 1.0 : 0.0;

  ierr = VecRestoreArray(species_["infiltrative"], &i_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(species_["edema"], &ed_ptr); CHKERRQ(ierr);

  // smooth
  ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / params_->grid_->n_[0];
  ierr = spec_ops_->weierstrassSmoother(species_["edema"], species_["edema"], params_, sigma_smooth); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode Tumor::computeSegmentation() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ierr = VecSet(seg_, 0); CHKERRQ(ierr);
  // compute seg_ of gm, wm, vt, bg, tumor
  std::vector<ScalarType> v;
  std::vector<ScalarType>::iterator seg_component;
  ScalarType *bg_ptr, *gm_ptr, *wm_ptr, *vt_ptr, *c_ptr, *csf_ptr, *seg_ptr, *p_ptr, *n_ptr, *ed_ptr;
  ierr = vecGetArray(mat_prop_->bg_, &bg_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(mat_prop_->wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(mat_prop_->vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(mat_prop_->csf_, &csf_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(seg_, &seg_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(c_t_, &c_ptr); CHKERRQ(ierr);

#ifdef CUDA
  computeTumorSegmentationCuda(bg_ptr, gm_ptr, wm_ptr, vt_ptr, csf_ptr, c_ptr, seg_ptr, params_->grid_->nl_);
#else
  //     ierr = VecGetArray (species_["proliferative"], &p_ptr);       CHKERRQ(ierr); CHKERRQ(ierr);
  //     ierr = VecGetArray (species_["necrotic"], &n_ptr);            CHKERRQ(ierr); CHKERRQ(ierr);

  //     for (int i = 0; i < params_->grid_->nl_; i++) {
  //         v.push_back (bg_ptr[i]);
  //         v.push_back (p_ptr[i]);
  //         v.push_back (n_ptr[i]);
  //         v.push_back (ed_ptr[i]);
  //         v.push_back (wm_ptr[i]);
  //         v.push_back (gm_ptr[i]);
  //         v.push_back (vt_ptr[i]);
  //         v.push_back (csf_ptr[i]);

  //         seg_component = std::max_element (v.begin(), v.end());
  //         seg_ptr[i] = std::distance (v.begin(), seg_component);

  //         v.clear();
  //     }
  // } else {
  ScalarType max = 0;
  ScalarType w, g, v, c;
  int ct = 0;
  for (int i = 0; i < params_->grid_->nl_; i++) {
    max = bg_ptr[i];
    ct = 0;
    w = wm_ptr[i] * (1 - c_ptr[i]);
    g = gm_ptr[i] * (1 - c_ptr[i]);
    v = csf_ptr[i] * (1 - c_ptr[i]);
    c = glm_ptr[i] * (1 - c_ptr[i]);
    if (c_ptr[i] > max) {max = c_ptr[i]; ct = 1;}
    if (w > max) {max = w; ct = 6;}
    if (g > max) {max = g; ct = 5;}
    if (v > max) {max = v; ct = 7;}
    if (c > max) {max = c; ct = 8;}
    seg_ptr[i] = ct;
    //v.push_back(bg_ptr[i]);
    //v.push_back(c_ptr[i]);
    //v.push_back(wm_ptr[i]);
    //v.push_back(gm_ptr[i]);
    //v.push_back(vt_ptr[i]);
    //v.push_back(csf_ptr[i]);

    //seg_component = std::max_element(v.begin(), v.end());
    //my_label   = std::distance(v.begin(), seg_component);

    //if (my_label == 2) {
    //  my_label = 6;
    //} else if (my_label == 3) {
    //  my_label = 5;
    //} else if (my_label == 4) {
    //  my_label = 7;
    //} else if (my_label == 5) {
    //  my_label = 8;
    //}

    //seg_ptr[i] = my_label;  

    //v.clear();
  }
  // }
#endif

  ierr = vecRestoreArray(mat_prop_->bg_, &bg_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(mat_prop_->wm_, &wm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(mat_prop_->vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(mat_prop_->csf_, &csf_ptr); CHKERRQ(ierr);
  //     ierr = VecRestoreArray (species_["proliferative"], &p_ptr);       CHKERRQ(ierr); CHKERRQ(ierr);
  //     ierr = VecRestoreArray (species_["necrotic"], &n_ptr);            CHKERRQ(ierr); CHKERRQ(ierr);
  // } else {
  ierr = vecRestoreArray(c_t_, &c_ptr); CHKERRQ(ierr);
  // }
  ierr = vecRestoreArray(seg_, &seg_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


PetscErrorCode Tumor::getTCRecon(Vec x) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *x_ptr, *seg_ptr;
  ierr = VecGetArray(x, &x_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(seg_, &seg_ptr); CHKERRQ(ierr);
  for (int i = 0; i < params_->grid_->nl_; i++) x_ptr[i] = (seg_ptr[i] == 1) ? 1 : 0;
  ierr = VecRestoreArray(x, &x_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(seg_, &seg_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode Tumor::computeSpeciesNorms() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType nrm;
  std::stringstream s;

  if (params_->tu_->model_ == 5) {
    ierr = VecNorm(species_["proliferative"], NORM_2, &nrm); CHKERRQ(ierr);
    s << "Proliferative cell concentration norm = " << nrm;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    ierr = VecNorm(species_["infiltrative"], NORM_2, &nrm); CHKERRQ(ierr);
    s << "Infiltrative cell concentration norm = " << nrm;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    ierr = VecNorm(species_["necrotic"], NORM_2, &nrm); CHKERRQ(ierr);
    s << "Necrotic cell concentration norm = " << nrm;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode Tumor::clipTumor() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *c_ptr;
  ierr = vecGetArray(c_t_, &c_ptr); CHKERRQ(ierr);

  if (params_->tu_->model_ == 5) {
    ScalarType *p_ptr, *i_ptr, *n_ptr;
    ierr = vecGetArray(species_["proliferative"], &p_ptr); CHKERRQ(ierr);
    ierr = vecGetArray(species_["infiltrative"], &i_ptr); CHKERRQ(ierr);
    ierr = vecGetArray(species_["necrotic"], &n_ptr); CHKERRQ(ierr);

#ifdef CUDA
    clipVectorCuda(p_ptr, params_->grid_->nl_);
    clipVectorCuda(i_ptr, params_->grid_->nl_);
    clipVectorCuda(n_ptr, params_->grid_->nl_);
#else
    for (int i = 0; i < params_->grid_->nl_; i++) {
      p_ptr[i] = (p_ptr[i] <= 0.) ? 0. : p_ptr[i];
      i_ptr[i] = (i_ptr[i] <= 0.) ? 0. : i_ptr[i];
      n_ptr[i] = (n_ptr[i] <= 0.) ? 0. : n_ptr[i];
    }
#endif

    ierr = vecRestoreArray(species_["proliferative"], &p_ptr); CHKERRQ(ierr);
    ierr = vecRestoreArray(species_["infiltrative"], &i_ptr); CHKERRQ(ierr);
    ierr = vecRestoreArray(species_["necrotic"], &n_ptr); CHKERRQ(ierr);

  } else {
#ifdef CUDA
    clipVectorCuda(c_ptr, params_->grid_->nl_);
    clipVectorAboveCuda(c_ptr, params_->grid_->nl_);
#else
    for (int i = 0; i < params_->grid_->nl_; i++) {
      c_ptr[i] = (c_ptr[i] <= 0.) ? 0. : c_ptr[i];
      c_ptr[i] = (c_ptr[i] > 1.) ? 1. : c_ptr[i];
    }
#endif
  }

  ierr = vecRestoreArray(c_t_, &c_ptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

Tumor::~Tumor() {
  PetscErrorCode ierr;
  ierr = VecDestroy(&c_t_);
  ierr = VecDestroy(&c_0_);
  ierr = VecDestroy(&p_t_);
  ierr = VecDestroy(&p_0_);
  if (p_ != nullptr)
    ierr = VecDestroy(&p_);

  for (int i = 0; i < 12; i++) {
    ierr = VecDestroy(&work_[i]);
  }
  delete[] work_;
  ierr = VecDestroy(&seg_);

  for (std::map<std::string, Vec>::iterator it = species_.begin(); it != species_.end(); it++) {
    ierr = VecDestroy(&it->second);
  }
}
