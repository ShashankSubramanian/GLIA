
// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
Solver::Solver(std::shared_ptr<SpectralOperators> spec_ops)
:
  params_(nullptr),
  solver_interface_(nullptr),
  spec_ops_(spec_ops),
  tumor_(nullptr),
  wm_(nullptr),
  gm_(nullptr),
  csf_(nullptr),
  ve_(nullptr),
  glm_(nullptr),
  tmp_(nullptr),
  data_t1_(nullptr),
  data_t0_(nullptr),
  data_support_(nullptr),
  data_comps_(nullptr),
  obs_filter_(nullptr),
  v_(nullptr) {

  solver_interface_ = std::make_shared<TumorSolverInterface>(n_misc, spec_ops, nullptr, nullptr);
  tumor_ = std::shared_ptr<Tumor> tumor = solver_interface->getTumor();
  EventRegistry::initialize();
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  params_ = params;
  ierr = solver_interface_->setOptimizerSettings(params->opt_); CHKERRQ (ierr);

  // create tmp vector according to distributed grid
  ierr = VecCreate(PETSC_COMM_WORLD, &tmp_); CHKERRQ (ierr);
  ierr = VecSetSizes(tmp_, params_->grid_->nl_, params_->grid_->ng_); CHKERRQ (ierr);
  ierr = setupVec(tmp_); CHKERRQ (ierr);
  ierr = VecSet(tmp_, 0.0); CHKERRQ (ierr);
  // create p_rec vector according to unknowns inverted for
  int np = params_->tu_->np_;
  int nk = (params_->tu_->diffusivity_inversion_) ? params_->tu_->nk_ : 0;
  ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p_rec_); CHKERRQ (ierr);
  ierr = setupVec (p_rec_, SEQ); CHKERRQ (ierr);

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::readAtlas(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ScalarType sigma_smooth = params_->smoothing_factor_ * 2 * M_PI / params_->n_[0];

  if(!params_->path_->seg_.empty()) {
    dataIn (tmp_, params_, params_->path_->seg_);
    // TODO(K): populate to wm, gm, csf, ve
  } else {
    if(!params_->path_->wm_.empty()) {
      ierr = VecDuplicate(tmp_, &wm_); CHKERRQ (ierr);
      dataIn (wm_, params_, params_->path_->wm_);
    }
    if(!params_->path_->gm_.empty()) {
      ierr = VecDuplicate(tmp_, &gm_); CHKERRQ (ierr);
      dataIn (gm_, params_, params_->path_->gm_);
    }
    if(!params_->path_->csf_.empty()) {
      ierr = VecDuplicate(tmp_, &csf_); CHKERRQ (ierr);
      dataIn (csf_, params_, params_->path_->csf_);
    }
    if(!params_->path_->ve_.empty()) {
      ierr = VecDuplicate(tmp_, &ve_); CHKERRQ (ierr);
      dataIn (ve_, params_, params_->path_->ve_);
    }
  }
  // mass effect
  if(!params_->path_->glm_.empty() && params_->model_ >= 4) {
      ierr = VecDuplicate(tmp_, &glm_); CHKERRQ (ierr);
      dataIn (glm_, params_, params_->path_->glm_);
  }
  // smooth
  if (gm_  != nullptr) {ierr = spec_ops->weierstrassSmoother (gm_, gm_, params_, sigma_smooth); CHKERRQ (ierr);}
  if (wm_  != nullptr) {ierr = spec_ops->weierstrassSmoother (wm_, wm_, params_, sigma_smooth); CHKERRQ (ierr);}
  if (csf_ != nullptr) {ierr = spec_ops->weierstrassSmoother (csf_, csf_, params_, sigma_smooth); CHKERRQ (ierr);}
  if (ve_  != nullptr) {ierr = spec_ops->weierstrassSmoother (ve_, ve_, params_, sigma_smooth); CHKERRQ (ierr);}
  if (glm_ != nullptr) {ierr = spec_ops->weierstrassSmoother (glm_, glm_, params_, sigma_smooth); CHKERRQ(ierr);}

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::readData(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ScalarType sigma_smooth = params_->smoothing_factor_ * 2 * M_PI / params_->n_[0];

  if(!params_->path_->data_t1_.empty()) {
    ierr = VecDuplicate(tmp_, &data_t1_); CHKERRQ (ierr);
    dataIn (data_t1_, params_, params_->path_->data_t1_);
    ierr = spec_ops->weierstrassSmoother (data_t1_, data_t1_, params_, sigma_smooth); CHKERRQ (ierr);
  }
  bool read_supp;
  if(!params_->path_->data_support_.empty()) {
    std::string file, path, ext;
    ierr = getFileName(path, file, ext, params_->path_->data_support_); CHKERRQ(ierr);
    read_supp = (strcmp(ext.c_str(),".nc") == 0) || (strcmp(ext.c_str(),".nii.gz") == 0); // file ends with *.nc or *.nii.gz?
    if(read_supp) {
      ierr = VecDuplicate(tmp_, &data_support_); CHKERRQ (ierr);
      dataIn (data_support_, params_, params_->path_->data_support_);
    }
  } else {
    data_support_ = data_t1_;
  }
  if(read_supp && !params_->path_->data_comps_.empty()) {
    ierr = VecDuplicate(tmp_, &data_comps_); CHKERRQ (ierr);
    dataIn (data_comps_, params_, params_->path_->data_comps_);
  }
  if(!params_->path_->obs_filter_.empty()) {
    ierr = VecDuplicate(tmp_, &obs_filter_); CHKERRQ (ierr);
    dataIn (obs_filter_, params_, params_->path_->obs_filter_);
  }

  ScalarType *ptr;
  ScalarType c0_min, c0_max;
  std::stringstream ss;
  if(!params_->path_->data_t0_.empty()) {
    ierr = VecDuplicate(tmp_, &data_t0_); CHKERRQ (ierr);
    dataIn (data_t0_, params_, params_->path_->data_t0_);
    ierr = VecMin(data_t0_, NULL, &c0_min); CHKERRQ (ierr);
    if (c0_min < 0) {
      ss << " tumor init is aliased with min " << c0_min << "; clipping and smoothing..."; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
      ierr = vecGetArray(data_t0_, &ptr); CHKERRQ (ierr);
      #ifdef CUDA
          clipVectorCuda(ptr, params_->grid_->nl_);
      #else
          for (int i = 0; i < params_->grid_->nl_; i++)
              ptr[i] = (ptr[i] <= 0.) ? 0. : ptr[i];
      #endif
      ierr = vecRestoreArray(data_t0_, &ptr); CHKERRQ (ierr);
    }
    // smooth a little bit because sometimes registration outputs have high gradients
    if(data_t0_ != nullptr) {ierr = spec_ops->weierstrassSmoother (data_t0_, data_t0_, params_, sigma_smooth); CHKERRQ (ierr);}
    ierr = VecMax (data_t0_, NULL, &c0_max); CHKERRQ (ierr);
    ierr = VecScale (data_t0_, (1.0 / c0_max)); CHKERRQ (ierr);
    ierr = dataOut (data_t0_, params, "c0True.nc"); CHKERRQ (ierr);
  } else {
      ierr = VecSet(data_t0_, 0.);        CHKERRQ (ierr);
  }

  PetscFunctionReturn(ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========                   InverseL2Solver                 ======== #### */
/* #### ------------------------------------------------------------------- #### */


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL2Solver::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  Solver::initialize(params);


  PetscFunctionReturn(ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========                   InverseL1Solver                 ======== #### */
/* #### ------------------------------------------------------------------- #### */


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL1Solver::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  Solver::initialize(params);


  PetscFunctionReturn(ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========          InverseReactionDiffusionSolver           ======== #### */
/* #### ------------------------------------------------------------------- #### */


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseReactionDiffusionSolver::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  Solver::initialize(params);


  PetscFunctionReturn(ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========               InverseMassEffectSolver             ======== #### */
/* #### ------------------------------------------------------------------- #### */


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  Solver::initialize(params);


  PetscFunctionReturn(ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========             InverseMultiSpeciesSolver             ======== #### */
/* #### ------------------------------------------------------------------- #### */


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMultiSpeciesSolver::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  Solver::initialize(params);


  PetscFunctionReturn(ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========                      TestSuite                    ======== #### */
/* #### ------------------------------------------------------------------- #### */


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TestSuite::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  Solver::initialize(params);


  PetscFunctionReturn(ierr);
}
