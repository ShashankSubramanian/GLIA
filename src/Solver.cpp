
// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
Solver::Solver(std::shared_ptr<SpectralOperators> spec_ops)
:
  params_(nullptr),
  solver_interface_(nullptr),
  spec_ops_(spec_ops),
  tumor_(nullptr),
  custom_obs_(false),
  warmstart_p_(false),
  synthetic_(false),
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
  velocity_(nullptr) {

  solver_interface_ = std::make_shared<TumorSolverInterface>(params_, spec_ops, nullptr, nullptr);
  tumor_ = std::shared_ptr<Tumor> tumor = solver_interface->getTumor();
  EventRegistry::initialize();
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  // set parameters, populate to optimizer
  params_ = params;
  ierr = solver_interface_->setOptimizerSettings(params_->opt_); CHKERRQ (ierr);
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

  // read  healthy segmentation wm, gm, csf, (ve) for simulation
  ierr = readAtlas(params_); CHKERRQ(ierr);
  ierr = solver_interface_->updateTumorCoefficients (wm_, gm_, glm_, csf_, nullptr); // TODO(K): can we get rid of bg?

  int fwd_temp = params_->forward_flag_;
  if(synthetic_) {
    // data t1 and data t0 is generated synthetically using user given cm and tumor model
    ierr = readUserCMs(); CHKERRQ(ierr); // TODO(K): implement
    ierr = generateSyntheticData (); CHKERRQ(ierr);  // TODO(K): implement
    data_support_ = data_t1_;
  } else {
    // read in target data (t1 and/or t0); observation operator
    ierr = readData(); CHKERRQ(ierr);
  }
  // reset forward flag so that time-history can be stored now if solver is in inverse-mode
  params_->forward_flag_ = fwd_temp;


  warmstart_p_ = !params_->path_->pvec_.empty();
  custom_obs_ = !params_->path_->obs_filter_.empty();
  synthetic_ = params_->syn_flag_;

  // error handling for some configuration inconsistencies
  std::stringstream;
  if (warmstart_p  && params_->path_->phi_.empty()) {
    ss << " ERROR: Initial guess for p is given but no coordinates for phi. Exiting. "; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    exit(0);
  }
  if (params_->inject_coarse_sol_ && (!warmstart_p || params_->path_->phi_.empty())) {
    ss << " ERROR: Trying to inject coarse solution, but p/phi not specified. Exiting. "; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    exit(-1);
  }

  #ifdef CUDA
    cudaPrintDeviceMemory ();
  #endif

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::readAtlas() {
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
PetscErrorCode Solver::readData() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  std::stringstream ss;
  ScalarType sigma_smooth = params_->smoothing_factor_ * 2 * M_PI / params_->grid_->n_[0];
  ScalarType sig_data = smooth_fac_data_ * 2 * M_PI / params->grid->n_[0];
  ScalarType min, max;

  if(!params_->path_->data_t1_.empty()) {
    ierr = VecDuplicate(tmp_, &data_t1_); CHKERRQ (ierr);
    dataIn (data_t1_, params_, params_->path_->data_t1_);
    if(smooth_fac_data_ > 0) {
      ierr = spec_ops->weierstrassSmoother (data_t1_, data_t1_, params_, sig_data); CHKERRQ (ierr);
      ss << " smoothing c(1) with factor: "<<smooth_fac_data_<<", and sigma: "<<sig_data; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    }
    // make obs threshold relaltive
    if(params_->relative_obs_threshold_) {
      ierr = VecMax (data_t1_, NULL, &max); CHKERRQ (ierr);
      params_->obs_threshold_1_ *= max;
      ss << " Changing observation threshold for d_1 to thr_1: "<<params_->obs_threshold_1_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    }
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
    ss << " Reading custom observation mask"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }

  ScalarType *ptr;
  if(!params_->path_->data_t0_.empty()) {
    ierr = VecDuplicate(tmp_, &data_t0_); CHKERRQ (ierr);
    dataIn (data_t0_, params_, params_->path_->data_t0_);
    ierr = VecMin(data_t0_, NULL, &min); CHKERRQ (ierr);
    if (min < 0) {
      ss << " tumor init is aliased with min " << min << "; clipping and smoothing..."; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
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
    if(data_t0_ != nullptr) {
      if(smooth_fac_data_ > 0) {
        ierr = spec_ops->weierstrassSmoother (data_t0_, data_t0_, params_, sig_data); CHKERRQ (ierr);
        ss << " smoothing c(1) with factor: "<<smooth_fac_data_<<", and sigma: "<<sig_data; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
      }
    }
    ierr = VecMax (data_t0_, NULL, &max); CHKERRQ (ierr);
    ierr = VecScale (data_t0_, (1.0 / max)); CHKERRQ (ierr);
    ierr = dataOut (data_t0_, params, "c0True.nc"); CHKERRQ (ierr);

    // make obs threshold relaltive
    if(params_->relative_obs_threshold_) {
      params_->obs_threshold_0_ *= max;
      ss << " Changing observation threshold for d_0 to thr_0: "<<params_->obs_threshold_0_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    }
  } else {
      ierr = VecSet(data_t0_, 0.);        CHKERRQ (ierr);
  }

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::readVelocity() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  if(params_->path_->velocity_x1_) {
    ierr = readVecField(tumor_->velocity_.get(), params_->path_->velocity_x1_, params_->path_->velocity_x2_, params_->path_->velocity_x3_, params_); CHKERRQ(ierr);
    ierr = tumor_->velocity_->scale(-1); CHKERRQ(ierr);
    Vec mag; ierr = VecDuplicate (data_t1_, &mag); CHKERRQ(ierr);
    ierr = tumor->velocity_->computeMagnitude(mag); CHKERRQ(ierr);
    dataOut (mag, params_, "velocity_mag.nc");
    ScalarType vnorm;
    ierr = VecNorm(mag, NORM_2, &vnorm); CHKERRQ(ierr);
    std::Stringstream ss;
    ss << " Given velocity read in: norm of velocity magnitude: "<<vnorm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ierr = VecDestroy(&mag); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::readDiffusionFiberTensor() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // TODO(K): implement reading in of DTI tensor.
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::readUserCMs() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  // TODO(K): implement reading in of different user cms, write out p_vec/phi file,
  //          and create c(0); store in data_t0_

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::createSynthetic() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  // TODO(K): implement synthetic data generation

  // if (n_misc->testcase_ == BRAINFARMF || n_misc->testcase_ == BRAINNEARMF) {
  //     ierr = createMFData (c_0, data, p_rec, solver_interface, n_misc, mri_path);
  // } else {
  //     ierr = generateSyntheticData (c_0, data, p_rec, solver_interface, n_misc, spec_ops, init_tumor_path, mri_path);
  // }

  // TODO(K): implement low frequency noise

  // if (n_misc->low_freq_noise_scale_ != 0) {
  //     ierr = applyLowFreqNoise (data, n_misc);
  //     Vec temp;
  //     ScalarType noise_err_norm, rel_noise_err_norm;
  //     ierr = VecDuplicate (data, &temp);      CHKERRQ (ierr);
  //     ierr = VecSet (temp, 0.);               CHKERRQ (ierr);
  //     ierr = VecCopy (data_nonoise, temp);    CHKERRQ (ierr);
  //     ierr = VecAXPY (temp, -1.0, data);      CHKERRQ (ierr);
  //     ierr = VecNorm (temp, NORM_2, &noise_err_norm);               CHKERRQ (ierr);  // diff btw noise corrupted signal and ground truth
  //     ierr = VecNorm (data_nonoise, NORM_2, &rel_noise_err_norm);   CHKERRQ (ierr);
  //     rel_noise_err_norm = noise_err_norm / rel_noise_err_norm;
  //     ss << " low frequency relative error = " << rel_noise_err_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  //
  //     // if (n_misc->writeOutput_)
  //     //     dataOut (data, n_misc, "dataNoise.nc");
  //
  //     if (temp != nullptr) {ierr = VecDestroy (&temp);          CHKERRQ (ierr); temp = nullptr;}

  // ss << " data generated with parameters: rho = " << n_misc->rho_ << " k = " << n_misc->k_ << " dt = " << n_misc->dt_ << " Nt = " << n_misc->nt_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // if (n_misc->model_ >= 4) {
  //     ss << " mass-effect forcing factor used = " << n_misc->forcing_factor_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  //     // write out p and phi so that they can be used if needed
  //     writeCheckpoint(tumor->p_true_, tumor->phi_, n_misc->writepath_.str(), "forward");
  //     ss << " ground truth phi and p written to file "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // }

  PetscFunctionReturn(ierr);
}




/* #### ------------------------------------------------------------------- #### */
/* #### ========                   ForwardSolver                   ======== #### */
/* #### ------------------------------------------------------------------- #### */


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode ForwardSolver::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  // set and populate parameters; read material properties; read data
  Solver::initialize(params);  CHKERRQ(ierr);


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

  // set and populate parameters; read material properties; read data
  Solver::initialize(params);  CHKERRQ(ierr);



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

  // set and populate parameters; read material properties; read data
  Solver::initialize(params);  CHKERRQ(ierr);


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

  // set and populate parameters; read material properties; read data
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

  // set and populate parameters; read material properties; read data
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
