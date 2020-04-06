
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
  mri_(nullptr),
  tmp_(nullptr),
  data_t1_(nullptr),
  data_t0_(nullptr),
  data_support_(nullptr),
  data_comps_(nullptr),
  obs_filter_(nullptr),
  velocity_(nullptr) {

  solver_interface_ = std::make_shared<TumorSolverInterface>(params_, spec_ops, nullptr, nullptr);
  tumor_ = std::shared_ptr<Tumor> tumor = solver_interface->getTumor();
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::initialize(std::shared_ptr<Parameters> params) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream;

  // === set parameters, populate to optimizer
  params_ = params;
  ierr = solver_interface_->setOptimizerSettings(params_->opt_); CHKERRQ (ierr);
  // === create tmp vector according to distributed grid
  ierr = VecCreate(PETSC_COMM_WORLD, &tmp_); CHKERRQ (ierr);
  ierr = VecSetSizes(tmp_, params_->grid_->nl_, params_->grid_->ng_); CHKERRQ (ierr);
  ierr = setupVec(tmp_); CHKERRQ (ierr);
  ierr = VecSet(tmp_, 0.0); CHKERRQ (ierr);
  // === create p_rec vector according to unknowns inverted for
  int np = params_->tu_->np_;
  int nk = (params_->tu_->diffusivity_inversion_) ? params_->tu_->nk_ : 0;
  ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p_rec_); CHKERRQ (ierr);
  ierr = setupVec (p_rec_, SEQ); CHKERRQ (ierr);

  warmstart_p_ = !params_->path_->pvec_.empty();
  custom_obs_ = !params_->path_->obs_filter_.empty();
  synthetic_ = params_->syn_flag_;
  has_dt0_ = params_->two_time_points_ || !params_->path_->data_t0_.empty();

  // === error handling for some configuration inconsistencies
  if (warmstart_p_  && params_->path_->phi_.empty()) {
    ss << " ERROR: Initial guess for p is given but no coordinates for phi. Exiting. "; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    exit(0);
  }
  if (params_->inject_coarse_sol_ && (!warmstart_p_ || params_->path_->phi_.empty())) {
    ss << " ERROR: Trying to inject coarse solution, but p/phi not specified. Exiting. "; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    exit(-1);
  }

  // === read brain: healthy segmentation wm, gm, csf, (ve) for simulation
  ierr = readAtlas(params_); CHKERRQ(ierr);

  // === read in user given velocity
  ierr = readVelocity(); CHKERRQ(ierr);

  // === advect healthy material properties with read in velocity, if given
  if (params_->pre_adv_time_ > 0 && velocity_ != nullptr) {
    ss << " pre-advecting material properties with velocity to time t="<<params_->pre_adv_time_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    if(!params_->path_->mri_.empty()) {
      ierr = VecDuplicate (data_t1_, &mri_); CHKERRQ (ierr);
      dataIn (mri_, params_, params_->path_->mri_);
    }
    ierr = solver_interface_->getPdeOperators()->preAdvection(wm_, gm_, csf_, mri_, params_->pre_adv_time_); CHKERRQ(ierr);
	}

  ierr = solver_interface_->updateTumorCoefficients (wm_, gm_, glm_, csf_, nullptr); CHKERRQ(ierr); // TODO(K): can we get rid of bg?
  ierr = tumor_->mat_prop_->setAtlas(gm_, wm_, glm_, csf_, nullptr); CHKERRQ(ierr); // TODO(K): can we get rid of bg?

  // === read data: generate synthetic or read real
  if(synthetic_) {
    int fwd_temp = params_->forward_flag_; // temporarily disable time_history
    // data t1 and data t0 is generated synthetically using user given cm and tumor model
    ierr = generateSyntheticData(); CHKERRQ(ierr);  // TODO(K): implement
    data_support_ = data_t1_;
    params_->forward_flag_ = fwd_temp; // restore mode, i.e., allow inverse solver to store time_history
  } else {
    // read in target data (t1 and/or t0); observation operator
    ierr = readData(); CHKERRQ(ierr);
  }

  // === set observation operator
  if(custom_obs_) {
    ierr = tumor_->obs_->setCustomFilter (obs_filter_, 1);
    ss << " Setting custom observation mask"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  } else {
    ierr = tumor_->obs_->setDefaultFilter (data_t1_, 1, params_->obs_threshold_1_); CHKERRQ(ierr);
    if(has_dt0_) {ierr = tumor_->obs_->setDefaultFilter (data_t0_, 0, params_->obs_threshold_0_); CHKERRQ(ierr);}
    ss << " Setting default observation mask based on input data (d1) and threshold " << tumor_->obs_->threshold_1_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << " Setting default observation mask based on input data (d0) and threshold " << tumor_->obs_->threshold_0_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }

  // === apply observation operator to data
  ierr = tumor_->obs_->apply (data_t1_, data_t1_), 1; CHKERRQ (ierr);
  ierr = tumor_->obs_->apply (data_support_, data_support_, 1); CHKERRQ (ierr);
  if(has_dt0_) {ierr = tumor_->obs_->apply (data_t0_, data_t0_, 0); CHKERRQ (ierr);}


  #ifdef CUDA
    cudaPrintDeviceMemory ();
  #endif

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ss << " Inversion with tumor parameters: rho = " << params_->tu_->rho_ << " k = " << params_->tu_->k_ << " dt = " << params_->tu_->dt_ << " Nt = " << params_->tu_->nt_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  ss << " Results in: " << params_->path_->writepath_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  // TODO(K) implement below:
  // ierr = VecNorm (p_rec, NORM_2, &prec_norm);                            CHKERRQ (ierr);
  // ss << " reconstructed p norm: " << prec_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // if (n_misc->diffusivity_inversion_ && !n_misc->reaction_inversion_) {
  //     ierr = VecGetArray (p_rec, &prec_ptr);                             CHKERRQ (ierr);
  //     ss << " k1: " << (n_misc->nk_ > 0 ? prec_ptr[n_misc->np_] : 0);     ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  //     ss << " k2: " << (n_misc->nk_ > 1 ? prec_ptr[n_misc->np_ + 1] : 0); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  //     ss << " k3: " << (n_misc->nk_ > 2 ? prec_ptr[n_misc->np_ + 2] : 0); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  //     ierr = VecRestoreArray (p_rec, &prec_ptr);                         CHKERRQ (ierr);
  // }
  // if (mri_path != NULL && mri_path[0] != '\0')
  //     n_misc->transport_mri_ = true;
  // ierr = computeError (l2_rel_error, error_norm_c0, p_rec, data_nonoise, data, c_0, solver_interface, n_misc, mri_path, init_tumor_path);
  // ss << " l2-error in reconstruction: " << l2_rel_error; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // ss << " --------------  RECONST P -----------------";  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // if (procid == 0) {
  //     ierr = VecView (p_rec, PETSC_VIEWER_STDOUT_SELF);                   CHKERRQ (ierr);
  // }
  // ss << " --------------  -------------- -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  //
  // ierr = computeSegmentation(tumor, n_misc, spec_ops);      // Writes segmentation with c0 and c1
  //
  // std::stringstream sstm;
  // sstm << n_misc->writepath_ .str().c_str() << "reconP.dat";
  // std::ofstream ofile;
  //
  // //write reconstructed p into text file
  // if (procid == 0) {
  //     ofile.open (sstm.str().c_str());
  //     ierr = VecGetArray (p_rec, &prec_ptr);                             CHKERRQ (ierr);
  //     int np = n_misc->np_;
  //     int nk = (n_misc->diffusivity_inversion_) ? n_misc->nk_ : 0;
  //     for (int i = 0; i < np + nk; i++)
  //         ofile << prec_ptr[i] << std::endl;
  //     ierr = VecRestoreArray (p_rec, &prec_ptr);                         CHKERRQ (ierr);
  //     ofile.close ();
  // }
  //
  // // write p to bin file
  // std::string fname = n_misc->writepath_ .str() + "p-final.bin";
  // writeBIN(p_rec, fname);



  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::predict() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  if(params_->pred_->enabled_) {
    ss << " Predicting future tumor growth..."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    if(params_->time_history_off_) {
      params_->tu_->dt_ = params_->pred_->dt_;
      // set c(0)
      if (params_->use_c0_) {ierr = VecCopy(data_t0, tumor_->c_0_); CHKERRQ(ierr);}
      else                  {ierr = tumor_->phi_->apply(tumor_->c_0_, p_rec_);}

      // predict tumor growth at different (user defined) times
      for(int i = 0; i < params_->pred_->t_pred_.size(); ++i) {
        params_->tu_->nt_ = (int) (params_->pred_->t_pred_[i] / params_->pred_->dt_); // number of time steps
        // if different brain to perform prediction is given, read in and reset atlas
        if(params_->pred_->wm_path.size() >= i && !params_->pred_->wm_path_[i].empty()) {
          params_->path_wm_ = params_->pred_->wm_path_[i];
          params_->path_gm_ = params_->pred_->gm_path_[i];
          params_->path_csf_ = params_->pred_->csf_path_[i];
          ierr = tuMSGstd(" .. reading in atlas brain to perform prediction."); CHKERRQ(ierr);
          ierr = readAtlas(); CHKERRQ(ierr);
          ierr = solver_interface_->updateTumorCoefficients (wm_, gm_, nullptr, csf_, nullptr); // TODO(K) is this safe, giving nullptr?
          ierr = tumor_->mat_prop_->setAtlas(gm_, wm_, nullptr, csf_, nullptr); CHKERRQ(ierr);
        } else {
          ierr = tumor_->mat_prop_->resetValues(); CHKERRQ(ierr);
        }
        ierr = solver_interface_->getPdeOperators()->solveState (0); CHKERRQ(ierr); // forward solve
        ss << "c_pred_at_[t=" << params_->pred_->t_pred_[i] <<"]";
        dataOut (tumor_->c_t_, params_, ss.str()+params_->path_->ext_); ss.str(""); ss.clear();
        ss << " .. prediction complete for t = "<<params_->pred_->t_pred_[i]; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

        // if given: compute error to ground truth at predicted time
        if(!params_->pred_->true_data_path_[i].empty()) {
          Vec true_dat_t1;
          ScalarType obs_c_norm, obs_data_norm, data_norm;
          ierr = VecDuplicate(data_t1_, &true_dat_t1); CHKERRQ (ierr);
          dataIn(true_dat_t1, params_, params_->pred_->true_data_path_[i]);
          ierr = VecNorm(true_dat_t1, NORM_2, &data_norm); CHKERRQ (ierr);
          // error everywhere
          ierr = VecCopy(tumor_->c_t_, tmp_); CHKERRQ (ierr);
          ierr = VecAXPY(tmp_, -1.0, true_dat_t1); CHKERRQ (ierr);
          ss << "res_at_[t=" << params_->pred_->t_pred_[i] <<"]";
          dataOut(tmp_, params_, ss.str()+params_->path_->ext_); ss.str(""); ss.clear();
          ierr = VecNorm (tmp_, NORM_2, &obs_c_norm); CHKERRQ (ierr);
          obs_c_norm /= data_norm;
          ss << " .. rel. l2-error (everywhere) (T=" << params_->pred_->t_pred_[i] << ") : " << obs_c_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
          // error at observation points
          ierr = VecCopy(tumor_->c_t_, tmp_); CHKERRQ (ierr);
          ierr = tumor_->obs_->apply (tmp_, tmp_); CHKERRQ (ierr);
          ierr = tumor_->obs_->apply (true_dat_t1, true_dat_t1); CHKERRQ (ierr);
          ierr = VecNorm (true_dat_t1, NORM_2, &obs_data_norm); CHKERRQ (ierr);
          ierr = VecAXPY (tmp_, -1.0, true_dat_t1); CHKERRQ (ierr);
          ss << "res_obs_at_[t=" << params_->pred_->t_pred_[i] <<"]";
          dataOut (tmp_, params_, ss.str()+params_->path_->ext_); ss.str(""); ss.clear();
          ierr = VecNorm (tmp_, NORM_2, &obs_c_norm); CHKERRQ (ierr);
          obs_c_norm /= obs_data_norm;
          ss << " .. rel. l2-error (at observation points) (T=" << params_->pred_->t_pred_[i] << ") : " << obs_c_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
          if(true_dat_t1 != nullptr) {ierr = VecDestroy (&true_dat_t1); CHKERRQ (ierr);}
        }
      }
    }
  }
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
PetscErrorCode Solver::createSynthetic() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  // save parameters
  ScalarType rho_temp = params_->tu_->rho_,
             k_temp = params_->tu_->k_,
             dt_temp = params_->tu_->dt_;
  int nt_temp = params_->tu_->nt_;
  // set to synthetic parameters
  params_->tu_->rho_ = params_->tu_->rho_data_;
  params_->tu_->k_ = params_->tu_->k_data_;
  params_->tu_->dt_ = params_->tu_->dt_data_;
  params_->tu_->nt_ = params_->tu_->nt_data_;

  ScalarType sigma_smooth = 2.0 * M_PI / n_misc->n_[0];
  ierr = VecCreateSeq (PETSC_COMM_SELF, params_->tu_->np_ + params_->get_nk(), &p_rec_); CHKERRQ (ierr);
  ierr = setupVec (p_rec_, SEQ); CHKERRQ (ierr);

  // allocate t1 and t0 data:
  if(data_t1_ == nullptr) {ierr = VecDuplicate (tmp_, &data_t1_); CHKERRQ (ierr);}
  if(data_t0_ == nullptr) {ierr = VecDuplicate (tmp_, &data_t0_); CHKERRQ (ierr);}
  ierr = VecSet (data_t0_, 0.); CHKERRQ (ierr);

  std::array<ScalarType, 3> cm_tmp;
  ScalarType scale = 1; int count = 0 ;
  // insert user defined tumor foci
  for(auto& cm : params_->syn_->user_cms_) {
    count++;
    ierr = VecSet (tmp_, 0.); CHKERRQ (ierr);
    cm_tmp[0] = (2 * M_PI / 256 * cm[0]);
    cm_tmp[1] = (2 * M_PI / 256 * cm[1]);
    cm_tmp[2] = (2 * M_PI / 256 * cm[2]);
    scale = (cm[3] < 0) ? 1 : cm[3]; // p activation stored in last entry

    ierr = tumor_->phi_->setGaussians (cm_tmp, params_->tu_->phi_sigma_, params_->tu_->phi_spacing_factor_, params_->tu_->np_);
    ierr = tumor_->phi_->setValues (tumor_->mat_prop_);
    ierr = tumor_->setTrueP (params_, scale);
    ss << " --------------  SYNTHETIC TRUE P (CM"<<count<<") -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    if (procid == 0) { ierr = VecView (tumor_->p_true_, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}
    ss << " --------------  -------------- -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ierr = tumor_->phi_->apply (tmp_, tumor_->p_true_); CHKERRQ (ierr);
    ierr = VecAXPY (data_t0_, 1.0, tmp_); CHKERRQ (ierr);
    ss << "p-syn-sm"<<count;
    writeCheckpoint(tumor_->p_true_, tumor_->phi_, params_->path_->writepath_, ss.str()); ss.str(""); ss.clear();
  }

  ScalarType max, min;
  ierr = VecMax(data_t0_, NULL, &max); CHKERRQ (ierr);
  ierr = VecMin(data_t0_, NULL, &min); CHKERRQ (ierr);

  #ifdef POSITIVITY
      ierr = enforcePositivity (data_t0_, params_);
  #endif
  if (params_->write_output_) {
      ierr = dataOut(data_t0_, n_misc, "c0_true_syn.nc"); CHKERRQ (ierr);
  }
  ierr = VecMax(data_t0_, NULL, &max); CHKERRQ (ierr);
  ierr = VecMin(data_t0_, NULL, &min); CHKERRQ (ierr);
  ss << " Synthetic data c(0) max and min : " << max << " " << min; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

  if (params_->model_ == 5) {
      std::map<std::string, Vec> species;
      ierr = solver_interface_->solveForward (data_t1_, data_t0_, &species); CHKERRQ (ierr);
  } else {
      ierr = solver_interface->solveForward (data_t1_, data_t0_); CHKERRQ (ierr);
  }
  ierr = VecMax(data_t1_, NULL, &max); CHKERRQ (ierr);
  ierr = VecMin(data_t1_, NULL, &min); CHKERRQ (ierr);
  ss << " Synthetic data c(1) max and min (before observation) : " << max << " " << min; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  if (params_->write_output_) {
    dataOut (c_t, n_misc, "c1_true_syn_before_observation.nc");
  }


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
  //     // if (params_->write_output_)
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

  // restore parameters
  params_->tu_->rho_ = rho_temp;
  params_->tu_->k_ = k_temp;
  params_->tu_->dt_ = dt_temp;
  params_->tu_->nt_ = nt_temp;

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::initializeGaussians() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ss << " Initialize Gaussian basis functions."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  if(warmstart_p_) {
    ss << "  .. solver warmstart: using p_vec and Gaussians from file."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ierr = tumor->phi_->setGaussians (params_->path_->phi_); CHKERRQ (ierr); // overwrites with custom phis
    ierr = tumor->phi_->setValues (tumor_->mat_prop_); CHKERRQ (ierr);
    ierr = readPVec(&p_rec_, n_misc->np_ + params_->get_nk() + params_->get_nr(), params_->tu_->np_, params_->path_->p_vec_); CHKERRQ (ierr);
  } else {
    if (!params_->path_->data_support_data_.empty()) { // read Gaussian centers and comp labels from txt file
      ss << "  .. reading Gaussian centers and component data from file."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
      ierr = tumor->phi_->setGaussians (params_->path_->data_support_data_, true); CHKERRQ (ierr);
    } else if(!params_->path_->data_support_.empty()) { // read Gaussian centers and comp labels from nc files
      ss << "  .. reading Gaussian centers from .nc image file."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
      ierr = tumor->phi_->setGaussians (data_support_); CHKERRQ (ierr);
    } else {
      ss << " Error: Cannot set Gaussians: expecting user input data -support_data_path *.nc or *.txt. Exiting."; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
      exit(1);
    }
    // set phi values and re-initialize p_vec
    ierr = tumor_->phi_->setValues (tumor_->mat_prop_); CHKERRQ (ierr);
    if (p_rec_ != nullptr) {ierr = VecDestroy (&p_rec_); CHKERRQ (ierr); p_rec_ = nullptr;}
    ierr = VecCreateSeq (PETSC_COMM_SELF, n_misc->np_ + params_->get_nk() + params_->get_nr(), &p_rec_); CHKERRQ (ierr); // TODO(K): check size: do we need nk or nk+nr?
    ierr = setupVec (p_rec_, SEQ); CHKERRQ (ierr);
  }

  PetscFunctionReturn(ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========                   ForwardSolver                   ======== #### */
/* #### ------------------------------------------------------------------- #### */


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode ForwardSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // no-op
  std::stringstream ss;
  ss << " Forward solve completed. Exiting."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
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
  std::stringstream ss;

  // set and populate parameters; read material properties; read data
  Solver::initialize(params);  CHKERRQ(ierr);

  // === set Gaussians
  if(inject_coarse_sol_) {
    ss << " Error: injecting coarse level solution is not supported for L2 inversion. Ignoring input."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }
  ierr = initializeGaussians(); CHKERRQ(ierr);
  ierr = solver_interface_->setParams (p_rec_, nullptr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
      ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
      ierr = solver_interface_->setInitialGuess (0.); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL2Solver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Beginning Inversion for Non-Sparse TIL, and Diffusion."); CHKERRQ(ierr);
  Solver::run();  CHKERRQ(ierr);
  ierr = solver_interface_->solveInverse(p_rec_, data_t1_, nullptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL2Solver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

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
  std::stringstream ss;

  // set and populate parameters; read material properties; read data
  Solver::initialize(params);  CHKERRQ(ierr);

  // read connected components; set sparsity level
  if(!params_->path_->data_comps_data_.empty()) {
    readConCompDat(tumor_->phi_->component_weights_, tumor_->phi_->component_centers_, params_->path_->data_comps_data_);
    int nnc = 0; for (auto w : tumor_->phi_->component_weights_) if (w >= 1E-3) nnc++; // number of significant components
    ss << " Setting sparsity level to "<< params_->sparsity_level_<< " x n_components (w > 1E-3) + n_components (w < 1E-3) = " << params_->sparsity_level_ << " x " << nnc <<" + " << (tumor_->phi_->component_weights_.size() - nnc) << " = " <<  params_->sparsity_level_ * nnc + (tumor_->phi_->component_weights_.size() - nnc); ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    params_->sparsity_level_ =  params_->sparsity_level_ * nnc + (tumor_->phi_->component_weights_.size() - nnc) ;
  }

  // === set Gaussians
  if(!warmstart_p_) { // set component labels
    if(!params_->path_->data_comps_.empty()) {
      ss << "  Setting component data from .nc image file."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
      tumor_->phi_->setLabels (data_comps_); CHKERRQ (ierr);
    }
  }

  // === inject coarse level solution
  if(!inject_coarse_sol_) {
    ierr = initializeGaussians(); CHKERRQ(ierr);
  } else {
    ss << " Injecting coarse level solution (adopting p_vec and Gaussians)."; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    Vec coarse_sol = nullptr;
    int np_save = n_misc->np_, np_coarse = 0; // save np, since overwritten in read function
    std::vector<ScalarType> coarse_sol_centers;
    ierr = readPhiMesh(coarse_sol_centers, params_, params_->path_->phi_, false); CHKERRQ (ierr);
    ierr = readPVec(&coarse_sol, params_->tu_->np_ + params_->get_nk() + params_->get_nk(), params_->tu_->np_, params_->path_->pvec_); CHKERRQ (ierr);
    np_coarse = params_->tu_->np_;
    params_->tu_->np_ = np_save; // reset to correct value
    // find coarse centers in centers_ of current Phi
    int xc, yc, zc, xf, yf, zf;
    ScalarType *xf_ptr, *xc_ptr;
    ScalarType hx =  2.0 * M_PI / n_misc->n_[0];
    ScalarType hy =  2.0 * M_PI / n_misc->n_[1];
    ScalarType hz =  2.0 * M_PI / n_misc->n_[2];
    ierr = VecGetArray(p_rec_, &xf_ptr); CHKERRQ (ierr);
    ierr = VecGetArray(coarse_sol, &xc_ptr); CHKERRQ (ierr);
    for (int j = 0; j < np_coarse; ++j) {
      for (int i = 0; i < n_misc->np_; ++i) {
        xc = (int)std::round(coarse_sol_centers[3*j + 0]/hx);
        yc = (int)std::round(coarse_sol_centers[3*j + 1]/hy);
        zc = (int)std::round(coarse_sol_centers[3*j + 2]/hz);
        xf = (int)std::round(tumor->phi_->centers_[3*i + 0]/hx);
        yf = (int)std::round(tumor->phi_->centers_[3*i + 1]/hy);
        zf = (int)std::round(tumor->phi_->centers_[3*i + 2]/hz);
        if(xc == xf && yc == yf && zc == zf) {
          xf_ptr[i] = 2 * xc_ptr[j]; // set initial guess (times 2 since sigma is halfed in every level)
          params_->support_.push_back(i); // add to support
        }
      }
    }
    ierr = VecRestoreArray (p_rec_, &xf_ptr); CHKERRQ (ierr);
    ierr = VecRestoreArray (coarse_sol, &xc_ptr); CHKERRQ (ierr);
    if (coarse_sol != nullptr) {ierr = VecDestroy(&coarse_sol); CHKERRQ(ierr); coarse_sol = nullptr;}
  }

  ierr = solver_interface_->setParams (p_rec_, nullptr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
      ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
      ierr = solver_interface_->setInitialGuess (0.); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL1Solver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Beginning Inversion for Sparse TIL, and Diffusion/Reaction."); CHKERRQ(ierr);
  Solver::run();  CHKERRQ(ierr);
  ierr = solver_interface_->solveInverseCoSaMp(p_rec_, data_t1_, nullptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL1Solver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

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

  ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  ierr = solver_interface_->setParams (p_rec_, nullptr);
  ierr = solver_interface_->setInitialGuess (0.); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseReactionDiffusionSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ierr = tuMSGwarn(" Beginning Reaction/Diffusion Inversion."); CHKERRQ(ierr);
  if (!warmstart_p && params_->path_->data_t0_.empty()) {
    ss << " Error: c(0) needs to be set, read in p and Gaussians. Exiting."; ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    exit(1);
  }
  ierr = solver_interface_->solveInverseReacDiff (p_rec_, data_t1_, nullptr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseReactionDiffusionSolver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

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

  // TODO(K): read mass effect data
  // std::shared_ptr<MData> m_data = std::make_shared<MData> (data, n_misc, spec_ops);
  // if (n_misc->model_ == 4) {
  //     n_misc->invert_mass_effect_ = 1;
  //     // m_data->readData(tumor->mat_prop_->gm_, tumor->mat_prop_->wm_, tumor->mat_prop_->csf_, tumor->mat_prop_->glm_);  // copies synthetic data to m_data
  //     m_data->readData(p_gm_path, p_wm_path, p_csf_path, p_glm_path);         // reads patient data
  //     ierr = solver_interface->setMassEffectData(m_data->gm_, m_data->wm_, m_data->csf_, m_data->glm_);   // sets derivative ops data
  //     ierr = solver_interface->updateTumorCoefficients(wm, gm, glm, csf, bg);                            // reset matprop to undeformed
  // }


  ierr = solver_interface_->setParams (p_rec_, nullptr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
      ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
      ierr = solver_interface_->setInitialGuess (0.); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  if(has_dt0_) {
      ierr = VecCopy(data_t0_, tumor_->c_0_); CHKERRQ(ierr);
  } else {
      ierr = tumor_->phi_->apply(tumor_->c_0_, p_rec_); CHKERRQ(ierr);
  }

  ierr = tuMSGwarn(" Beginning Mass Effect Inversion."); CHKERRQ(ierr);
  ierr = solver_interface_->solveInverseMassEffect (&gamma_, data_t0_, nullptr);

  // Reset mat-props and diffusion and reaction operators, tumor IC does not change
  ierr = tumor_->mat_prop_->resetValues ();  CHKERRQ (ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->velocity_->set(0.);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

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


  ierr = solver_interface_->setParams (p_rec_, nullptr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
      ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
      ierr = solver_interface_->setInitialGuess (0.); CHKERRQ(ierr);
  }
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

  ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  ierr = solver_interface_->setParams (p_rec_, nullptr);
  ierr = solver_interface_->setInitialGuess (0.); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);

  PetscFunctionReturn(ierr);
}
