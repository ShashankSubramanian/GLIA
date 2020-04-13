#include "Solver.h"
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <vector>
#include "TumorSolverInterface.h"
#include "Utils.h"

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
Solver::Solver()
    : params_(nullptr),
      solver_interface_(nullptr),
      spec_ops_(nullptr),
      tumor_(nullptr),
      custom_obs_(false),
      warmstart_p_(false),
      wm_(nullptr),
      gm_(nullptr),
      vt_(nullptr),
      csf_(nullptr),
      mri_(nullptr),
      tmp_(nullptr),
      data_t1_(nullptr),
      data_t0_(nullptr),
      p_rec_(nullptr),
      data_support_(nullptr),
      data_comps_(nullptr),
      obs_filter_(nullptr),
      velocity_(nullptr) {}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  std::stringstream ss;
  ss << "    grid size: " << params->grid_->n_[0] << "x" << params->grid_->n_[1] << "x" << params->grid_->n_[2];
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); 
  ss.str(""); ss.clear();
  
  // === set parameters, initialize solver interface
  spec_ops_ = spec_ops;
  params_ = params;
  app_settings_ = app_settings;
  solver_interface_ = std::make_shared<TumorSolverInterface>(params_, spec_ops_, nullptr, nullptr);
  tumor_ = solver_interface_->getTumor();
  // === create tmp vector according to distributed grid
  ierr = VecCreate(PETSC_COMM_WORLD, &tmp_); CHKERRQ(ierr);
  ierr = VecSetSizes(tmp_, params_->grid_->nl_, params_->grid_->ng_); CHKERRQ(ierr);
  ierr = setupVec(tmp_); CHKERRQ(ierr);
  ierr = VecSet(tmp_, 0.0); CHKERRQ(ierr);
  // === create p_rec vector according to unknowns inverted for
  ierr = VecCreateSeq(PETSC_COMM_SELF, params_->tu_->np_ + params_->get_nk() + params_->get_nr(), &p_rec_); CHKERRQ(ierr);
  ierr = setupVec(p_rec_, SEQ); CHKERRQ(ierr);

  warmstart_p_ = !app_settings_->path_->pvec_.empty();
  custom_obs_ = !app_settings_->path_->obs_filter_.empty();
  has_dt0_ = params_->tu_->two_time_points_ || !app_settings_->path_->data_t0_.empty();

  // === error handling for some configuration inconsistencies
  if (warmstart_p_ && app_settings_->path_->phi_.empty()) {
    ss << " ERROR: Initial guess for p is given but no coordinates for phi. Exiting. ";
    ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    exit(0);
  }
  if (app_settings_->inject_solution_ && (!warmstart_p_ || app_settings_->path_->phi_.empty())) {
    ss << " ERROR: Trying to inject coarse solution, but p/phi not specified. Exiting. ";
    ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    exit(-1);
  }

  // === read brain: healthy segmentation wm, gm, csf, (ve) for simulation
  ierr = readAtlas(); CHKERRQ(ierr);

  // === read in user given velocity
  ierr = readVelocity(); CHKERRQ(ierr);

  // === advect healthy material properties with read in velocity, if given
  if (app_settings_->syn_->pre_adv_time_ > 0 && velocity_ != nullptr) {
    ss << " pre-advecting material properties with velocity to time t=" << app_settings_->syn_->pre_adv_time_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    if (!app_settings_->path_->mri_.empty()) {
      ierr = VecDuplicate(tmp_, &mri_); CHKERRQ(ierr);
      ierr = dataIn(mri_, params_, app_settings_->path_->mri_); CHKERRQ(ierr);
    }
    // TODO(K): pde-operators has no preAdvection
    // ierr = solver_interface_->getPdeOperators()->preAdvection(wm_, gm_, vt_, mri_, app_settings_->syn_->pre_adv_time_); CHKERRQ(ierr);
  } else if (params_->tu_->transport_mri_) {
    // the forward solve will transport the mri if mass effect is enabled
    if (mri_ == nullptr) {
      ierr = VecDuplicate(tmp_, &mri_); CHKERRQ(ierr);
      ierr = dataIn(mri_, params_, app_settings_->path_->mri_); CHKERRQ(ierr);
      if (tumor_->mat_prop_->mri_ == nullptr) {
        tumor_->mat_prop_->mri_ = mri_;
      }
    }
  }

  ierr = solver_interface_->updateTumorCoefficients(wm_, gm_, csf_, vt_, nullptr); CHKERRQ(ierr);
  ierr = tumor_->mat_prop_->setAtlas(gm_, wm_, csf_, vt_, nullptr); CHKERRQ(ierr);

  // === read data: generate synthetic or read real
  if (app_settings_->syn_->enabled_) {
    // data t1 and data t0 is generated synthetically using user given cm and tumor model
    ierr = createSynthetic(); CHKERRQ(ierr);
    data_support_ = data_t1_;
  } else {
    // read in target data (t1 and/or t0); observation operator
    ierr = readData(); CHKERRQ(ierr);
  }

  // === set observation operator
  if (custom_obs_) {
    ierr = tumor_->obs_->setCustomFilter(obs_filter_, 1);
    ss << " Setting custom observation mask";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  } else {
    ierr = tumor_->obs_->setDefaultFilter(data_t1_, 1, params_->tu_->obs_threshold_1_); CHKERRQ(ierr);
    if (has_dt0_) {
      ierr = tumor_->obs_->setDefaultFilter(data_t0_, 0, params_->tu_->obs_threshold_0_); CHKERRQ(ierr);
    }
    ss << " Setting default observation mask based on input data (d1) and threshold " << tumor_->obs_->threshold_1_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    ss << " Setting default observation mask based on input data (d0) and threshold " << tumor_->obs_->threshold_0_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  }

  // === apply observation operator to data
  ierr = tumor_->obs_->apply(data_t1_, data_t1_), 1; CHKERRQ(ierr);
  ierr = tumor_->obs_->apply(data_support_, data_support_, 1); CHKERRQ(ierr);
  if (has_dt0_) {
    ierr = tumor_->obs_->apply(data_t0_, data_t0_, 0); CHKERRQ(ierr);
  }

#ifdef CUDA
  cudaPrintDeviceMemory();
#endif

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ss << " Inversion with tumor parameters: rho = " << params_->tu_->rho_ << " k = " << params_->tu_->k_ << " dt = " << params_->tu_->dt_ << " Nt = " << params_->tu_->nt_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  ss << " Results in: " << params_->tu_->writepath_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  ScalarType prec_norm;
  ierr = VecNorm(p_rec_, NORM_2, &prec_norm); CHKERRQ(ierr);
  ss << " Norm of p reconstruction: " << prec_norm;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  ScalarType *prec_ptr;
  if (params_->opt_->diffusivity_inversion_ && !params_->opt_->reaction_inversion_) {
    ierr = VecGetArray(p_rec_, &prec_ptr); CHKERRQ(ierr);
    ss << " k1: " << (params_->tu_->nk_ > 0 ? prec_ptr[params_->tu_->np_] : 0);
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    ss << " k2: " << (params_->tu_->nk_ > 1 ? prec_ptr[params_->tu_->np_ + 1] : 0);
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    ss << " k3: " << (params_->tu_->nk_ > 2 ? prec_ptr[params_->tu_->np_ + 2] : 0);
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    ierr = VecRestoreArray(p_rec_, &prec_ptr); CHKERRQ(ierr);
  }
  params_->tu_->transport_mri_ = !app_settings_->path_->mri_.empty();
  ierr = tumor_->phi_->apply(tumor_->c_0_, p_rec_);
  // === segmentation
  ierr = tumor_->computeSegmentation(); CHKERRQ(ierr);
  ss << "seg_rec_final";
  ierr = dataOut(tumor_->seg_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
  ss.str(std::string());
  ss.clear();

  // === compute errors
  ScalarType *c0_ptr;
  ierr = tumor_->phi_->apply(tumor_->c_0_, p_rec_);
  if (params_->tu_->write_output_) {
    ierr = dataOut(tumor_->c_0_, params_, "c0_rec" + params_->tu_->ext_); CHKERRQ(ierr);
  }

  if (params_->tu_->transport_mri_) {
    if (mri_ == nullptr) {
      ierr = VecDuplicate(tmp_, &mri_); CHKERRQ(ierr);
      ierr = dataIn(mri_, params_, app_settings_->path_->mri_); CHKERRQ(ierr);
      if (tumor_->mat_prop_->mri_ == nullptr) {
        tumor_->mat_prop_->mri_ = mri_;
      }
    }
  }
  ierr = solver_interface_->solveForward(tmp_, tumor_->c_0_); CHKERRQ(ierr);

  ScalarType max, min;
  ierr = VecMax(tmp_, NULL, &max); CHKERRQ(ierr);
  ierr = VecMin(tmp_, NULL, &min); CHKERRQ(ierr);
  ss << " Reconstructed c(1) max and min : " << max << " " << min;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  if (params_->tu_->write_output_) ierr = dataOut(tmp_, params_, "c1_rec" + params_->tu_->ext_); CHKERRQ(ierr);

  // copy c(1)
  ScalarType data_norm, error_norm, error_norm_0;
  Vec c1_obs;
  ierr = VecDuplicate(tmp_, &c1_obs); CHKERRQ(ierr);
  ierr = VecCopy(tmp_, c1_obs); CHKERRQ(ierr);

  // c(1): error everywhere
  ierr = VecAXPY(tmp_, -1.0, data_t1_); CHKERRQ(ierr);
  ierr = VecNorm(data_t1_, NORM_2, &data_norm); CHKERRQ(ierr);
  ierr = VecNorm(tmp_, NORM_2, &error_norm); CHKERRQ(ierr);
  ss << " t=1: l2-error (everywhere): " << error_norm;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  error_norm /= data_norm;
  ss << " t=1: rel. l2-error (everywhere): " << error_norm;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();

  // c(1): error at observation points
  ierr = tumor_->obs_->apply(c1_obs, c1_obs, 1); CHKERRQ(ierr);
  ierr = tumor_->obs_->apply(tmp_, data_t1_, 1); CHKERRQ(ierr);
  ierr = VecAXPY(c1_obs, -1.0, tmp_); CHKERRQ(ierr);
  ierr = VecNorm(tmp_, NORM_2, &data_norm); CHKERRQ(ierr);
  ierr = VecNorm(c1_obs, NORM_2, &error_norm); CHKERRQ(ierr);
  ss << " t=1: l2-error (at observation points): " << error_norm;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  error_norm /= data_norm;
  ss << " t=1: rel. l2-error (at observation points): " << error_norm;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  ierr = VecDestroy(&c1_obs); CHKERRQ(ierr);

  // c(0): error everywhere
  ierr = VecCopy(tumor_->c_0_, tmp_); CHKERRQ(ierr);
  ierr = VecAXPY(tmp_, -1.0, data_t0_); CHKERRQ(ierr);
  ierr = VecNorm(data_t0_, NORM_2, &data_norm); CHKERRQ(ierr);
  ierr = VecNorm(tmp_, NORM_2, &error_norm_0); CHKERRQ(ierr);
  ss << " t=0: l2-error (everywhere): " << error_norm_0;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  error_norm_0 /= data_norm;
  ss << " t=0: rel. l2-error (everywhere): " << error_norm_0;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();

  // write file
  if (procid == 0) {
    std::ofstream opfile;
    opfile.open(params_->tu_->writepath_ + "reconstruction_info.dat");
    opfile << "rho k c1_rel c0_rel \n";
    opfile << params_->tu_->rho_ << " " << params_->tu_->k_ << " " << error_norm << " " << error_norm_0 << std::endl;
    opfile.flush();
    opfile.close();
    // print reconstructed p_vec
    ss << " --------------  RECONST P -----------------";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    ierr = VecView(p_rec_, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
    ss << " --------------  -------------- -----------------";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  }

  // === final prediction
  ierr = predict(); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::predict() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  if (app_settings_->pred_->enabled_) {
    ss << " Predicting future tumor growth...";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    if (params_->tu_->time_history_off_) {
      params_->tu_->dt_ = app_settings_->pred_->dt_;
      // set c(0)
      if (params_->tu_->use_c0_) {
        ierr = VecCopy(data_t0_, tumor_->c_0_); CHKERRQ(ierr);
      } else {
        ierr = tumor_->phi_->apply(tumor_->c_0_, p_rec_);
      }

      // predict tumor growth at different (user defined) times
      for (int i = 0; i < app_settings_->pred_->t_pred_.size(); ++i) {
        params_->tu_->nt_ = (int)(app_settings_->pred_->t_pred_[i] / app_settings_->pred_->dt_);  // number of time steps
        // if different brain to perform prediction is given, read in and reset atlas
        if (app_settings_->pred_->wm_path_.size() >= i && !app_settings_->pred_->wm_path_[i].empty()) {
          app_settings_->path_->wm_ = app_settings_->pred_->wm_path_[i];
          app_settings_->path_->gm_ = app_settings_->pred_->gm_path_[i];
          app_settings_->path_->vt_ = app_settings_->pred_->vt_path_[i];
          ierr = tuMSGstd(" .. reading in atlas brain to perform prediction."); CHKERRQ(ierr);
          ierr = readAtlas(); CHKERRQ(ierr);
          ierr = solver_interface_->updateTumorCoefficients(wm_, gm_, csf_, vt_, nullptr);
          ierr = tumor_->mat_prop_->setAtlas(gm_, wm_, csf_, vt_, nullptr); CHKERRQ(ierr);
        } else {
          ierr = tumor_->mat_prop_->resetValues(); CHKERRQ(ierr);
        }
        ierr = solver_interface_->getPdeOperators()->solveState(0); CHKERRQ(ierr);
        ss << "c_pred_at_[t=" << app_settings_->pred_->t_pred_[i] << "]";
        ierr = dataOut(tumor_->c_t_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
        ss.str("");
        ss.clear();
        ss << " .. prediction complete for t = " << app_settings_->pred_->t_pred_[i];
        ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
        ss.str("");
        ss.clear();

        // if given: compute error to ground truth at predicted time
        if (!app_settings_->pred_->true_data_path_[i].empty()) {
          Vec true_dat_t1;
          ScalarType obs_c_norm, obs_data_norm, data_norm;
          ierr = VecDuplicate(data_t1_, &true_dat_t1); CHKERRQ(ierr);
          ierr = dataIn(true_dat_t1, params_, app_settings_->pred_->true_data_path_[i]); CHKERRQ(ierr);
          ierr = VecNorm(true_dat_t1, NORM_2, &data_norm); CHKERRQ(ierr);
          // error everywhere
          ierr = VecCopy(tumor_->c_t_, tmp_); CHKERRQ(ierr);
          ierr = VecAXPY(tmp_, -1.0, true_dat_t1); CHKERRQ(ierr);
          ss << "res_at_[t=" << app_settings_->pred_->t_pred_[i] << "]";
          ierr = dataOut(tmp_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
          ss.str("");
          ss.clear();
          ierr = VecNorm(tmp_, NORM_2, &obs_c_norm); CHKERRQ(ierr);
          obs_c_norm /= data_norm;
          ss << " .. rel. l2-error (everywhere) (T=" << app_settings_->pred_->t_pred_[i] << ") : " << obs_c_norm;
          ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
          ss.str("");
          ss.clear();
          // error at observation points
          ierr = VecCopy(tumor_->c_t_, tmp_); CHKERRQ(ierr);
          ierr = tumor_->obs_->apply(tmp_, tmp_); CHKERRQ(ierr);
          ierr = tumor_->obs_->apply(true_dat_t1, true_dat_t1); CHKERRQ(ierr);
          ierr = VecNorm(true_dat_t1, NORM_2, &obs_data_norm); CHKERRQ(ierr);
          ierr = VecAXPY(tmp_, -1.0, true_dat_t1); CHKERRQ(ierr);
          ss << "res_obs_at_[t=" << app_settings_->pred_->t_pred_[i] << "]";
          ierr = dataOut(tmp_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
          ss.str("");
          ss.clear();
          ierr = VecNorm(tmp_, NORM_2, &obs_c_norm); CHKERRQ(ierr);
          obs_c_norm /= obs_data_norm;
          ss << " .. rel. l2-error (at observation points) (T=" << app_settings_->pred_->t_pred_[i] << ") : " << obs_c_norm;
          ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
          ss.str("");
          ss.clear();
          if (true_dat_t1 != nullptr) {
            ierr = VecDestroy(&true_dat_t1); CHKERRQ(ierr);
          }
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

  ScalarType sigma_smooth = params_->tu_->smoothing_factor_ * 2 * M_PI / params_->grid_->n_[0];

  if (!app_settings_->path_->seg_.empty()) {
    ierr = dataIn(tmp_, params_, app_settings_->path_->seg_); CHKERRQ(ierr);
    // TODO(K): populate to wm, gm, csf, ve
  } else {
    if (!app_settings_->path_->wm_.empty()) {
      ierr = VecDuplicate(tmp_, &wm_); CHKERRQ(ierr);
      ierr = dataIn(wm_, params_, app_settings_->path_->wm_); CHKERRQ(ierr);
    }
    if (!app_settings_->path_->gm_.empty()) {
      ierr = VecDuplicate(tmp_, &gm_); CHKERRQ(ierr);
      ierr = dataIn(gm_, params_, app_settings_->path_->gm_); CHKERRQ(ierr);
    }
    if (!app_settings_->path_->vt_.empty()) {
      ierr = VecDuplicate(tmp_, &vt_); CHKERRQ(ierr);
      ierr = dataIn(vt_, params_, app_settings_->path_->vt_); CHKERRQ(ierr);
    }
    // if(params_->tu_->model >= 4)
    if (!app_settings_->path_->csf_.empty()) {
      ierr = VecDuplicate(tmp_, &csf_); CHKERRQ(ierr);
      ierr = dataIn(csf_, params_, app_settings_->path_->csf_); CHKERRQ(ierr);
    }
  }
  // smooth
  if (gm_ != nullptr) {
    ierr = spec_ops_->weierstrassSmoother(gm_, gm_, params_, sigma_smooth); CHKERRQ(ierr);
  }
  if (wm_ != nullptr) {
    ierr = spec_ops_->weierstrassSmoother(wm_, wm_, params_, sigma_smooth); CHKERRQ(ierr);
  }
  if (vt_ != nullptr) {
    ierr = spec_ops_->weierstrassSmoother(vt_, vt_, params_, sigma_smooth); CHKERRQ(ierr);
  }
  if (csf_ != nullptr) {
    ierr = spec_ops_->weierstrassSmoother(csf_, csf_, params_, sigma_smooth); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::readData() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  std::stringstream ss;
  // ScalarType sigma_smooth = params_->smoothing_factor_ * 2 * M_PI / params_->grid_->n_[0];
  ScalarType sig_data = params_->tu_->smoothing_factor_data_ * 2 * M_PI / params_->grid_->n_[0];
  ScalarType min, max;

  if (!app_settings_->path_->data_t1_.empty()) {
    ierr = VecDuplicate(tmp_, &data_t1_); CHKERRQ(ierr);
    ierr = dataIn(data_t1_, params_, app_settings_->path_->data_t1_); CHKERRQ(ierr);
    if (params_->tu_->smoothing_factor_data_ > 0) {
      ierr = spec_ops_->weierstrassSmoother(data_t1_, data_t1_, params_, sig_data); CHKERRQ(ierr);
      ss << " smoothing c(1) with factor: " << params_->tu_->smoothing_factor_data_ << ", and sigma: " << sig_data;
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
    }
    // make obs threshold relaltive
    if (params_->tu_->relative_obs_threshold_) {
      ierr = VecMax(data_t1_, NULL, &max); CHKERRQ(ierr);
      params_->tu_->obs_threshold_1_ *= max;
      ss << " Changing observation threshold for d_1 to thr_1: " << params_->tu_->obs_threshold_1_;
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
    }
  }
  bool read_supp;
  if (!app_settings_->path_->data_support_.empty()) {
    std::string file, path, ext;
    ierr = getFileName(path, file, ext, app_settings_->path_->data_support_); CHKERRQ(ierr);
    read_supp = (strcmp(ext.c_str(), ".nc") == 0) || (strcmp(ext.c_str(), ".nii.gz") == 0);  // file ends with *.nc or *.nii.gz?
    if (read_supp) {
      ierr = VecDuplicate(tmp_, &data_support_); CHKERRQ(ierr);
      ierr = dataIn(data_support_, params_, app_settings_->path_->data_support_); CHKERRQ(ierr);
    }
  } else {
    data_support_ = data_t1_;
  }
  if (read_supp && !app_settings_->path_->data_comps_.empty()) {
    ierr = VecDuplicate(tmp_, &data_comps_); CHKERRQ(ierr);
    ierr = dataIn(data_comps_, params_, app_settings_->path_->data_comps_); CHKERRQ(ierr);
  }
  if (!app_settings_->path_->obs_filter_.empty()) {
    ierr = VecDuplicate(tmp_, &obs_filter_); CHKERRQ(ierr);
    ierr = dataIn(obs_filter_, params_, app_settings_->path_->obs_filter_); CHKERRQ(ierr);
    ss << " Reading custom observation mask";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  }

  ScalarType *ptr;
  if (!app_settings_->path_->data_t0_.empty()) {
    ierr = VecDuplicate(tmp_, &data_t0_); CHKERRQ(ierr);
    ierr = dataIn(data_t0_, params_, app_settings_->path_->data_t0_); CHKERRQ(ierr);
    ierr = VecMin(data_t0_, NULL, &min); CHKERRQ(ierr);
    if (min < 0) {
      ss << " tumor init is aliased with min " << min << "; clipping and smoothing...";
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      ierr = vecGetArray(data_t0_, &ptr); CHKERRQ(ierr);
#ifdef CUDA
      clipVectorCuda(ptr, params_->grid_->nl_);
#else
      for (int i = 0; i < params_->grid_->nl_; i++) ptr[i] = (ptr[i] <= 0.) ? 0. : ptr[i];
#endif
      ierr = vecRestoreArray(data_t0_, &ptr); CHKERRQ(ierr);
    }
    // smooth a little bit because sometimes registration outputs have high gradients
    if (data_t0_ != nullptr) {
      if (params_->tu_->smoothing_factor_data_ > 0) {
        ierr = spec_ops_->weierstrassSmoother(data_t0_, data_t0_, params_, sig_data); CHKERRQ(ierr);
        ss << " smoothing c(1) with factor: " << params_->tu_->smoothing_factor_data_ << ", and sigma: " << sig_data;
        ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
        ss.str("");
        ss.clear();
      }
    }
    ierr = VecMax(data_t0_, NULL, &max); CHKERRQ(ierr);
    ierr = VecScale(data_t0_, (1.0 / max)); CHKERRQ(ierr);
    ierr = dataOut(data_t0_, params_, "c0True.nc"); CHKERRQ(ierr);

    // make obs threshold relaltive
    if (params_->tu_->relative_obs_threshold_) {
      params_->tu_->obs_threshold_0_ *= max;
      ss << " Changing observation threshold for d_0 to thr_0: " << params_->tu_->obs_threshold_0_;
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
    }
  } else {
    ierr = VecSet(data_t0_, 0.); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Solver::readVelocity() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  if (!app_settings_->path_->velocity_x1_.empty()) {
    // TODO(K) readVecFiled not implemented, copy from alzh branch
    ierr = readVecField(tumor_->velocity_.get(), app_settings_->path_->velocity_x1_, app_settings_->path_->velocity_x2_, app_settings_->path_->velocity_x3_, params_); CHKERRQ(ierr);
    ierr = tumor_->velocity_->scale(-1); CHKERRQ(ierr);
    Vec mag;
    ierr = VecDuplicate(data_t1_, &mag); CHKERRQ(ierr);
    ierr = tumor_->velocity_->computeMagnitude(mag); CHKERRQ(ierr);
    ierr = dataOut(mag, params_, "velocity_mag.nc"); CHKERRQ(ierr);
    ScalarType vnorm;
    ierr = VecNorm(mag, NORM_2, &vnorm); CHKERRQ(ierr);
    std::stringstream ss;
    ss << " Given velocity read in: norm of velocity magnitude: " << vnorm;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
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
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  // save parameters
  ScalarType rho_temp = params_->tu_->rho_, k_temp = params_->tu_->k_, dt_temp = params_->tu_->dt_;
  int nt_temp = params_->tu_->nt_;
  // set to synthetic parameters
  params_->tu_->rho_ = app_settings_->syn_->rho_;
  params_->tu_->k_ = app_settings_->syn_->k_;
  params_->tu_->dt_ = app_settings_->syn_->dt_;
  params_->tu_->nt_ = app_settings_->syn_->nt_;

  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);

  // allocate t1 and t0 data:
  if (data_t1_ == nullptr) {
    ierr = VecDuplicate(tmp_, &data_t1_); CHKERRQ(ierr);
  }
  if (data_t0_ == nullptr) {
    ierr = VecDuplicate(tmp_, &data_t0_); CHKERRQ(ierr);
  }
  ierr = VecSet(data_t0_, 0.); CHKERRQ(ierr);

  std::array<ScalarType, 3> cm_tmp;
  ScalarType scale = 1;
  int count = 0;
  // insert user defined tumor foci
  for (auto &cm : app_settings_->syn_->user_cms_) {
    count++;
    ierr = VecSet(tmp_, 0.); CHKERRQ(ierr);
    cm_tmp[0] = (2 * M_PI / 256 * cm[0]);
    cm_tmp[1] = (2 * M_PI / 256 * cm[1]);
    cm_tmp[2] = (2 * M_PI / 256 * cm[2]);
    scale = (cm[3] < 0) ? 1 : cm[3];  // p activation stored in last entry

    ierr = tumor_->phi_->setGaussians(cm_tmp, params_->tu_->phi_sigma_, params_->tu_->phi_spacing_factor_, params_->tu_->np_);
    ierr = tumor_->phi_->setValues(tumor_->mat_prop_);

    // set p_rec_ according to user centers
    ierr = VecSet(p_rec_, scale); CHKERRQ(ierr);
    // ss << " --------------  Synthetic p_vec (using cm"<<count<<") -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    // ss << " --------------  -------------- -----------------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ierr = tumor_->phi_->apply(tmp_, p_rec_); CHKERRQ(ierr);
    ierr = VecAXPY(data_t0_, 1.0, tmp_); CHKERRQ(ierr);
    ss << "p-syn-sm" << count;
    writeCheckpoint(p_rec_, tumor_->phi_, params_->tu_->writepath_, ss.str());
    ss.str("");
    ss.clear();
  }

  ScalarType max, min;
  if (params_->tu_->write_output_) {
    ierr = dataOut(data_t0_, params_, "c0_true_syn" + params_->tu_->ext_); CHKERRQ(ierr);
  }
  ierr = VecMax(data_t0_, NULL, &max); CHKERRQ(ierr);
  ierr = VecMin(data_t0_, NULL, &min); CHKERRQ(ierr);
  ss << " Synthetic data c(0) max and min : " << max << " " << min;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();

  if (params_->tu_->model_ == 5) {
    std::map<std::string, Vec> species;
    ierr = solver_interface_->solveForward(data_t1_, data_t0_, &species); CHKERRQ(ierr);
  } else {
    ierr = solver_interface_->solveForward(data_t1_, data_t0_); CHKERRQ(ierr);
  }
  ierr = VecMax(data_t1_, NULL, &max); CHKERRQ(ierr);
  ierr = VecMin(data_t1_, NULL, &min); CHKERRQ(ierr);
  ss << " Synthetic data c(1) max and min (before observation) : " << max << " " << min;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  if (params_->tu_->write_output_) {
    ierr = dataOut(data_t1_, params_, "c1_true_syn_before_observation" + params_->tu_->ext_); CHKERRQ(ierr);
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
  //     rel_noise_err_norm = noise_err_norm / rel_noise_err_norm; CHKERRQ(ierr);
  //     ss << " low frequency relative error = " << rel_noise_err_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  //
  //     // if (params_->tu_->write_output_)
  //     //     dataOut (data, n_misc, "dataNoise.nc");
  //
  //     if (temp != nullptr) {ierr = VecDestroy (&temp);          CHKERRQ (ierr); temp = nullptr;}
  // ss << " data generated with parameters: rho = " << n_misc->rho_ << " k = " << n_misc->k_ << " dt = " << n_misc->dt_ << " Nt = " << n_misc->nt_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  //     ss << " mass-effect forcing factor used = " << n_misc->forcing_factor_; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  //     // write out p and phi so that they can be used if needed
  //     writeCheckpoint(tumor->p_rec_, tumor->phi_, n_misc->writepath_.str(), "forward"); CHKERRQ(ierr);
  //     ss << " ground truth phi and p written to file "; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // }
  ss << " Synthetic data solve parameters: r = " << params_->tu_->rho_ << ", k = " << params_->tu_->k_;
  if (params_->tu_->model_ >= 4)
    ss << ", g = " << params_->tu_->forcing_factor_;
  ss << ", nt = " << params_->tu_->nt_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();

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

  std::stringstream ss;
  ss << " Initialize Gaussian basis functions.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  if (warmstart_p_) {
    ss << "  .. solver warmstart: using p_vec and Gaussians from file.";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    ierr = tumor_->phi_->setGaussians(app_settings_->path_->phi_); CHKERRQ(ierr);
    ierr = tumor_->phi_->setValues(tumor_->mat_prop_); CHKERRQ(ierr);
    ierr = readPVec(&p_rec_, params_->tu_->np_ + params_->get_nk() + params_->get_nr(), params_->tu_->np_, app_settings_->path_->pvec_); CHKERRQ(ierr);
  } else {
    if (!app_settings_->path_->data_support_data_.empty()) {  // read Gaussian centers and comp labels from txt file
      ss << "  .. reading Gaussian centers and component data from file.";
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      ierr = tumor_->phi_->setGaussians(app_settings_->path_->data_support_data_, true); CHKERRQ(ierr);
    } else if (!app_settings_->path_->data_support_.empty()) {  // read Gaussian centers and comp labels from nc files
      ss << "  .. reading Gaussian centers from .nc image file.";
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      ierr = tumor_->phi_->setGaussians(data_support_); CHKERRQ(ierr);
    } else if (app_settings_->syn_->enabled_) {
      // synthetic data generation in data_t1_
      ss << "  .. setting Gaussians with synthetic data.";
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      ierr = tumor_->phi_->setGaussians(data_t1_); CHKERRQ(ierr);
    } else {
      ss << " Error: Cannot set Gaussians: expecting user input data -support_data_path *.nc or *.txt. Exiting.";
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      exit(1);
    }
    // set phi values and re-initialize p_vec
    ierr = tumor_->phi_->setValues(tumor_->mat_prop_); CHKERRQ(ierr);
    if (p_rec_ != nullptr) {
      ierr = VecDestroy(&p_rec_); CHKERRQ(ierr);
      p_rec_ = nullptr;
    }
    ierr = VecCreateSeq(PETSC_COMM_SELF, params_->tu_->np_ + params_->get_nk() + params_->get_nr(), &p_rec_); CHKERRQ(ierr);
    ierr = setupVec(p_rec_, SEQ); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========                   ForwardSolver                   ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode ForwardSolver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ierr = tuMSGwarn(" Initializing Forward Solver."); CHKERRQ(ierr);
  // switch off time history
  params->tu_->time_history_off_;
  ierr = tuMSGstd(" .. switching off time history."); CHKERRQ(ierr);
  // set and populate parameters; read material properties; read data
  params->tu_->np_ = 1;
  // transport mri if needed
  params->tu_->transport_mri_ = !app_settings->path_->mri_.empty();
  ierr = Solver::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode ForwardSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // no-op
  std::stringstream ss;
  ss << " Forward solve completed. Exiting.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  PetscFunctionReturn(ierr);
}

PetscErrorCode ForwardSolver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========                   InverseL2Solver                 ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL2Solver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ierr = tuMSGwarn(" Initializing Inversion for Non-Sparse TIL, and Diffusion."); CHKERRQ(ierr);
  // set and populate parameters; read material properties; read data
  ierr = Solver::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  // === set Gaussians
  if (app_settings_->inject_solution_) {
    ss << " Error: injecting coarse level solution is not supported for L2 inversion. Ignoring input.";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  }
  ierr = initializeGaussians(); CHKERRQ(ierr);
  ierr = solver_interface_->setParams(p_rec_, nullptr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
    ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
    ierr = solver_interface_->setInitialGuess(0.); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL2Solver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Beginning Inversion for Non-Sparse TIL, and Diffusion."); CHKERRQ(ierr);
  Solver::run(); CHKERRQ(ierr);
  ierr = solver_interface_->solveInverse(p_rec_, data_t1_, nullptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL2Solver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Finalizing Inversion for Non-Sparse TIL, and Diffusion."); CHKERRQ(ierr);
  ierr = Solver::finalize(); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========                   InverseL1Solver                 ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL1Solver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ierr = tuMSGwarn(" Initializing Inversion for Sparse TIL, and Reaction/Diffusion."); CHKERRQ(ierr);
  // set and populate parameters; read material properties; read data
  ierr = Solver::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  // read connected components; set sparsity level
  if (!app_settings_->path_->data_comps_data_.empty()) {
    readConCompDat(tumor_->phi_->component_weights_, tumor_->phi_->component_centers_, app_settings_->path_->data_comps_data_);
    int nnc = 0;
    for (auto w : tumor_->phi_->component_weights_)
      if (w >= 1E-3) nnc++;  // number of significant components
    ss << " Setting sparsity level to " << params_->tu_->sparsity_level_ << " x n_components (w > 1E-3) + n_components (w < 1E-3) = " << params_->tu_->sparsity_level_ << " x " << nnc << " + "
       << (tumor_->phi_->component_weights_.size() - nnc) << " = " << params_->tu_->sparsity_level_ * nnc + (tumor_->phi_->component_weights_.size() - nnc);
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    params_->tu_->sparsity_level_ = params_->tu_->sparsity_level_ * nnc + (tumor_->phi_->component_weights_.size() - nnc);
  }

  // === set Gaussians
  if (!warmstart_p_) {  // set component labels
    if (!app_settings_->path_->data_comps_.empty()) {
      ss << "  Setting component data from .nc image file.";
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      tumor_->phi_->setLabels(data_comps_); CHKERRQ(ierr);
    }
  }

  // === inject coarse level solution
  if (!app_settings_->inject_solution_) {
    ierr = initializeGaussians(); CHKERRQ(ierr);
  } else {
    ss << " Injecting coarse level solution (adopting p_vec and Gaussians).";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    Vec coarse_sol = nullptr;
    int np_save = params_->tu_->np_, np_coarse = 0;  // save np, since overwritten in read function
    std::vector<ScalarType> coarse_sol_centers;
    ierr = readPhiMesh(coarse_sol_centers, params_, app_settings_->path_->phi_, false); CHKERRQ(ierr);
    ierr = readPVec(&coarse_sol, params_->tu_->np_ + params_->get_nk() + params_->get_nk(), params_->tu_->np_, app_settings_->path_->pvec_); CHKERRQ(ierr);
    np_coarse = params_->tu_->np_;
    params_->tu_->np_ = np_save;  // reset to correct value
    // find coarse centers in centers_ of current Phi
    int xc, yc, zc, xf, yf, zf;
    ScalarType *xf_ptr, *xc_ptr;
    ScalarType hx = 2.0 * M_PI / params_->grid_->n_[0];
    ScalarType hy = 2.0 * M_PI / params_->grid_->n_[1];
    ScalarType hz = 2.0 * M_PI / params_->grid_->n_[2];
    ierr = VecGetArray(p_rec_, &xf_ptr); CHKERRQ(ierr);
    ierr = VecGetArray(coarse_sol, &xc_ptr); CHKERRQ(ierr);
    for (int j = 0; j < np_coarse; ++j) {
      for (int i = 0; i < params_->tu_->np_; ++i) {
        xc = (int)std::round(coarse_sol_centers[3 * j + 0] / hx);
        yc = (int)std::round(coarse_sol_centers[3 * j + 1] / hy);
        zc = (int)std::round(coarse_sol_centers[3 * j + 2] / hz);
        xf = (int)std::round(tumor_->phi_->centers_[3 * i + 0] / hx);
        yf = (int)std::round(tumor_->phi_->centers_[3 * i + 1] / hy);
        zf = (int)std::round(tumor_->phi_->centers_[3 * i + 2] / hz);
        if (xc == xf && yc == yf && zc == zf) {
          xf_ptr[i] = 2 * xc_ptr[j];       // set initial guess (times 2 since sigma is halfed in every level)
          params_->tu_->support_.push_back(i);  // add to support
        }
      }
    }
    ierr = VecRestoreArray(p_rec_, &xf_ptr); CHKERRQ(ierr);
    ierr = VecRestoreArray(coarse_sol, &xc_ptr); CHKERRQ(ierr);
    if (coarse_sol != nullptr) {
      ierr = VecDestroy(&coarse_sol); CHKERRQ(ierr);
      coarse_sol = nullptr;
    }
  }

  ierr = solver_interface_->setParams(p_rec_, nullptr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
    ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
    ierr = solver_interface_->setInitialGuess(0.); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL1Solver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Beginning Inversion for Sparse TIL, and Diffusion/Reaction."); CHKERRQ(ierr);
  Solver::run(); CHKERRQ(ierr);
  ierr = solver_interface_->solveInverseCoSaMp(p_rec_, data_t1_, nullptr); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL1Solver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Finalizing Inversion for Sparse TIL, and Reaction/Diffusion."); CHKERRQ(ierr);
  ierr = Solver::finalize(); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========          InverseReactionDiffusionSolver           ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseReactionDiffusionSolver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Initializing Reaction/Diffusion Inversion."); CHKERRQ(ierr);
  // set and populate parameters; read material properties; read data
  ierr = Solver::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  ierr = solver_interface_->setParams(p_rec_, nullptr);
  ierr = solver_interface_->setInitialGuess(0.); CHKERRQ(ierr);
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
  if (!warmstart_p_ && app_settings_->path_->data_t0_.empty()) {
    ss << " Error: c(0) needs to be set, read in p and Gaussians. Exiting.";
    ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    exit(1);
  }
  ierr = solver_interface_->solveInverseReacDiff(p_rec_, data_t1_, nullptr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseReactionDiffusionSolver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Finalizing Reaction/Diffusion Inversion."); CHKERRQ(ierr);
  ierr = Solver::finalize(); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========               InverseMassEffectSolver             ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  ierr = tuMSGwarn(" Initializing Mass Effect Inversion."); CHKERRQ(ierr);

  // set and populate parameters; read material properties; read data
  ierr = Solver::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  // read mass effect patient data
  ierr = readPatient(); CHKERRQ(ierr);
  params_->opt_->invert_mass_effect_ = true;  // enable mass effect inversion in optimizer
  // set patient material properties
  ierr = solver_interface_->setMassEffectData(p_gm_, p_wm_, p_vt_, p_csf_); CHKERRQ(ierr);
  // ierr = solver_interface_->updateTumorCoefficients(wm_, gm_, csf_, vt_, nullptr);       // TODO(K) I think this is not needed, double check with S

  ierr = solver_interface_->setParams(p_rec_, nullptr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
    ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
    ierr = solver_interface_->setInitialGuess(0.); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  if (has_dt0_) {
    ierr = VecCopy(data_t0_, tumor_->c_0_); CHKERRQ(ierr);
  } else {
    ierr = tumor_->phi_->apply(tumor_->c_0_, p_rec_); CHKERRQ(ierr);
  }

  ierr = tuMSGwarn(" Beginning Mass Effect Inversion."); CHKERRQ(ierr);
  ierr = solver_interface_->solveInverseMassEffect(&gamma_, data_t0_, nullptr);

  // Reset mat-props and diffusion and reaction operators, tumor IC does not change
  ierr = tumor_->mat_prop_->resetValues(); CHKERRQ(ierr);
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

  std::stringstream ss;

  ierr = tuMSGwarn(" Finalizing Mass Effect Inversion."); CHKERRQ(ierr);
  ierr = Solver::finalize(); CHKERRQ(ierr);

  if (params_->tu_->write_output_) {
    ierr = tumor_->computeSegmentation(); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "seg_rec_final";
    ierr = dataOut(tumor_->seg_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "c_rec_final";
    ierr = dataOut(tumor_->c_t_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "vt_rec_final";
    ierr = dataOut(tumor_->mat_prop_->vt_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "csf_rec_final";
    ierr = dataOut(tumor_->mat_prop_->csf_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "wm_rec_final";
    ierr = dataOut(tumor_->mat_prop_->wm_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "gm_rec_final";
    ierr = dataOut(tumor_->mat_prop_->gm_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    Vec mag = nullptr;
    ierr = solver_interface_->getPdeOperators()->getModelSpecificVector(&mag);
    ierr = tumor_->displacement_->computeMagnitude(mag);
    ss << "displacement_rec_final";
    ierr = dataOut(mag, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ScalarType mag_norm, mm;
    ierr = VecNorm(mag, NORM_2, &mag_norm); CHKERRQ(ierr);
    ierr = VecMax(mag, NULL, &mm); CHKERRQ(ierr);
    ss << " Norm of displacement: " << mag_norm << "; max of displacement: " << mm;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    if (tumor_->mat_prop_->mri_ != nullptr) {
      ss << "mri_rec_final";
      ierr = dataOut(tumor_->mat_prop_->mri_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
    }
  }

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::readPatient() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ScalarType sigma_smooth = params_->tu_->smoothing_factor_ * 2 * M_PI / params_->grid_->n_[0];

  if (!app_settings_->path_->p_seg_.empty()) {
    ierr = dataIn(tmp_, params_, app_settings_->path_->p_seg_); CHKERRQ(ierr);
    // TODO(K): populate to wm, gm, csf, ve
  } else {
    if (!app_settings_->path_->p_wm_.empty()) {
      ierr = VecDuplicate(tmp_, &p_wm_); CHKERRQ(ierr);
      ierr = dataIn(p_wm_, params_, app_settings_->path_->p_wm_); CHKERRQ(ierr);
    }
    if (!app_settings_->path_->p_gm_.empty()) {
      ierr = VecDuplicate(tmp_, &p_gm_); CHKERRQ(ierr);
      ierr = dataIn(p_gm_, params_, app_settings_->path_->p_gm_); CHKERRQ(ierr);
    }
    if (!app_settings_->path_->p_vt_.empty()) {
      ierr = VecDuplicate(tmp_, &p_vt_); CHKERRQ(ierr);
      ierr = dataIn(p_vt_, params_, app_settings_->path_->p_vt_); CHKERRQ(ierr);
    }
    if (!app_settings_->path_->p_csf_.empty()) {
      ierr = VecDuplicate(tmp_, &p_csf_); CHKERRQ(ierr);
      ierr = dataIn(p_csf_, params_, app_settings_->path_->p_csf_); CHKERRQ(ierr);
    }
  }
  // smooth
  if (p_gm_ != nullptr) {
    ierr = spec_ops_->weierstrassSmoother(p_gm_, p_gm_, params_, sigma_smooth); CHKERRQ(ierr);
  }
  if (p_wm_ != nullptr) {
    ierr = spec_ops_->weierstrassSmoother(p_wm_, p_wm_, params_, sigma_smooth); CHKERRQ(ierr);
  }
  if (p_vt_ != nullptr) {
    ierr = spec_ops_->weierstrassSmoother(p_vt_, p_vt_, params_, sigma_smooth); CHKERRQ(ierr);
  }
  if (p_csf_ != nullptr) {
    ierr = spec_ops_->weierstrassSmoother(p_csf_, p_csf_, params_, sigma_smooth); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========             MultiSpeciesSolver             ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode MultiSpeciesSolver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  ierr = tuMSGwarn(" Initializing Multi Species Forward Solver."); CHKERRQ(ierr);

  params->tu_->time_history_off_ = true;
  ierr = Solver::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  // ierr = solver_interface_->setParams (p_rec_, nullptr);
  // ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  // ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode MultiSpeciesSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  if (has_dt0_) {
    ierr = VecCopy(data_t0_, tumor_->c_0_); CHKERRQ(ierr);
  } else {
    ierr = tumor_->phi_->apply(tumor_->c_0_, p_rec_); CHKERRQ(ierr);
  }

  ierr = tuMSGwarn(" Beginning Multi Species Forward Solve."); CHKERRQ(ierr);
  // TODO(K): call multi species inverse solver

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode MultiSpeciesSolver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Finalizing Multi Species Forward Solve."); CHKERRQ(ierr);
  ierr = Solver::finalize(); CHKERRQ(ierr);

  std::stringstream ss;
  if (params_->tu_->write_output_) {
    ierr = tumor_->computeSegmentation(); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "seg_rec_final";
    ierr = dataOut(tumor_->seg_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "c_rec_final";
    ierr = dataOut(tumor_->c_t_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "vt_rec_final";
    ierr = dataOut(tumor_->mat_prop_->vt_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "csf_rec_final";
    ierr = dataOut(tumor_->mat_prop_->csf_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "wm_rec_final";
    ierr = dataOut(tumor_->mat_prop_->wm_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "gm_rec_final";
    ierr = dataOut(tumor_->mat_prop_->gm_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    Vec mag = nullptr;
    ierr = solver_interface_->getPdeOperators()->getModelSpecificVector(&mag);
    ierr = tumor_->displacement_->computeMagnitude(mag);
    ss << "displacement_rec_final";
    ierr = dataOut(mag, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ScalarType mag_norm, mm;
    ierr = VecNorm(mag, NORM_2, &mag_norm); CHKERRQ(ierr);
    ierr = VecMax(mag, NULL, &mm); CHKERRQ(ierr);
    ss << " Norm of displacement: " << mag_norm << "; max of displacement: " << mm;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    if (tumor_->mat_prop_->mri_ != nullptr) {
      ss << "mri_rec_final";
      ierr = dataOut(tumor_->mat_prop_->mri_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
    }
  }

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========                      TestSuite                    ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TestSuite::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = Solver::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  ierr = solver_interface_->setParams(p_rec_, nullptr);
  ierr = solver_interface_->setInitialGuess(0.); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);

  PetscFunctionReturn(ierr);
}

PetscErrorCode TestSuite::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  PetscFunctionReturn(ierr);
}

PetscErrorCode TestSuite::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  
  PetscFunctionReturn(ierr);
}
