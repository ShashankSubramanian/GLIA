
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <vector>

#include "Utils.h"
#include "SolverInterface.h"

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
SolverInterface::SolverInterface()
: params_(nullptr),
  derivative_operators_(nullptr),
  pde_operators_(nullptr),
  spec_ops_(nullptr),
  tumor_(nullptr),
  custom_obs_(false),
  warmstart_p_(false),
  data_t1_from_seg_(false),
  n_inv_(0),
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
  velocity_(nullptr),
  data_(nullptr)
{}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
SolverInterface::~SolverInterface() {
  if(wm_  != nullptr) VecDestroy(&wm_);
  if(gm_  != nullptr) VecDestroy(&gm_);
  if(vt_  != nullptr) VecDestroy(&vt_);
  if(csf_ != nullptr) VecDestroy(&csf_);
  if(mri_ != nullptr) VecDestroy(&mri_);
  if(tmp_ != nullptr) VecDestroy(&tmp_);
  if(data_t1_ != nullptr) {VecDestroy(&data_t1_); data_t1_ = nullptr;}
  if(data_t0_ != nullptr) {VecDestroy(&data_t0_); data_t0_ = nullptr;}
  if(!app_settings_->syn_->enabled_ && !app_settings_->path_->data_support_.empty()) {
    if(data_support_ != nullptr) VecDestroy(&data_support_);
  }
  if(data_comps_ != nullptr) VecDestroy(&data_comps_);
  if(obs_filter_ != nullptr) VecDestroy(&obs_filter_);
  if(p_rec_ != nullptr) VecDestroy(&p_rec_);
  if(velocity_ != nullptr) VecDestroy(&velocity_);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  std::stringstream ss;
  ss << "    grid size: " << params->grid_->n_[0] << "x" << params->grid_->n_[1] << "x" << params->grid_->n_[2];
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str(""); ss.clear();

  // === set parameters
  spec_ops_ = spec_ops;
  params_ = params;
  app_settings_ = app_settings;
  data_ = std::make_shared<Data>();

  // === create tmp vector according to distributed grid
  ierr = VecCreate(PETSC_COMM_WORLD, &tmp_); CHKERRQ(ierr);
  ierr = VecSetSizes(tmp_, params_->grid_->nl_, params_->grid_->ng_); CHKERRQ(ierr);
  ierr = setupVec(tmp_); CHKERRQ(ierr);
  ierr = VecSet(tmp_, 0.0); CHKERRQ(ierr);

  // === create p_rec vector according to unknowns inverted for
  // ierr = VecCreateSeq(PETSC_COMM_SELF, params_->tu_->np_ + params_->get_nk(), &p_rec_); CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF, params_->tu_->np_, &p_rec_); CHKERRQ(ierr);
  ierr = setupVec(p_rec_, SEQ); CHKERRQ(ierr);

  // === initialize tumor, phi, mat_prop
  tumor_ = std::make_shared<Tumor>(params_, spec_ops_);
  ierr = tumor_->initialize(p_rec_, params_, spec_ops_, nullptr, nullptr); CHKERRQ(ierr);
  // === initialize pde- and derivative operators
  ierr = initializeOperators(); CHKERRQ(ierr);

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
    // ierr = pde_operators_->preAdvection(wm_, gm_, vt_, mri_, app_settings_->syn_->pre_adv_time_); CHKERRQ(ierr);
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
  // update diffusion coefficient, reaction coefficient, phi with material properties
  ierr = updateTumorCoefficients(wm_, gm_, csf_, vt_, nullptr); CHKERRQ(ierr);
  ierr = tumor_->mat_prop_->setAtlas(gm_, wm_, csf_, vt_, nullptr); CHKERRQ(ierr);
  #ifdef CUDA
  cudaPrintDeviceMemory();
  #endif
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::run() {
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
PetscErrorCode SolverInterface::finalize() {
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
  // transport mri
  if (params_->tu_->transport_mri_) {
    if (mri_ == nullptr) {
      ierr = VecDuplicate(tmp_, &mri_); CHKERRQ(ierr);
      ierr = dataIn(mri_, params_, app_settings_->path_->mri_); CHKERRQ(ierr);
      if (tumor_->mat_prop_->mri_ == nullptr) {
        tumor_->mat_prop_->mri_ = mri_;
      }
    }
  }
  ierr = pde_operators_->solveState(0); CHKERRQ(ierr);
  ierr = VecCopy(tumor_->c_t_, tmp_); CHKERRQ(ierr);

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
  ierr = VecAXPY(tmp_, -1.0, data_->dt1()); CHKERRQ(ierr);
  ierr = VecNorm(data_->dt1(), NORM_2, &data_norm); CHKERRQ(ierr);
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
  ierr = tumor_->obs_->apply(tmp_, data_->dt1(), 1); CHKERRQ(ierr);
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
  if(data_t0_ != nullptr) {
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
  } else {
    ierr = tuMSGstd(" Cannot compute errors for TIL, since TIL is nullptr."); CHKERRQ(ierr);
  }

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
PetscErrorCode SolverInterface::predict() {
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
        ierr = VecCopy(data_->dt0(), tumor_->c_0_); CHKERRQ(ierr);
      } else {
        ierr = tumor_->phi_->apply(tumor_->c_0_, p_rec_); CHKERRQ(ierr);
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
          ierr = updateTumorCoefficients(wm_, gm_, csf_, vt_, nullptr);
          ierr = tumor_->mat_prop_->setAtlas(gm_, wm_, csf_, vt_, nullptr); CHKERRQ(ierr);
        } else {
          ierr = tumor_->mat_prop_->resetValues(); CHKERRQ(ierr);
        }
        ierr = pde_operators_->solveState(0); CHKERRQ(ierr);
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
          ierr = VecDuplicate(data_->dt1(), &true_dat_t1); CHKERRQ(ierr);
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
PetscErrorCode SolverInterface::setupData() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  // === read data: generate synthetic or read real
  if (app_settings_->syn_->enabled_) {
    // data t1 and data t0 is generated synthetically using user given cm and tumor model
    ierr = createSynthetic(); CHKERRQ(ierr);
    data_support_ = data_t1_;
  } else {
    // read in target data (t1 and/or t0); observation operator
    ierr = readData(); CHKERRQ(ierr);
  }
  data_->set(data_t1_, data_t0_);
  // === set observation operator
  if (custom_obs_) {
    ierr = tumor_->obs_->setCustomFilter(obs_filter_, 1);
    ss << " Setting custom observation mask";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  } else {
    ierr = tumor_->obs_->setDefaultFilter(data_->dt1(), 1, params_->tu_->obs_threshold_1_); CHKERRQ(ierr);
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
  ierr = tumor_->obs_->apply(data_->dt1(), data_->dt1()), 1; CHKERRQ(ierr);
  ierr = tumor_->obs_->apply(data_support_, data_support_, 1); CHKERRQ(ierr);
  if (has_dt0_) {
    ierr = tumor_->obs_->apply(data_t0_, data_t0_, 0); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::readAtlas() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ScalarType sigma_smooth = params_->tu_->smoothing_factor_ * 2 * M_PI / params_->grid_->n_[0];

  if (!app_settings_->path_->seg_.empty()) {
    ierr = dataIn(tmp_, params_, app_settings_->path_->seg_); CHKERRQ(ierr);
    if(app_settings_->atlas_seg_[0] <= 0 || app_settings_->atlas_seg_[1] <= 0 || app_settings_->atlas_seg_[2] <= 0) {
      ierr = tuMSGwarn(" Error: The segmentation must at least have WM, GM, VT."); CHKERRQ(ierr);
      exit(0);
    } else {
      ierr = VecDuplicate(tmp_, &wm_); CHKERRQ(ierr);
      ierr = VecDuplicate(tmp_, &gm_); CHKERRQ(ierr);
      ierr = VecDuplicate(tmp_, &vt_); CHKERRQ(ierr);
      csf_ = nullptr; data_t1_ = nullptr;
      if (app_settings_->atlas_seg_[3] > 0) {
        ierr = VecDuplicate(tmp_, &csf_); CHKERRQ(ierr);
      }
      if (app_settings_->atlas_seg_[4] > 0 || (app_settings_->atlas_seg_[5] > 0 && app_settings_->atlas_seg_[6] > 0)) {
        ierr = VecDuplicate(tmp_, &data_t1_); CHKERRQ(ierr);
        data_t1_from_seg_ = true;
      }
    }
    ierr = splitSegmentation(tmp_, wm_, gm_, vt_, csf_, data_t1_, params_->grid_->nl_, app_settings_->atlas_seg_); CHKERRQ(ierr);

    // TODO(K): test read in from segmentation
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
PetscErrorCode SolverInterface::readData() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  std::stringstream ss;
  // ScalarType sigma_smooth = params_->smoothing_factor_ * 2 * M_PI / params_->grid_->n_[0];
  ScalarType sig_data = params_->tu_->smoothing_factor_data_ * 2 * M_PI / params_->grid_->n_[0];
  ScalarType min, max;

  if (!app_settings_->path_->data_t1_.empty()) {
    if(!data_t1_from_seg_) { // If false, t=1 data has already been read from segmentation in readAtlas
      ierr = VecDuplicate(tmp_, &data_t1_); CHKERRQ(ierr);
      ierr = dataIn(data_t1_, params_, app_settings_->path_->data_t1_); CHKERRQ(ierr);
    }
    if (params_->tu_->smoothing_factor_data_ > 0) {
      ierr = spec_ops_->weierstrassSmoother(data_t1_, data_t1_, params_, sig_data); CHKERRQ(ierr);
      ss << " smoothing c(1) with factor: " << params_->tu_->smoothing_factor_data_ << ", and sigma: " << sig_data;
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
    }
    // make obs threshold relative
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
  } else {data_t0_ = nullptr;}
  // else { ierr = VecSet(data_t0_, 0.); CHKERRQ(ierr);} K: there should be no else
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::readVelocity() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  if (!app_settings_->path_->velocity_x1_.empty()) {
    ierr = readVecField(tumor_->velocity_.get(), app_settings_->path_->velocity_x1_, app_settings_->path_->velocity_x2_, app_settings_->path_->velocity_x3_, params_); CHKERRQ(ierr);
    ierr = tumor_->velocity_->scale(-1); CHKERRQ(ierr);
    Vec mag;
    ierr = VecDuplicate(data_->dt1(), &mag); CHKERRQ(ierr);
    ierr = tumor_->velocity_->computeMagnitude(mag); CHKERRQ(ierr);
    ierr = dataOut(mag, params_, "velocity_mag.nc"); CHKERRQ(ierr);
    ScalarType vnorm;
    ierr = VecNorm(mag, NORM_2, &vnorm); CHKERRQ(ierr);
    std::stringstream ss;
    ss << " Given velocity read in: norm of velocity magnitude: " << vnorm;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str("");ss.clear();
    params_->opt_->adv_velocity_set_ = true;
    ierr = VecDestroy(&mag); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::readDiffusionFiberTensor() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // TODO(K): implement reading in of DTI tensor.
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::createSynthetic() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  // save parameters
  ScalarType rho_temp = params_->tu_->rho_, k_temp = params_->tu_->k_, dt_temp = params_->tu_->dt_;
  ScalarType forcing_factor_temp = params_->tu_->forcing_factor_;
  int nt_temp = params_->tu_->nt_;
  // set to synthetic parameters
  params_->tu_->rho_ = app_settings_->syn_->rho_;
  params_->tu_->k_ = app_settings_->syn_->k_;
  params_->tu_->dt_ = app_settings_->syn_->dt_;
  params_->tu_->nt_ = app_settings_->syn_->nt_;
  params_->tu_->forcing_factor_ = app_settings_->syn_->forcing_factor_;

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

  std::map<std::string, Vec> species;
  if (params_->tu_->model_ == 5) {
    ierr = VecCopy(data_t0_, tumor_->c_0_); CHKERRQ(ierr);
    ierr = pde_operators_->solveState(0); CHKERRQ(ierr);
    ierr = VecCopy(tumor_->c_t_, data_t1_); CHKERRQ(ierr);
    species = tumor_->species_;
  } else {
    ierr = VecCopy(data_t0_, tumor_->c_0_); CHKERRQ(ierr);
    ierr = pde_operators_->solveState(0); CHKERRQ(ierr);
    ierr = VecCopy(tumor_->c_t_, data_t1_); CHKERRQ(ierr);
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
  params_->tu_->forcing_factor_ = forcing_factor_temp;

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::initializeGaussians() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  std::stringstream ss;
  ss << " Initialize Gaussian basis functions.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  // warmstart p given but  it's not a coarse solution
  if (warmstart_p_ && !app_settings_->inject_solution_) {
    ss << " .. solver warmstart: using p_vec and Gaussians from file.";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    ierr = tumor_->phi_->setGaussians(app_settings_->path_->phi_); CHKERRQ(ierr);
    ierr = tumor_->phi_->setValues(tumor_->mat_prop_); CHKERRQ(ierr);
    ierr = readPVec(&p_rec_, params_->tu_->np_ + params_->get_nk() + params_->get_nr(), params_->tu_->np_, app_settings_->path_->pvec_); CHKERRQ(ierr);
  // either no warmstart p given or coarse solution is injected
  } else {
    if (!app_settings_->path_->data_support_data_.empty()) {  // read Gaussian centers and comp labels from txt file
      ss << " .. using centers from .dat file.";
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      ierr = tumor_->phi_->setGaussians(app_settings_->path_->data_support_data_, true); CHKERRQ(ierr);
    } else if (app_settings_->syn_->enabled_) {
      // synthetic data generation in data_t1_
      ss << " .. using synthetic data.";
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      ierr = tumor_->phi_->setGaussians(data_->dt1()); CHKERRQ(ierr);
    } else if (data_support_ != nullptr) {  // read Gaussian centers and comp labels from nc files
      if(data_t1_from_seg_) {
        ss << " .. using tumor data from segmentation file.";
      } else {
        ss << " .. using data_support.nc file..";
      }
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      ierr = tumor_->phi_->setGaussians(data_support_); CHKERRQ(ierr);
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
    ierr = VecCreateSeq(PETSC_COMM_SELF, params_->tu_->np_ + n_inv_, &p_rec_); CHKERRQ(ierr);
    ierr = setupVec(p_rec_, SEQ); CHKERRQ(ierr);
    ierr = resetOperators(p_rec_); CHKERRQ(ierr);
    ss << " .. creating p_vec of size " << params_->tu_->np_ + n_inv_ << ", where np = " << params_->tu_->np_ << " is the number of selected Gaussians; nk+nr = " << n_inv_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    n_inv_ += params_->tu_->np_;
  }

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::initializeOperators() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // === initialize pde- and derivative operators
  switch (params_->tu_->model_) {
    case 1: {
        pde_operators_ = std::make_shared<PdeOperatorsRD>(tumor_, params_, spec_ops_);
        if (params_->opt_->cross_entropy_loss_) {
          derivative_operators_ = std::make_shared<DerivativeOperatorsKL>(pde_operators_, params_, tumor_);
        } else {
          derivative_operators_ = std::make_shared<DerivativeOperatorsRD>(pde_operators_, params_, tumor_);
        }
    break;
    }
    case 2: {
      pde_operators_ = std::make_shared<PdeOperatorsRD>(tumor_, params_, spec_ops_);
      derivative_operators_ = std::make_shared<DerivativeOperatorsRD>(pde_operators_, params_, tumor_);
      break;
    }
    case 3: {
      pde_operators_ = std::make_shared<PdeOperatorsRD>(tumor_, params_, spec_ops_);
      derivative_operators_ = std::make_shared<DerivativeOperatorsRDObj>(pde_operators_, params_, tumor_);
      break;
    }
    case 4: {
      pde_operators_ = std::make_shared<PdeOperatorsMassEffect>(tumor_, params_, spec_ops_);
      derivative_operators_ = std::make_shared<DerivativeOperatorsMassEffect>(pde_operators_, params_, tumor_);
      break;
    }
    case 5: {
      pde_operators_ = std::make_shared<PdeOperatorsMultiSpecies>(tumor_, params_, spec_ops_);
      derivative_operators_ = std::make_shared<DerivativeOperatorsRD>(pde_operators_, params_, tumor_);
      break;
    }
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::resetOperators(Vec p, bool ninv_changed, bool nt_changed) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tumor_->setParams(p, params_, ninv_changed); CHKERRQ(ierr);
  if(ninv_changed) {
    ierr = derivative_operators_->reset(p, pde_operators_, params_, tumor_); CHKERRQ(ierr);
  }
  if(nt_changed) {
    ierr = pde_operators_->reset(params_, tumor_); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::updateTumorCoefficients(Vec wm, Vec gm, Vec csf, Vec vt, Vec bg) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (wm == nullptr) {
    ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector wm is nullptr."); CHKERRQ(ierr);
  }
  if (gm == nullptr) {
    ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector gm is nullptr."); CHKERRQ(ierr);
  }
  if (vt == nullptr) {
    ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector vt is nullptr."); CHKERRQ(ierr);
  }
  if (csf == nullptr) {
    ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector csf is nullptr."); CHKERRQ(ierr);
  }
  Event e("update-tumor-coefficients"); // timing
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ierr = tumor_->mat_prop_->setValuesCustom(gm, wm, csf, vt, bg, params_); CHKERRQ(ierr);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_); CHKERRQ(ierr);
  ierr = tumor_->phi_->setValues(tumor_->mat_prop_); CHKERRQ(ierr);
  ierr = pde_operators_->diff_solver_->precFactor(); CHKERRQ(ierr);
  self_exec_time += MPI_Wtime();       // timing
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========            Helper Functions for SIBIA             ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::setDistMeassureSimulationGeoImages(Vec wm, Vec gm, Vec csf, Vec bg) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (wm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureSimulationGeoImages) Vector wm is nullptr."); CHKERRQ(ierr);
  }
  if (gm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureSimulationGeoImages) Vector gm is nullptr."); CHKERRQ(ierr);
  }
  if (csf == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureSimulationGeoImages) Vector csf is nullptr."); CHKERRQ(ierr);
  }
  /** Sets the image vectors for the simulation geometry material properties
   *  - MOVING PATIENT: mA(0) (= initial helathy atlas)
   *  - MOVING ATLAS:   mA(1) (= initial helathy patient)
   */
  return derivative_operators_->setDistMeassureSimulationGeoImages(wm, gm, csf, nullptr, nullptr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::setDistMeassureTargetDataImages(Vec wm, Vec gm, Vec csf, Vec bg) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (wm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureTargetDataImages) Vector wm is nullptr."); CHKERRQ(ierr);
  }
  if (gm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureTargetDataImages) Vector gm is nullptr."); CHKERRQ(ierr);
  }
  if (csf == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureTargetDataImages) Vector csf is nullptr."); CHKERRQ(ierr);
  }
  /** Sets the image vectors for the simulation geometry material properties
   * Sets the image vectors for the target (patient) geometry material properties
   *  - MOVING PATIENT: mP(1) (= advected patient)
   *  - MOVING ATLAS:   mR    (= patient data)
   */
  return derivative_operators_->setDistMeassureTargetDataImages(wm, gm, csf, nullptr, nullptr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::setDistMeassureDiffImages(Vec wm, Vec gm, Vec csf, Vec bg) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (wm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureDiffImages) Vector wm is nullptr."); CHKERRQ(ierr);
  }
  if (gm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureDiffImages) Vector gm is nullptr."); CHKERRQ(ierr);
  }
  if (csf == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureDiffImages) Vector csf is nullptr."); CHKERRQ(ierr);
  }
  /** Sets the image vectors for the simulation geometry material properties
   * Sets the image vectors for the distance measure difference
   *  - MOVING PATIENT: || mA(0)(1-c(1)) - mP(1) ||^2
   *  - MOVING ATLAS:   || mA(1)(1-c(1)) - mR    ||^2
   */
  return derivative_operators_->setDistMeassureDiffImages(wm, gm, csf, nullptr, nullptr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::computeTumorContributionRegistration(Vec q1, Vec q2, Vec q4) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (pde_operators_ != nullptr) {
    ierr = pde_operators_->computeTumorContributionRegistration(q1, q2, nullptr, q4); CHKERRQ(ierr);
  } else {
    ierr = tuMSGwarn("Error: (in computeTumorContributionRegistration()) PdeOperators not initialized. Exiting .."); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========            Helper Functions for Cython            ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::smooth(Vec x, ScalarType num_voxels) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ScalarType sigma_smooth = num_voxels * 2.0 * M_PI / params_->grid_->n_[0];
  ierr = spec_ops_->weierstrassSmoother(x, x, params_, sigma_smooth); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::readNetCDF(Vec A, std::string filename) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ierr = dataIn(A, params_, filename.c_str()); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::writeNetCDF(Vec A, std::string filename) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ierr = dataOut(A, params_, filename.c_str()); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::initializeEvent() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  EventRegistry::initialize();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SolverInterface::finalizeEvent() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  EventRegistry::finalize();
  if (procid == 0) {
    EventRegistry r;
    r.print();
    r.print("TumorSolverTimings.log", true);
  }
  PetscFunctionReturn(ierr);
}
