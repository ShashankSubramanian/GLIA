#include "PdeOperators.h"

/* #### --------------------------------------------------------------------------- #### */
/* #### ========              PDE Ops. Multispecies effect model           ======== #### */
/* #### --------------------------------------------------------------------------- #### */
PetscErrorCode PdeOperatorsMultiSpecies::reset(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  // no-op so far

  params_ = params;
  if (tumor != nullptr) tumor_ = tumor;

  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::computeReactionRate(Vec m) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-reaction");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType *ox_ptr, *m_ptr, *rho_ptr;
  ierr = vecGetArray(m, &m_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->species_["oxygen"], &ox_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);
#ifdef CUDA
  //computeReactionRateCuda(m_ptr, ox_ptr, rho_ptr, params_->tu_->ox_hypoxia_, params_->grid_->nl_);
  computeReactionRateCuda(m_ptr, ox_ptr, rho_ptr, params_->tu_->ox_hypoxia_, params_->grid_->nl_, params_->tu_->ox_inv_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) m_ptr[i] = rho_ptr[i] * (1 / (1 + std::exp(-100 * (ox_ptr[i] - params_->tu_->ox_hypoxia_))));
#endif

  ierr = vecRestoreArray(m, &m_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->species_["oxygen"], &ox_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::computeTransition(Vec alpha, Vec beta) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-transition");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType *ox_ptr, *alpha_ptr, *beta_ptr, *p_ptr, *i_ptr;
  //ScalarType thres = 0.9;
  ierr = vecGetArray(alpha, &alpha_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(beta, &beta_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->species_["oxygen"], &ox_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->species_["proliferative"], &p_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->species_["infiltrative"], &i_ptr); CHKERRQ(ierr);

#ifdef CUDA
  //computeTransitionCuda(alpha_ptr, beta_ptr, ox_ptr, p_ptr, i_ptr, params_->tu_->alpha_0_, params_->tu_->beta_0_, params_->tu_->ox_inv_, thres, params_->grid_->nl_);
  computeTransitionCuda(alpha_ptr, beta_ptr, ox_ptr, p_ptr, i_ptr, params_->tu_->alpha_0_, params_->tu_->beta_0_, params_->tu_->ox_inv_, params_->tu_->sigma_b_, params_->grid_->nl_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    alpha_ptr[i] = params_->tu_->alpha_0_ * (1 / (1 + std::exp(100 * (ox_ptr[i] - params_->tu_->ox_inv_))));
    beta_ptr[i] = params_->tu_->beta_0_ * ox_ptr[i];
  }
#endif

  ierr = vecRestoreArray(alpha, &alpha_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(beta, &beta_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->species_["oxygen"], &ox_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->species_["proliferative"], &p_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->species_["infiltrative"], &i_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::computeThesholder(Vec h) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-thresholder");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType *ox_ptr, *h_ptr;
  ierr = vecGetArray(h, &h_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->species_["oxygen"], &ox_ptr); CHKERRQ(ierr);

#ifdef CUDA
  computeThesholderCuda(h_ptr, ox_ptr, params_->tu_->ox_hypoxia_, params_->grid_->nl_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) h_ptr[i] = (1 / (1 + std::exp(100 * (ox_ptr[i] - params_->tu_->ox_hypoxia_))));
#endif

  ierr = vecRestoreArray(h, &h_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->species_["oxygen"], &ox_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::computeSources(Vec p, Vec i, Vec n, Vec O, ScalarType dt) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-sources");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ierr = computeReactionRate(tumor_->work_[0]); CHKERRQ(ierr);
  ierr = computeTransition(tumor_->work_[1], tumor_->work_[2]); CHKERRQ(ierr);
  ierr = computeThesholder(tumor_->work_[3]); CHKERRQ(ierr);

  ScalarType *p_ptr, *i_ptr, *n_ptr, *al_ptr, *bet_ptr, *h_ptr, *m_ptr, *di_ptr;
  ScalarType *gm_ptr, *wm_ptr;
  ScalarType *ox_ptr;

  ierr = vecGetArray(p, &p_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(i, &i_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(n, &n_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->work_[0], &m_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->work_[1], &al_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->work_[2], &bet_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->work_[3], &h_ptr); CHKERRQ(ierr);

  ierr = vecGetArray(tumor_->mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->wm_, &wm_ptr); CHKERRQ(ierr);

  ierr = vecGetArray(O, &ox_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->work_[11], &di_ptr); CHKERRQ(ierr);

#ifdef CUDA
  computeSourcesCuda(p_ptr, i_ptr, n_ptr, m_ptr, al_ptr, bet_ptr, h_ptr, gm_ptr, wm_ptr, ox_ptr, di_ptr, dt, params_->tu_->death_rate_, params_->tu_->ox_source_, params_->tu_->ox_consumption_, params_->grid_->nl_);
#else
  ScalarType p_temp, i_temp, frac_1, frac_2;
  ScalarType ox_heal = 1;
  ScalarType reac_ratio = 1;
  ScalarType death_ratio = 1;
  for (int i = 0; i < params_->grid_->nl_; i++) {
    p_temp = p_ptr[i];
    i_temp = i_ptr[i];
    p_ptr[i] += dt * (m_ptr[i] * p_ptr[i] * (1. - p_ptr[i]) - al_ptr[i] * p_ptr[i] + bet_ptr[i] * i_ptr[i] - params_->tu_->death_rate_ * h_ptr[i] * p_ptr[i]);
    i_ptr[i] += dt * (reac_ratio * m_ptr[i] * i_ptr[i] * (1. - i_ptr[i]) + al_ptr[i] * p_temp - bet_ptr[i] * i_ptr[i] - death_ratio * params_->tu_->death_rate_ * h_ptr[i] * i_ptr[i]);
    n_ptr[i] += dt * (h_ptr[i] * params_->tu_->death_rate_ * (p_temp + death_ratio * i_temp + gm_ptr[i] + wm_ptr[i]));
    ox_ptr[i] += dt * (-params_->tu_->ox_consumption_ * p_temp + params_->tu_->ox_source_ * (ox_heal - ox_ptr[i]) * (gm_ptr[i] + wm_ptr[i]));
    ox_ptr[i] = (ox_ptr[i] <= 0.) ? 0. : ox_ptr[i];

    // conserve healthy cells
    if (gm_ptr[i] > 0.01 || wm_ptr[i] > 0.01) {
      frac_1 = gm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
      frac_2 = wm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
    } else {
      frac_1 = 0.;
      frac_2 = 0.;
    }
    frac_1 = (std::isnan(frac_1)) ? 0. : frac_1;
    frac_2 = (std::isnan(frac_2)) ? 0. : frac_2;
    gm_ptr[i] += -dt * (frac_1 * (m_ptr[i] * p_temp * (1. - p_temp) + reac_ratio * m_ptr[i] * i_temp * (1. - i_temp) + di_ptr[i]) + h_ptr[i] * params_->tu_->death_rate_ * gm_ptr[i]);
    wm_ptr[i] += -dt * (frac_2 * (m_ptr[i] * p_temp * (1. - p_temp) + reac_ratio * m_ptr[i] * i_temp * (1. - i_temp) + di_ptr[i]) + h_ptr[i] * params_->tu_->death_rate_ * wm_ptr[i]);
  }
#endif

  ierr = vecRestoreArray(p, &p_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(i, &i_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(n, &n_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->work_[0], &m_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->work_[1], &al_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->work_[2], &bet_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->work_[3], &h_ptr); CHKERRQ(ierr);

  ierr = vecRestoreArray(tumor_->mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->wm_, &wm_ptr); CHKERRQ(ierr);

  ierr = vecRestoreArray(O, &ox_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->work_[11], &di_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::updateReacAndDiffCoefficients(Vec seg, std::shared_ptr<Tumor> tumor) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType *bg_ptr, *gm_ptr, *vt_ptr, *csf_ptr, *rho_ptr, *k_ptr;
  ierr = vecGetArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->k_->kxx_, &k_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->bg_, &bg_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->mat_prop_->csf_, &csf_ptr); CHKERRQ(ierr);

  ScalarType temp = 1.;
  ScalarType gm_k_scale = 1 - params_->tu_->k_gm_wm_ratio_;
  ScalarType gm_r_scale = 1 - params_->tu_->r_gm_wm_ratio_;
#ifdef CUDA
  updateReacAndDiffCoefficientsCuda(rho_ptr, k_ptr, bg_ptr, gm_ptr, vt_ptr, csf_ptr, params_->tu_->rho_, params_->tu_->k_, gm_r_scale, gm_k_scale, params_->grid_->nl_);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    temp = (1 - (bg_ptr[i] + gm_r_scale * gm_ptr[i] + vt_ptr[i] + csf_ptr[i]));
    temp = (temp < 0) ? 0 : temp;
    rho_ptr[i] = temp * params_->tu_->rho_;
    temp = (1 - (bg_ptr[i] + gm_k_scale * gm_ptr[i] + vt_ptr[i] + csf_ptr[i]));
    temp = (temp < 0) ? 0 : temp;
    k_ptr[i] = temp * params_->tu_->k_;
  }
#endif

  ierr = vecRestoreArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->k_->kxx_, &k_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->bg_, &bg_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->gm_, &gm_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->vt_, &vt_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->mat_prop_->csf_, &csf_ptr); CHKERRQ(ierr);

  // copy kxx to other directions
  ierr = VecCopy(tumor_->k_->kxx_, tumor_->k_->kyy_); CHKERRQ(ierr);
  ierr = VecCopy(tumor_->k_->kxx_, tumor_->k_->kzz_); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::solveState(int linearized) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  std::stringstream ss;
  std::stringstream s;
  Event e("tumor-solve-state");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType dt = params_->tu_->dt_;
  int nt = params_->tu_->nt_;

  params_->tu_->statistics_.nb_state_solves++;

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  ierr = displacement_old_->set(0); CHKERRQ(ierr);

  ierr = VecCopy(tumor_->c_0_, tumor_->species_["proliferative"]); CHKERRQ(ierr);
  ierr = VecCopy(tumor_->c_0_, tumor_->species_["infiltrative"]); CHKERRQ(ierr);
  // set infiltrative as a small fraction of proliferative; oxygen is max everywhere in the beginning - consider changing to (max - p) if needed
  //ierr = VecScale(tumor_->species_["infiltrative"], 0); CHKERRQ(ierr);

  ierr = VecScale(tumor_->species_["infiltrative"], params_->tu_->i0_c0_ratio_); CHKERRQ(ierr);
  ScalarType p0_c0_ratio_ = 1 - params_->tu_->i0_c0_ratio_;
  ierr = VecScale(tumor_->species_["proliferative"], p0_c0_ratio_); CHKERRQ(ierr);
  
  ierr = VecSet(tumor_->species_["oxygen"], 1.); CHKERRQ(ierr);
  ierr = VecSet(tumor_->species_["necrotic"], 0.); CHKERRQ(ierr);
  ierr = VecSet(tumor_->species_["edema"], 0.); CHKERRQ(ierr);
  


  ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / params_->grid_->n_[0];
  // smooth i_t to keep aliasing to a minimum
  // ierr = spec_ops_->weierstrassSmoother (tumor_->species_["infiltrative"], tumor_->species_["infiltrative"], params_, sigma_smooth);     CHKERRQ (ierr);

  ierr = tumor_->clipTumor(); CHKERRQ(ierr);

  // no healthy cells where tumor is maximum
  ierr = VecWAXPY(tumor_->c_t_, 1., tumor_->species_["proliferative"], tumor_->species_["infiltrative"]); CHKERRQ(ierr);
  ierr = tumor_->mat_prop_->filterTumor(tumor_->c_t_); CHKERRQ(ierr);

  ScalarType k1, k2, k3, r1, r2, r3;
  k1 = params_->tu_->k_;
  k2 = params_->tu_->k_gm_wm_ratio_ * params_->tu_->k_;
  k3 = 0;
  r1 = params_->tu_->rho_;
  r2 = params_->tu_->r_gm_wm_ratio_ * params_->tu_->rho_;
  r3 = 0;

  // force compute
  ierr = VecCopy(tumor_->species_["proliferative"], tumor_->c_t_); CHKERRQ(ierr);

  if (params_->tu_->forcing_factor_ > 0) {
    ierr = tumor_->computeForce(tumor_->c_t_);
    // displacement compute through elasticity solve
    ierr = elasticity_solver_->solve(tumor_->displacement_, tumor_->force_);
    // copy displacement to old vector
    ierr = displacement_old_->copy(tumor_->displacement_);
  }

  diff_ksp_itr_state_ = 0;
  ScalarType vel_max;
  ScalarType cfl;
  ScalarType vel_x_norm, vel_y_norm, vel_z_norm;

  bool flag_smooth_velocity = true;
  bool write_output_and_break = false;

  ScalarType max_cfl = 8;

  for (int i = 0; i <= nt; i++) {
    s << "Time step = " << i;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    // compute CFL
    ierr = tumor_->computeEdema(); CHKERRQ(ierr);
    ierr = tumor_->computeSegmentation(); CHKERRQ(ierr);
    ierr = tumor_->velocity_->computeMagnitude(magnitude_);
    ierr = VecMax(magnitude_, NULL, &vel_max); CHKERRQ(ierr);
    cfl = dt * vel_max / params_->grid_->h_[0];
    s << "CFL = " << cfl;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
    // Adaptively time step if CFL is too large
    if (cfl >= max_cfl) {
      s << "CFL is too large (>=" << max_cfl << "); consider using smaller forcing factor; exiting solver...";
      ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();
      write_output_and_break = true;
    }

    if ((params_->tu_->write_output_ && i % 50 == 0) || write_output_and_break) {
      ss << "velocity_t[" << i << "].nc";
      dataOut(magnitude_, params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ierr = displacement_old_->computeMagnitude(magnitude_);
      ss << "displacement_t[" << i << "].nc";
      dataOut(magnitude_, params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ss << "seg_t[" << i << "].nc";
      dataOut(tumor_->seg_, params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ss << "c_t[" << i << "].nc";
      dataOut(tumor_->c_t_, params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ss << "p_t[" << i << "].nc";
      dataOut(tumor_->species_["proliferative"], params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ss << "i_t[" << i << "].nc";
      dataOut(tumor_->species_["infiltrative"], params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ss << "n_t[" << i << "].nc";
      dataOut(tumor_->species_["necrotic"], params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      ss << "o_t[" << i << "].nc";
      dataOut(tumor_->species_["oxygen"], params_, ss.str().c_str());
      ss.str(std::string());
      ss.clear();
      if (params_->tu_->verbosity_ > 2) {
        //     ss << "rho_t[" << i << "].nc";
        //     dataOut (tumor_->rho_->rho_vec_, params_, ss.str().c_str());
        //     ss.str(std::string()); ss.clear();
        ss << "m_t[" << i << "].nc";
        dataOut(tumor_->work_[0], params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "ed_t[" << i << "].nc";
        dataOut(tumor_->species_["edema"], params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "kxx_t[" << i << "].nc";
        dataOut(tumor_->k_->kxx_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        // ss << "lam_t[" << i << "].nc";
        // dataOut (elasticity_solver_->ctx_->lam_, params_, ss.str().c_str());
        // ss.str(std::string()); ss.clear();
        // ss << "mu_t[" << i << "].nc";
        // dataOut (elasticity_solver_->ctx_->mu_, params_, ss.str().c_str());
        // ss.str(std::string()); ss.clear();
        // ss << "scr_t[" << i << "].nc";
        // dataOut (elasticity_solver_->ctx_->screen_, params_, ss.str().c_str());
        // ss.str(std::string()); ss.clear();
        ierr = tumor_->force_->computeMagnitude(magnitude_);
        ss << "force_t[" << i << "].nc";
        dataOut(magnitude_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "vt_t[" << i << "].nc";
        dataOut(tumor_->mat_prop_->vt_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "csf_t[" << i << "].nc";
        dataOut(tumor_->mat_prop_->csf_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "wm_t[" << i << "].nc";
        dataOut(tumor_->mat_prop_->wm_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
        ss << "gm_t[" << i << "].nc";
        dataOut(tumor_->mat_prop_->gm_, params_, ss.str().c_str());
        ss.str(std::string());
        ss.clear();
      }
    }

    if (write_output_and_break) break;
    // ------------------------------------------------ advection  ------------------------------------------------

    // Update diffusivity and reaction coefficient
    ierr = updateReacAndDiffCoefficients(tumor_->seg_, tumor_); CHKERRQ(ierr);
    ierr = tumor_->k_->updateIsotropicCoefficients(k1, k2, k3, tumor_->mat_prop_, params_); CHKERRQ(ierr);

    // need to update prefactors for diffusion KSP preconditioner, as k changed
    ierr = diff_solver_->precFactor(); CHKERRQ(ierr);

    if (params_->tu_->forcing_factor_ > 0) {
      // Advection of tumor and healthy tissue
      // first compute trajectories for semi-Lagrangian solve as velocity is changing every itr
      adv_solver_->trajectoryIsComputed_ = false;
      ierr = adv_solver_->solve(tumor_->mat_prop_->gm_, tumor_->velocity_, dt); CHKERRQ(ierr);
      ierr = adv_solver_->solve(tumor_->mat_prop_->wm_, tumor_->velocity_, dt); CHKERRQ(ierr);
      adv_solver_->advection_mode_ = 2;  // pure advection for vt
      ierr = adv_solver_->solve(tumor_->mat_prop_->vt_, tumor_->velocity_, dt); CHKERRQ(ierr);
      ierr = adv_solver_->solve(tumor_->mat_prop_->csf_, tumor_->velocity_, dt); CHKERRQ(ierr);
      adv_solver_->advection_mode_ = 1;  // reset to mass conservation
      ierr = adv_solver_->solve(tumor_->species_["proliferative"], tumor_->velocity_, dt); CHKERRQ(ierr);
      ierr = adv_solver_->solve(tumor_->species_["infiltrative"], tumor_->velocity_, dt); CHKERRQ(ierr);
      ierr = adv_solver_->solve(tumor_->species_["necrotic"], tumor_->velocity_, dt); CHKERRQ(ierr);
    }

    // All solves complete except elasticity: clip values to ensure positivity
    // clip healthy tissues
    ierr = tumor_->mat_prop_->clipHealthyTissues(); CHKERRQ(ierr);
    // clip tumor : single-precision advection seems to have issues if this is not clipped.
    ierr = tumor_->clipTumor(); CHKERRQ(ierr);

    // smooth infiltrative to avoid aliasing
    ierr = spec_ops_->weierstrassSmoother(tumor_->species_["infiltrative"], tumor_->species_["infiltrative"], params_, sigma_smooth); CHKERRQ(ierr);

    // compute Di to be used for healthy cell evolution equations: make sure work[11] is not used till sources are computed
    ierr = VecCopy(tumor_->species_["infiltrative"], tumor_->work_[11]); CHKERRQ(ierr);
    ierr = tumor_->k_->applyD(tumor_->work_[11], tumor_->work_[11]); CHKERRQ(ierr);

    // ------------------------------------------------ diffusion  ------------------------------------------------
    ierr = diff_solver_->solve(tumor_->species_["infiltrative"], dt);
    diff_ksp_itr_state_ += diff_solver_->ksp_itr_; CHKERRQ(ierr);
    // ierr = diff_solver_->solve (tumor_->species_["oxygen"], dt);               diff_ksp_itr_state_ += diff_solver_->ksp_itr_;   CHKERRQ (ierr);

    // ------------------------------------------------ explicit source terms for all equations (includes reaction source)  ------------------------------------------------
    ierr = computeSources(tumor_->species_["proliferative"], tumor_->species_["infiltrative"], tumor_->species_["necrotic"], tumor_->species_["oxygen"], dt); CHKERRQ(ierr);

    // set tumor core as c_t_
    ierr = VecWAXPY(tumor_->c_t_, 1., tumor_->species_["proliferative"], tumor_->species_["necrotic"]); CHKERRQ(ierr);
    ierr = VecAXPY(tumor_->c_t_, 1., tumor_->species_["infiltrative"]); CHKERRQ(ierr);

    if (params_->tu_->forcing_factor_ > 0) {
      // ------------------------------------------------ elasticity update ------------------------------------------------
      // force compute
      ierr = tumor_->computeForce(tumor_->c_t_);
      // displacement compute through elasticity solve: Linv(force_) = displacement_
      ierr = elasticity_solver_->solve(tumor_->displacement_, tumor_->force_);
      // compute velocity
      ierr = VecWAXPY(tumor_->velocity_->x_, -1.0, displacement_old_->x_, tumor_->displacement_->x_); CHKERRQ(ierr);
      ierr = VecWAXPY(tumor_->velocity_->y_, -1.0, displacement_old_->y_, tumor_->displacement_->y_); CHKERRQ(ierr);
      ierr = VecWAXPY(tumor_->velocity_->z_, -1.0, displacement_old_->z_, tumor_->displacement_->z_); CHKERRQ(ierr);
      ierr = VecScale(tumor_->velocity_->x_, (1.0 / dt)); CHKERRQ(ierr);
      ierr = VecScale(tumor_->velocity_->y_, (1.0 / dt)); CHKERRQ(ierr);
      ierr = VecScale(tumor_->velocity_->z_, (1.0 / dt)); CHKERRQ(ierr);

      // smooth the velocity
      if (flag_smooth_velocity) {
        ierr = spec_ops_->weierstrassSmoother(tumor_->velocity_->x_, tumor_->velocity_->x_, params_, sigma_smooth); CHKERRQ(ierr);
        ierr = spec_ops_->weierstrassSmoother(tumor_->velocity_->y_, tumor_->velocity_->y_, params_, sigma_smooth); CHKERRQ(ierr);
        ierr = spec_ops_->weierstrassSmoother(tumor_->velocity_->z_, tumor_->velocity_->z_, params_, sigma_smooth); CHKERRQ(ierr);
      }

      ierr = VecNorm(tumor_->velocity_->x_, NORM_2, &vel_x_norm); CHKERRQ(ierr);
      ierr = VecNorm(tumor_->velocity_->y_, NORM_2, &vel_y_norm); CHKERRQ(ierr);
      ierr = VecNorm(tumor_->velocity_->z_, NORM_2, &vel_z_norm); CHKERRQ(ierr);
      s << "Norm of velocity (x,y,z) = (" << vel_x_norm << ", " << vel_y_norm << ", " << vel_z_norm << ")";
      ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
      s.str("");
      s.clear();

      // copy displacement to old vector
      ierr = displacement_old_->copy(tumor_->displacement_);
    }
  }

  if (params_->tu_->verbosity_ >= 3) {
    s << " Accumulated KSP itr for state eqn = " << diff_ksp_itr_state_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

#ifdef CUDA
  cudaPrintDeviceMemory();
#endif

  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}
