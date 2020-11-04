#include "PdeOperators.h"

/* #### --------------------------------------------------------------------------- #### */
/* #### ========                 PDE Ops. default RD Model                 ======== #### */
/* #### --------------------------------------------------------------------------- #### */
PetscErrorCode PdeOperatorsRD::reset(std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  resizeTimeHistory(params);
  params_ = params;
  if (tumor != nullptr) tumor_ = tumor;

  PetscFunctionReturn(ierr);
}

PdeOperatorsRD::PdeOperatorsRD(std::shared_ptr<Tumor> tumor, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : PdeOperators(tumor, params, spec_ops) {
  PetscErrorCode ierr = 0;
  ScalarType dt = params_->tu_->dt_;
  int nt = params_->tu_->nt_;

  if (!params->tu_->time_history_off_ && params_->tu_->model_ < 4) {
    c_.resize(nt + 1);  // Time history of tumor
    p_.resize(nt + 1);  // Time history of adjoints
    if (params->tu_->adjoint_store_) {
      // store half-time history to avoid unecessary diffusion solves
      c_half_.resize(nt);
    }
    ierr = VecCreate(PETSC_COMM_WORLD, &c_[0]);
    ierr = VecSetSizes(c_[0], params->grid_->nl_, params->grid_->ng_);
    ierr = setupVec(c_[0]);
    ierr = VecCreate(PETSC_COMM_WORLD, &p_[0]);
    ierr = VecSetSizes(p_[0], params->grid_->nl_, params->grid_->ng_);
    ierr = setupVec(p_[0]);

    for (int i = 1; i < nt + 1; i++) {
      ierr = VecDuplicate(c_[0], &c_[i]);
      ierr = VecDuplicate(p_[0], &p_[i]);
    }
    for (int i = 0; i < nt + 1; i++) {
      ierr = VecSet(c_[i], 0);
      ierr = VecSet(p_[i], 0);
    }
    if (params->tu_->adjoint_store_) {
      for (int i = 0; i < nt; i++) {
        ierr = VecDuplicate(c_[0], &c_half_[i]);
        ierr = VecSet(c_half_[i], 0.);
      }
    }
  }
}

PetscErrorCode PdeOperatorsRD::resizeTimeHistory(std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  if(params->tu_->time_history_off_) {ierr = tuMSGwarn(" Cannot resize time history: switched off."); CHKERRQ(ierr); PetscFunctionReturn(0);}

  ScalarType dt = params_->tu_->dt_;
  int nt = params->tu_->nt_;

  nt_ = nt;

  for (int i = 0; i < c_.size(); i++) {
    ierr = VecDestroy(&c_[i]);
    ierr = VecDestroy(&p_[i]);
    if (c_half_.size() > 0 && i != c_.size() - 1) ierr = VecDestroy(&c_half_[i]);
  }

  c_.resize(nt + 1);                               // Time history of tumor
  p_.resize(nt + 1);                               // Time history of adjoints
  if (params_->tu_->adjoint_store_) c_half_.resize(nt);  // Time history of half-time concs

  ierr = VecCreate(PETSC_COMM_WORLD, &c_[0]);
  ierr = VecSetSizes(c_[0], params->grid_->nl_, params->grid_->ng_);
  ierr = setupVec(c_[0]);
  ierr = VecCreate(PETSC_COMM_WORLD, &p_[0]);
  ierr = VecSetSizes(p_[0], params->grid_->nl_, params->grid_->ng_);
  ierr = setupVec(p_[0]);

  for (int i = 1; i < nt + 1; i++) {
    ierr = VecDuplicate(c_[0], &c_[i]);
    ierr = VecDuplicate(p_[0], &p_[i]);
  }

  for (int i = 0; i < nt + 1; i++) {
    ierr = VecSet(c_[i], 0);
    ierr = VecSet(p_[i], 0);
  }

  if (params_->tu_->adjoint_store_) {
    for (int i = 0; i < nt; i++) {
      ierr = VecDuplicate(c_[0], &c_half_[i]);
      ierr = VecSet(c_half_[i], 0.);
    }
  }

  PetscFunctionReturn(ierr);
}


PetscErrorCode PdeOperatorsRD::reaction(int linearized, int iter) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-reaction");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType *c_t_ptr, *rho_ptr;
  ScalarType *c_ptr;
  ScalarType factor, alph;
  ScalarType dt = params_->tu_->dt_;

  ierr = vecGetArray(tumor_->c_t_, &c_t_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);
  if (linearized != 0) {
    ierr = vecGetArray(c_[iter], &c_ptr); CHKERRQ(ierr);
  }

#ifdef CUDA
  logisticReactionCuda(c_t_ptr, rho_ptr, c_ptr, dt, params_->grid_->nl_, linearized);
#else
  if (linearized == 0) {
    for (int i = 0; i < params_->grid_->nl_; i++) {
      factor = std::exp(rho_ptr[i] * dt);
      alph = c_t_ptr[i] / (1.0 - c_t_ptr[i]);
      if (std::isinf(alph))
        c_t_ptr[i] = 1.0;
      else
        c_t_ptr[i] = alph * factor / (alph * factor + 1.0);
    }
  } else {
    for (int i = 0; i < params_->grid_->nl_; i++) {
      factor = std::exp(rho_ptr[i] * dt);
      alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
      c_t_ptr[i] = c_t_ptr[i] * factor / (alph * alph);
    }
  }
#endif
  ierr = vecRestoreArray(tumor_->c_t_, &c_t_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);
  if (linearized != 0) {
    ierr = vecRestoreArray(c_[iter], &c_ptr); CHKERRQ(ierr);
  }

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsRD::solveIncremental(Vec c_tilde, std::vector<Vec> c_history, ScalarType dt, int iter, int mode) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-incr-fwd-secdiff-solve");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  Vec temp = tumor_->work_[11];
  // c_tilde = c_tilde + dt / 2 * (Dc^i+1 + Dc^i)
  if (mode == 1) {
    // first split
    // temp is c(i) + c(i+1)
    ierr = VecWAXPY(temp, 1., c_history[iter], c_history[iter + 1]); CHKERRQ(ierr);
    // temp is 0.5 * (c(i) + c(i+1))
    ierr = VecScale(temp, 0.5); CHKERRQ(ierr);
    // temp is 0.5 * c(i+1) + 1.5 * c(i)
    ierr = VecAXPY(temp, 1.0, c_history[iter]); CHKERRQ(ierr);
    // apply D with secondary coefficients
    ierr = tumor_->k_->applyDWithSecondaryCoeffs(temp, temp); CHKERRQ(ierr);
    // update c_tilde
    ierr = VecAXPY(c_tilde, dt / 2, temp); CHKERRQ(ierr);
  } else {
    // second split
    // temp is c(i) + c(i+1)
    ierr = VecWAXPY(temp, 1., c_history[iter], c_history[iter + 1]); CHKERRQ(ierr);
    // temp is 0.5 * (c(i) + c(i+1))
    ierr = VecScale(temp, 0.5); CHKERRQ(ierr);
    // temp is 0.5 * c(i) + 1.5 * c(i+1)
    ierr = VecAXPY(temp, 1.0, c_history[iter + 1]); CHKERRQ(ierr);
    // apply D with secondary coefficients
    ierr = tumor_->k_->applyDWithSecondaryCoeffs(temp, temp); CHKERRQ(ierr);
    // update c_tilde
    ierr = VecAXPY(c_tilde, dt / 2, temp); CHKERRQ(ierr);
  }

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsRD::solveState(int linearized) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-solve-state");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType dt = params_->tu_->dt_;
  int nt = params_->tu_->nt_;

  params_->tu_->statistics_.nb_state_solves++;

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  ierr = VecCopy(tumor_->c_0_, tumor_->c_t_); CHKERRQ(ierr);
  if (linearized == 0 && !params_->tu_->time_history_off_) {
    ierr = VecCopy(tumor_->c_t_, c_[0]); CHKERRQ(ierr);
  }

  diff_ksp_itr_state_ = 0;

  /* linearized = 0 -- state equation
     linearized = 1 -- linearized state equation
     linearized = 2 -- linearized state equation with diffusivity inversion
                       for hessian application
  */

  for (int i = 0; i < nt; i++) {
    if (linearized == 2) {
      // eliminating incremental forward for Hpk k_tilde calculation during hessian apply
      // since i+0.5 does not exist, we average i and i+1 to approximate this psuedo time
      ierr = solveIncremental(tumor_->c_t_, c_, dt / 2, i, 1);
    }

    if (params_->tu_->order_ == 2) {
      diff_solver_->solve(tumor_->c_t_, dt / 2.0);
      diff_ksp_itr_state_ += diff_solver_->ksp_itr_;
      if (linearized == 0 && params_->tu_->adjoint_store_ && !params_->tu_->time_history_off_) {
        ierr = VecCopy(tumor_->c_t_, c_half_[i]); CHKERRQ(ierr);
      }
      ierr = reaction(linearized, i);
      diff_solver_->solve(tumor_->c_t_, dt / 2.0);
      diff_ksp_itr_state_ += diff_solver_->ksp_itr_;

      // diff inv for incr fwd
      if (linearized == 2) {
        // eliminating incremental forward for Hpk k_tilde calculation during hessian apply
        // since i+0.5 does not exist, we average i and i+1 to approximate this psuedo time
        ierr = solveIncremental(tumor_->c_t_, c_, dt / 2, i, 2);
      }
    } else {
      diff_solver_->solve(tumor_->c_t_, dt);
      diff_ksp_itr_state_ += diff_solver_->ksp_itr_;
      if (linearized == 0 && params_->tu_->adjoint_store_ && !params_->tu_->time_history_off_) {
        ierr = VecCopy(tumor_->c_t_, c_half_[i]); CHKERRQ(ierr);
      }
      ierr = reaction(linearized, i);
    }

    // Copy current conc to use for the adjoint equation
    if (linearized == 0 && !params_->tu_->time_history_off_) {
      ierr = VecCopy(tumor_->c_t_, c_[i + 1]); CHKERRQ(ierr);
    }
  }

  std::stringstream s;
  if (params_->tu_->verbosity_ >= 3) {
    s << " Accumulated KSP itr for state eqn = " << diff_ksp_itr_state_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsRD::reactionAdjoint(int linearized, int iter) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-reaction-adj");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType *p_0_ptr, *rho_ptr;
  ScalarType *c_ptr;
  ScalarType factor, alph;
  ScalarType dt = params_->tu_->dt_;

  Vec temp = tumor_->work_[11];
  // reaction adjoint needs c_ at half time step.
  ierr = VecCopy(c_[iter], temp); CHKERRQ(ierr);
  if (params_->tu_->adjoint_store_) {
    // half time-step is already stored
    ierr = VecCopy(c_half_[iter], temp); CHKERRQ(ierr);
  } else {
    if (params_->tu_->order_ == 2) {
      diff_solver_->solve(temp, dt / 2.0);
      diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
    } else {
      diff_solver_->solve(temp, dt);
      diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
    }
  }

  ierr = vecGetArray(tumor_->p_0_, &p_0_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);
  ierr = vecGetArray(temp, &c_ptr); CHKERRQ(ierr);

#ifdef CUDA
  logisticReactionCuda(p_0_ptr, rho_ptr, c_ptr, dt, params_->grid_->nl_, linearized);
#else
  for (int i = 0; i < params_->grid_->nl_; i++) {
    factor = std::exp(rho_ptr[i] * dt);
    alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
    p_0_ptr[i] = p_0_ptr[i] * factor / (alph * alph);
  }
#endif

  ierr = vecRestoreArray(tumor_->p_0_, &p_0_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(tumor_->rho_->rho_vec_, &rho_ptr); CHKERRQ(ierr);
  ierr = vecRestoreArray(temp, &c_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsRD::solveAdjoint(int linearized) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-solve-adj");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType dt = params_->tu_->dt_;
  int nt = params_->tu_->nt_;
  params_->tu_->statistics_.nb_adjoint_solves++;

  ierr = VecCopy(tumor_->p_t_, tumor_->p_0_); CHKERRQ(ierr);
  if (linearized == 1) {
    ierr = VecCopy(tumor_->p_0_, p_[nt]); CHKERRQ(ierr);
  }
  diff_ksp_itr_adj_ = 0;
  for (int i = 0; i < nt; i++) {
    if (params_->tu_->order_ == 2) {
      diff_solver_->solve(tumor_->p_0_, dt / 2.0);
      diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
      ierr = reactionAdjoint(linearized, nt - i - 1);
      diff_solver_->solve(tumor_->p_0_, dt / 2.0);
      diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
    } else {
      diff_solver_->solve(tumor_->p_0_, dt);
      diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
      ierr = reactionAdjoint(linearized, nt - i - 1);
    }
    // Copy current adjoint time point to use in additional term for moving-atlas formulation
    // if (linearized == 1) {
    ierr = VecCopy(tumor_->p_0_, p_[nt - i - 1]); CHKERRQ(ierr);
    // }
  }

  std::stringstream s;
  if (params_->tu_->verbosity_ >= 3) {
    s << " Accumulated KSP itr for adjoint eqn = " << diff_ksp_itr_adj_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode PdeOperatorsRD::computeTumorContributionRegistration(Vec q1, Vec q2, Vec q3, Vec q4) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-compute-q");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ScalarType integration_weight = params_->tu_->dt_;
  ScalarType *c_ptr, *p_ptr, *r_ptr;

  // clear
  if (q1 != nullptr) {
    ierr = VecSet(q1, 0.0); CHKERRQ(ierr);
  }
  if (q2 != nullptr) {
    ierr = VecSet(q2, 0.0); CHKERRQ(ierr);
  }
  if (q3 != nullptr) {
    ierr = VecSet(q3, 0.0); CHKERRQ(ierr);
  }
  if (q4 != nullptr) {
    ierr = VecSet(q4, 0.0); CHKERRQ(ierr);
  }
  // compute numerical time integration using trapezoidal rule
  for (int i = 0; i < this->nt_ + 1; i++) {
    // integration weight for chain trapezoidal rule
    if (i == 0 || i == this->nt_) integration_weight *= 0.5;

    // compute x = k_bar * (grad c)^T grad \alpha, where k_bar = dK / dm
    ierr = tumor_->k_->compute_dKdm_gradc_gradp((q1 != nullptr) ? tumor_->work_[8] : nullptr, (q2 != nullptr) ? tumor_->work_[9] : nullptr, (q3 != nullptr) ? tumor_->work_[10] : nullptr,
                                                (q4 != nullptr) ? tumor_->work_[11] : nullptr, c_[i], p_[i], params_->grid_->plan_); CHKERRQ(ierr);

    // compute y = c(1-c) * \alpha
    ierr = VecGetArray(c_[i], &c_ptr); CHKERRQ(ierr);
    ierr = VecGetArray(p_[i], &p_ptr); CHKERRQ(ierr);
    ierr = VecGetArray(tumor_->work_[0], &r_ptr); CHKERRQ(ierr);
    for (int j = 0; j < params_->grid_->nl_; j++) {
      r_ptr[j] = c_ptr[j] * (1 - c_ptr[j]) * p_ptr[j];
    }
    ierr = VecRestoreArray(c_[i], &c_ptr); CHKERRQ(ierr);
    ierr = VecRestoreArray(p_[i], &p_ptr); CHKERRQ(ierr);
    ierr = VecRestoreArray(tumor_->work_[0], &r_ptr); CHKERRQ(ierr);
    // compute rho_bar * c(1-c) * \alpha, where rho_bar = dR / dm
    // this function adds to q1, q2, q3, q4 via AXPY, has to be called after the diff coeff function
    ierr = tumor_->rho_->applydRdm((q1 != nullptr) ? tumor_->work_[8] : nullptr, (q2 != nullptr) ? tumor_->work_[9] : nullptr, (q3 != nullptr) ? tumor_->work_[10] : nullptr,
                                   (q4 != nullptr) ? tumor_->work_[11] : nullptr, tumor_->work_[0]); CHKERRQ(ierr);

    // numerical time integration using trapezoidal rule
    if (q1 != nullptr) {
      ierr = VecAXPY(q1, integration_weight, tumor_->work_[8]); CHKERRQ(ierr);
    }
    if (q2 != nullptr) {
      ierr = VecAXPY(q2, integration_weight, tumor_->work_[9]); CHKERRQ(ierr);
    }
    if (q3 != nullptr) {
      ierr = VecAXPY(q3, integration_weight, tumor_->work_[10]); CHKERRQ(ierr);
    }
    if (q4 != nullptr) {
      ierr = VecAXPY(q4, integration_weight, tumor_->work_[11]); CHKERRQ(ierr);
    }
    // use weight 1 for inner points
    if (i == 0) integration_weight *= 0.5;
  }

  // compute norm of q, additional information, not needed
  std::stringstream s;
  ScalarType norm_q = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;
  if (q1 != nullptr) {
    ierr = VecNorm(q1, NORM_2, &tmp1);
    norm_q += tmp1; CHKERRQ(ierr);
  }
  if (q2 != nullptr) {
    ierr = VecNorm(q2, NORM_2, &tmp2);
    norm_q += tmp2; CHKERRQ(ierr);
  }
  if (q3 != nullptr) {
    ierr = VecNorm(q3, NORM_2, &tmp3);
    norm_q += tmp3; CHKERRQ(ierr);
  }
  if (q4 != nullptr) {
    ierr = VecNorm(q4, NORM_2, &tmp4);
    norm_q += tmp4; CHKERRQ(ierr);
  }
  s << " ||q||_2 = l2q_1 + l2q_2 + l2q_3 + l2q_4 = " << norm_q << " = " << tmp1 << " + " << tmp2 << " + " << tmp3 << " + " << tmp4;
  ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
  s.str("");
  s.clear();

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PdeOperatorsRD::~PdeOperatorsRD() {
  PetscErrorCode ierr = 0;
  if (!params_->tu_->time_history_off_) {
    // use c_.size() not nt
    for (int i = 0; i < c_.size(); i++) {
      ierr = VecDestroy(&c_[i]);
      ierr = VecDestroy(&p_[i]);
      if (c_half_.size() > 0 && i != nt_) ierr = VecDestroy(&c_half_[i]);
    }
  }
}



/* #### ----------------------------------------------------------------------------------- #### */
/* #### ========    PDE Operators for Reaction/DIffusion w/ explicit. Advection    ======== #### */
/* #### ----------------------------------------------------------------------------------- #### */

PdeOperatorsRDAdv::PdeOperatorsRDAdv(std::shared_ptr<Tumor> tumor, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops)
: PdeOperatorsRD(tumor, params, spec_ops) {
  PetscErrorCode ierr = 0;
  ScalarType dt = params_->tu_->dt_;
  int nt = params_->tu_->nt_;

  // first order splitting
  params->tu_->order_ = 1;
  // initialize advection solver
  if(params->tu_->adv_velocity_set_) {
    adv_solver_ = std::make_shared<SemiLagrangianSolver> (params, tumor, spec_ops);
  }
}

PetscErrorCode PdeOperatorsRDAdv::solveState(int linearized) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-solve-state");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType dt = params_->tu_->dt_;
  int nt = params_->tu_->nt_;

  params_->tu_->statistics_.nb_state_solves++;

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  ierr = VecCopy(tumor_->c_0_, tumor_->c_t_); CHKERRQ(ierr);
  if (linearized == 0 && !params_->tu_->time_history_off_) {
    ierr = VecCopy(tumor_->c_t_, c_[0]); CHKERRQ(ierr);
  }

  diff_ksp_itr_state_ = 0;

  // reset mat prob to atlas
  ierr = tumor_->mat_prop_->resetValues(); CHKERRQ(ierr);

  /* linearized = 0 -- state equation
     linearized = 1 -- linearized state equation
     linearized = 2 -- linearized state equation with diffusivity inversion
                       for hessian application
  */

  for (int i = 0; i < nt; i++) {

    ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->kf_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
    ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
    //tumor_->phi_->setValues (tumor_->mat_prop_);  // update the phi values, i.e., update the filter
    diff_solver_->precFactor();   // need to update prefactors for diffusion KSP preconditioner, as k changed


    if (linearized == 2) {
      // eliminating incremental forward for Hpk k_tilde calculation during hessian apply
      // since i+0.5 does not exist, we average i and i+1 to approximate this psuedo time
      ierr = solveIncremental(tumor_->c_t_, c_, dt / 2, i, 1);
    }

    // advection of healthy tissue
    if (params_->tu_->adv_velocity_set_) {
        adv_solver_->advection_mode_ = 1;  //  mass conservation
        // adv_solver_->advection_mode_ = 2;    // pure advection
        ierr = adv_solver_->solve(tumor_->mat_prop_->gm_, tumor_->velocity_, dt); CHKERRQ(ierr);
        ierr = adv_solver_->solve(tumor_->mat_prop_->wm_, tumor_->velocity_, dt); CHKERRQ(ierr);
        ierr = adv_solver_->solve(tumor_->mat_prop_->vt_, tumor_->velocity_, dt); CHKERRQ(ierr);
        ierr = adv_solver_->solve(tumor_->mat_prop_->kfxx_, tumor_->velocity_, dt); CHKERRQ(ierr);
        ierr = adv_solver_->solve(tumor_->mat_prop_->kfxy_, tumor_->velocity_, dt); CHKERRQ(ierr);
        ierr = adv_solver_->solve(tumor_->mat_prop_->kfxz_, tumor_->velocity_, dt); CHKERRQ(ierr);
        ierr = adv_solver_->solve(tumor_->mat_prop_->kfyy_, tumor_->velocity_, dt); CHKERRQ(ierr);
        ierr = adv_solver_->solve(tumor_->mat_prop_->kfyz_, tumor_->velocity_, dt); CHKERRQ(ierr);
        ierr = adv_solver_->solve(tumor_->mat_prop_->kfzz_, tumor_->velocity_, dt); CHKERRQ(ierr);
    
        if(tumor_->mat_prop_->csf_ != nullptr) {
          ierr = adv_solver_->solve(tumor_->mat_prop_->csf_, tumor_->velocity_, dt); CHKERRQ(ierr);
        }
        ierr = adv_solver_->solve(tumor_->c_t_, tumor_->velocity_, dt); CHKERRQ(ierr);
    }

    if (params_->tu_->order_ == 2) {
      diff_solver_->solve(tumor_->c_t_, dt / 2.0);
      diff_ksp_itr_state_ += diff_solver_->ksp_itr_;
      if (linearized == 0 && params_->tu_->adjoint_store_ && !params_->tu_->time_history_off_) {
        ierr = VecCopy(tumor_->c_t_, c_half_[i]); CHKERRQ(ierr);
      }
      ierr = reaction(linearized, i);
      diff_solver_->solve(tumor_->c_t_, dt / 2.0);
      diff_ksp_itr_state_ += diff_solver_->ksp_itr_;

      // diff inv for incr fwd
      if (linearized == 2) {
        // eliminating incremental forward for Hpk k_tilde calculation during hessian apply
        // since i+0.5 does not exist, we average i and i+1 to approximate this psuedo time
        ierr = solveIncremental(tumor_->c_t_, c_, dt / 2, i, 2);
      }
    } else {
      diff_solver_->solve(tumor_->c_t_, dt);
      diff_ksp_itr_state_ += diff_solver_->ksp_itr_;
      if (linearized == 0 && params_->tu_->adjoint_store_ && !params_->tu_->time_history_off_) {
        ierr = VecCopy(tumor_->c_t_, c_half_[i]); CHKERRQ(ierr);
      }
      ierr = reaction(linearized, i);
    }

    // Copy current conc to use for the adjoint equation
    if (linearized == 0 && !params_->tu_->time_history_off_) {
      ierr = VecCopy(tumor_->c_t_, c_[i + 1]); CHKERRQ(ierr);
    }
  }

  std::stringstream s;
  if (params_->tu_->verbosity_ >= 3) {
    s << " Accumulated KSP itr for state eqn = " << diff_ksp_itr_state_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);
    s.str("");
    s.clear();
  }

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}


PetscErrorCode PdeOperatorsRDAdv::preAdvection (Vec &wm, Vec &gm, Vec &csf, Vec &mri, ScalarType adv_time) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  ScalarType dt = params_->tu_->dt_;
  int nt = adv_time/dt;
  std::stringstream ss;
  ss << " Advecting using nt="<<nt<<" time steps"; tuMSGstd(ss.str());

  for (int i = 0; i < nt; i++) {
    // advection of healthy tissue
    if (params_->tu_->adv_velocity_set_) {
      //adv_solver_->advection_mode_ = 1;  //  mass conservation
      adv_solver_->advection_mode_ = 2;  // pure advection
      ierr = adv_solver_->solve (gm, tumor_->velocity_, dt); CHKERRQ(ierr);
      ierr = adv_solver_->solve (wm, tumor_->velocity_, dt); CHKERRQ(ierr);
      ierr = adv_solver_->solve (csf, tumor_->velocity_, dt); CHKERRQ(ierr);
      if(mri != nullptr) {
        ierr = adv_solver_->solve (mri, tumor_->velocity_, dt); CHKERRQ(ierr);
      }
      ierr = tumor_->mat_prop_->clipHealthyTissues(); CHKERRQ(ierr);
    }
  }
  // ierr = dataOut (wm, params_, "wm_atlas_adv.nc"); CHKERRQ(ierr);
  // ierr = dataOut (gm, params_, "gm_atlas_adv.nc"); CHKERRQ(ierr);
  // ierr = dataOut (csf, params_, "csf_atlas_adv.nc"); CHKERRQ(ierr);
  // if(mri != nullptr) {
      // ierr = dataOut (mri, params_, "mri_atlas_adv.nc"); CHKERRQ(ierr);
  // }
  PetscFunctionReturn (ierr);
}
