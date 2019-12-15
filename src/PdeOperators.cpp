#include "PdeOperators.h"

/* #### ------------------------------------------------------------------- #### */
/* #### ======== RESET (CHANGE SIZE OF WORK VECTORS, TIME HISTORY) ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode PdeOperatorsRD::reset (std::shared_ptr <NMisc> n_misc, std::shared_ptr<Tumor> tumor) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    resizeTimeHistory(n_misc);
    n_misc_ = n_misc;
    if (tumor != nullptr) tumor_ = tumor;

    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsMassEffect::reset (std::shared_ptr <NMisc> n_misc, std::shared_ptr<Tumor> tumor) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    // no-op so far

    n_misc_ = n_misc;
    if (tumor != nullptr) tumor_ = tumor;

    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::reset (std::shared_ptr <NMisc> n_misc, std::shared_ptr<Tumor> tumor) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    // no-op so far

    n_misc_ = n_misc;
    if (tumor != nullptr) tumor_ = tumor;

    PetscFunctionReturn (ierr);
}


PdeOperatorsRD::PdeOperatorsRD (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops)
        : PdeOperators (tumor, n_misc, spec_ops) {

    PetscErrorCode ierr = 0;
    ScalarType dt = n_misc_->dt_;
    int nt = n_misc->nt_;

    if (!n_misc->forward_flag_) {
        c_.resize (nt + 1);                         //Time history of tumor
        p_.resize (nt + 1);                         //Time history of adjoints
        if (n_misc->adjoint_store_) {
            // store half-time history to avoid unecessary diffusion solves
            c_half_.resize (nt);
        }
        ierr = VecCreate (PETSC_COMM_WORLD, &c_[0]);
        ierr = VecSetSizes (c_[0], n_misc->n_local_, n_misc->n_global_);
        ierr = setupVec (c_[0]);
        ierr = VecCreate (PETSC_COMM_WORLD, &p_[0]);
        ierr = VecSetSizes (p_[0], n_misc->n_local_, n_misc->n_global_);
        ierr = setupVec (p_[0]);

        for (int i = 1; i < nt + 1; i++) {
            ierr = VecDuplicate (c_[0], &c_[i]);
            ierr = VecDuplicate (p_[0], &p_[i]);
        }
        for (int i = 0; i < nt + 1; i++) {
            ierr = VecSet (c_[i], 0);
            ierr = VecSet (p_[i], 0);
        }
        if (n_misc->adjoint_store_) {
            for (int i = 0; i < nt; i++) {
                ierr = VecDuplicate (c_[0], &c_half_[i]);
                ierr = VecSet (c_half_[i], 0.);
            }
        }
    }
}


PetscErrorCode PdeOperatorsRD::resizeTimeHistory (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ScalarType dt = n_misc_->dt_;
    int nt = n_misc->nt_;

    nt_ = nt;

    for (int i = 0; i < c_.size(); i++) {
        ierr = VecDestroy (&c_[i]);
        ierr = VecDestroy (&p_[i]);
        if (c_half_.size() > 0 && i != c_.size() - 1) ierr = VecDestroy (&c_half_[i]);
    }

    c_.resize (nt + 1);                         //Time history of tumor
    p_.resize (nt + 1);                         //Time history of adjoints
    if (n_misc->adjoint_store_) c_half_.resize (nt);                        //Time history of half-time concs

    ierr = VecCreate (PETSC_COMM_WORLD, &c_[0]);
    ierr = VecSetSizes (c_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = setupVec (c_[0]);
    ierr = VecCreate (PETSC_COMM_WORLD, &p_[0]);
    ierr = VecSetSizes (p_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = setupVec (p_[0]);

    for (int i = 1; i < nt + 1; i++) {
        ierr = VecDuplicate (c_[0], &c_[i]);
        ierr = VecDuplicate (p_[0], &p_[i]);
    }

    for (int i = 0; i < nt + 1; i++) {
        ierr = VecSet (c_[i], 0);
        ierr = VecSet (p_[i], 0);
    }

    if (n_misc->adjoint_store_) {
        for (int i = 0; i < nt; i++) {
            ierr = VecDuplicate (c_[0], &c_half_[i]);
            ierr = VecSet (c_half_[i], 0.);
        }
    }

    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsRD::reaction (int linearized, int iter) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-reaction");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    ScalarType *c_t_ptr, *rho_ptr;
    ScalarType *c_ptr;
    ScalarType factor, alph;
    ScalarType dt = n_misc_->dt_;

    ierr = vecGetArray (tumor_->c_t_, &c_t_ptr);                 CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);
    if (linearized != 0) {
        ierr = vecGetArray (c_[iter], &c_ptr);                       CHKERRQ (ierr);
    }

    #ifdef CUDA
        logisticReactionCuda (c_t_ptr, rho_ptr, c_ptr, dt, n_misc_->n_local_, linearized);
    #else
        if (linearized == 0) {
            for (int i = 0; i < n_misc_->n_local_; i++) {
                factor = std::exp (rho_ptr[i] * dt);
                alph = c_t_ptr[i] / (1.0 - c_t_ptr[i]);
                if (std::isinf(alph)) c_t_ptr[i] = 1.0;
                else c_t_ptr[i] = alph * factor / (alph * factor + 1.0);
            }
        } else {
            for (int i = 0; i < n_misc_->n_local_; i++) {
                factor = std::exp (rho_ptr[i] * dt);
                alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
                c_t_ptr[i] = c_t_ptr[i] * factor / (alph * alph);
            }
        }
    #endif
    ierr = vecRestoreArray (tumor_->c_t_, &c_t_ptr);                 CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);
    if (linearized != 0) {
        ierr = vecRestoreArray (c_[iter], &c_ptr);                       CHKERRQ (ierr);
    }

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsRD::solveIncremental (Vec c_tilde, std::vector<Vec> c_history, ScalarType dt, int iter, int mode) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-incr-fwd-secdiff-solve");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    Vec temp = tumor_->work_[11];
    // c_tilde = c_tilde + dt / 2 * (Dc^i+1 + Dc^i)
    if (mode == 1) {
        // first split
        // temp is c(i) + c(i+1)
        ierr = VecWAXPY (temp, 1., c_history[iter], c_history[iter + 1]);           CHKERRQ (ierr);
        // temp is 0.5 * (c(i) + c(i+1))
        ierr = VecScale (temp, 0.5);                                                CHKERRQ (ierr);
        // temp is 0.5 * c(i+1) + 1.5 * c(i)
        ierr = VecAXPY (temp, 1.0, c_history[iter]);                                CHKERRQ (ierr);
        // apply D with secondary coefficients
        ierr = tumor_->k_->applyDWithSecondaryCoeffs (temp, temp);                  CHKERRQ (ierr);
        // update c_tilde
        ierr = VecAXPY (c_tilde, dt / 2, temp);                                     CHKERRQ (ierr);
    } else {
        // second split
        // temp is c(i) + c(i+1)
        ierr = VecWAXPY (temp, 1., c_history[iter], c_history[iter + 1]);           CHKERRQ (ierr);
        // temp is 0.5 * (c(i) + c(i+1))
        ierr = VecScale (temp, 0.5);                                                CHKERRQ (ierr);
        // temp is 0.5 * c(i) + 1.5 * c(i+1)
        ierr = VecAXPY (temp, 1.0, c_history[iter + 1]);                            CHKERRQ (ierr);
        // apply D with secondary coefficients
        ierr = tumor_->k_->applyDWithSecondaryCoeffs (temp, temp);                  CHKERRQ (ierr);
        // update c_tilde
        ierr = VecAXPY (c_tilde, dt / 2, temp);                                     CHKERRQ (ierr);
    }

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsRD::solveState (int linearized) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-solve-state");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    ScalarType dt = n_misc_->dt_;
    int nt = n_misc_->nt_;

    n_misc_->statistics_.nb_state_solves++;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    //enforce positivity : hack
    if (!linearized) {
        #ifdef POSITIVITY
            ierr = enforcePositivity (tumor_->c_0_, n_misc_);
        #endif
    }

    ierr = VecCopy (tumor_->c_0_, tumor_->c_t_);                 CHKERRQ (ierr);
    if (linearized == 0 && !n_misc_->forward_flag_) {
        ierr = VecCopy (tumor_->c_t_, c_[0]);                    CHKERRQ (ierr);
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
            ierr = solveIncremental (tumor_->c_t_, c_, dt / 2, i, 1);
        }

        if (n_misc_->order_ == 2) {
            diff_solver_->solve (tumor_->c_t_, dt / 2.0);   diff_ksp_itr_state_ += diff_solver_->ksp_itr_;
            if (linearized == 0 && n_misc_->adjoint_store_ && !n_misc_->forward_flag_) {
                ierr = VecCopy (tumor_->c_t_, c_half_[i]);                    CHKERRQ (ierr);
            }
            ierr = reaction (linearized, i);
            diff_solver_->solve (tumor_->c_t_, dt / 2.0);   diff_ksp_itr_state_ += diff_solver_->ksp_itr_;

            // diff inv for incr fwd
            if (linearized == 2) {
                // eliminating incremental forward for Hpk k_tilde calculation during hessian apply
                // since i+0.5 does not exist, we average i and i+1 to approximate this psuedo time
                ierr = solveIncremental (tumor_->c_t_, c_, dt / 2, i, 2);
            }
        } else {
            diff_solver_->solve (tumor_->c_t_, dt);         diff_ksp_itr_state_ += diff_solver_->ksp_itr_;
            if (linearized == 0 && n_misc_->adjoint_store_ && !n_misc_->forward_flag_) {
                ierr = VecCopy (tumor_->c_t_, c_half_[i]);                    CHKERRQ (ierr);
            }
            ierr = reaction (linearized, i);
        }
        //enforce positivity : hack
        if (!linearized) {
            #ifdef POSITIVITY
                ierr = enforcePositivity (tumor_->c_t_, n_misc_);
            #endif
        }
        //Copy current conc to use for the adjoint equation
        if (linearized == 0 && !n_misc_->forward_flag_) {
            ierr = VecCopy (tumor_->c_t_, c_[i + 1]);            CHKERRQ (ierr);
        }
    }

    std::stringstream s;
    if (n_misc_->verbosity_ >= 3) {
        s << " Accumulated KSP itr for state eqn = " << diff_ksp_itr_state_;
        ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsRD::reactionAdjoint (int linearized, int iter) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-reaction-adj");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();


    ScalarType *p_0_ptr, *rho_ptr;
    ScalarType *c_ptr;
    ScalarType factor, alph;
    ScalarType dt = n_misc_->dt_;

    Vec temp = tumor_->work_[11];
    //reaction adjoint needs c_ at half time step.
    ierr = VecCopy (c_[iter], temp);                             CHKERRQ (ierr);
    if (n_misc_->adjoint_store_) {
        // half time-step is already stored
        ierr = VecCopy (c_half_[iter], temp);                    CHKERRQ (ierr);
    } else {
        if (n_misc_->order_ == 2) {
            diff_solver_->solve (temp, dt / 2.0);   diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
        } else {
            diff_solver_->solve (temp, dt);         diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
        }
    }

    ierr = vecGetArray (tumor_->p_0_, &p_0_ptr);                 CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);
    ierr = vecGetArray (temp, &c_ptr);                           CHKERRQ (ierr);

    #ifdef CUDA
        logisticReactionCuda (p_0_ptr, rho_ptr, c_ptr, dt, n_misc_->n_local_, linearized);
    #else
        for (int i = 0; i < n_misc_->n_local_; i++) {
            factor = std::exp (rho_ptr[i] * dt);
            alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
            p_0_ptr[i] = p_0_ptr[i] * factor / (alph * alph);
        }
    #endif

    ierr = vecRestoreArray (tumor_->p_0_, &p_0_ptr);             CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);   CHKERRQ (ierr);
    ierr = vecRestoreArray (temp, &c_ptr);                       CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsRD::solveAdjoint (int linearized) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-solve-adj");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    ScalarType dt = n_misc_->dt_;
    int nt = n_misc_->nt_;
    n_misc_->statistics_.nb_adjoint_solves++;

    ierr = VecCopy (tumor_->p_t_, tumor_->p_0_);                 CHKERRQ (ierr);
    if (linearized == 1) {
        ierr = VecCopy (tumor_->p_0_, p_[nt]);                   CHKERRQ (ierr);
    }
    diff_ksp_itr_adj_ = 0;
    for (int i = 0; i < nt; i++) {
        if (n_misc_->order_ == 2) {
            diff_solver_->solve (tumor_->p_0_, dt / 2.0);       diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
            ierr = reactionAdjoint (linearized, nt - i - 1);
            diff_solver_->solve (tumor_->p_0_, dt / 2.0);       diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
        } else {
            diff_solver_->solve (tumor_->p_0_, dt);             diff_ksp_itr_adj_ += diff_solver_->ksp_itr_;
            ierr = reactionAdjoint (linearized, nt - i - 1);
        }
        //Copy current adjoint time point to use in additional term for moving-atlas formulation
        // if (linearized == 1) {
        ierr = VecCopy (tumor_->p_0_, p_[nt - i - 1]);            CHKERRQ (ierr);
        // }
    }

    std::stringstream s;
    if (n_misc_->verbosity_ >= 3) {
        s << " Accumulated KSP itr for adjoint eqn = " << diff_ksp_itr_adj_;
        ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsRD::computeTumorContributionRegistration(Vec q1, Vec q2, Vec q3, Vec q4) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tumor-compute-q");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  ScalarType integration_weight = n_misc_->dt_;
  ScalarType *c_ptr, *p_ptr, *r_ptr;

  // clear
  if(q1 != nullptr) {ierr = VecSet(q1, 0.0);                     CHKERRQ(ierr);}
  if(q2 != nullptr) {ierr = VecSet(q2, 0.0);                     CHKERRQ(ierr);}
  if(q3 != nullptr) {ierr = VecSet(q3, 0.0);                     CHKERRQ(ierr);}
  if(q4 != nullptr) {ierr = VecSet(q4, 0.0);                     CHKERRQ(ierr);}
  // compute numerical time integration using trapezoidal rule
  for (int i = 0; i < this->nt_ + 1; i++) {
    // integration weight for chain trapezoidal rule
    if (i == 0 || i == this->nt_) integration_weight *= 0.5;

    // compute x = k_bar * (grad c)^T grad \alpha, where k_bar = dK / dm
    ierr = tumor_->k_->compute_dKdm_gradc_gradp(
      (q1 != nullptr) ? tumor_->work_[8]  : nullptr,
      (q2 != nullptr) ? tumor_->work_[9]  : nullptr,
      (q3 != nullptr) ? tumor_->work_[10] : nullptr,
      (q4 != nullptr) ? tumor_->work_[11] : nullptr,
      c_[i], p_[i], n_misc_->plan_);                              CHKERRQ(ierr);

    // compute y = c(1-c) * \alpha
    ierr = VecGetArray (c_[i], &c_ptr);                          CHKERRQ (ierr);
    ierr = VecGetArray (p_[i], &p_ptr);                          CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->work_[0], &r_ptr);               CHKERRQ (ierr);
    for (int j = 0; j < n_misc_->n_local_; j++) {
      r_ptr[j] = c_ptr[j] * (1 - c_ptr[j]) * p_ptr[j];
    }
    ierr = VecRestoreArray (c_[i], &c_ptr);                      CHKERRQ (ierr);
    ierr = VecRestoreArray (p_[i], &p_ptr);                      CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->work_[0], &r_ptr);           CHKERRQ (ierr);
    // compute rho_bar * c(1-c) * \alpha, where rho_bar = dR / dm
    // this function adds to q1, q2, q3, q4 via AXPY, has to be called after the diff coeff function
    ierr = tumor_->rho_->applydRdm(
      (q1 != nullptr) ? tumor_->work_[8]  : nullptr,
      (q2 != nullptr) ? tumor_->work_[9]  : nullptr,
      (q3 != nullptr) ? tumor_->work_[10] : nullptr,
      (q4 != nullptr) ? tumor_->work_[11] : nullptr,
       tumor_->work_[0]);                                        CHKERRQ (ierr);

    // numerical time integration using trapezoidal rule
    if(q1 != nullptr) {ierr = VecAXPY (q1, integration_weight, tumor_->work_[8]);   CHKERRQ (ierr);}
    if(q2 != nullptr) {ierr = VecAXPY (q2, integration_weight, tumor_->work_[9]);   CHKERRQ (ierr);}
    if(q3 != nullptr) {ierr = VecAXPY (q3, integration_weight, tumor_->work_[10]);  CHKERRQ (ierr);}
    if(q4 != nullptr) {ierr = VecAXPY (q4, integration_weight, tumor_->work_[11]);  CHKERRQ (ierr);}
    // use weight 1 for inner points
    if (i == 0) integration_weight *= 0.5;
  }

  // compute norm of q, additional information, not needed
  std::stringstream s; ScalarType norm_q = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;
  if(q1 != nullptr) {ierr = VecNorm (q1, NORM_2, &tmp1); norm_q += tmp1;            CHKERRQ (ierr);}
  if(q2 != nullptr) {ierr = VecNorm (q2, NORM_2, &tmp2); norm_q += tmp2;            CHKERRQ (ierr);}
  if(q3 != nullptr) {ierr = VecNorm (q3, NORM_2, &tmp3); norm_q += tmp3;            CHKERRQ (ierr);}
  if(q4 != nullptr) {ierr = VecNorm (q4, NORM_2, &tmp4); norm_q += tmp4;            CHKERRQ (ierr);}
  s << " ||q||_2 = l2q_1 + l2q_2 + l2q_3 + l2q_4 = " << norm_q << " = " << tmp1 << " + " << tmp2 << " + " << tmp3 << " + " << tmp4;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();

  self_exec_time += MPI_Wtime();
  //accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings (t); e.stop ();
  PetscFunctionReturn (ierr);
}

PdeOperatorsRD::~PdeOperatorsRD () {
    PetscErrorCode ierr = 0;
    if (!n_misc_->forward_flag_) {  
        // use c_.size() not nt      
        for (int i = 0; i < c_.size(); i++) {
            ierr = VecDestroy (&c_[i]);
            ierr = VecDestroy (&p_[i]);
            if (c_half_.size() > 0 && i != nt_) ierr = VecDestroy (&c_half_[i]);
        }
    }
}

PetscErrorCode PdeOperatorsMassEffect::conserveHealthyTissues () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-conserve-healthy");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    // gm, wm is conserved with rhs g/(g + w) * (Dc + Rc) : treated explicity
    ScalarType *c_ptr, *rho_ptr, *gm_ptr, *wm_ptr, *scale_gm_ptr, *scale_wm_ptr, *sum_ptr;
    ScalarType dt = n_misc_->dt_;
    ierr = VecCopy (tumor_->c_t_, temp_[1]);                        CHKERRQ (ierr);
    ierr = tumor_->k_->applyD (temp_[0], temp_[1]);                 CHKERRQ (ierr);     // Dc
    ierr = VecPointwiseMult (temp_[1], tumor_->c_t_, tumor_->c_t_);     CHKERRQ (ierr);
    ierr = VecAYPX (temp_[1], -1.0, tumor_->c_t_);       	    CHKERRQ (ierr);
    ierr = VecPointwiseMult (temp_[1], temp_[1], tumor_->rho_->rho_vec_);   CHKERRQ (ierr); // Rc
    ierr = VecAXPY (temp_[0], 1.0, temp_[1]);                       CHKERRQ (ierr);         // (Rc + Dc) in temp_[0]

    ierr = vecGetArray (tumor_->mat_prop_->gm_, &gm_ptr);           CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->mat_prop_->wm_, &wm_ptr);           CHKERRQ (ierr);
    ierr = vecGetArray (temp_[0], &sum_ptr);                        CHKERRQ (ierr);
    ierr = vecGetArray (temp_[1], &scale_gm_ptr);                   CHKERRQ (ierr);
    ierr = vecGetArray (temp_[2], &scale_wm_ptr);                   CHKERRQ (ierr);

    #ifdef CUDA
        conserveHealthyTissuesCuda (gm_ptr, wm_ptr, sum_ptr, scale_gm_ptr, scale_wm_ptr, dt, n_misc_->n_local_);
    #else
        for (int i = 0; i < n_misc_->n_local_; i++) {
            scale_gm_ptr[i] = 0.0;
            scale_wm_ptr[i] = 0.0;

            if (gm_ptr[i] > 0.01 || wm_ptr[i] > 0.01) {
                scale_gm_ptr[i] = -1.0 * dt * gm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
                scale_wm_ptr[i] = -1.0 * dt * wm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
            }

            scale_gm_ptr[i] = (std::isnan (scale_gm_ptr[i])) ? 0.0 : scale_gm_ptr[i];
            scale_wm_ptr[i] = (std::isnan (scale_wm_ptr[i])) ? 0.0 : scale_wm_ptr[i];

            gm_ptr[i] += scale_gm_ptr[i] * sum_ptr[i];
            wm_ptr[i] += scale_wm_ptr[i] * sum_ptr[i];
        }
    #endif

    ierr = vecRestoreArray (tumor_->mat_prop_->gm_, &gm_ptr);           CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->mat_prop_->wm_, &wm_ptr);           CHKERRQ (ierr);
    ierr = vecRestoreArray (temp_[0], &sum_ptr);                        CHKERRQ (ierr);
    ierr = vecRestoreArray (temp_[1], &scale_gm_ptr);                   CHKERRQ (ierr);
    ierr = vecRestoreArray (temp_[2], &scale_wm_ptr);                   CHKERRQ (ierr);


    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsMassEffect::updateReacAndDiffCoefficients (Vec seg, std::shared_ptr<Tumor> tumor) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ScalarType *seg_ptr, *rho_ptr, *k_ptr;
    ierr = VecSet (tumor_->rho_->rho_vec_, 0.);                 CHKERRQ (ierr);
    ierr = VecSet (tumor_->k_->kxx_, 0.);                       CHKERRQ (ierr);

    ierr = VecGetArray (seg, &seg_ptr);                         CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);      CHKERRQ (ierr);
    // ierr = VecGetArray (tumor_->k_->kxx_, &k_ptr);              CHKERRQ (ierr);

    for (int i = 0; i < n_misc_->n_local_; i++) {
        if (std::abs(seg_ptr[i] - 1) < 1E-3 || std::abs(seg_ptr[i] - 2) < 1E-3) {
            // 1 is tumor, 2 is wm
            rho_ptr[i] = n_misc_->rho_;
            // k_ptr[i] = n_misc_->k_;
        }
    }

    ierr = VecRestoreArray (seg, &seg_ptr);                         CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);      CHKERRQ (ierr);
    // ierr = VecRestoreArray (tumor_->k_->kxx_, &k_ptr);              CHKERRQ (ierr);

    // smooth them now
    ScalarType sigma_smooth = n_misc_->smoothing_factor_ * 2 * M_PI / n_misc_->n_[0];
    ierr = spec_ops_->weierstrassSmoother (tumor_->rho_->rho_vec_, tumor_->rho_->rho_vec_, n_misc_, sigma_smooth);
    // ierr = spec_ops_->weierstrassSmoother (tumor_->k_->kxx_, tumor_->k_->kxx_, n_misc_, sigma_smooth);

    // copy kxx to other directions
    // ierr = VecCopy (tumor_->k_->kxx_, tumor_->k_->kyy_);            CHKERRQ (ierr);
    // ierr = VecCopy (tumor_->k_->kxx_, tumor_->k_->kzz_);            CHKERRQ (ierr);

    // ignore the avg for now since it won't change much and the preconditioner does not have much effect on
    // the diffusion solver

    PetscFunctionReturn (ierr);
}


PetscErrorCode PdeOperatorsMassEffect::solveState (int linearized) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-solve-state");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    ScalarType dt = n_misc_->dt_;
    int nt = n_misc_->nt_;
    n_misc_->statistics_.nb_state_solves++;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    ierr = VecCopy (tumor_->c_0_, tumor_->c_t_);                 CHKERRQ (ierr);

    ScalarType k1, k2, k3, r1, r2, r3;
    k1 = n_misc_->k_;
    k2 = n_misc_->k_gm_wm_ratio_ * n_misc_->k_; k3 = 0;
    r1 = n_misc_->rho_;
    r2 = n_misc_->r_gm_wm_ratio_ * n_misc_->rho_; r3 = 0;

      
    // filter matprop
    ierr = tumor_->mat_prop_->filterTumor (tumor_->c_t_);                                                                           CHKERRQ (ierr);
    // force compute
    ierr = tumor_->computeForce (tumor_->c_t_);
    // displacement compute through elasticity solve
    ierr = elasticity_solver_->solve (displacement_old_, tumor_->force_);

    std::stringstream ss;    
    ScalarType vel_max;
    ScalarType cfl;
    std::stringstream s;
    ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / n_misc_->n_[0];
    bool flag_smooth_velocity = true;
    bool write_output_and_break = false;

    ScalarType max_cfl = 8;

    for (int i = 0; i < nt + 1; i++) {
        if (n_misc_->verbosity_ > 1) {
            s << "Time step = " << i; ierr = tuMSGstd (s.str()); CHKERRQ (ierr); s.str (""); s.clear ();
        }

        // compute CFL
        ierr = tumor_->computeSegmentation ();                                    CHKERRQ (ierr);
        ierr = tumor_->velocity_->computeMagnitude (magnitude_);
        ierr = VecMax (magnitude_, NULL, &vel_max);      CHKERRQ (ierr);
        cfl = dt * vel_max / n_misc_->h_[0];
        if (n_misc_->verbosity_ > 1) {
            s << "CFL = " << cfl;
            ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
            s.str (""); s.clear ();
        }
        // Adaptively time step if CFL is too large
        if (cfl >= max_cfl) {
            // TODO: is adaptive time-stepping necessary? Because if cfl is already this number; then numerical oscillations
            // have already propogated and dt keeps dropping to very small values -- very bad accuracy from semi-Lagrangian.
            // instead suggest to the user to have a smaller forcing factor. Large forcing factor also stretches the tumor
            // concentration too much which calls for higher E_tumor which can be very unrealistic. If mass-effect still does 
            // not match, then rho/kappa might be highly inaccurate.

            // keep re-size of time history commented for now
            // dt *= 0.5;
            // nt = i + 2. * (n_misc_->nt_ - i - 1) + 1;
            // n_misc_->dt_ = dt;
            // n_misc_->nt_ = nt;
            // s << "CFL too large -- Changing dt to " << dt << " and nt to " << nt << "\n";
            // ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
            // s.str (""); s.clear ();
            // if (nt >= 200) {
            //     s << "Number of time-steps too large, consider using smaller forcing factor; exiting solver...";
            //     ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
            //     s.str (""); s.clear ();
            //     break;
            // }
            s << "CFL is too large (>=" << max_cfl << "); consider using smaller forcing factor; exiting solver...";
            ierr = tuMSGwarn (s.str());                                                CHKERRQ(ierr);
            s.str (""); s.clear ();
            write_output_and_break = true;
        }

        if ((n_misc_->writeOutput_ && n_misc_->verbosity_ > 1 && i % 5 == 0) || write_output_and_break) {
            ss << "velocity_t[" << i << "].nc";
            dataOut (magnitude_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ierr = displacement_old_->computeMagnitude(magnitude_);
            ss << "displacement_t[" << i << "].nc";
            dataOut (magnitude_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "seg_t[" << i << "].nc";
            dataOut (tumor_->seg_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "c_t[" << i << "].nc";
            dataOut (tumor_->c_t_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            if (n_misc_->verbosity_ > 2) {
                ss << "rho_t[" << i << "].nc";
                dataOut (tumor_->rho_->rho_vec_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "kxx_t[" << i << "].nc";
                dataOut (tumor_->k_->kxx_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "lam_t[" << i << "].nc";
                dataOut (elasticity_solver_->ctx_->lam_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "mu_t[" << i << "].nc";
                dataOut (elasticity_solver_->ctx_->mu_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "scr_t[" << i << "].nc";
                dataOut (elasticity_solver_->ctx_->screen_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ierr = tumor_->force_->computeMagnitude(magnitude_);
                ss << "force_t[" << i << "].nc";
                dataOut (magnitude_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "vt_t[" << i << "].nc";
                dataOut (tumor_->mat_prop_->csf_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "csf_t[" << i << "].nc";
                dataOut (tumor_->mat_prop_->glm_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "wm_t[" << i << "].nc";
                dataOut (tumor_->mat_prop_->wm_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "gm_t[" << i << "].nc";
                dataOut (tumor_->mat_prop_->gm_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
            }
        }

        if (write_output_and_break) break;

        // Update diffusivity and reaction coefficient
        ierr = updateReacAndDiffCoefficients (tumor_->seg_, tumor_);              CHKERRQ (ierr);
        ierr = tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, tumor_->mat_prop_, n_misc_);    CHKERRQ(ierr);

        // need to update prefactors for diffusion KSP preconditioner, as k changed
        ierr = diff_solver_->precFactor();                                                          CHKERRQ(ierr);

        // Advection of tumor and healthy tissue
        // first compute trajectories for semi-Lagrangian solve as velocity is changing every itr
        adv_solver_->trajectoryIsComputed_ = false;
        ierr = adv_solver_->solve (tumor_->mat_prop_->gm_, tumor_->velocity_, dt);                  CHKERRQ(ierr);
        ierr = adv_solver_->solve (tumor_->mat_prop_->wm_, tumor_->velocity_, dt);                  CHKERRQ(ierr);
        adv_solver_->advection_mode_ = 2;  // pure advection for csf
        ierr = adv_solver_->solve (tumor_->mat_prop_->csf_, tumor_->velocity_, dt);                 CHKERRQ(ierr);
        ierr = adv_solver_->solve (tumor_->mat_prop_->glm_, tumor_->velocity_, dt);                 CHKERRQ(ierr); 
        adv_solver_->advection_mode_ = 1;  // reset to mass conservation
        ierr = adv_solver_->solve (tumor_->c_t_, tumor_->velocity_, dt);                            CHKERRQ(ierr);

        // All solves complete except elasticity: clip values to ensure positivity
        // clip healthy tissues
        ierr = tumor_->mat_prop_->clipHealthyTissues ();                          CHKERRQ (ierr);

        // Diffusion of tumor
        ierr = diff_solver_->solve (tumor_->c_t_, dt);

        // Reaction of tumor
        ierr = reaction (linearized, i);                                                            CHKERRQ(ierr);
        // Mass conservation of healthy: modified gm and wm to account for cell death   
        ierr = conserveHealthyTissues ();                                                           CHKERRQ(ierr);

        // force compute
        ierr = tumor_->computeForce (tumor_->c_t_);                                                 CHKERRQ(ierr);
        // displacement compute through elasticity solve: Linv(force_) = displacement_
        ierr = elasticity_solver_->solve (tumor_->displacement_, tumor_->force_);                   CHKERRQ(ierr);
        // compute velocity
        ierr = VecWAXPY (tumor_->velocity_->x_, -1.0, displacement_old_->x_, tumor_->displacement_->x_);     CHKERRQ (ierr);
        ierr = VecWAXPY (tumor_->velocity_->y_, -1.0, displacement_old_->y_, tumor_->displacement_->y_);     CHKERRQ (ierr);
        ierr = VecWAXPY (tumor_->velocity_->z_, -1.0, displacement_old_->z_, tumor_->displacement_->z_);     CHKERRQ (ierr);
        ierr = VecScale (tumor_->velocity_->x_, (1.0 / dt));                                                CHKERRQ (ierr);
        ierr = VecScale (tumor_->velocity_->y_, (1.0 / dt));                                                CHKERRQ (ierr);
        ierr = VecScale (tumor_->velocity_->z_, (1.0 / dt));                                                CHKERRQ (ierr);

        // smooth the velocity
        if (flag_smooth_velocity) {
            ierr = spec_ops_->weierstrassSmoother (tumor_->velocity_->x_, tumor_->velocity_->x_, n_misc_, sigma_smooth);     CHKERRQ (ierr);
            ierr = spec_ops_->weierstrassSmoother (tumor_->velocity_->y_, tumor_->velocity_->y_, n_misc_, sigma_smooth);     CHKERRQ (ierr);
            ierr = spec_ops_->weierstrassSmoother (tumor_->velocity_->z_, tumor_->velocity_->z_, n_misc_, sigma_smooth);     CHKERRQ (ierr);
        }

        ScalarType vel_x_norm, vel_y_norm, vel_z_norm;
        ierr = VecNorm (tumor_->velocity_->x_, NORM_2, &vel_x_norm);        CHKERRQ (ierr);
        ierr = VecNorm (tumor_->velocity_->y_, NORM_2, &vel_y_norm);        CHKERRQ (ierr);
        ierr = VecNorm (tumor_->velocity_->z_, NORM_2, &vel_z_norm);        CHKERRQ (ierr);
        if (n_misc_->verbosity_ > 1) {
            s << "Norm of velocity (x,y,z) = (" << vel_x_norm << ", " << vel_y_norm << ", " << vel_z_norm << ")";
            ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
            s.str (""); s.clear ();
        }

        // copy displacement to old vector
        ierr = displacement_old_->copy (tumor_->displacement_);
    }

    if (n_misc_->verbosity_ >= 3) {
        s << " Accumulated KSP itr for state eqn = " << diff_ksp_itr_state_;
        ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }

    #ifdef CUDA
        if (n_misc_->verbosity_ > 1) cudaPrintDeviceMemory ();
    #endif


    if ((n_misc_->writeOutput_ && n_misc_->verbosity_ > 1 && !write_output_and_break)) {
        // for mass-effect inversion, write the very last one too. TODO: change loc of print statements instead.
        ss.str(std::string()); ss.clear();
        ss << "c_final.nc";
        dataOut (tumor_->c_t_, n_misc_, ss.str().c_str());
        ss.str(std::string()); ss.clear();
        ss << "vt_final.nc";
        dataOut (tumor_->mat_prop_->csf_, n_misc_, ss.str().c_str());
        ss.str(std::string()); ss.clear();
        ss << "csf_final.nc";
        dataOut (tumor_->mat_prop_->glm_, n_misc_, ss.str().c_str());
        ss.str(std::string()); ss.clear();
        ss << "wm_final.nc";
        dataOut (tumor_->mat_prop_->wm_, n_misc_, ss.str().c_str());
        ss.str(std::string()); ss.clear();
        ss << "gm_final.nc";
        dataOut (tumor_->mat_prop_->gm_, n_misc_, ss.str().c_str());
        ss.str(std::string()); ss.clear();
    }

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode enforcePositivity (Vec c, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ScalarType *c_ptr;
    ierr = VecGetArray (c, &c_ptr);                              CHKERRQ (ierr);
    for (int i = 0; i < n_misc->n_local_; i++) {
        c_ptr[i] = (c_ptr[i] < 0.0) ? 0.0 : c_ptr[i];
        c_ptr[i] = (c_ptr[i] > 1.0) ? 1.0 : c_ptr[i];
    }
    ierr = VecRestoreArray (c, &c_ptr);                          CHKERRQ (ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode checkClipping (Vec c, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    std::stringstream s;
    ScalarType max, min;
    ierr = VecMax (c, NULL, &max);  CHKERRQ (ierr);
    ierr = VecMin (c, NULL, &min);  CHKERRQ (ierr);
    ScalarType tol = 0.;
    s << " ---------- tumor c(0) bounds: max = " << max << ", min = " << min << " ----------- ";
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();
    if (max > 1 || min < tol) {
        #ifdef POSITIVITY
            s << " ---------- warning: tumor c(0) is clipped! max = " << max << ", min = " << min << " ----------- ";
            ierr = tuMSGwarn(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();
        #endif
    }
    PetscFunctionReturn (ierr);
}


PetscErrorCode PdeOperatorsMultiSpecies::computeReactionRate (Vec m) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-reaction");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    ScalarType *ox_ptr, *m_ptr, *rho_ptr;
    ierr = vecGetArray (m, &m_ptr);                                 CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->species_["oxygen"], &ox_ptr);       CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);          CHKERRQ (ierr);
    #ifdef CUDA
        computeReactionRateCuda (m_ptr, ox_ptr, rho_ptr, n_misc_->ox_hypoxia_, n_misc_->n_local_);
    #else
        for (int i = 0; i < n_misc_->n_local_; i++) 
            m_ptr[i] = rho_ptr[i] * (1 / (1 + std::exp(-100 * (ox_ptr[i] - n_misc_->ox_hypoxia_))));
    #endif

    ierr = vecRestoreArray (m, &m_ptr);                                 CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->species_["oxygen"], &ox_ptr);       CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);          CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::computeTransition (Vec alpha, Vec beta) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-transition");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    ScalarType *ox_ptr, *alpha_ptr, *beta_ptr, *p_ptr, *i_ptr;
    ScalarType thres = 0.9;
    ierr = vecGetArray (alpha, &alpha_ptr);                         CHKERRQ (ierr);
    ierr = vecGetArray (beta, &beta_ptr);                           CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->species_["oxygen"], &ox_ptr);       CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->species_["proliferative"], &p_ptr);       CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->species_["infiltrative"], &i_ptr);        CHKERRQ (ierr);

    #ifdef CUDA
        computeTransitionCuda (alpha_ptr, beta_ptr, ox_ptr, p_ptr, i_ptr, n_misc_->alpha_0_, n_misc_->beta_0_, n_misc_->ox_inv_, thres, n_misc_->n_local_);
    #else
        for (int i = 0; i < n_misc_->n_local_; i++) {
            alpha_ptr[i] = n_misc_->alpha_0_ * (1 / (1 + std::exp(100 * (ox_ptr[i] - n_misc_->ox_inv_))));
            beta_ptr[i] = n_misc_->beta_0_ * ox_ptr[i];
        }
    #endif

    ierr = vecRestoreArray (alpha, &alpha_ptr);                         CHKERRQ (ierr);
    ierr = vecRestoreArray (beta, &beta_ptr);                           CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->species_["oxygen"], &ox_ptr);       CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->species_["proliferative"], &p_ptr);       CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->species_["infiltrative"], &i_ptr);        CHKERRQ (ierr);


    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::computeThesholder (Vec h) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-thresholder");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    ScalarType *ox_ptr, *h_ptr;
    ierr = vecGetArray (h, &h_ptr);                                 CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->species_["oxygen"], &ox_ptr);       CHKERRQ (ierr);

    #ifdef CUDA
        computeThesholderCuda (h_ptr, ox_ptr, n_misc_->ox_hypoxia_, n_misc_->n_local_);
    #else
        for (int i = 0; i < n_misc_->n_local_; i++) 
            h_ptr[i] = (1 / (1 + std::exp(100 * (ox_ptr[i] - n_misc_->ox_hypoxia_))));
    #endif

    ierr = vecRestoreArray (h, &h_ptr);                                 CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->species_["oxygen"], &ox_ptr);       CHKERRQ (ierr);


    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::computeSources (Vec p, Vec i, Vec n, Vec O, ScalarType dt) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-sources");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    
    ierr = computeReactionRate (tumor_->work_[0]);                                    CHKERRQ (ierr);
    ierr = computeTransition (tumor_->work_[1], tumor_->work_[2]);                    CHKERRQ (ierr);
    ierr = computeThesholder (tumor_->work_[3]);                                      CHKERRQ (ierr);

    ScalarType *p_ptr, *i_ptr, *n_ptr, *al_ptr, *bet_ptr, *h_ptr, *m_ptr, *di_ptr;
    ScalarType *gm_ptr, *wm_ptr;
    ScalarType *ox_ptr;

    ierr = vecGetArray (p, &p_ptr);                                                   CHKERRQ (ierr);
    ierr = vecGetArray (i, &i_ptr);                                                   CHKERRQ (ierr);
    ierr = vecGetArray (n, &n_ptr);                                                   CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->work_[0], &m_ptr);                                    CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->work_[1], &al_ptr);                                   CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->work_[2], &bet_ptr);                                  CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->work_[3], &h_ptr);                                    CHKERRQ (ierr);

    ierr = vecGetArray (tumor_->mat_prop_->gm_, &gm_ptr);                             CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->mat_prop_->wm_, &wm_ptr);                             CHKERRQ (ierr);

    ierr = vecGetArray (O, &ox_ptr);                                                  CHKERRQ (ierr);
    ierr = vecGetArray (tumor_->work_[11], &di_ptr);                                  CHKERRQ (ierr);

    #ifdef CUDA
        computeSourcesCuda (p_ptr, i_ptr, n_ptr, m_ptr, al_ptr, bet_ptr, h_ptr, gm_ptr, wm_ptr, ox_ptr, di_ptr, dt, 
            n_misc_->death_rate_, n_misc_->ox_source_, n_misc_->ox_consumption_, n_misc_->n_local_);
    #else
        ScalarType p_temp, i_temp, frac_1, frac_2;
        ScalarType ox_heal = 1.;
        ScalarType reac_ratio = 0;
        ScalarType death_ratio = 1;
        for (int i = 0; i < n_misc_->n_local_; i++) {
            p_temp = p_ptr[i]; i_temp = i_ptr[i];
            p_ptr[i] += dt * (m_ptr[i] * p_ptr[i] * (1. - p_ptr[i]) - al_ptr[i] * p_ptr[i] + bet_ptr[i] * i_ptr[i] - 
                                n_misc_->death_rate_ * h_ptr[i] * p_ptr[i]);
            i_ptr[i] += dt * (reac_ratio * m_ptr[i] * i_ptr[i] * (1. - i_ptr[i]) + al_ptr[i] * p_temp - bet_ptr[i] * i_ptr[i] - 
                                death_ratio * n_misc_->death_rate_ * h_ptr[i] * i_ptr[i]);
            n_ptr[i] += dt * (h_ptr[i] * n_misc_->death_rate_ * (p_temp + death_ratio * i_temp + gm_ptr[i] + wm_ptr[i]));
            ox_ptr[i] += dt * (-n_misc_->ox_consumption_ * p_temp + n_misc_->ox_source_ * (ox_heal - ox_ptr[i]) * (gm_ptr[i] + wm_ptr[i]));
            ox_ptr[i] = (ox_ptr[i] <= 0.) ? 0. : ox_ptr[i];

            // conserve healthy cells
            if (gm_ptr[i] > 0.01 || wm_ptr[i] > 0.01) {
                frac_1 = gm_ptr[i] / (gm_ptr[i] + wm_ptr[i]); frac_2 = wm_ptr[i] / (gm_ptr[i] + wm_ptr[i]);
            } else {
                frac_1 = 0.; frac_2 = 0.;
            }
            frac_1 = (std::isnan(frac_1)) ? 0. : frac_1;
            frac_2 = (std::isnan(frac_2)) ? 0. : frac_2;
            gm_ptr[i] += -dt * (frac_1 * (m_ptr[i] * p_temp * (1. - p_temp) + reac_ratio * m_ptr[i] * i_temp * (1. - i_temp) + di_ptr[i])
                             + h_ptr[i] * n_misc_->death_rate_ * gm_ptr[i]); 
            wm_ptr[i] += -dt * (frac_2 * (m_ptr[i] * p_temp * (1. - p_temp) + reac_ratio * m_ptr[i] * i_temp * (1. - i_temp) + di_ptr[i])
                             + h_ptr[i] * n_misc_->death_rate_ * wm_ptr[i]); 
        }
    #endif

    ierr = vecRestoreArray (p, &p_ptr);                                                   CHKERRQ (ierr);
    ierr = vecRestoreArray (i, &i_ptr);                                                   CHKERRQ (ierr);
    ierr = vecRestoreArray (n, &n_ptr);                                                   CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->work_[0], &m_ptr);                                    CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->work_[1], &al_ptr);                                   CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->work_[2], &bet_ptr);                                  CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->work_[3], &h_ptr);                                    CHKERRQ (ierr);

    ierr = vecRestoreArray (tumor_->mat_prop_->gm_, &gm_ptr);                             CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->mat_prop_->wm_, &wm_ptr);                             CHKERRQ (ierr);

    ierr = vecRestoreArray (O, &ox_ptr);                                                  CHKERRQ (ierr);
    ierr = vecRestoreArray (tumor_->work_[11], &di_ptr);                                  CHKERRQ (ierr);


    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::updateReacAndDiffCoefficients (Vec seg, std::shared_ptr<Tumor> tumor) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ScalarType *seg_ptr, *rho_ptr, *k_ptr;
    ierr = VecSet (tumor_->rho_->rho_vec_, 0.);                 CHKERRQ (ierr);
    ierr = VecSet (tumor_->k_->kxx_, 0.);                       CHKERRQ (ierr);

    ierr = VecGetArray (seg, &seg_ptr);                         CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);      CHKERRQ (ierr);
    // ierr = VecGetArray (tumor_->k_->kxx_, &k_ptr);              CHKERRQ (ierr);

    for (int i = 0; i < n_misc_->n_local_; i++) {
        if (std::abs(seg_ptr[i] - 1) < 1E-3 || std::abs(seg_ptr[i] - 2) < 1E-3) {
            // 1 is tumor, 2 is wm
            rho_ptr[i] = n_misc_->rho_;
            // k_ptr[i] = n_misc_->k_;
        }
    }

    ierr = VecRestoreArray (seg, &seg_ptr);                         CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);      CHKERRQ (ierr);
    // ierr = VecRestoreArray (tumor_->k_->kxx_, &k_ptr);              CHKERRQ (ierr);

    // smooth them now
    ScalarType sigma_smooth = n_misc_->smoothing_factor_ * 2 * M_PI / n_misc_->n_[0];
    ierr = spec_ops_->weierstrassSmoother (tumor_->rho_->rho_vec_, tumor_->rho_->rho_vec_, n_misc_, sigma_smooth);
    // ierr = spec_ops_->weierstrassSmoother (tumor_->k_->kxx_, tumor_->k_->kxx_, n_misc_, sigma_smooth);

    // copy kxx to other directions
    // ierr = VecCopy (tumor_->k_->kxx_, tumor_->k_->kyy_);            CHKERRQ (ierr);
    // ierr = VecCopy (tumor_->k_->kxx_, tumor_->k_->kzz_);            CHKERRQ (ierr);

    // ignore the avg for now since it won't change much and the preconditioner does not have much effect on
    // the diffusion solver

    PetscFunctionReturn (ierr);
}

PetscErrorCode PdeOperatorsMultiSpecies::solveState (int linearized) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-solve-state");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    ScalarType dt = n_misc_->dt_;
    int nt = n_misc_->nt_;

    n_misc_->statistics_.nb_state_solves++;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    ierr = VecCopy (tumor_->c_0_, tumor_->species_["proliferative"]);                     CHKERRQ (ierr);
    ierr = VecCopy (tumor_->c_0_, tumor_->species_["infiltrative"]);                      CHKERRQ (ierr);
    // set infiltrative as a small fraction of proliferative; oxygen is max everywhere in the beginning - consider changing to (max - p) if needed
    ierr = VecScale (tumor_->species_["infiltrative"], 0);                              CHKERRQ (ierr); 

    ierr = VecSet (tumor_->species_["oxygen"], 1.);                                       CHKERRQ (ierr);

    ScalarType sigma_smooth = 1.0 * 2.0 * M_PI / n_misc_->n_[0];
    // smooth i_t to keep aliasing to a minimum
    // ierr = spec_ops_->weierstrassSmoother (tumor_->species_["infiltrative"], tumor_->species_["infiltrative"], n_misc_, sigma_smooth);     CHKERRQ (ierr);

    ierr = tumor_->clipTumor();                                                                 CHKERRQ (ierr);

    // no healthy cells where tumor is maximum
    ierr = VecWAXPY (tumor_->c_t_, 1., tumor_->species_["proliferative"], tumor_->species_["infiltrative"]);                        CHKERRQ (ierr);
    ierr = tumor_->mat_prop_->filterTumor (tumor_->c_t_);                                                                           CHKERRQ (ierr);

    ScalarType k1, k2, k3, r1, r2, r3;
    k1 = n_misc_->k_;
    k2 = n_misc_->k_gm_wm_ratio_ * n_misc_->k_; k3 = 0;
    r1 = n_misc_->rho_;
    r2 = n_misc_->r_gm_wm_ratio_ * n_misc_->rho_; r3 = 0;

    // force compute
    ierr = VecCopy (tumor_->species_["proliferative"], tumor_->c_t_);                        CHKERRQ (ierr);

    if (n_misc_->forcing_factor_ > 0) {
        ierr = tumor_->computeForce (tumor_->c_t_);
        // displacement compute through elasticity solve
        ierr = elasticity_solver_->solve (tumor_->displacement_, tumor_->force_);
        // copy displacement to old vector
        ierr = displacement_old_->copy (tumor_->displacement_);
    }

    diff_ksp_itr_state_ = 0;
    ScalarType vel_max;
    ScalarType cfl;
    std::stringstream ss;
    ScalarType vel_x_norm, vel_y_norm, vel_z_norm;
    std::stringstream s;

    bool flag_smooth_velocity = true;
    bool write_output_and_break = false;

    ScalarType max_cfl = 8;

    for (int i = 0; i <= nt; i++) {
        s << "Time step = " << i;
        ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
        s.str (""); s.clear ();
        // compute CFL
        ierr = tumor_->computeEdema ();                                           CHKERRQ (ierr);
        ierr = tumor_->computeSegmentation ();                                    CHKERRQ (ierr);
        ierr = tumor_->velocity_->computeMagnitude (magnitude_);
        ierr = VecMax (magnitude_, NULL, &vel_max);      CHKERRQ (ierr);
        cfl = dt * vel_max / n_misc_->h_[0];
        s << "CFL = " << cfl;
        ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
        s.str (""); s.clear ();
        // Adaptively time step if CFL is too large
        if (cfl >= max_cfl) {
            // TODO: is adaptive time-stepping necessary? Because if cfl is already this number; then numerical oscillations
            // have already propogated and dt keeps dropping to very small values -- very bad accuracy from semi-Lagrangian.
            // instead suggest to the user to have a smaller forcing factor. Large forcing factor also stretches the tumor
            // concentration too much which calls for higher E_tumor which can be very unrealistic. If mass-effect still does 
            // not match, then rho/kappa might be highly inaccurate.

            // keep re-size of time history commented for now
            // dt *= 0.5;
            // nt = i + 2. * (n_misc_->nt_ - i - 1) + 1;
            // n_misc_->dt_ = dt;
            // n_misc_->nt_ = nt;
            // s << "CFL too large -- Changing dt to " << dt << " and nt to " << nt << "\n";
            // ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
            // s.str (""); s.clear ();
            // if (nt >= 200) {
            //     s << "Number of time-steps too large, consider using smaller forcing factor; exiting solver...";
            //     ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
            //     s.str (""); s.clear ();
            //     break;
            // }
            s << "CFL is too large (>=" << max_cfl << "); consider using smaller forcing factor; exiting solver...";
            ierr = tuMSGwarn (s.str());                                                CHKERRQ(ierr);
            s.str (""); s.clear ();
            write_output_and_break = true;
        }

        if ((n_misc_->writeOutput_ && i % 5 == 0) || write_output_and_break) {
            ss << "velocity_t[" << i << "].nc";
            dataOut (magnitude_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ierr = displacement_old_->computeMagnitude(magnitude_);
            ss << "displacement_t[" << i << "].nc";
            dataOut (magnitude_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "seg_t[" << i << "].nc";
            dataOut (tumor_->seg_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "c_t[" << i << "].nc";
            dataOut (tumor_->c_t_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "p_t[" << i << "].nc";
            dataOut (tumor_->species_["proliferative"], n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "i_t[" << i << "].nc";
            dataOut (tumor_->species_["infiltrative"], n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "n_t[" << i << "].nc";
            dataOut (tumor_->species_["necrotic"], n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "o_t[" << i << "].nc";
            dataOut (tumor_->species_["oxygen"], n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            if (n_misc_->verbosity_ > 2) {
            //     ss << "rho_t[" << i << "].nc";
            //     dataOut (tumor_->rho_->rho_vec_, n_misc_, ss.str().c_str());
            //     ss.str(std::string()); ss.clear();
                ss << "m_t[" << i << "].nc";
                dataOut (tumor_->work_[0], n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "ed_t[" << i << "].nc";
                dataOut (tumor_->species_["edema"], n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "kxx_t[" << i << "].nc";
                dataOut (tumor_->k_->kxx_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                // ss << "lam_t[" << i << "].nc";
                // dataOut (elasticity_solver_->ctx_->lam_, n_misc_, ss.str().c_str());
                // ss.str(std::string()); ss.clear();
                // ss << "mu_t[" << i << "].nc";
                // dataOut (elasticity_solver_->ctx_->mu_, n_misc_, ss.str().c_str());
                // ss.str(std::string()); ss.clear();
                // ss << "scr_t[" << i << "].nc";
                // dataOut (elasticity_solver_->ctx_->screen_, n_misc_, ss.str().c_str());
                // ss.str(std::string()); ss.clear();
                ierr = tumor_->force_->computeMagnitude(magnitude_);
                ss << "force_t[" << i << "].nc";
                dataOut (magnitude_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "vt_t[" << i << "].nc";
                dataOut (tumor_->mat_prop_->csf_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "csf_t[" << i << "].nc";
                dataOut (tumor_->mat_prop_->glm_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "wm_t[" << i << "].nc";
                dataOut (tumor_->mat_prop_->wm_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
                ss << "gm_t[" << i << "].nc";
                dataOut (tumor_->mat_prop_->gm_, n_misc_, ss.str().c_str());
                ss.str(std::string()); ss.clear();
            }
        }

        if (write_output_and_break) break;
        // ------------------------------------------------ advection  ------------------------------------------------

        // Update diffusivity and reaction coefficient
        ierr = updateReacAndDiffCoefficients (tumor_->seg_, tumor_);                                CHKERRQ (ierr);
        ierr = tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, tumor_->mat_prop_, n_misc_);    CHKERRQ (ierr);

        // need to update prefactors for diffusion KSP preconditioner, as k changed
        ierr = diff_solver_->precFactor();                                                          CHKERRQ (ierr);

        if (n_misc_->forcing_factor_ > 0) {
            // Advection of tumor and healthy tissue
            // first compute trajectories for semi-Lagrangian solve as velocity is changing every itr
            adv_solver_->trajectoryIsComputed_ = false;
            ierr = adv_solver_->solve (tumor_->mat_prop_->gm_, tumor_->velocity_, dt);                  CHKERRQ (ierr);
            ierr = adv_solver_->solve (tumor_->mat_prop_->wm_, tumor_->velocity_, dt);                  CHKERRQ (ierr);
            adv_solver_->advection_mode_ = 2;  // pure advection for csf
            ierr = adv_solver_->solve (tumor_->mat_prop_->csf_, tumor_->velocity_, dt);                 CHKERRQ (ierr);
            ierr = adv_solver_->solve (tumor_->mat_prop_->glm_, tumor_->velocity_, dt);                 CHKERRQ (ierr); 
            adv_solver_->advection_mode_ = 1;  // reset to mass conservation
            ierr = adv_solver_->solve (tumor_->species_["proliferative"], tumor_->velocity_, dt);       CHKERRQ (ierr);
            ierr = adv_solver_->solve (tumor_->species_["infiltrative"], tumor_->velocity_, dt);        CHKERRQ (ierr);
            ierr = adv_solver_->solve (tumor_->species_["necrotic"], tumor_->velocity_, dt);            CHKERRQ (ierr);  
        }

        // All solves complete except elasticity: clip values to ensure positivity
        // clip healthy tissues
        ierr = tumor_->mat_prop_->clipHealthyTissues ();                          CHKERRQ (ierr);
        // clip tumor : single-precision advection seems to have issues if this is not clipped.
        ierr = tumor_->clipTumor();                                                                 CHKERRQ (ierr);
          
        // compute Di to be used for healthy cell evolution equations: make sure work[11] is not used till sources are computed
        ierr = VecCopy (tumor_->species_["infiltrative"], tumor_->work_[11]);      CHKERRQ (ierr);
        ierr = tumor_->k_->applyD (tumor_->work_[11], tumor_->work_[11]);          CHKERRQ (ierr);

        // ------------------------------------------------ diffusion  ------------------------------------------------
        ierr = diff_solver_->solve (tumor_->species_["infiltrative"], dt);         diff_ksp_itr_state_ += diff_solver_->ksp_itr_;   CHKERRQ (ierr);
        // ierr = diff_solver_->solve (tumor_->species_["oxygen"], dt);               diff_ksp_itr_state_ += diff_solver_->ksp_itr_;   CHKERRQ (ierr);
        
        // ------------------------------------------------ explicit source terms for all equations (includes reaction source)  ------------------------------------------------
        ierr = computeSources (tumor_->species_["proliferative"], tumor_->species_["infiltrative"], tumor_->species_["necrotic"], 
                                tumor_->species_["oxygen"], dt);                                                                    CHKERRQ (ierr);

        // set tumor core as c_t_
        ierr = VecWAXPY (tumor_->c_t_, 1., tumor_->species_["proliferative"], tumor_->species_["necrotic"]);                        CHKERRQ (ierr);

        if (n_misc_->forcing_factor_ > 0) {
            // ------------------------------------------------ elasticity update ------------------------------------------------ 
            // force compute
            ierr = tumor_->computeForce (tumor_->c_t_);
            // displacement compute through elasticity solve: Linv(force_) = displacement_
            ierr = elasticity_solver_->solve (tumor_->displacement_, tumor_->force_);
            // compute velocity
            ierr = VecWAXPY (tumor_->velocity_->x_, -1.0, displacement_old_->x_, tumor_->displacement_->x_);     CHKERRQ (ierr);
            ierr = VecWAXPY (tumor_->velocity_->y_, -1.0, displacement_old_->y_, tumor_->displacement_->y_);     CHKERRQ (ierr);
            ierr = VecWAXPY (tumor_->velocity_->z_, -1.0, displacement_old_->z_, tumor_->displacement_->z_);     CHKERRQ (ierr);
            ierr = VecScale (tumor_->velocity_->x_, (1.0 / dt));                                                CHKERRQ (ierr);
            ierr = VecScale (tumor_->velocity_->y_, (1.0 / dt));                                                CHKERRQ (ierr);
            ierr = VecScale (tumor_->velocity_->z_, (1.0 / dt));                                                CHKERRQ (ierr);

            // smooth the velocity
            if (flag_smooth_velocity) {
                ierr = spec_ops_->weierstrassSmoother (tumor_->velocity_->x_, tumor_->velocity_->x_, n_misc_, sigma_smooth);     CHKERRQ (ierr);
                ierr = spec_ops_->weierstrassSmoother (tumor_->velocity_->y_, tumor_->velocity_->y_, n_misc_, sigma_smooth);     CHKERRQ (ierr);
                ierr = spec_ops_->weierstrassSmoother (tumor_->velocity_->z_, tumor_->velocity_->z_, n_misc_, sigma_smooth);     CHKERRQ (ierr);
            }

            ierr = VecNorm (tumor_->velocity_->x_, NORM_2, &vel_x_norm);        CHKERRQ (ierr);
            ierr = VecNorm (tumor_->velocity_->y_, NORM_2, &vel_y_norm);        CHKERRQ (ierr);
            ierr = VecNorm (tumor_->velocity_->z_, NORM_2, &vel_z_norm);        CHKERRQ (ierr);
            s << "Norm of velocity (x,y,z) = (" << vel_x_norm << ", " << vel_y_norm << ", " << vel_z_norm << ")";
            ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
            s.str (""); s.clear ();

            // copy displacement to old vector
            ierr = displacement_old_->copy (tumor_->displacement_);
        }
    }

    if (n_misc_->verbosity_ >= 3) {
        s << " Accumulated KSP itr for state eqn = " << diff_ksp_itr_state_;
        ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }

    #ifdef CUDA
        cudaPrintDeviceMemory ();
    #endif

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

