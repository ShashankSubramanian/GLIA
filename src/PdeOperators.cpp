#include "PdeOperators.h"

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

    PetscFunctionReturn (0);
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

    #ifdef CUDA
    ierr = VecCUDAGetArrayReadWrite (tumor_->c_t_, &c_t_ptr);                 CHKERRQ (ierr);
    ierr = VecCUDAGetArrayReadWrite (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);
    if (linearized != 0) {
        ierr = VecCUDAGetArrayReadWrite (c_[iter], &c_ptr);                       CHKERRQ (ierr);
    }

    logisticReactionCuda (c_t_ptr, rho_ptr, c_ptr, dt, n_misc_->n_local_, linearized);

    ierr = VecCUDARestoreArrayReadWrite (tumor_->c_t_, &c_t_ptr);                 CHKERRQ (ierr);
    ierr = VecCUDARestoreArrayReadWrite (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);
    if (linearized != 0) {
        ierr = VecCUDARestoreArrayReadWrite (c_[iter], &c_ptr);                       CHKERRQ (ierr);
    }

    #else
    ierr = VecGetArray (tumor_->c_t_, &c_t_ptr);                 CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);

    if (linearized == 0) {
        for (int i = 0; i < n_misc_->n_local_; i++) {
            factor = std::exp (rho_ptr[i] * dt);
            alph = (1.0 - c_t_ptr[i]) / c_t_ptr[i];
            c_t_ptr[i] = factor / (factor + alph);
        }
    } else {
        ierr = VecGetArray (c_[iter], &c_ptr);                       CHKERRQ (ierr);
        for (int i = 0; i < n_misc_->n_local_; i++) {
            factor = std::exp (rho_ptr[i] * dt);
            alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
            c_t_ptr[i] = c_t_ptr[i] * factor / (alph * alph);
        }
        ierr = VecRestoreArray (c_[iter], &c_ptr);                   CHKERRQ (ierr);
    }

    ierr = VecRestoreArray (tumor_->c_t_, &c_t_ptr);             CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);   CHKERRQ (ierr);
    
    #endif

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (0);
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
    PetscFunctionReturn (0);
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
    PetscFunctionReturn (0);
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

    #ifdef CUDA
    ierr = VecCUDAGetArrayReadWrite (tumor_->p_0_, &p_0_ptr);                 CHKERRQ (ierr);
    ierr = VecCUDAGetArrayReadWrite (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);
    ierr = VecCUDAGetArrayReadWrite (temp, &c_ptr);                           CHKERRQ (ierr);

    logisticReactionCuda (p_0_ptr, rho_ptr, c_ptr, dt, n_misc_->n_local_, linearized);

    ierr = VecCUDARestoreArrayReadWrite (tumor_->p_0_, &p_0_ptr);             CHKERRQ (ierr);
    ierr = VecCUDARestoreArrayReadWrite (tumor_->rho_->rho_vec_, &rho_ptr);   CHKERRQ (ierr);
    ierr = VecCUDARestoreArrayReadWrite (temp, &c_ptr);                       CHKERRQ (ierr);

    #else
    ierr = VecGetArray (tumor_->p_0_, &p_0_ptr);                 CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);
    ierr = VecGetArray (temp, &c_ptr);                           CHKERRQ (ierr);

    for (int i = 0; i < n_misc_->n_local_; i++) {
        factor = std::exp (rho_ptr[i] * dt);
        alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
        p_0_ptr[i] = p_0_ptr[i] * factor / (alph * alph);
    }

    ierr = VecRestoreArray (tumor_->p_0_, &p_0_ptr);             CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);   CHKERRQ (ierr);
    ierr = VecRestoreArray (temp, &c_ptr);                       CHKERRQ (ierr);
    #endif

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (0);
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
    PetscFunctionReturn (0);
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
  PetscFunctionReturn (0);
}

PdeOperatorsRD::~PdeOperatorsRD () {
    PetscErrorCode ierr = 0;
    if (!n_misc_->forward_flag_) {        
        for (int i = 0; i < nt_ + 1; i++) {
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
    ScalarType *c_ptr, *rho_ptr;

    // Dc
    ierr = VecCopy (tumor_->c_t_, temp_[1]);                        CHKERRQ (ierr);
    ierr = tumor_->k_->applyD (temp_[0], temp_[1]);

    // Rc = rho * c * (1 - c)
    ierr = VecGetArray (temp_[1], &c_ptr);                          CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);          CHKERRQ (ierr);
    ScalarType dt = n_misc_->dt_;
    for (int i = 0; i < n_misc_->n_local_; i++) {
        c_ptr[i] = rho_ptr[i] * c_ptr[i] * (1. - c_ptr[i]);
    }
    ierr = VecRestoreArray (temp_[1], &c_ptr);                      CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);      CHKERRQ (ierr);

    // R + D
    ierr = VecAXPY (temp_[0], 1.0, temp_[1]);                       CHKERRQ (ierr);

    // scaling
    ScalarType *gm_ptr, *wm_ptr, *scale_gm_ptr, *scale_wm_ptr, *sum_ptr;
    ierr = VecGetArray (tumor_->mat_prop_->gm_, &gm_ptr);           CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->mat_prop_->wm_, &wm_ptr);           CHKERRQ (ierr);
    ierr = VecGetArray (temp_[0], &sum_ptr);                        CHKERRQ (ierr);
    ierr = VecGetArray (temp_[1], &scale_gm_ptr);                   CHKERRQ (ierr);
    ierr = VecGetArray (temp_[2], &scale_wm_ptr);                   CHKERRQ (ierr);

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

    ierr = VecRestoreArray (tumor_->mat_prop_->gm_, &gm_ptr);           CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->mat_prop_->wm_, &wm_ptr);           CHKERRQ (ierr);
    ierr = VecRestoreArray (temp_[0], &sum_ptr);                        CHKERRQ (ierr);
    ierr = VecRestoreArray (temp_[1], &scale_gm_ptr);                   CHKERRQ (ierr);
    ierr = VecRestoreArray (temp_[2], &scale_wm_ptr);                   CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (0);
    PetscFunctionReturn (0);
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


    std::shared_ptr<VecField> displacement_old = std::make_shared<VecField> (n_misc_->n_local_, n_misc_->n_global_);  
    // force compute
    ierr = tumor_->computeForce (tumor_->c_t_);
    // displacement compute through elasticity solve
    ierr = elasticity_solver_->solve (displacement_old, tumor_->force_);

    std::stringstream ss;    
    ScalarType vel_max;
    ScalarType cfl;

    for (int i = 0; i < nt + 1; i++) {
        PCOUT << "Time step = " << i << std::endl;
        if (n_misc_->writeOutput_ && i % 10 == 0) {
            ierr = displacement_old->computeMagnitude();
            ierr = tumor_->force_->computeMagnitude();
            ss << "displacement_t[" << i << "].nc";
            dataOut (displacement_old->magnitude_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "force_t[" << i << "].nc";
            dataOut (tumor_->force_->magnitude_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "csf_t[" << i << "].nc";
            dataOut (tumor_->mat_prop_->csf_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "wm_t[" << i << "].nc";
            dataOut (tumor_->mat_prop_->wm_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "seg_t[" << i << "].nc";
            ierr = tumor_->computeSegmentation ();
            dataOut (tumor_->seg_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "c_t[" << i << "].nc";
            dataOut (tumor_->c_t_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "velocity_t[" << i << "].nc";
            dataOut (tumor_->velocity_->magnitude_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
        }
        // Update diffusivity and reaction coefficient
        ierr = tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, tumor_->mat_prop_, n_misc_);    CHKERRQ(ierr);
        ierr = tumor_->rho_->updateIsotropicCoefficients (r1, r2, r3, tumor_->mat_prop_, n_misc_);  CHKERRQ(ierr);
        // need to update prefactors for diffusion KSP preconditioner, as k changed
        ierr = diff_solver_->precFactor();                                                          CHKERRQ(ierr);

        // Advection of tumor and healthy tissue
        // first compute trajectories for semi-Lagrangian solve as velocity is changing every itr
        adv_solver_->trajectoryIsComputed_ = false;
        ierr = adv_solver_->solve (tumor_->mat_prop_->gm_, tumor_->velocity_, dt);                  CHKERRQ(ierr);
        ierr = adv_solver_->solve (tumor_->mat_prop_->wm_, tumor_->velocity_, dt);                  CHKERRQ(ierr);
        adv_solver_->advection_mode_ = 2;  // pure advection for csf
        ierr = adv_solver_->solve (tumor_->mat_prop_->csf_, tumor_->velocity_, dt);                 CHKERRQ(ierr); adv_solver_->advection_mode_ = 1;  // reset to mass conservation
        ierr = adv_solver_->solve (tumor_->c_t_, tumor_->velocity_, dt);                            CHKERRQ(ierr);

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
        ierr = VecWAXPY (tumor_->velocity_->x_, -1.0, displacement_old->x_, tumor_->displacement_->x_);     CHKERRQ (ierr);
        ierr = VecWAXPY (tumor_->velocity_->y_, -1.0, displacement_old->y_, tumor_->displacement_->y_);     CHKERRQ (ierr);
        ierr = VecWAXPY (tumor_->velocity_->z_, -1.0, displacement_old->z_, tumor_->displacement_->z_);     CHKERRQ (ierr);
        ierr = VecScale (tumor_->velocity_->x_, (1.0 / dt));                                                CHKERRQ (ierr);
        ierr = VecScale (tumor_->velocity_->y_, (1.0 / dt));                                                CHKERRQ (ierr);
        ierr = VecScale (tumor_->velocity_->z_, (1.0 / dt));                                                CHKERRQ (ierr);
        ScalarType vel_x_norm, vel_y_norm, vel_z_norm;
        ierr = VecNorm (tumor_->velocity_->x_, NORM_2, &vel_x_norm);        CHKERRQ (ierr);
        ierr = VecNorm (tumor_->velocity_->y_, NORM_2, &vel_y_norm);        CHKERRQ (ierr);
        ierr = VecNorm (tumor_->velocity_->z_, NORM_2, &vel_z_norm);        CHKERRQ (ierr);
        PCOUT << "Norm of velocity (x,y,z) = (" << vel_x_norm << ", " << vel_y_norm << ", " << vel_z_norm << ")\n";
        // compute CFL
        ierr = tumor_->velocity_->computeMagnitude ();
        ierr = VecMax (tumor_->velocity_->magnitude_, NULL, &vel_max);      CHKERRQ (ierr);
        cfl = dt * vel_max / n_misc_->h_[0];
        PCOUT << "CFL = " << cfl << "\n\n";
        // Adaptively time step if CFL is too large
        if (cfl > 0.5) {
            // // TODO: resize time history
            // dt *= 0.5;
            // nt = i + 2. * (n_misc_->nt_ - i - 1) + 1;
            // n_misc_->dt_ = dt;
            // n_misc_->nt_ = nt;

            // PCOUT << "CFL too large -- Changing dt to " << dt << " and nt to " << nt << "\n";
            PCOUT << "CFL too large: exiting...\n"; break;
        }

        // copy displacement to old vector
        ierr = displacement_old->copy (tumor_->displacement_);
    }

    self_exec_time += MPI_Wtime();
    //accumulateTimers (t, t, self_exec_time);
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (0);
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
    PetscFunctionReturn (0);
}

PetscErrorCode checkClipping (Vec c, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    ScalarType max, min;
    ierr = VecMax (c, NULL, &max);  CHKERRQ (ierr);
    ierr = VecMin (c, NULL, &min);  CHKERRQ (ierr);
    ScalarType tol = 0.;
    PCOUT << "[---------- Tumor IC bounds: Max = " << max << ", Min = " << min << " -----------]" << std::endl;
    if (max > 1 || min < tol) {
        #ifdef POSITIVITY
            PCOUT << "[---------- Warning! Tumor IC is clipped: Max = " << max << ", Min = " << min << "! -----------]" << std::endl;
        // #else
            // PCOUT << "[---------- Warning! Tumor IC is out of bounds and not clipped: Max = " << max << ", Min = " << min << "! -----------]" << std::endl;
        #endif
    }
    PetscFunctionReturn (0);
}
