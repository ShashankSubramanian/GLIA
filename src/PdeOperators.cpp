#include "PdeOperators.h"

PdeOperatorsRD::PdeOperatorsRD (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc)
        : PdeOperators (tumor, n_misc) {

    PetscErrorCode ierr = 0;
    double dt = n_misc_->dt_;
    int nt = n_misc->nt_;

    c_.resize (nt + 1);                         //Time history of tumor
    p_.resize (nt + 1);                         //Time history of adjoints

    ierr = VecCreate (PETSC_COMM_WORLD, &c_[0]);
    ierr = VecSetSizes (c_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (c_[0]);
    ierr = VecCreate (PETSC_COMM_WORLD, &p_[0]);
    ierr = VecSetSizes (p_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (p_[0]);

    for (int i = 1; i < nt + 1; i++) {
        ierr = VecDuplicate (c_[0], &c_[i]);
        ierr = VecDuplicate (p_[0], &p_[i]);
    }
    for (int i = 0; i < nt + 1; i++) {
        ierr = VecSet (c_[i], 0);
        ierr = VecSet (p_[i], 0);
    }
}


PetscErrorCode PdeOperatorsRD::resizeTimeHistory (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    double dt = n_misc_->dt_;
    int nt = n_misc->nt_;

    nt_ = nt;

    for (int i = 0; i < c_.size(); i++) {
        ierr = VecDestroy (&c_[i]);
        ierr = VecDestroy (&p_[i]);
    }

    c_.resize (nt + 1);                         //Time history of tumor
    p_.resize (nt + 1);                         //Time history of adjoints

    ierr = VecCreate (PETSC_COMM_WORLD, &c_[0]);
    ierr = VecSetSizes (c_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (c_[0]);
    ierr = VecCreate (PETSC_COMM_WORLD, &p_[0]);
    ierr = VecSetSizes (p_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (p_[0]);

    for (int i = 1; i < nt + 1; i++) {
        ierr = VecDuplicate (c_[0], &c_[i]);
        ierr = VecDuplicate (p_[0], &p_[i]);
    }

    for (int i = 0; i < nt + 1; i++) {
        ierr = VecSet (c_[i], 0);
        ierr = VecSet (p_[i], 0);
    }

    PetscFunctionReturn (0);
}

PetscErrorCode PdeOperatorsRD::reaction (int linearized, int iter) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-reaction");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    double *c_t_ptr, *rho_ptr;
    double *c_ptr;
    double factor, alph;
    double dt = n_misc_->dt_;
    ierr = VecGetArray (tumor_->c_t_, &c_t_ptr);                 CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);

    ierr = VecGetArray (c_[iter], &c_ptr);                       CHKERRQ (ierr);

    for (int i = 0; i < n_misc_->n_local_; i++) {
        if (linearized == 0) {
            factor = std::exp (rho_ptr[i] * dt);
            alph = (1.0 - c_t_ptr[i]) / c_t_ptr[i];
            c_t_ptr[i] = factor / (factor + alph);
        }
        else {
            factor = std::exp (rho_ptr[i] * dt);
            alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
            c_t_ptr[i] = c_t_ptr[i] * factor / (alph * alph);
        }
    }

    ierr = VecRestoreArray (tumor_->c_t_, &c_t_ptr);             CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);   CHKERRQ (ierr);
    ierr = VecRestoreArray (c_[iter], &c_ptr);                   CHKERRQ (ierr);

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

    double dt = n_misc_->dt_;
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
    if (linearized == 0) {
        ierr = VecCopy (tumor_->c_t_, c_[0]);                    CHKERRQ (ierr);
    }

    for (int i = 0; i < nt; i++) {
        diff_solver_->solve (tumor_->c_t_, dt / 2.0);
        ierr = reaction (linearized, i);
        diff_solver_->solve (tumor_->c_t_, dt / 2.0);

        //enforce positivity : hack
        if (!linearized) {
            #ifdef POSITIVITY
                ierr = enforcePositivity (tumor_->c_t_, n_misc_);
            #endif
        }

        //Copy current conc to use for the adjoint equation
        if (linearized == 0) {
            ierr = VecCopy (tumor_->c_t_, c_[i + 1]);            CHKERRQ (ierr);
        }
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


    double *p_0_ptr, *rho_ptr;
    double *c_ptr;
    double factor, alph;
    double dt = n_misc_->dt_;

    Vec temp = tumor_->work_[11];
    //reaction adjoint needs c_ at half time step.
    ierr = VecCopy (c_[iter], temp);                             CHKERRQ (ierr);
    diff_solver_->solve (temp, dt / 2.0);

    ierr = VecGetArray (tumor_->p_0_, &p_0_ptr);                 CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->rho_->rho_vec_, &rho_ptr);       CHKERRQ (ierr);
    ierr = VecGetArray (temp, &c_ptr);                           CHKERRQ (ierr);

    for (int i = 0; i < n_misc_->n_local_; i++) {
        if (linearized == 1) {
            factor = std::exp (rho_ptr[i] * dt);
            alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
            p_0_ptr[i] = p_0_ptr[i] * factor / (alph * alph);
        }
        else { //Gauss - Newton method
            factor = std::exp (rho_ptr[i] * dt);
            alph = (c_ptr[i] * factor + 1.0 - c_ptr[i]);
            p_0_ptr[i] = p_0_ptr[i] * factor / (alph * alph);
        }
    }

    ierr = VecRestoreArray (tumor_->p_0_, &p_0_ptr);             CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->rho_->rho_vec_, &rho_ptr);   CHKERRQ (ierr);
    ierr = VecRestoreArray (temp, &c_ptr);                       CHKERRQ (ierr);

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

    double dt = n_misc_->dt_;
    int nt = n_misc_->nt_;
    n_misc_->statistics_.nb_adjoint_solves++;

    ierr = VecCopy (tumor_->p_t_, tumor_->p_0_);                 CHKERRQ (ierr);
    if (linearized == 1) {
        ierr = VecCopy (tumor_->p_0_, p_[nt]);                   CHKERRQ (ierr);
    }
    for (int i = 0; i < nt; i++) {
        diff_solver_->solve (tumor_->p_0_, dt / 2.0);
        ierr = reactionAdjoint (linearized, nt - i - 1);
        diff_solver_->solve (tumor_->p_0_, dt / 2.0);
        //Copy current adjoint time point to use in additional term for moving-atlas formulation
        if (linearized == 1) {
            ierr = VecCopy (tumor_->p_0_, p_[nt - i - 1]);            CHKERRQ (ierr);
        }
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
  double integration_weight = n_misc_->dt_;
  PetscScalar *c_ptr, *p_ptr, *r_ptr;

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
  std::stringstream s; PetscScalar norm_q = 0, tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;
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
    for (int i = 0; i < nt_ + 1; i++) {
        ierr = VecDestroy (&c_[i]);
        ierr = VecDestroy (&p_[i]);
    }
}

PetscErrorCode PdeOperatorsMassEffect::solveState (int linearized) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tumor-solve-state");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    double dt = n_misc_->dt_;
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
    if (linearized == 0) {
        ierr = VecCopy (tumor_->c_t_, c_[0]);                    CHKERRQ (ierr);
    }

    double k1, k2, k3, r1, r2, r3;
    k1 = n_misc_->k_;
    k2 = n_misc_->k_gm_wm_ratio_ * n_misc_->k_; k3 = 0;
    r1 = n_misc_->rho_;
    r2 = n_misc_->r_gm_wm_ratio_ * n_misc_->rho_; r3 = 0;


    std::shared_ptr<VecField> displacement_old = std::make_shared<VecField> (n_misc_->n_local_, n_misc_->n_global_);  
    // force compute
    ierr = tumor_->computeForce (tumor_->c_t_);
    // displacement compute through elasticity solve
    ierr = elasticity_solver_->solve (displacement_old, tumor_->force_);

    if (n_misc_->writeOutput_) {
        ierr = displacement_old->computeMagnitude();
        ierr = tumor_->force_->computeMagnitude();
        dataOut (displacement_old->magnitude_, n_misc_, "displacement_t[0].nc");
        dataOut (tumor_->force_->magnitude_, n_misc_, "force_t[0].nc");
        dataOut (tumor_->mat_prop_->csf_, n_misc_, "csf_t[0].nc");
        ierr = tumor_->computeSegmentation ();
        dataOut (tumor_->seg_, n_misc_, "seg_t[0].nc");
    }

    std::stringstream ss;
    double vel_max;

    for (int i = 0; i < nt; i++) {
        PCOUT << "Time = " << i << std::endl;
        // Update diffusivity and reaction coefficient
        ierr = tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, tumor_->mat_prop_, n_misc_);    CHKERRQ(ierr);
        ierr = tumor_->rho_->updateIsotropicCoefficients (r1, r2, r3, tumor_->mat_prop_, n_misc_);
        // need to update prefactors for diffusion KSP preconditioner, as k changed
        ierr = diff_solver_->precFactor();

        // Advection of tumor and healthy tissue
        ierr = adv_solver_->solve (tumor_->mat_prop_->gm_, tumor_->velocity_, dt);
        ierr = adv_solver_->solve (tumor_->mat_prop_->wm_, tumor_->velocity_, dt);
        ierr = adv_solver_->solve (tumor_->mat_prop_->csf_, tumor_->velocity_, dt);
        ierr = adv_solver_->solve (tumor_->c_t_, tumor_->velocity_, dt);

        // Mass conservation of healthy -- TODO

        // Diffusion of tumor
        ierr = diff_solver_->solve (tumor_->c_t_, dt);

        // Reaction of tumor
        ierr = reaction (linearized, i);

        // force compute
        ierr = tumor_->computeForce (tumor_->c_t_);
        // displacement compute through elasticity solve: Linv(force_) = displacement_
        ierr = elasticity_solver_->solve (tumor_->displacement_, tumor_->force_);

        // compute velocity
        ierr = VecWAXPY (tumor_->velocity_->x_, -1.0, displacement_old->x_, tumor_->displacement_->x_);     CHKERRQ (ierr);
        ierr = VecWAXPY (tumor_->velocity_->y_, -1.0, displacement_old->y_, tumor_->displacement_->y_);     CHKERRQ (ierr);
        ierr = VecWAXPY (tumor_->velocity_->z_, -1.0, displacement_old->z_, tumor_->displacement_->z_);     CHKERRQ (ierr);
        ierr = VecScale (tumor_->velocity_->x_, (1.0 / dt));                                                CHKERRQ (ierr);
        ierr = VecScale (tumor_->velocity_->y_, (1.0 / dt));                                                CHKERRQ (ierr);
        ierr = VecScale (tumor_->velocity_->z_, (1.0 / dt));                                                CHKERRQ (ierr);

        double vel_x_norm, vel_y_norm, vel_z_norm;
        ierr = VecNorm (tumor_->velocity_->x_, NORM_2, &vel_x_norm);        CHKERRQ (ierr);
        ierr = VecNorm (tumor_->velocity_->y_, NORM_2, &vel_y_norm);        CHKERRQ (ierr);
        ierr = VecNorm (tumor_->velocity_->z_, NORM_2, &vel_z_norm);        CHKERRQ (ierr);
        PCOUT << "Norm of velocity (x,y,z) : (" << vel_x_norm << ", " << vel_y_norm << ", " << vel_z_norm << ")\n";

        // compute CFL
        ierr = tumor_->velocity_->computeMagnitude ();
        ierr = VecMax (tumor_->velocity_->magnitude_, NULL, &vel_max);      CHKERRQ (ierr);
        PCOUT << "CFL: " << dt * vel_max / n_misc_->h_[0] << "\n\n";

        // copy displacement to old vector
        ierr = displacement_old->copy (tumor_->displacement_);

        if (n_misc_->writeOutput_) {
            ierr = displacement_old->computeMagnitude();
            ierr = tumor_->force_->computeMagnitude();
            ss << "displacement_t[" << i + 1 << "].nc";
            dataOut (displacement_old->magnitude_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "force_t[" << i + 1 << "].nc";
            dataOut (tumor_->force_->magnitude_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "csf_t[" << i + 1 << "].nc";
            dataOut (tumor_->mat_prop_->csf_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
            ss << "seg_t[" << i + 1 << "].nc";
            ierr = tumor_->computeSegmentation ();
            dataOut (tumor_->seg_, n_misc_, ss.str().c_str());
            ss.str(std::string()); ss.clear();
        }

        //enforce positivity : hack
        if (!linearized) {
            #ifdef POSITIVITY
                ierr = enforcePositivity (tumor_->c_t_, n_misc_);
            #endif
        }

        //Copy current conc to use for the adjoint equation
        if (linearized == 0) {
            ierr = VecCopy (tumor_->c_t_, c_[i + 1]);            CHKERRQ (ierr);
        }
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
    double *c_ptr;
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
    double max, min;
    ierr = VecMax (c, NULL, &max);  CHKERRQ (ierr);
    ierr = VecMin (c, NULL, &min);  CHKERRQ (ierr);
    double tol = 0.;
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
