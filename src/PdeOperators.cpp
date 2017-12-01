#include "PdeOperators.h"

PdeOperatorsRD::PdeOperatorsRD (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc)
        : PdeOperators (tumor, n_misc) {

    PetscErrorCode ierr = 0;
    double dt = n_misc_->dt_;
    int nt = n_misc->nt_;
    c_ = (Vec *) malloc (sizeof (Vec *) * (nt));    //Stores all (nt) tumor concentrations for adjoint eqns
    p_ = (Vec *) malloc (sizeof (Vec *) * (nt));    //Stores all (nt) tumor adjoint time points
    ierr = VecCreate (PETSC_COMM_WORLD, &c_[0]);
    ierr = VecSetSizes (c_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (c_[0]);
    ierr = VecCreate (PETSC_COMM_WORLD, &p_[0]);
    ierr = VecSetSizes (p_[0], n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (p_[0]);
    for (int i = 1; i < nt; i++) {
        ierr = VecDuplicate (c_[0], &c_[i]);
        ierr = VecDuplicate (p_[0], &p_[i]);
    }
    for (int i = 0; i < nt; i++) {
        ierr = VecSet (c_[i], 0);
        ierr = VecSet (p_[i], 0);
    }
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
    accumulateTimers (t, t, self_exec_time);
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

    //enforce positivity : hack
    if (!linearized) {
        #ifdef POSITIVITY
            ierr = enforcePositivity (tumor_->c_0_, n_misc_);
        #endif
    }

    ierr = VecCopy (tumor_->c_0_, tumor_->c_t_);                 CHKERRQ (ierr);

    for (int i = 0; i < nt; i++) {
        //Copy current conc to use for the adjoint equation
        if (linearized == 0) {
            ierr = VecCopy (tumor_->c_t_, c_[i]);                CHKERRQ (ierr);
        }
        diff_solver_->solve (tumor_->c_t_, dt / 2.0);
        ierr = reaction (linearized, i);
        diff_solver_->solve (tumor_->c_t_, dt / 2.0);

        //enforce positivity : hack
        if (!linearized) {
            #ifdef POSITIVITY
                ierr = enforcePositivity (tumor_->c_t_, n_misc_);
            #endif
        }
    }

    self_exec_time += MPI_Wtime();
    accumulateTimers (t, t, self_exec_time);
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

    Vec temp = tumor_->k_->temp_[0];
    //reaction adjoint needs c_ at half time step. 
    ierr = VecCopy (c_[iter], temp);                            CHKERRQ (ierr);
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
    accumulateTimers (t, t, self_exec_time);
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
    for (int i = 0; i < nt; i++) {
        diff_solver_->solve (tumor_->p_0_, dt / 2.0);
        ierr = reactionAdjoint (linearized, nt - i - 1);
        diff_solver_->solve (tumor_->p_0_, dt / 2.0);
        //Copy current adjoint time point to use in additional term for moving-atlas formulation
        if (linearized == 1) {
            ierr = VecCopy (tumor_->p_0_, p_[i]);                CHKERRQ (ierr);
        }
    }

    self_exec_time += MPI_Wtime();
    accumulateTimers (t, t, self_exec_time); e.addTimings (t); e.stop ();
    PetscFunctionReturn (0);
}

PetscErrorCode PdeOperatorsRD::computeVaryingMatProbContribution(Vec q) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tumor-compute-q");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  double integration_weight = n_misc_->dt_;
  PetscScalar *c_ptr, *p_ptr, *r_ptr;

  // clear
  ierr = VecSet (q , 0);                                         CHKERRQ (ierr);
  // compute numerical time integration using trapezoidal rule
  for (int i = 0; i < this->nt_; i++) {
    // integration weight for chain trapezoidal rule
    if (j == 0 || j == this->nt_-1) integration_weight *= 0.5;

    // compute x = k_bar * (grad c)^T grad \alpha, where k_bar = dK / dm
    ierr = tumor_->k_->compute_dKdm_gradc_gradp(tumor_->k_->temp_[0], c_[i], p_[i], n_misc_->plan_); CHKERRQ(ierr);
    // compute y = c(1-c) * \alpha
    ierr = VecGetArray (c_[i], &c_ptr);                          CHKERRQ (ierr);
    ierr = VecGetArray (p_[i], &p_ptr);                          CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->k_->temp_[1], &r_ptr);           CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
      r_ptr[i] = c_ptr[i] * (1 - c_ptr[i]) * p_ptr[i];
    }
    ierr = VecRestoreArray (c_[i], &c_ptr);                      CHKERRQ (ierr);
    ierr = VecRestoreArray (p_[i], &p_ptr);                      CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->k_->temp_[1], &r_ptr);       CHKERRQ (ierr);
    // compute rhp_bar * c(1-c) * \alpha, where rho_bar = dR / dm
    ierr = tumor_->k_->applydRdm(tumor_->k_->temp_[1]);          CHKERRQ (ierr);
    // add
    ierr = VecAXPY (tumor_->k_->temp_[0], 1.0, tumor_->k_->temp_[1]); CHKERRQ (ierr);
    // numerical time integration using trapezoidal rule
    ierr = VecAXPY (q, integration_weight, tumor_->k_->temp_[0]);CHKERRQ (ierr);
    // use weight 1 for inner points
    if (j == 0) integration_weight *= 0.5;
  }

  self_exec_time += MPI_Wtime();
  accumulateTimers (t, t, self_exec_time); e.addTimings (t); e.stop ();
  PetscFunctionReturn (0);
}

PdeOperatorsRD::~PdeOperatorsRD () {
    PetscErrorCode ierr = 0;
    for (int i = 0; i < this->nt_; i++) {
        ierr = VecDestroy (&c_[i]);
        ierr = VecDestroy (&p_[i]);
    }
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
