#include "DerivativeOperators.h"
#include "Utils.h"
#include <petsc/private/vecimpl.h>

/* #### ------------------------------------------------------------------- #### */
/* #### ========         RESET (CHANGE SIZE OF WORK VECTORS)       ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperators::reset (Vec p, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc, std::shared_ptr<Tumor> tumor) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    // delete and re-create p vectors
    if (ptemp_ != nullptr)     {ierr = VecDestroy (&ptemp_);     CHKERRQ (ierr); ptemp_ = nullptr;}
    if (p_current_ != nullptr) {ierr = VecDestroy (&p_current_); CHKERRQ (ierr); p_current_ = nullptr;}
    ierr = VecDuplicate        (p, &ptemp_);                     CHKERRQ (ierr);
    ierr = VecDuplicate        (p, &p_current_);                 CHKERRQ (ierr);
    if (temp_ != nullptr)      {ierr = VecSet (temp_, 0.0);      CHKERRQ (ierr);}

    pde_operators_ = pde_operators;
    tumor_         = tumor;
    n_misc_        = n_misc;
    PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsPos::reset (Vec p, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc, std::shared_ptr<Tumor> tumor) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    // c(0) does not change size, just zero out
    if (temp_phip_ != nullptr)      {ierr = VecSet (temp_phip_, 0.0);      CHKERRQ (ierr);}
    if (temp_phiptilde_ != nullptr) {ierr = VecSet (temp_phiptilde_, 0.0); CHKERRQ (ierr);}
    // call base class reset function
    DerivativeOperators::reset(p, pde_operators, n_misc, tumor);
    PetscFunctionReturn (ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========          STANDARD REACTION DIFFUSION (MP)         ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsRD::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_obj_evals++;
    ScalarType *x_ptr, k1, k2, k3;

    int x_sz;

    
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      x->lock = 0;
    }
    #endif


    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    if (n_misc_->diffusivity_inversion_ || n_misc_->flag_reaction_inv_) {
      #ifndef SERIAL
        TU_assert(false, "Inversion for diffusivity only supported for serial p.");
      #endif
      ierr = VecGetArray (x, &x_ptr);                                       CHKERRQ (ierr);
      #ifdef POSITIVITY_DIFF_COEF
        //Positivity clipping in diffusio coefficient
        x_ptr[n_misc_->np_] = x_ptr[n_misc_->np_] > 0 ? x_ptr[n_misc_->np_] : 0;
        if (n_misc_->nk_ > 1)
          x_ptr[n_misc_->np_ + 1] = x_ptr[n_misc_->np_ + 1] > 0 ? x_ptr[n_misc_->np_ + 1] : 0;
        if (n_misc_->nk_ > 2)
          x_ptr[n_misc_->np_ + 2] = x_ptr[n_misc_->np_ + 2] > 0 ? x_ptr[n_misc_->np_ + 2] : 0;
      #endif
      k1 = x_ptr[n_misc_->np_];
      k2 = (n_misc_->nk_ > 1) ? x_ptr[n_misc_->np_ + 1] : 0;
      k3 = (n_misc_->nk_ > 2) ? x_ptr[n_misc_->np_ + 2] : 0;
      ierr = VecRestoreArray (x, &x_ptr);                                   CHKERRQ (ierr);
      ierr = tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, tumor_->mat_prop_, n_misc_);    CHKERRQ(ierr);
      // need to update prefactors for diffusion KSP preconditioner, as k changed
      pde_operators_->diff_solver_->precFactor();
    }

    ScalarType r1, r2, r3;
    if (n_misc_->flag_reaction_inv_) {
      ierr = VecGetArray(x, &x_ptr);                                        CHKERRQ(ierr);
      r1 = x_ptr[n_misc_->np_ + n_misc_->nk_];
      r2 = (n_misc_->nr_ > 1) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 1] : 0;
      r3 = (n_misc_->nr_ > 2) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 2] : 0;
      ierr = tumor_->rho_->updateIsotropicCoefficients (r1, r2, r3, tumor_->mat_prop_, n_misc_);
      ierr = VecRestoreArray(x, &x_ptr);                                    CHKERRQ(ierr);
    }

    std::stringstream s;
    if (n_misc_->verbosity_ >= 3) {
      if (n_misc_->diffusivity_inversion_ || n_misc_->flag_reaction_inv_) {
        s << " Diffusivity guess = (" << k1 << ", " << k2 << ", " << k3 << ")";
        ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
      }
      if (n_misc_->flag_reaction_inv_) {
        s << " Reaction  guess   = (" << r1 << ", " << r2 << ", " << r3 << ")";
        ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
      }
    }


    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);
    ierr = VecDot (temp_, temp_, J);                                CHKERRQ (ierr);

    /*Regularization term*/
    PetscReal reg = 0;
    if (n_misc_->regularization_norm_ == L1) {
      ierr = VecNorm (x, NORM_1, &reg);                             CHKERRQ (ierr);
      reg *= n_misc_->lambda_;
    } else if (n_misc_->regularization_norm_ == wL2) {
      ierr = VecPointwiseMult (ptemp_, tumor_->weights_, x);          CHKERRQ (ierr);
      ierr = VecDot (x, ptemp_, &reg);                                CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_;
    } else if (n_misc_->regularization_norm_ == L2){  //In tumor space, so scale norm by lebesque measure
      ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);             CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_;
      reg *= n_misc_->lebesgue_measure_;
    } else if (n_misc_->regularization_norm_ == L2b){
      // Reg term only on the initial condition. Leave out the diffusivity.
      ierr = VecGetArray (x, &x_ptr);                               CHKERRQ (ierr);
      for (int i = 0; i < n_misc_->np_; i++) {
        reg += x_ptr[i] * x_ptr[i];
      }
      ierr = VecRestoreArray (x, &x_ptr);                           CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_;
    }


    (*J) *= n_misc_->lebesgue_measure_;


    s << "  J(p) = Dc(c) + S(c0) = "<< std::setprecision(12) << 0.5*(*J)+reg <<" = " << std::setprecision(12)<< 0.5*(*J) <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();


    (*J) *= 0.5;
    (*J) += reg;

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      x->lock = lock_state;
    }
    #endif

    PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRD::evaluateGradient (Vec dJ, Vec x, Vec data){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ScalarType *x_ptr, *p_ptr;
    std::bitset<3> XYZ; XYZ[0] = 1; XYZ[1] = 1; XYZ[2] = 1;
    n_misc_->statistics_.nb_grad_evals++;
    Event e ("tumor-eval-grad");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    ScalarType k1, k2, k3;

    int x_sz;

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      x->lock = 0;
    }
    #endif

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    if (n_misc_->diffusivity_inversion_ || n_misc_->flag_reaction_inv_) {
      #ifndef SERIAL
        TU_assert(false, "Inversion for diffusivity only supported for serial p.");
      #endif
      ierr = VecGetArray (x, &x_ptr);                                       CHKERRQ (ierr);
      #ifdef POSITIVITY_DIFF_COEF
        //Positivity clipping in diffusio coefficient
        x_ptr[n_misc_->np_] = x_ptr[n_misc_->np_] > 0 ? x_ptr[n_misc_->np_] : 0;
        if (n_misc_->nk_ > 1)
          x_ptr[n_misc_->np_ + 1] = x_ptr[n_misc_->np_ + 1] > 0 ? x_ptr[n_misc_->np_ + 1] : 0;
        if (n_misc_->nk_ > 2)
          x_ptr[n_misc_->np_ + 2] = x_ptr[n_misc_->np_ + 2] > 0 ? x_ptr[n_misc_->np_ + 2] : 0;
      #endif
      k1 = x_ptr[n_misc_->np_];
      k2 = (n_misc_->nk_ > 1) ? x_ptr[n_misc_->np_ + 1] : 0;
      k3 = (n_misc_->nk_ > 2) ? x_ptr[n_misc_->np_ + 2] : 0;
      ierr = VecRestoreArray (x, &x_ptr);                                   CHKERRQ (ierr);
      ierr = tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, tumor_->mat_prop_, n_misc_);    CHKERRQ(ierr);
      // need to update prefactors for diffusion KSP preconditioner, as k changed
      pde_operators_->diff_solver_->precFactor();
    }

    ScalarType r1, r2, r3;
    if (n_misc_->flag_reaction_inv_) {
      ierr = VecGetArray(x, &x_ptr);                                        CHKERRQ(ierr);
      r1 = x_ptr[n_misc_->np_ + n_misc_->nk_];
      r2 = (n_misc_->nr_ > 1) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 1] : 0;
      r3 = (n_misc_->nr_ > 2) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 2] : 0;
      ierr = tumor_->rho_->updateIsotropicCoefficients (r1, r2, r3, tumor_->mat_prop_, n_misc_);
      ierr = VecRestoreArray(x, &x_ptr);                                    CHKERRQ(ierr);
    }

    /* ------------------ */
    /* (1) compute grad_p */
    // c = Phi(p), solve state
    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);
    // final cond adjoint
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);
    // solve adjoint
    ierr = pde_operators_->solveAdjoint (1);
    // compute gradient
    if (!n_misc_->phi_store_) {
      // restructure phi compute because it is now expensive
      // assume that reg norm is L2 for now
      // TODO: change to normal if reg norm is not L2

      // p0 = p0 - beta * phi * p
      ierr = VecAXPY (tumor_->p_0_, -n_misc_->beta_, tumor_->c_0_);   CHKERRQ (ierr);
      // dJ is phiT p0 - beta * phiT * phi * p
      ierr = tumor_->phi_->applyTranspose (dJ, tumor_->p_0_);        CHKERRQ (ierr);
      // dJ is beta * phiT * phi * p - phiT * p0
      ierr = VecScale (dJ, -n_misc_->lebesgue_measure_);                         CHKERRQ (ierr);

    } else {
      ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
      ierr = VecScale (ptemp_, n_misc_->lebesgue_measure_);           CHKERRQ (ierr);

      // Gradient according to reg parameter chosen
      if (n_misc_->regularization_norm_ == L1) {
        ierr = VecCopy (ptemp_, dJ);                                  CHKERRQ (ierr);
        ierr = VecScale (dJ, -1.0);                                   CHKERRQ (ierr);
      } else if (n_misc_->regularization_norm_ == wL2) {
        ierr = VecPointwiseMult (dJ, tumor_->weights_, x);              CHKERRQ (ierr);
        ierr = VecScale (dJ, n_misc_->beta_);                           CHKERRQ (ierr);
        ierr = VecAXPY (dJ, -1.0, ptemp_);                              CHKERRQ (ierr);
      } else if (n_misc_->regularization_norm_ == L2){
        ierr = tumor_->phi_->applyTranspose (dJ, tumor_->c_0_);
        ierr = VecScale (dJ, n_misc_->beta_ * n_misc_->lebesgue_measure_);                         CHKERRQ (ierr);
        ierr = VecAXPY (dJ, -1.0, ptemp_);                            CHKERRQ (ierr);
      } else if (n_misc_->regularization_norm_ == L2b){
        ierr = VecCopy (x, dJ);                                       CHKERRQ (ierr);
        ierr = VecScale (dJ, n_misc_->beta_);                         CHKERRQ (ierr);
        ierr = VecAXPY (dJ, -1.0, ptemp_);                            CHKERRQ (ierr);
      }
    }

    ScalarType temp_scalar;
    /* ------------------------- */
    /* INVERSION FOR DIFFUSIVITY */
    /* ------------------------- */
    /* (2) compute grad_k   int_T int_Omega { m_i * (grad c)^T grad alpha } dx dt */
    ScalarType integration_weight = 1.0;
    if (n_misc_->diffusivity_inversion_ || n_misc_->flag_reaction_inv_) {
      ierr = VecSet(temp_, 0.0);                                      CHKERRQ (ierr);
      // compute numerical time integration using trapezoidal rule
      for (int i = 0; i < n_misc_->nt_ + 1; i++) {
        // integration weight for chain trapezoidal rule
        if (i == 0 || i == n_misc_->nt_) integration_weight = 0.5;
        else integration_weight = 1.0;

        // compute x = (grad c)^T grad \alpha
        // compute gradient of state variable c(t)
        pde_operators_->spec_ops_->computeGradient (tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], &XYZ, t.data());
        // compute gradient of adjoint variable p(t)
        pde_operators_->spec_ops_->computeGradient (tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], &XYZ, t.data());
        // scalar product (grad c)^T grad \alpha
        ierr = VecPointwiseMult (tumor_->work_[0], tumor_->work_[1], tumor_->work_[4]);  CHKERRQ (ierr);  // c_x * \alpha_x
        ierr = VecPointwiseMult (tumor_->work_[1], tumor_->work_[2], tumor_->work_[5]);  CHKERRQ (ierr);  // c_y * \alpha_y
        ierr = VecAXPY (tumor_->work_[0], 1.0,  tumor_->work_[1]);                       CHKERRQ (ierr);
        ierr = VecPointwiseMult (tumor_->work_[1], tumor_->work_[3], tumor_->work_[6]);  CHKERRQ (ierr);  // c_z * \alpha_z
        ierr = VecAXPY (tumor_->work_[0], 1.0,  tumor_->work_[1]);                       CHKERRQ (ierr);  // result in tumor_->work_[0]

        // numerical time integration using trapezoidal rule
        ierr = VecAXPY (temp_, n_misc_->dt_ * integration_weight, tumor_->work_[0]);     CHKERRQ (ierr);
      }
      // time integration of [ int_0 (grad c)^T grad alpha dt ] done, result in temp_
      // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
      ierr = VecGetArray(dJ, &x_ptr);                                                  CHKERRQ (ierr);
      ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[n_misc_->np_]);              CHKERRQ(ierr);
      x_ptr[n_misc_->np_] *= n_misc_->lebesgue_measure_;

      if (n_misc_->nk_ == 1) {
        // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
        // other tissue types using n_misc - Hence, the gradient will change accordingly.
        // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar);              CHKERRQ(ierr);
        temp_scalar *= n_misc_->lebesgue_measure_;
        temp_scalar *= n_misc_->k_gm_wm_ratio_;    // this ratio will control the diffusivity in gm
        x_ptr[n_misc_->np_] += temp_scalar;
      }

      if (n_misc_->nk_ > 1) {
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[n_misc_->np_ + 1]);        CHKERRQ(ierr);
        x_ptr[n_misc_->np_ + 1] *= n_misc_->lebesgue_measure_;
      }
      if (n_misc_->nk_ > 2) {
        ierr = VecDot(tumor_->mat_prop_->glm_, temp_, &x_ptr[n_misc_->np_ + 2]);       CHKERRQ(ierr);
        x_ptr[n_misc_->np_ + 2] *= n_misc_->lebesgue_measure_;
      }
      ierr = VecRestoreArray(dJ, &x_ptr);                                              CHKERRQ (ierr);
    }

    /* INVERSION FOR REACTION COEFFICIENT */
    integration_weight = 1.0;
     if (n_misc_->flag_reaction_inv_) {
      ierr = VecSet(temp_, 0.0);                                      CHKERRQ (ierr);
      // compute numerical time integration using trapezoidal rule
      for (int i = 0; i < n_misc_->nt_ + 1; i++) {
        // integration weight for chain trapezoidal rule
        if (i == 0 || i == n_misc_->nt_) integration_weight = 0.5;
        else integration_weight = 1.0;

        ierr = VecPointwiseMult (tumor_->work_[0], pde_operators_->c_[i], pde_operators_->c_[i]);  CHKERRQ (ierr); // work is c*c
        ierr = VecAXPY (tumor_->work_[0], -1.0, pde_operators_->c_[i]);                            CHKERRQ (ierr); // work is c*c - c
        ierr = VecPointwiseMult (tumor_->work_[0], pde_operators_->p_[i], tumor_->work_[0]);       CHKERRQ (ierr); // work is a * (c*c - c)

        // numerical time integration using trapezoidal rule
        ierr = VecAXPY (temp_, n_misc_->dt_ * integration_weight, tumor_->work_[0]);     CHKERRQ (ierr);
      }
      // time integration of [ int_0 (grad c)^T grad alpha dt ] done, result in temp_
      // integration over omega (i.e., inner product, as periodic boundary)

      ierr = VecGetArray(dJ, &x_ptr);                                                                  CHKERRQ (ierr);
      ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[n_misc_->np_ +  n_misc_->nk_]);              CHKERRQ(ierr);
      x_ptr[n_misc_->np_ +  n_misc_->nk_] *= n_misc_->lebesgue_measure_;

      if (n_misc_->nr_ == 1) {
        // Inverting for only one parameters a.k.a reaction in WM. Provide user with the option of setting a reaction for
        // other tissue types using n_misc - Hence, the gradient will change accordingly.
        // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar);              CHKERRQ(ierr);
        temp_scalar *= n_misc_->lebesgue_measure_;
        temp_scalar *= n_misc_->r_gm_wm_ratio_;    // this ratio will control the reaction coefficient in gm
        x_ptr[n_misc_->np_ +  n_misc_->nk_] += temp_scalar;
      }

      if (n_misc_->nr_ > 1) {
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[n_misc_->np_ +  n_misc_->nk_ + 1]);              CHKERRQ(ierr);
        x_ptr[n_misc_->np_ +  n_misc_->nk_ + 1] *= n_misc_->lebesgue_measure_;
      }

      if (n_misc_->nr_ > 2) {
        ierr = VecDot(tumor_->mat_prop_->glm_, temp_, &x_ptr[n_misc_->np_ +  n_misc_->nk_ + 2]);              CHKERRQ(ierr);
        x_ptr[n_misc_->np_ +  n_misc_->nk_ + 2] *= n_misc_->lebesgue_measure_;
      }

      ierr = VecRestoreArray(dJ, &x_ptr);                                              CHKERRQ (ierr);
    }


    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      x->lock = lock_state;
    }
    #endif

    // timing
    self_exec_time += MPI_Wtime(); t[5] = self_exec_time; e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}

// saves on forward solve
PetscErrorCode DerivativeOperatorsRD::evaluateObjectiveAndGradient (PetscReal *J, Vec dJ, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_obj_evals++;
    n_misc_->statistics_.nb_grad_evals++;
    std::bitset<3> XYZ; XYZ[0] = 1; XYZ[1] = 1; XYZ[2] = 1;
    Event e ("tumor-eval-objandgrad");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    ScalarType *x_ptr, k1, k2, k3;

    int x_sz;

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      x->lock = 0;
    }
    #endif

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);


    if (n_misc_->diffusivity_inversion_ || n_misc_->flag_reaction_inv_) { // if solveForParameters is happening always invert for diffusivity
      #ifndef SERIAL
        TU_assert(false, "Inversion for diffusivity only supported for serial p.");
      #endif
      ierr = VecGetArray (x, &x_ptr);                                       CHKERRQ (ierr);
      #ifdef POSITIVITY_DIFF_COEF
        //Positivity clipping in diffusio coefficient
        x_ptr[n_misc_->np_] = x_ptr[n_misc_->np_] > 0 ? x_ptr[n_misc_->np_] : 0;
        if (n_misc_->nk_ > 1)
          x_ptr[n_misc_->np_ + 1] = x_ptr[n_misc_->np_ + 1] > 0 ? x_ptr[n_misc_->np_ + 1] : 0;
        if (n_misc_->nk_ > 2)
          x_ptr[n_misc_->np_ + 2] = x_ptr[n_misc_->np_ + 2] > 0 ? x_ptr[n_misc_->np_ + 2] : 0;
      #endif
      k1 = x_ptr[n_misc_->np_];
      k2 = (n_misc_->nk_ > 1) ? x_ptr[n_misc_->np_ + 1] : 0;
      k3 = (n_misc_->nk_ > 2) ? x_ptr[n_misc_->np_ + 2] : 0;
      ierr = VecRestoreArray (x, &x_ptr);                                   CHKERRQ (ierr);
      ierr = tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, tumor_->mat_prop_, n_misc_);    CHKERRQ(ierr);
      // need to update prefactors for diffusion KSP preconditioner, as k changed
      pde_operators_->diff_solver_->precFactor();
    }

    ScalarType r1, r2, r3;
    if (n_misc_->flag_reaction_inv_) {
      ierr = VecGetArray(x, &x_ptr);                                        CHKERRQ(ierr);
      r1 = x_ptr[n_misc_->np_ + n_misc_->nk_];
      r2 = (n_misc_->nr_ > 1) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 1] : 0;
      r3 = (n_misc_->nr_ > 2) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 2] : 0;
      ierr = tumor_->rho_->updateIsotropicCoefficients (r1, r2, r3, tumor_->mat_prop_, n_misc_);
      ierr = VecRestoreArray(x, &x_ptr);                                    CHKERRQ(ierr);
    }

    std::stringstream s;
    if (n_misc_->verbosity_ >= 3) {
      if (n_misc_->diffusivity_inversion_ || n_misc_->flag_reaction_inv_) {
        s << " Diffusivity guess = (" << k1 << ", " << k2 << ", " << k3 << ")";
        ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
      }
      if (n_misc_->flag_reaction_inv_) {
        s << " Reaction  guess   = (" << r1 << ", " << r2 << ", " << r3 << ")";
        ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
      }
    }

    // solve state
    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);

    // c(1) - d
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);
    // mismatch, squared residual norm
    ierr = VecDot (temp_, temp_, J);                                CHKERRQ (ierr);
    // solve adjoint
    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);
    ierr = pde_operators_->solveAdjoint (1);

    if (!n_misc_->phi_store_) {
      // restructure phi compute because it is now expensive
      // assume that reg norm is L2 for now
      // TODO: change to normal if reg norm is not L2

      // p0 = p0 - beta * phi * p
      ierr = VecAXPY (tumor_->p_0_, -n_misc_->beta_, tumor_->c_0_);   CHKERRQ (ierr);
      // dJ is phiT p0 - beta * phiT * phi * p
      ierr = tumor_->phi_->applyTranspose (dJ, tumor_->p_0_);        CHKERRQ (ierr);
      // dJ is beta * phiT * phi * p - phiT * p0
      ierr = VecScale (dJ, -n_misc_->lebesgue_measure_);                         CHKERRQ (ierr);

    } else {
      ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
      ierr = VecScale (ptemp_, n_misc_->lebesgue_measure_);           CHKERRQ (ierr);

      // Gradient according to reg parameter chosen
      if (n_misc_->regularization_norm_ == L1) {
        ierr = VecCopy (ptemp_, dJ);                                  CHKERRQ (ierr);
        ierr = VecScale (dJ, -1.0);                                   CHKERRQ (ierr);
      } else if (n_misc_->regularization_norm_ == wL2) {
        ierr = VecPointwiseMult (dJ, tumor_->weights_, x);              CHKERRQ (ierr);
        ierr = VecScale (dJ, n_misc_->beta_);                           CHKERRQ (ierr);
        ierr = VecAXPY (dJ, -1.0, ptemp_);                              CHKERRQ (ierr);
      } else if (n_misc_->regularization_norm_ == L2){
        ierr = tumor_->phi_->applyTranspose (dJ, tumor_->c_0_);
        ierr = VecScale (dJ, n_misc_->beta_ * n_misc_->lebesgue_measure_);                         CHKERRQ (ierr);
        ierr = VecAXPY (dJ, -1.0, ptemp_);                            CHKERRQ (ierr);
      } else if (n_misc_->regularization_norm_ == L2b){
        ierr = VecCopy (x, dJ);                                       CHKERRQ (ierr);
        ierr = VecScale (dJ, n_misc_->beta_);                         CHKERRQ (ierr);
        ierr = VecAXPY (dJ, -1.0, ptemp_);                            CHKERRQ (ierr);
      }
    }

    // regularization
    PetscReal reg;
    if (n_misc_->regularization_norm_ == L1) {
      ierr = VecNorm (x, NORM_1, &reg);                             CHKERRQ (ierr);
      reg *= n_misc_->lambda_;
    } else if (n_misc_->regularization_norm_ == wL2) {
      ierr = VecPointwiseMult (ptemp_, tumor_->weights_, x);          CHKERRQ (ierr);
      ierr = VecDot (x, ptemp_, &reg);                                CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_;
    } else if (n_misc_->regularization_norm_ == L2){
      ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);             CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_;
      reg *= n_misc_->lebesgue_measure_;
    } else if (n_misc_->regularization_norm_ == L2b){
      // Reg term only on the initial condition. Leave out the diffusivity.
      ierr = VecGetArray (x, &x_ptr);                               CHKERRQ (ierr);
      for (int i = 0; i < n_misc_->np_; i++) {
        reg += x_ptr[i] * x_ptr[i];
      }
      ierr = VecRestoreArray (x, &x_ptr);                           CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_;
    }


    (*J) *= n_misc_->lebesgue_measure_;

    s << "  J(p) = Dc(c) + S(c0) = "<< std::setprecision(12) << 0.5*(*J)+reg <<" = " << std::setprecision(12)<< 0.5*(*J) <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    // objective function value
    (*J) *= 0.5;
    (*J) += reg;



    ScalarType temp_scalar;
    /* ------------------------- */
    /* INVERSION FOR DIFFUSIVITY */
    /* ------------------------- */
    /* (2) compute grad_k   int_T int_Omega { m_i * (grad c)^T grad alpha } dx dt */
    ScalarType integration_weight = 1.0;
    if (n_misc_->diffusivity_inversion_ || n_misc_->flag_reaction_inv_) {
      ierr = VecSet(temp_, 0.0);                                      CHKERRQ (ierr);
      // compute numerical time integration using trapezoidal rule
      for (int i = 0; i < n_misc_->nt_ + 1; i++) {
        // integration weight for chain trapezoidal rule
        if (i == 0 || i == n_misc_->nt_) integration_weight = 0.5;
        else integration_weight = 1.0;

        // compute x = (grad c)^T grad \alpha
        // compute gradient of state variable c(t)
        pde_operators_->spec_ops_->computeGradient (tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], &XYZ, t.data());
        // compute gradient of adjoint variable p(t)
        pde_operators_->spec_ops_->computeGradient (tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], &XYZ, t.data());
        // scalar product (grad c)^T grad \alpha
        ierr = VecPointwiseMult (tumor_->work_[0], tumor_->work_[1], tumor_->work_[4]);  CHKERRQ (ierr);  // c_x * \alpha_x
        ierr = VecPointwiseMult (tumor_->work_[1], tumor_->work_[2], tumor_->work_[5]);  CHKERRQ (ierr);  // c_y * \alpha_y
        ierr = VecAXPY (tumor_->work_[0], 1.0,  tumor_->work_[1]);                       CHKERRQ (ierr);
        ierr = VecPointwiseMult (tumor_->work_[1], tumor_->work_[3], tumor_->work_[6]);  CHKERRQ (ierr);  // c_z * \alpha_z
        ierr = VecAXPY (tumor_->work_[0], 1.0,  tumor_->work_[1]);                       CHKERRQ (ierr);  // result in tumor_->work_[0]

        // numerical time integration using trapezoidal rule
        ierr = VecAXPY (temp_, n_misc_->dt_ * integration_weight, tumor_->work_[0]);     CHKERRQ (ierr);
      }
      // time integration of [ int_0 (grad c)^T grad alpha dt ] done, result in temp_
      // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
      ierr = VecGetArray(dJ, &x_ptr);                                                  CHKERRQ (ierr);
      ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[n_misc_->np_]);              CHKERRQ(ierr);
      x_ptr[n_misc_->np_] *= n_misc_->lebesgue_measure_;

      if (n_misc_->nk_ == 1) {
        // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
        // other tissue types using n_misc - Hence, the gradient will change accordingly.
        // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar);              CHKERRQ(ierr);
        temp_scalar *= n_misc_->lebesgue_measure_;
        temp_scalar *= n_misc_->k_gm_wm_ratio_;    // this ratio will control the diffusivity in gm
        x_ptr[n_misc_->np_] += temp_scalar;
      }

      if (n_misc_->nk_ > 1) {
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[n_misc_->np_ + 1]);        CHKERRQ(ierr);
        x_ptr[n_misc_->np_ + 1] *= n_misc_->lebesgue_measure_;
      }
      if (n_misc_->nk_ > 2) {
        ierr = VecDot(tumor_->mat_prop_->glm_, temp_, &x_ptr[n_misc_->np_ + 2]);       CHKERRQ(ierr);
        x_ptr[n_misc_->np_ + 2] *= n_misc_->lebesgue_measure_;
      }
      ierr = VecRestoreArray(dJ, &x_ptr);                                              CHKERRQ (ierr);
    }

    /* INVERSION FOR REACTION COEFFICIENT */
    integration_weight = 1.0;
    if (n_misc_->flag_reaction_inv_) {
      ierr = VecSet(temp_, 0.0);                                      CHKERRQ (ierr);
      // compute numerical time integration using trapezoidal rule
      for (int i = 0; i < n_misc_->nt_ + 1; i++) {
        // integration weight for chain trapezoidal rule
        if (i == 0 || i == n_misc_->nt_) integration_weight = 0.5;
        else integration_weight = 1.0;

        ierr = VecPointwiseMult (tumor_->work_[0], pde_operators_->c_[i], pde_operators_->c_[i]);  CHKERRQ (ierr); // work is c*c
        ierr = VecAXPY (tumor_->work_[0], -1.0, pde_operators_->c_[i]);                            CHKERRQ (ierr); // work is c*c - c
        ierr = VecPointwiseMult (tumor_->work_[0], pde_operators_->p_[i], tumor_->work_[0]);       CHKERRQ (ierr); // work is a * (c*c - c)

        // numerical time integration using trapezoidal rule
        ierr = VecAXPY (temp_, n_misc_->dt_ * integration_weight, tumor_->work_[0]);     CHKERRQ (ierr);
      }
      // time integration of [ int_0 (grad c)^T grad alpha dt ] done, result in temp_
      // integration over omega (i.e., inner product, as periodic boundary)

      ierr = VecGetArray(dJ, &x_ptr);                                                                  CHKERRQ (ierr);
      ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[n_misc_->np_ +  n_misc_->nk_]);              CHKERRQ(ierr);
      x_ptr[n_misc_->np_ +  n_misc_->nk_] *= n_misc_->lebesgue_measure_;

      if (n_misc_->nr_ == 1) {
        // Inverting for only one parameters a.k.a reaction in WM. Provide user with the option of setting a reaction for
        // other tissue types using n_misc - Hence, the gradient will change accordingly.
        // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar);              CHKERRQ(ierr);
        temp_scalar *= n_misc_->lebesgue_measure_;
        temp_scalar *= n_misc_->r_gm_wm_ratio_;    // this ratio will control the reaction coefficient in gm
        x_ptr[n_misc_->np_ +  n_misc_->nk_] += temp_scalar;
      }

      if (n_misc_->nr_ > 1) {
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[n_misc_->np_ +  n_misc_->nk_ + 1]);              CHKERRQ(ierr);
        x_ptr[n_misc_->np_ +  n_misc_->nk_ + 1] *= n_misc_->lebesgue_measure_;
      }

      if (n_misc_->nr_ > 2) {
        ierr = VecDot(tumor_->mat_prop_->glm_, temp_, &x_ptr[n_misc_->np_ +  n_misc_->nk_ + 2]);              CHKERRQ(ierr);
        x_ptr[n_misc_->np_ +  n_misc_->nk_ + 2] *= n_misc_->lebesgue_measure_;
      }

      ierr = VecRestoreArray(dJ, &x_ptr);                                              CHKERRQ (ierr);
    }

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      x->lock = lock_state;
    }
    #endif

    // timing
    self_exec_time += MPI_Wtime(); t[5] = self_exec_time; e.addTimings (t); e.stop ();


    PetscFunctionReturn (ierr);

    // PetscFunctionBegin;
    // PetscErrorCode ierr = 0;
    // n_misc_->statistics_.nb_obj_evals++;
    // n_misc_->statistics_.nb_grad_evals++;
    // ierr = evaluateObjective (J, x, data);                        CHKERRQ(ierr);
    // ierr = evaluateGradient (dJ, x, data);                        CHKERRQ(ierr);
    // PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRD::evaluateHessian (Vec y, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_hessian_evals++;

    std::bitset<3> XYZ; XYZ[0] = 1; XYZ[1] = 1; XYZ[2] = 1;
    Event e ("tumor-eval-hessian");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    ScalarType *y_ptr;

    if (n_misc_->diffusivity_inversion_) {
      /* HESSIAN WITH DIFFUSIVITY INVERSION
        Hx = [Hpp p_tilde + Hpk k_tilde; Hkp p_tiilde + Hkk k_tilde]
        Each Matvec is computed separately by eliminating the
        incremental forward and adjoint equations and the result is added into y = Hx
      */
      //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
      // --------------- Compute Hpp * p_tilde -------------------
      // Solve incr fwd with k_tilde = 0 and c0_tilde = \phi * p_tilde
      ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
      ierr = pde_operators_->solveState (1);
      // Solve incr adj with alpha1_tilde = -OT * O * c1_tilde
      ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
      ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
      ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);
      ierr = pde_operators_->solveAdjoint (2);
      // Matvec is \beta\phiT\phi p_tilde - \phiT \alpha0_tilde
      ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
      ierr = tumor_->phi_->applyTranspose (y, tumor_->c_0_);          CHKERRQ (ierr);
      ierr = VecScale (y, n_misc_->beta_);                            CHKERRQ (ierr);
      ierr = VecAXPY (y, -1.0, ptemp_);                               CHKERRQ (ierr);
      ierr = VecScale (y, n_misc_->lebesgue_measure_);                CHKERRQ (ierr);

      //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
      // --------------- Compute Hkp * p_tilde -- \int \int m_i \grad c . \grad \alpha_tilde -------------------
      ScalarType integration_weight = 1.0;
      ScalarType temp_scalar = 0.;
      ierr = VecSet(temp_, 0.0);                                      CHKERRQ (ierr);
      // compute numerical time integration using trapezoidal rule
      for (int i = 0; i < n_misc_->nt_ + 1; i++) {
        // integration weight for chain trapezoidal rule
        if (i == 0 || i == n_misc_->nt_) integration_weight = 0.5;
        else integration_weight = 1.0;

        // compute x = (grad c)^T grad \alpha_tilde
        // compute gradient of c(t)
        pde_operators_->spec_ops_->computeGradient (tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], &XYZ, t.data());
        // compute gradient of \alpha_tilde(t)
        pde_operators_->spec_ops_->computeGradient (tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], &XYZ, t.data());
        // scalar product (grad c)^T grad \alpha_tilde
        ierr = VecPointwiseMult (tumor_->work_[0], tumor_->work_[1], tumor_->work_[4]);  CHKERRQ (ierr);  // c_x * \alpha_x
        ierr = VecPointwiseMult (tumor_->work_[1], tumor_->work_[2], tumor_->work_[5]);  CHKERRQ (ierr);  // c_y * \alpha_y
        ierr = VecAXPY (tumor_->work_[0], 1.0,  tumor_->work_[1]);                       CHKERRQ (ierr);
        ierr = VecPointwiseMult (tumor_->work_[1], tumor_->work_[3], tumor_->work_[6]);  CHKERRQ (ierr);  // c_z * \alpha_z
        ierr = VecAXPY (tumor_->work_[0], 1.0,  tumor_->work_[1]);                       CHKERRQ (ierr);  // result in tumor_->work_[0]

        // numerical time integration using trapezoidal rule
        ierr = VecAXPY (temp_, n_misc_->dt_ * integration_weight, tumor_->work_[0]);     CHKERRQ (ierr);
      }
      // time integration of [ int_0 (grad c)^T grad alpha_tilde dt ] done, result in temp_
      // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
      ierr = VecGetArray(y, &y_ptr);                                                  CHKERRQ (ierr);
      ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &y_ptr[n_misc_->np_]);              CHKERRQ(ierr);
      y_ptr[n_misc_->np_] *= n_misc_->lebesgue_measure_;

      if (n_misc_->nk_ == 1) {
        // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
        // other tissue types using n_misc - Hence, the gradient will change accordingly.
        // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar);              CHKERRQ(ierr);
        temp_scalar *= n_misc_->lebesgue_measure_;
        temp_scalar *= n_misc_->k_gm_wm_ratio_;    // this ratio will control the diffusivity in gm
        y_ptr[n_misc_->np_] += temp_scalar;
      }

      if (n_misc_->nk_ > 1) {
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &y_ptr[n_misc_->np_ + 1]);        CHKERRQ(ierr);
        y_ptr[n_misc_->np_ + 1] *= n_misc_->lebesgue_measure_;
      }
      if (n_misc_->nk_ > 2) {
        ierr = VecDot(tumor_->mat_prop_->glm_, temp_, &y_ptr[n_misc_->np_ + 2]);       CHKERRQ(ierr);
        y_ptr[n_misc_->np_ + 2] *= n_misc_->lebesgue_measure_;
      }
      ierr = VecRestoreArray(y, &y_ptr);                                              CHKERRQ (ierr);

      //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
      // --------------- Compute Hpk * k_tilde -- -\phiT \alpha0_tilde -------------------
      // Set c0_tilde to zero
      ierr = VecSet (tumor_->c_0_, 0.);                               CHKERRQ (ierr);
      // solve tumor incr fwd with k_tilde
      // get the update on kappa -- this is used in tandem with the actual kappa in
      // the incr fwd solves and hence we cannot re-use the diffusivity vectors
      // TODO: here, it is assumed that the update is isotropic updates- this has
      // to be modified later is anisotropy is included
      ScalarType k1, k2, k3;
      ierr = VecGetArray (x, &y_ptr);
      k1 = y_ptr[n_misc_->np_];
      k2 = (n_misc_->nk_ > 1) ? y_ptr[n_misc_->np_ + 1] : 0.;
      k3 = (n_misc_->nk_ > 2) ? y_ptr[n_misc_->np_ + 2] : 0.;
      ierr = tumor_->k_->setSecondaryCoefficients (k1, k2, k3, tumor_->mat_prop_, n_misc_);                    CHKERRQ(ierr);
      ierr = VecRestoreArray (x, &y_ptr);

      ierr = pde_operators_->solveState (2);                          CHKERRQ (ierr);
      // Solve incr adj with alpha1_tilde = -OT * O * c1_tilde
      ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
      ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
      ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);
      ierr = pde_operators_->solveAdjoint (2);
      // Matvec is  - \phiT \alpha0_tilde
      ierr = VecSet (ptemp_, 0.);                                     CHKERRQ (ierr);
      ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);     CHKERRQ (ierr);
      ierr = VecScale (ptemp_, -n_misc_->lebesgue_measure_);          CHKERRQ (ierr);
      // Add Hpk k_tilde to Hpp p_tilde:  Note the kappa/rho components are zero
      // so are unchanged in y
      ierr = VecAXPY (y, 1.0, ptemp_);                                CHKERRQ (ierr);


      //  -------------------  -------------------  -------------------  -------------------  -------------------  -------------------
      // --------------- Compute Hkk * k_tilde -- \int \int mi \grad c \grad \alpha_tilde -------------------
      integration_weight = 1.0;
      ierr = VecSet(temp_, 0.0);                                      CHKERRQ (ierr);
      // compute numerical time integration using trapezoidal rule
      for (int i = 0; i < n_misc_->nt_ + 1; i++) {
        // integration weight for chain trapezoidal rule
        if (i == 0 || i == n_misc_->nt_) integration_weight = 0.5;
        else integration_weight = 1.0;

        // compute x = (grad c)^T grad \alpha_tilde
        // compute gradient of c(t)
        pde_operators_->spec_ops_->computeGradient (tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], &XYZ, t.data());
        // compute gradient of \alpha_tilde(t)
        pde_operators_->spec_ops_->computeGradient (tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], &XYZ, t.data());
        // scalar product (grad c)^T grad \alpha_tilde
        ierr = VecPointwiseMult (tumor_->work_[0], tumor_->work_[1], tumor_->work_[4]);  CHKERRQ (ierr);  // c_x * \alpha_x
        ierr = VecPointwiseMult (tumor_->work_[1], tumor_->work_[2], tumor_->work_[5]);  CHKERRQ (ierr);  // c_y * \alpha_y
        ierr = VecAXPY (tumor_->work_[0], 1.0,  tumor_->work_[1]);                       CHKERRQ (ierr);
        ierr = VecPointwiseMult (tumor_->work_[1], tumor_->work_[3], tumor_->work_[6]);  CHKERRQ (ierr);  // c_z * \alpha_z
        ierr = VecAXPY (tumor_->work_[0], 1.0,  tumor_->work_[1]);                       CHKERRQ (ierr);  // result in tumor_->work_[0]

        // numerical time integration using trapezoidal rule
        ierr = VecAXPY (temp_, n_misc_->dt_ * integration_weight, tumor_->work_[0]);     CHKERRQ (ierr);
      }
      // time integration of [ int_0 (grad c)^T grad alpha_tilde dt ] done, result in temp_
      // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
      ierr = VecGetArray(y, &y_ptr);                                                  CHKERRQ (ierr);
      ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &temp_scalar);                          CHKERRQ (ierr);
      temp_scalar *= n_misc_->lebesgue_measure_;
      y_ptr[n_misc_->np_] += temp_scalar;

      if (n_misc_->nk_ == 1) {
        // Inverting for only one parameters a.k.a diffusivity in WM. Provide user with the option of setting a diffusivity for
        // other tissue types using n_misc - Hence, the gradient will change accordingly.
        // Implicitly assuming there's no glm. TODO: remove glm from all subsequent iterations of the solver.
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar);            CHKERRQ (ierr);
        temp_scalar *= n_misc_->lebesgue_measure_;
        temp_scalar *= n_misc_->k_gm_wm_ratio_;    // this ratio will control the diffusivity in gm
        y_ptr[n_misc_->np_] += temp_scalar;
      }

      if (n_misc_->nk_ > 1) {
        ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &temp_scalar);            CHKERRQ (ierr);
        temp_scalar *= n_misc_->lebesgue_measure_;
        y_ptr[n_misc_->np_ + 1] += temp_scalar;
      }
      if (n_misc_->nk_ > 2) {
        ierr = VecDot(tumor_->mat_prop_->glm_, temp_, &temp_scalar);           CHKERRQ (ierr);
        temp_scalar *= n_misc_->lebesgue_measure_;
        y_ptr[n_misc_->np_ + 2] += temp_scalar;
      }
      ierr = VecRestoreArray(y, &y_ptr);                                       CHKERRQ (ierr);
    }
    else {
      ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
      ierr = pde_operators_->solveState (1);

      ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
      ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
      ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);

      ierr = pde_operators_->solveAdjoint (2);

      ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
      ierr = VecScale (ptemp_, n_misc_->lebesgue_measure_);           CHKERRQ (ierr);

      //No hessian info for L1 for now
      if (n_misc_->regularization_norm_ == wL2) {
        ierr = VecPointwiseMult (y, tumor_->weights_, x);               CHKERRQ (ierr);
        ierr = VecScale (y, n_misc_->beta_);                            CHKERRQ (ierr);
        ierr = VecAXPY (y, -1.0, ptemp_);                               CHKERRQ (ierr);
      } else if (n_misc_->regularization_norm_ == L2b){
        ierr = VecCopy (x, y);                                       CHKERRQ (ierr);
        ierr = VecScale (y, n_misc_->beta_);                         CHKERRQ (ierr);
        ierr = VecAXPY (y, -1.0, ptemp_);                            CHKERRQ (ierr);
      } else {
        ierr = tumor_->phi_->applyTranspose (y, tumor_->c_0_);
        ierr = VecScale (y, n_misc_->beta_ * n_misc_->lebesgue_measure_);                            CHKERRQ (ierr);
        ierr = VecAXPY (y, -1.0, ptemp_);                               CHKERRQ (ierr);
      }
    }
    self_exec_time += MPI_Wtime(); t[5] = self_exec_time; e.addTimings (t); e.stop ();

    PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRD::evaluateConstantHessianApproximation  (Vec y, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = tumor_->phi_->applyTranspose (y, tumor_->c_0_);          CHKERRQ (ierr);
    ierr = VecScale (y, n_misc_->beta_);                            CHKERRQ (ierr);
    PetscFunctionReturn (ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========       POSITIVITY/SIGMOID PARAMETRIZATION          ======== #### */
/* #### ------------------------------------------------------------------- #### */

PetscErrorCode DerivativeOperatorsPos::sigmoid (Vec temp, Vec input) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = VecCopy (input, temp);                                  CHKERRQ (ierr);

    ierr = VecScale (temp, -1.0);                                  CHKERRQ (ierr);
    ierr = VecShift (temp, n_misc_->exp_shift_);                   CHKERRQ (ierr);
    ierr = VecExp (temp);                                          CHKERRQ (ierr);
    ierr = VecShift (temp, 1.0);                                   CHKERRQ (ierr);
    ierr = VecReciprocal (temp);                                   CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}


PetscErrorCode DerivativeOperatorsPos::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_obj_evals++;

    ierr = tumor_->phi_->apply (temp_phip_, x);                     CHKERRQ (ierr);
    ierr = sigmoid (tumor_->c_0_, temp_phip_);
    ierr = pde_operators_->solveState (0);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);
    ierr = VecDot (temp_, temp_, J);                                CHKERRQ (ierr);

    PetscReal reg;
    ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);               CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->beta_;

    (*J) *= 0.5;
    (*J) += reg;

    ierr = VecDot (x, x, &reg);                                     CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->penalty_;

    (*J) += reg;

    PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsPos::evaluateGradient (Vec dJ, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_grad_evals++;

    ierr = tumor_->phi_->apply (temp_phip_, x);                     CHKERRQ (ierr);
    ierr = sigmoid (tumor_->c_0_, temp_phip_);
    ierr = pde_operators_->solveState (0);


    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);

    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);

    ierr = pde_operators_->solveAdjoint (1);

    ierr = VecCopy (temp_phip_, temp_);                             CHKERRQ (ierr);
    ScalarType *temp_ptr, *p_ptr;
    ierr = VecGetArray (temp_, &temp_ptr);                          CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        temp_ptr[i] = (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-temp_ptr[i] + n_misc_->exp_shift_);

        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }
    }
    ierr = VecRestoreArray (temp_, &temp_ptr);                      CHKERRQ (ierr);

    ierr = tumor_->phi_->applyTranspose (ptemp_, temp_);
    ierr = VecScale (ptemp_, n_misc_->beta_);                       CHKERRQ (ierr);

    ierr = VecCopy (ptemp_, dJ);                                    CHKERRQ (ierr);

    ierr = VecCopy (temp_phip_, temp_);                             CHKERRQ (ierr);
    ierr = VecGetArray (temp_, &temp_ptr);                          CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->p_0_, &p_ptr);                      CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        temp_ptr[i] = p_ptr[i] *
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-temp_ptr[i] + n_misc_->exp_shift_);
        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }
    }


    ierr = VecRestoreArray (temp_, &temp_ptr);                      CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->p_0_, &p_ptr);                  CHKERRQ (ierr);

    ierr = tumor_->phi_->applyTranspose (ptemp_, temp_);
    ierr = VecAXPY (dJ, -1.0, ptemp_);                              CHKERRQ (ierr);
    ierr = VecAXPY (dJ, n_misc_->penalty_, x);                      CHKERRQ (ierr);
}

PetscErrorCode DerivativeOperatorsPos::evaluateHessian (Vec y, Vec x) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_hessian_evals++;

    ierr = tumor_->phi_->apply (temp_phip_, p_current_);            CHKERRQ (ierr);
    ierr = VecCopy (temp_phip_, tumor_->c_0_);                      CHKERRQ (ierr);
    ScalarType *temp_ptr;
    ierr = VecGetArray (tumor_->c_0_, &temp_ptr);                   CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        temp_ptr[i] = (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-temp_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-temp_ptr[i] + n_misc_->exp_shift_);
        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 1.0;
        }
    }
    ierr = VecRestoreArray (tumor_->c_0_, &temp_ptr);               CHKERRQ (ierr);
    ierr = tumor_->phi_->apply (temp_phiptilde_, x);                CHKERRQ (ierr);
    ierr = VecPointwiseMult (tumor_->c_0_, tumor_->c_0_, temp_phiptilde_);    CHKERRQ (ierr);
    ierr = pde_operators_->solveState (1);

    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);

    ierr = pde_operators_->solveAdjoint (2);

    ScalarType *phip_ptr, *phiptilde_ptr, *p_ptr;
    ierr = VecGetArray (temp_phip_, &phip_ptr);                     CHKERRQ (ierr);
    ierr = VecGetArray (temp_phiptilde_, &phiptilde_ptr);           CHKERRQ (ierr);
    ierr = VecGetArray (temp_, &temp_ptr);                          CHKERRQ (ierr);
    ierr = VecGetArray (tumor_->p_0_, &p_ptr);                      CHKERRQ (ierr);
    for (int i = 0; i < n_misc_->n_local_; i++) {
        temp_ptr[i] = -n_misc_->beta_ *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_) *
                        phiptilde_ptr[i];

        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }

        temp_ptr[i] += 3.0 * n_misc_->beta_ *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_) *
                        phiptilde_ptr[i];

        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }

        temp_ptr[i] += -p_ptr[i] *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        (1 / (1 + exp(-phip_ptr[i] + n_misc_->exp_shift_))) *
                        exp(-phip_ptr[i] + n_misc_->exp_shift_);

        if (std::isnan(temp_ptr[i])) {
            temp_ptr[i] = 0.0;
        }
    }
    ierr = VecRestoreArray (temp_phip_, &phip_ptr);                  CHKERRQ (ierr);
    ierr = VecRestoreArray (temp_phiptilde_, &phiptilde_ptr);        CHKERRQ (ierr);
    ierr = VecRestoreArray (temp_, &temp_ptr);                       CHKERRQ (ierr);
    ierr = VecRestoreArray (tumor_->p_0_, &p_ptr);                   CHKERRQ (ierr);


    ierr = tumor_->phi_->applyTranspose (ptemp_, temp_);
    ierr = VecCopy (x, y);                                          CHKERRQ (ierr);
    ierr = VecScale (y, n_misc_->penalty_);                         CHKERRQ (ierr);
    ierr = VecAXPY (y, 1.0, ptemp_);                                CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========    REACTION DIFFUSION W/ MODIFIED OBJECTIVE (MP)  ======== #### */
/* #### ========    REACTION DIFFUSION FOR MOVING ATLAS (MA)       ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsRDObj::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert (data != nullptr, "DerivativeOperatorsRDObj::evaluateObjective: requires non-null input data.");
    ScalarType misfit_tu = 0, misfit_brain = 0;
    PetscReal reg;
    n_misc_->statistics_.nb_obj_evals++;

    //compute c0
    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                CHKERRQ (ierr);
    // compute c1
    ierr = pde_operators_->solveState (0);                       CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);            CHKERRQ (ierr);
    // geometric coupling, update probability maps
    ierr = geometricCoupling(
      xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_,
      m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_,  m_geo_bg_,
      tumor_->c_t_, n_misc_);                                    CHKERRQ (ierr);
    // evaluate tumor distance meassure || c(1) - d ||
    ierr = VecAXPY (temp_, -1.0, data);                          CHKERRQ (ierr);
    ierr = VecDot (temp_, temp_, &misfit_tu);                    CHKERRQ (ierr);
    // evaluate brain tissue distance meassure || mR - mT ||, mR = mA0(1-c), mT = patient
    geometricCouplingAdjoint(&misfit_brain,
      xi_wm_, xi_gm_, xi_csf_, xi_glm_,  xi_bg_,
      m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_,  m_geo_bg_,
      m_data_wm_, m_data_gm_, m_data_csf_, m_data_glm_,  m_data_bg_); CHKERRQ (ierr);

    /*Regularization term*/
    if (n_misc_->regularization_norm_ == L1) {
      ierr = VecNorm (x, NORM_1, &reg);                           CHKERRQ (ierr);
      reg *= n_misc_->lambda_;
    } else if (n_misc_->regularization_norm_ == wL2) {
      ierr = VecPointwiseMult (ptemp_, tumor_->weights_, x);      CHKERRQ (ierr);
      ierr = VecDot (x, ptemp_, &reg);                            CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_;
    } else if (n_misc_->regularization_norm_ == L2){
      ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);           CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_ * n_misc_->lebesgue_measure_;
    } else if (n_misc_->regularization_norm_ == L2b){
      ierr = VecDot (x, x, &reg);                                   CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_;
    }

    // compute objective function value
    misfit_brain *= 0.5 * n_misc_->lebesgue_measure_;
    misfit_tu  *= 0.5 * n_misc_->lebesgue_measure_;
    (*J) = misfit_tu + misfit_brain;
    (*J) *= 1./nc_;
    (*J) += reg;

    std::stringstream s;
    s << "  J(p,m) = Dm(v,c) + Dc(c) + S(c0) = "<< std::setprecision(12) << (*J) <<" = " << std::setprecision(12) <<misfit_brain * 1./nc_ <<" + "<< std::setprecision(12)<< misfit_tu * 1./nc_ <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRDObj::evaluateGradient (Vec dJ, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ScalarType misfit_brain;
    n_misc_->statistics_.nb_grad_evals++;

    ScalarType dJ_val = 0, norm_alpha = 0, norm_phiTalpha = 0, norm_phiTphic0 = 0;
    ScalarType norm_adjfinal1 = 0., norm_adjfinal2 = 0., norm_c0 = 0., norm_c1 = 0., norm_d =0.;
    std::stringstream s;

    if (n_misc_->verbosity_ >= 2) {ierr = VecNorm (data, NORM_2, &norm_d);          CHKERRQ (ierr);}

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                CHKERRQ (ierr);
    if (n_misc_->verbosity_ >= 2) {ierr = VecNorm (tumor_->c_0_, NORM_2, &norm_c0); CHKERRQ (ierr);}
    ierr = pde_operators_->solveState (0);                       CHKERRQ (ierr);

    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);            CHKERRQ (ierr);
    if (n_misc_->verbosity_ >= 2) {ierr = VecNorm (tumor_->c_t_, NORM_2, &norm_c1);  CHKERRQ (ierr);}
    ierr = VecAXPY (temp_, -1.0, data);                          CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);            CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                        CHKERRQ (ierr);
    ierr = geometricCoupling(
      xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_,
      m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_,  m_geo_bg_,
      tumor_->c_t_, n_misc_);                                    CHKERRQ (ierr);
    // evaluate brain tissue distance meassure || mR - mT ||, mR = mA0(1-c), mT = patient
    geometricCouplingAdjoint(&misfit_brain,
      xi_wm_, xi_gm_, xi_csf_, xi_glm_,  xi_bg_,
      m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_,  m_geo_bg_,
      m_data_wm_, m_data_gm_, m_data_csf_, m_data_glm_,  m_data_bg_); CHKERRQ (ierr);
    // compute xi * mA0, add    -\xi * mA0 to adjoint final cond.
    if(m_geo_wm_ != nullptr) {
  		ierr = VecPointwiseMult (temp_, xi_wm_, m_geo_wm_);        CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(m_geo_gm_ != nullptr) {
      ierr = VecPointwiseMult (temp_, xi_gm_, m_geo_gm_);        CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(m_geo_csf_ != nullptr) {
      ierr = VecPointwiseMult (temp_, xi_csf_, m_geo_csf_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(m_geo_glm_ != nullptr) {
      ierr = VecPointwiseMult (temp_, xi_glm_, m_geo_glm_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
    if (n_misc_->verbosity_ >= 2) {ierr = VecNorm (tumor_->p_t_, NORM_2, &norm_adjfinal1); CHKERRQ (ierr);}
    ierr = VecScale (tumor_->p_t_, 1.0/nc_);                     CHKERRQ (ierr);
    if (n_misc_->verbosity_ >= 2) {ierr = VecNorm (tumor_->p_t_, NORM_2, &norm_adjfinal2); CHKERRQ (ierr);}

    // solve adjoint equation with specified final condition
    ierr = pde_operators_->solveAdjoint (1);
    // evaluate gradient
    ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);  CHKERRQ (ierr);

    ierr = VecScale (ptemp_, n_misc_->lebesgue_measure_);           CHKERRQ (ierr);

    if (n_misc_->verbosity_ >= 2) {ierr = VecNorm (ptemp_, NORM_2, &norm_phiTalpha);       CHKERRQ (ierr);}

    // gradient according to reg paramater
    if (n_misc_->regularization_norm_ == L1) {
      ierr = VecCopy (ptemp_, dJ);                               CHKERRQ (ierr);
      ierr = VecScale (dJ, -1.0);                                CHKERRQ (ierr);
    } else if (n_misc_->regularization_norm_ == wL2) {
      ierr = VecPointwiseMult (dJ, tumor_->weights_, x);         CHKERRQ (ierr);
      ierr = VecScale (dJ, n_misc_->beta_);                      CHKERRQ (ierr);
      ierr = VecAXPY (dJ, -1.0, ptemp_);                         CHKERRQ (ierr);
    } else if (n_misc_->regularization_norm_ == L2){
      ierr = tumor_->phi_->applyTranspose (dJ, tumor_->c_0_);
      ierr = VecScale (dJ, n_misc_->beta_ * n_misc_->lebesgue_measure_);                      CHKERRQ (ierr);
      if (n_misc_->verbosity_ >= 2) {ierr = VecNorm (dJ, NORM_2, &norm_phiTphic0);         CHKERRQ (ierr);}
      ierr = VecAXPY (dJ, -1.0, ptemp_);                         CHKERRQ (ierr);
    } else if (n_misc_->regularization_norm_ == L2b){
      ierr = VecCopy (x, dJ);                                    CHKERRQ (ierr);
      ierr = VecScale (dJ, n_misc_->beta_);                      CHKERRQ (ierr);
      ierr = VecAXPY (dJ, -1.0, ptemp_);                         CHKERRQ (ierr);
    }

    // TODO: add inversion for diffusivity

    // additional information
    ierr = VecNorm (dJ, NORM_2, &dJ_val);                         CHKERRQ(ierr);
    ierr = VecNorm (tumor_->p_0_, NORM_2, &norm_alpha);           CHKERRQ(ierr);

    if (n_misc_->verbosity_ >= 2) {
      s <<   "||phiTc0|| = " << std::scientific << std::setprecision(6) << norm_phiTphic0 << " ||phiTa(0)|| = " << std::scientific << std::setprecision(6) << norm_phiTalpha;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
      s <<   "||a(1)|| = " << std::scientific << std::setprecision(6) << norm_adjfinal1 << " ||a(1)||s = " << std::scientific << std::setprecision(6) << norm_adjfinal2<< " ||c(1)|| = " << std::scientific << std::setprecision(6) << norm_c1<< " ||c(0)|| = " << std::scientific << std::setprecision(6) << norm_c0<< " ||d|| = " << std::scientific << std::setprecision(6) << norm_d;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }
    s <<   "dJ(p,m) = "    << std::scientific << std::setprecision(6) << dJ_val         << " ||a(0)|| = "   << std::scientific << std::setprecision(6) << norm_alpha;      ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    PetscFunctionReturn (ierr);
}

// TODO: implement optimized version
PetscErrorCode DerivativeOperatorsRDObj::evaluateObjectiveAndGradient (PetscReal *J,Vec dJ, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_obj_evals++;
    n_misc_->statistics_.nb_grad_evals++;
    ierr = evaluateObjective (J, x, data);                        CHKERRQ(ierr);
    ierr = evaluateGradient (dJ, x, data);                        CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperatorsRDObj::evaluateHessian (Vec y, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_hessian_evals++;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                CHKERRQ (ierr);
    ierr = pde_operators_->solveState (1);                       CHKERRQ (ierr);
    // alpha(1) = - O^TO \tilde{c(1)}
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);            CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);            CHKERRQ (ierr);
    ierr = VecScale (tumor_->p_t_, -1.0);                        CHKERRQ (ierr);
    // alpha(1) = - O^TO \tilde{c(1)} - mA0 mA0 \tilde{c(1)}
    if(m_geo_wm_ != nullptr) {
  		ierr = VecPointwiseMult (temp_, m_geo_wm_, m_geo_wm_);     CHKERRQ (ierr);
      ierr = VecPointwiseMult (temp_, temp_, tumor_->c_t_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(m_geo_gm_ != nullptr) {
      ierr = VecPointwiseMult (temp_, m_geo_gm_, m_geo_gm_);     CHKERRQ (ierr);
      ierr = VecPointwiseMult (temp_, temp_, tumor_->c_t_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(m_geo_csf_ != nullptr) {
      ierr = VecPointwiseMult (temp_, m_geo_csf_, m_geo_csf_);   CHKERRQ (ierr);
      ierr = VecPointwiseMult (temp_, temp_, tumor_->c_t_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}
  	if(m_geo_glm_ != nullptr) {
      ierr = VecPointwiseMult (temp_, m_geo_glm_, m_geo_glm_);   CHKERRQ (ierr);
      ierr = VecPointwiseMult (temp_, temp_, tumor_->c_t_);      CHKERRQ (ierr);
      ierr = VecAXPY (tumor_->p_t_, -1.0, temp_);                CHKERRQ (ierr);
  	}

    ierr = VecScale (tumor_->p_t_, 1.0/nc_);                     CHKERRQ (ierr);
    ierr = pde_operators_->solveAdjoint (2);                     CHKERRQ (ierr);
    ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);  CHKERRQ (ierr);
    ierr = VecScale (ptemp_, n_misc_->lebesgue_measure_);           CHKERRQ (ierr);

    //No hessian info for L1 for now
    if (n_misc_->regularization_norm_ == wL2) {
      ierr = VecPointwiseMult (y, tumor_->weights_, x);          CHKERRQ (ierr);
      ierr = VecScale (y, n_misc_->beta_);                       CHKERRQ (ierr);
      ierr = VecAXPY (y, -1.0, ptemp_);                          CHKERRQ (ierr);
    } else if (n_misc_->regularization_norm_ == L2b){
        ierr = VecCopy (x, y);                                       CHKERRQ (ierr);
        ierr = VecScale (y, n_misc_->beta_);                         CHKERRQ (ierr);
        ierr = VecAXPY (y, -1.0, ptemp_);                            CHKERRQ (ierr);
    } else {
      ierr = tumor_->phi_->applyTranspose (y, tumor_->c_0_);
      ierr = VecScale (y, n_misc_->beta_ * n_misc_->lebesgue_measure_);                       CHKERRQ (ierr);
      ierr = VecAXPY (y, -1.0, ptemp_);                          CHKERRQ (ierr);
    }

    PetscFunctionReturn (ierr);
}


/* #### ------------------------------------------------------------------- #### */
/* #### ========                  BASE CLASS                       ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperators::checkGradient (Vec p, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    PCOUT << "\n\n----- Gradient check with taylor expansion ----- " << std::endl;

    ScalarType norm;
    ierr = VecNorm (p, NORM_2, &norm);                          CHKERRQ (ierr);

    PCOUT << "Gradient check performed at x with norm: " << norm << std::endl;
    ScalarType *x_ptr, k1, k2, k3;
    if (n_misc_->diffusivity_inversion_) {
      ierr = VecGetArray (p, &x_ptr);                             CHKERRQ (ierr);
      k1 = x_ptr[n_misc_->np_];
      k2 = (n_misc_->nk_ > 1) ? x_ptr[n_misc_->np_ + 1] : 0;
      k3 = (n_misc_->nk_ > 2) ? x_ptr[n_misc_->np_ + 2] : 0;
      PCOUT << "k1: " << k1 << " k2: " << k2 << " k3: " << k3 << std::endl;
      ierr = VecRestoreArray (p, &x_ptr);                         CHKERRQ (ierr);
    }

    if (n_misc_->flag_reaction_inv_) {
      ierr = VecGetArray (p, &x_ptr);                             CHKERRQ (ierr);
      k1 = x_ptr[n_misc_->np_ + n_misc_->nk_];
      k2 = (n_misc_->nr_ > 1) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 1] : 0;
      k3 = (n_misc_->nr_ > 2) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 2] : 0;
      PCOUT << "r1: " << k1 << " r2: " << k2 << " r3: " << k3 << std::endl;
      ierr = VecRestoreArray (p, &x_ptr);                         CHKERRQ (ierr);
    }


    ScalarType h[6];
    ScalarType J, J_taylor, J_p, diff;

    Vec dJ;
    Vec p_tilde;
    Vec p_new;
    ierr = VecDuplicate (p, &dJ);                               CHKERRQ (ierr);
    ierr = VecDuplicate (p, &p_tilde);                          CHKERRQ (ierr);
    ierr = VecDuplicate (p, &p_new);                            CHKERRQ (ierr);

    ierr = evaluateObjectiveAndGradient (&J_p, dJ, p, data);    CHKERRQ (ierr);

    PetscRandom rctx;
    #ifdef SERIAL
      ierr = PetscRandomCreate (PETSC_COMM_SELF, &rctx);          CHKERRQ (ierr);
    #else
      ierr = PetscRandomCreate (PETSC_COMM_WORLD, &rctx);         CHKERRQ (ierr);
    #endif
    ierr = PetscRandomSetFromOptions (rctx);                    CHKERRQ (ierr);
    ierr = VecSetRandom (p_tilde, rctx);                        CHKERRQ (ierr);

    for (int i = 0; i < 6; i++) {
        h[i] = 1E-5 * std::pow (10, -i);
        ierr = VecWAXPY (p_new, h[i], p_tilde, p);              CHKERRQ (ierr);
        ierr = evaluateObjective (&J, p_new, data);
        ierr = VecDot (dJ, p_tilde, &J_taylor);                 CHKERRQ (ierr);
        J_taylor *= h[i];
        J_taylor +=  J_p;
        diff = std::abs(J - J_taylor);
        PCOUT << "h[i]: " << h[i] << " |J - J_taylor|: " << diff << "  log10(diff) : " << log10(diff) << std::endl;
    }
    PCOUT << "\n\n";

    ierr = VecDestroy (&dJ);               CHKERRQ (ierr);
    ierr = VecDestroy (&p_tilde);          CHKERRQ (ierr);
    ierr = VecDestroy (&p_new);            CHKERRQ (ierr);
    ierr = PetscRandomDestroy (&rctx);     CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}

PetscErrorCode DerivativeOperators::checkHessian (Vec p, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    PCOUT << "\n\n----- Hessian check with taylor expansion ----- " << std::endl;

    ScalarType norm;
    ierr = VecNorm (p, NORM_2, &norm);                          CHKERRQ (ierr);

    PCOUT << "Hessian check performed at x with norm: " << norm << std::endl;
    ScalarType *x_ptr, k1, k2, k3;
    if (n_misc_->diffusivity_inversion_) {
      ierr = VecGetArray (p, &x_ptr);                             CHKERRQ (ierr);
      k1 = x_ptr[n_misc_->np_];
      k2 = (n_misc_->nk_ > 1) ? x_ptr[n_misc_->np_ + 1] : 0;
      k3 = (n_misc_->nk_ > 2) ? x_ptr[n_misc_->np_ + 2] : 0;
      PCOUT << "k1: " << k1 << " k2: " << k2 << " k3: " << k3 << std::endl;
      ierr = VecRestoreArray (p, &x_ptr);                         CHKERRQ (ierr);
    }

    if (n_misc_->flag_reaction_inv_) {
      ierr = VecGetArray (p, &x_ptr);                             CHKERRQ (ierr);
      k1 = x_ptr[n_misc_->np_ + n_misc_->nk_];
      k2 = (n_misc_->nr_ > 1) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 1] : 0;
      k3 = (n_misc_->nr_ > 2) ? x_ptr[n_misc_->np_ + n_misc_->nk_ + 2] : 0;
      PCOUT << "r1: " << k1 << " r2: " << k2 << " r3: " << k3 << std::endl;
      ierr = VecRestoreArray (p, &x_ptr);                         CHKERRQ (ierr);
    }

    ScalarType h[6] = {0, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10};
    ScalarType J, J_taylor, J_p, diff;

    Vec dJ, Hx, temp;
    Vec p_tilde;
    Vec p_new;
    ierr = VecDuplicate (p, &dJ);                               CHKERRQ (ierr);
    ierr = VecDuplicate (p, &temp);                               CHKERRQ (ierr);
    ierr = VecDuplicate (p, &Hx);                               CHKERRQ (ierr);
    ierr = VecDuplicate (p, &p_tilde);                          CHKERRQ (ierr);
    ierr = VecDuplicate (p, &p_new);                            CHKERRQ (ierr);

    ierr = evaluateObjectiveAndGradient (&J_p, dJ, p, data);    CHKERRQ (ierr);

    PetscRandom rctx;
    #ifdef SERIAL
      ierr = PetscRandomCreate (PETSC_COMM_SELF, &rctx);          CHKERRQ (ierr);
    #else
      ierr = PetscRandomCreate (PETSC_COMM_WORLD, &rctx);         CHKERRQ (ierr);
    #endif
    ierr = PetscRandomSetFromOptions (rctx);                    CHKERRQ (ierr);
    ierr = VecSetRandom (p_tilde, rctx);                        CHKERRQ (ierr);
    ierr = VecCopy (p_tilde, temp);                             CHKERRQ (ierr);
    ScalarType hess_term = 0.;
    for (int i = 0; i < 6; i++) {
        ierr = VecWAXPY (p_new, h[i], p_tilde, p);              CHKERRQ (ierr);
        ierr = evaluateObjective (&J, p_new, data);
        ierr = VecDot (dJ, p_tilde, &J_taylor);                 CHKERRQ (ierr);
        J_taylor *= h[i];
        J_taylor +=  J_p;
        // H(p)*p_tilde
        ierr = VecCopy (p_tilde, temp);                         CHKERRQ (ierr);
        ierr = VecScale (temp, h[i]);                           CHKERRQ (ierr);
        ierr = evaluateHessian (Hx, p_tilde);                   CHKERRQ (ierr);
        ierr = VecDot (p_tilde, Hx, &hess_term);                CHKERRQ (ierr);
        hess_term *= 0.5 * h[i] * h[i];
        J_taylor += hess_term;
        diff = std::abs(J - J_taylor);
        PCOUT << "|J - J_taylor|: " << diff << "  log10(diff) : " << log10(diff) << std::endl;
    }
    PCOUT << "\n\n";

    ierr = VecDestroy (&dJ);               CHKERRQ (ierr);
    ierr = VecDestroy (&temp);               CHKERRQ (ierr);
    ierr = VecDestroy (&Hx);               CHKERRQ (ierr);
    ierr = VecDestroy (&p_tilde);          CHKERRQ (ierr);
    ierr = VecDestroy (&p_new);            CHKERRQ (ierr);
    ierr = PetscRandomDestroy (&rctx);     CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}
