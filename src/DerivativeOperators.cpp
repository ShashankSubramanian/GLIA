#include "DerivativeOperators.h"
#include "Utils.h"

/* #### ------------------------------------------------------------------- #### */
/* #### ========          STANDARD REACTION DIFFUSION (MP)         ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsRD::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_obj_evals++;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);
    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
    ierr = VecAXPY (temp_, -1.0, data);                             CHKERRQ (ierr);
    ierr = VecDot (temp_, temp_, J);                                CHKERRQ (ierr);

    PetscReal reg;
    ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);               CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->beta_;


    std::stringstream s;
    s << "  J(p) = Dc(c) + S(c0) = "<< std::setprecision(12) << 0.5*(*J)+reg <<" = " << std::setprecision(12)<< 0.5*(*J) <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();

    (*J) *= 0.5;
    (*J) += reg;

    PetscFunctionReturn(0);
}

PetscErrorCode DerivativeOperatorsRD::evaluateGradient (Vec dJ, Vec x, Vec data){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    PetscScalar *x_ptr, *p_ptr;
    std::bitset<3> XYZ; XYZ[0] = 1; XYZ[1] = 1; XYZ[2] = 1;
    n_misc_->statistics_.nb_grad_evals++;
    Event e ("tumor-eval-grad");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

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
    ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
    ierr = tumor_->phi_->applyTranspose (dJ, tumor_->c_0_);
    ierr = VecScale (dJ, n_misc_->beta_);                           CHKERRQ (ierr);
    ierr = VecAXPY (dJ, -1.0, ptemp_);                              CHKERRQ (ierr);

    /* ------------------------- */
    /* INVERSION FOR DIFFUSIVITY */
    /* ------------------------- */
    /* (2) compute grad_k   int_T int_Omega { m_i * (grad c)^T grad alpha } dx dt */
    double integration_weight = 1.0;
    if (n_misc_->diffusivity_inversion_) {
      ierr = VecSet(temp_, 0.0);                                      CHKERRQ (ierr);
      // compute numerical time integration using trapezoidal rule
      for (int i = 0; i < n_misc_->nt_ + 1; i++) {
        // integration weight for chain trapezoidal rule
        if (i == 0 || i == n_misc_->nt_) integration_weight = 0.5;
        else integration_weight = 1.0;

        // compute x = (grad c)^T grad \alpha
        // compute gradient of state variable c(t)
        accfft_grad (tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], n_misc_->plan_, &XYZ, t.data());
        // compute gradient of adjoint variable p(t)
        accfft_grad (tumor_->work_[4], tumor_->work_[5], tumor_->work_[6], pde_operators_->p_[i], n_misc_->plan_, &XYZ, t.data());
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
      ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[n_misc_->np_ + 1]);          CHKERRQ(ierr);
      if (n_misc_->nk_ > 2) {
        ierr = VecDot(tumor_->mat_prop_->glm_, temp_, &x_ptr[n_misc_->np_ + 2]);       CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(dJ, &x_ptr);                                              CHKERRQ (ierr);
    }

    // if (n_misc_->diffusivity_inversion_) {
    //   ierr = VecSet(temp_, 0.0);                                      CHKERRQ (ierr);
    //   // compute numerical time integration using trapezoidal rule
    //   for (int i = 0; i < n_misc_->nt_ + 1; i++) {
    //     // integration weight for chain trapezoidal rule
    //     if (i == 0 || i == n_misc_->nt_) integration_weight = 0.5;
    //     else integration_weight = 1.0;

    //     // compute x = (grad c)^T grad \alpha
    //     // compute gradient of state variable c(t)
    //     accfft_grad (tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], pde_operators_->c_[i], n_misc_->plan_, &XYZ, t.data());
    //     ierr = VecPointwiseMult (tumor_->work_[1], tumor_->work_[1], tumor_->mat_prop_->wm_);  CHKERRQ (ierr);  
    //     ierr = VecPointwiseMult (tumor_->work_[2], tumor_->work_[2], tumor_->mat_prop_->wm_);  CHKERRQ (ierr);  
    //     ierr = VecPointwiseMult (tumor_->work_[3], tumor_->work_[3], tumor_->mat_prop_->wm_);  CHKERRQ (ierr);  
    //     accfft_divergence (tumor_->work_[0], tumor_->work_[1], tumor_->work_[2], tumor_->work_[3], n_misc_->plan_, t.data());
    //     ierr = VecPointwiseMult (tumor_->work_[0], tumor_->work_[0], pde_operators_->p_[i]);  CHKERRQ (ierr);  
        

    //     // numerical time integration using trapezoidal rule
    //     ierr = VecAXPY (temp_, n_misc_->dt_ * integration_weight, tumor_->work_[0]);     CHKERRQ (ierr);
    //   }
    //   // time integration of [ int_0 (grad c)^T grad alpha dt ] done, result in temp_
    //   // integration over omega (i.e., inner product, as periodic boundary and no lebesque measure in tumor code)
    //   ierr = VecGetArray(dJ, &x_ptr);                                                  CHKERRQ (ierr);
    //   ierr = VecDot(tumor_->mat_prop_->wm_, temp_, &x_ptr[n_misc_->np_]);              CHKERRQ(ierr);
    //   ierr = VecDot(tumor_->mat_prop_->gm_, temp_, &x_ptr[n_misc_->np_ + 1]);          CHKERRQ(ierr);
    //   if (n_misc_->nk_ > 2) {
    //     ierr = VecDot(tumor_->mat_prop_->glm_, temp_, &x_ptr[n_misc_->np_ + 2]);       CHKERRQ(ierr);
    //   }
    //   ierr = VecRestoreArray(dJ, &x_ptr);                                              CHKERRQ (ierr);
    // }

    // timing
    self_exec_time += MPI_Wtime(); t[5] = self_exec_time; e.addTimings (t); e.stop ();
    PetscFunctionReturn(0);
}

// saves on forward solve
PetscErrorCode DerivativeOperatorsRD::evaluateObjectiveAndGradient (PetscReal *J,Vec dJ, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_obj_evals++;
    n_misc_->statistics_.nb_grad_evals++;

    double *x_ptr;
    if (n_misc_->diffusivity_inversion_) {
      ierr = evaluateObjective (J, x, data);                          CHKERRQ(ierr);
      ierr = evaluateGradient (dJ, x, data);                          CHKERRQ(ierr);
     // TODO: implement efficient version
    } 
    else {
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
      ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
      ierr = tumor_->phi_->applyTranspose (dJ, tumor_->c_0_);
      ierr = VecScale (dJ, n_misc_->beta_);                           CHKERRQ (ierr);
      // gradient
      ierr = VecAXPY (dJ, -1.0, ptemp_);                              CHKERRQ (ierr);
      // regularization
      PetscReal reg;
      ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);               CHKERRQ (ierr);
      reg *= 0.5 * n_misc_->beta_;
      std::stringstream s;
      s << "  J(p) = Dc(c) + S(c0) = "<< std::setprecision(12) << 0.5*(*J)+reg <<" = " << std::setprecision(12)<< 0.5*(*J) <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
      // objective function value
      (*J) *= 0.5;
      (*J) += reg;
    }
    PetscFunctionReturn (0);
}

PetscErrorCode DerivativeOperatorsRD::evaluateHessian (Vec y, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_hessian_evals++;

    if (n_misc_->diffusivity_inversion_) {
      TU_assert(false, "not implemented."); CHKERRQ(ierr);     
    }
    else {
      ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
      ierr = pde_operators_->solveState (1);

      ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);               CHKERRQ (ierr);
      ierr = tumor_->obs_->apply (tumor_->p_t_, temp_);               CHKERRQ (ierr);
      ierr = VecScale (tumor_->p_t_, -1.0);                           CHKERRQ (ierr);

      ierr = pde_operators_->solveAdjoint (2);

      ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);
      ierr = tumor_->phi_->applyTranspose (y, tumor_->c_0_);
      ierr = VecScale (y, n_misc_->beta_);                            CHKERRQ (ierr);
      ierr = VecAXPY (y, -1.0, ptemp_);                               CHKERRQ (ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode DerivativeOperatorsRD::evaluateConstantHessianApproximation  (Vec y, Vec x){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                   CHKERRQ (ierr);
    ierr = tumor_->phi_->applyTranspose (y, tumor_->c_0_);          CHKERRQ (ierr);
    ierr = VecScale (y, n_misc_->beta_);                            CHKERRQ (ierr);
    PetscFunctionReturn(0);
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

    PetscFunctionReturn (0);
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

    PetscFunctionReturn(0);
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
    double *temp_ptr, *p_ptr;
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
    double *temp_ptr;
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

    double *phip_ptr, *phiptilde_ptr, *p_ptr;
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

    PetscFunctionReturn(0);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========    REACTION DIFFUSION W/ MODIFIED OBJECTIVE (MP)  ======== #### */
/* #### ========    REACTION DIFFUSION FOR MOVING ATLAS (MA)       ======== #### */
/* #### ------------------------------------------------------------------- #### */
PetscErrorCode DerivativeOperatorsRDObj::evaluateObjective (PetscReal *J, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert (data != nullptr, "DerivativeOperatorsRDObj::evaluateObjective: requires non-null input data.");
    PetscScalar misfit_tu = 0, misfit_brain = 0;
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
    // compute regularization
    ierr = VecDot (tumor_->c_0_, tumor_->c_0_, &reg);            CHKERRQ (ierr);
    reg *= 0.5 * n_misc_->beta_;
    // compute objective function value
    misfit_brain *= 0.5;
    misfit_tu  *= 0.5;
    (*J) = misfit_tu + misfit_brain;
    (*J) *= 1./nc_;
    (*J) += reg;

    std::stringstream s;
    s << "  J(p,m) = Dm(v,c) + Dc(c) + S(c0) = "<< std::setprecision(12) << (*J) <<" = " << std::setprecision(12) <<misfit_brain * 1./nc_ <<" + "<< std::setprecision(12)<< misfit_tu * 1./nc_ <<" + "<< std::setprecision(12) <<reg<<"";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    PetscFunctionReturn(0);
}

PetscErrorCode DerivativeOperatorsRDObj::evaluateGradient (Vec dJ, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    PetscScalar misfit_brain;
    n_misc_->statistics_.nb_grad_evals++;

    ierr = tumor_->phi_->apply (tumor_->c_0_, x);                CHKERRQ (ierr);
    ierr = pde_operators_->solveState (0);                       CHKERRQ (ierr);

    ierr = tumor_->obs_->apply (temp_, tumor_->c_t_);            CHKERRQ (ierr);
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
    ierr = VecScale (tumor_->p_t_, 1.0/nc_);                     CHKERRQ (ierr);
    // solve adjoint equation with specified final condition
    ierr = pde_operators_->solveAdjoint (1);
    // evaluate gradient
    ierr = tumor_->phi_->applyTranspose (ptemp_, tumor_->p_0_);  CHKERRQ (ierr);
    ierr = tumor_->phi_->applyTranspose (dJ, tumor_->c_0_);      CHKERRQ (ierr);
    ierr = VecScale (dJ, n_misc_->beta_);                        CHKERRQ (ierr);
    ierr = VecAXPY (dJ, -1.0, ptemp_);                           CHKERRQ (ierr);

    // additional information
    std::stringstream s; PetscScalar dJ_val = 0, norm_alpha = 0;
    ierr = VecNorm (dJ, NORM_2, &dJ_val);                         CHKERRQ(ierr);
    ierr = VecNorm (tumor_->p_0_, NORM_2, &norm_alpha);           CHKERRQ(ierr);
    s <<   "dJ(p,m) = "<< std::setprecision(12) << dJ_val << " ||a(0)||_2 = "<<norm_alpha;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    PetscFunctionReturn(0);
}

// TODO: implement optimized version
PetscErrorCode DerivativeOperatorsRDObj::evaluateObjectiveAndGradient (PetscReal *J,Vec dJ, Vec x, Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    n_misc_->statistics_.nb_obj_evals++;
    n_misc_->statistics_.nb_grad_evals++;
    ierr = evaluateObjective (J, x, data);                        CHKERRQ(ierr);
    ierr = evaluateGradient (dJ, x, data);                        CHKERRQ(ierr);
    PetscFunctionReturn(0);
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
    ierr = tumor_->phi_->applyTranspose (y, tumor_->c_0_);       CHKERRQ (ierr);
    ierr = VecScale (y, n_misc_->beta_);                         CHKERRQ (ierr);
    ierr = VecAXPY (y, -1.0, ptemp_);                            CHKERRQ (ierr);

    PetscFunctionReturn(0);
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

    double norm;
    ierr = VecNorm (p, NORM_2, &norm);                          CHKERRQ (ierr);

    PCOUT << "Gradient check performed at x with norm: " << norm << std::endl;
    double *x_ptr, k1, k2, k3;
    if (n_misc_->diffusivity_inversion_) {
      ierr = VecGetArray (p, &x_ptr);                             CHKERRQ (ierr);
      k1 = x_ptr[n_misc_->np_];
      k2 = (n_misc_->nk_ > 1) ? x_ptr[n_misc_->np_ + 1] : 0;
      k3 = (n_misc_->nk_ > 2) ? x_ptr[n_misc_->np_ + 2] : 0;
      PCOUT << "k1: " << k1 << " k2: " << k2 << " k3: " << k3 << std::endl;
      ierr = VecRestoreArray (p, &x_ptr);                         CHKERRQ (ierr);
    }


    double h[7] = {0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
    double J, J_taylor, J_p, diff;

    Vec dJ;
    Vec p_tilde;
    Vec p_new;
    ierr = VecDuplicate (p, &dJ);                               CHKERRQ (ierr);
    ierr = VecDuplicate (p, &p_tilde);                          CHKERRQ (ierr);
    ierr = VecDuplicate (p, &p_new);                            CHKERRQ (ierr);

    ierr = evaluateGradient (dJ, p, data);
    ierr = evaluateObjective(&J_p, p, data);

    PetscRandom rctx;
    #ifdef SERIAL
      ierr = PetscRandomCreate (PETSC_COMM_SELF, &rctx);          CHKERRQ (ierr);
    #else
      ierr = PetscRandomCreate (PETSC_COMM_WORLD, &rctx);         CHKERRQ (ierr);
    #endif
    ierr = PetscRandomSetFromOptions (rctx);                    CHKERRQ (ierr);
    ierr = VecSetRandom (p_tilde, rctx);                        CHKERRQ (ierr);

    for (int i = 0; i < 7; i++) {
        ierr = VecWAXPY (p_new, h[i], p_tilde, p);              CHKERRQ (ierr);
        ierr = evaluateObjective (&J, p_new, data);
        ierr = VecDot (dJ, p_tilde, &J_taylor);                 CHKERRQ (ierr);
        J_taylor *= h[i];
        J_taylor +=  J_p;
        diff = std::abs(J - J_taylor);
        PCOUT << "|J - J_taylor|: " << diff << "  log10(diff) : " << log10(diff) << std::endl;
    }
    PCOUT << "\n\n";
    PetscFunctionReturn (0);
}
