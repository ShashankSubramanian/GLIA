#include "Tumor.h"

Tumor::Tumor (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    mat_prop_ = std::make_shared<MatProp> (n_misc);
    k_ = std::make_shared<DiffCoef> (n_misc);
    rho_ = std::make_shared<ReacCoef> (n_misc);
    phi_ = std::make_shared<Phi> (n_misc);
    obs_ = std::make_shared<Obs> (n_misc);

    ierr = VecCreate (PETSC_COMM_WORLD, &c_t_);
    ierr = VecSetSizes (c_t_, n_misc->n_local_, n_misc->n_global_);
    ierr = VecSetFromOptions (c_t_);
    ierr = VecDuplicate (c_t_, &c_0_);
    ierr = VecDuplicate (c_t_, &p_0_);
    ierr = VecDuplicate (c_0_, &p_t_);

    ierr = VecSet (c_t_, 0);
    ierr = VecSet (c_0_, 0);
    ierr = VecSet (p_0_, 0);
    ierr = VecSet (p_t_, 0);
}

PetscErrorCode Tumor::initialize (Vec p, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ierr = mat_prop_->setValues (n_misc);
    ierr = k_->setValues (n_misc->k_, n_misc->k_gm_wm_ratio_, n_misc->k_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = rho_->setValues (n_misc->rho_, n_misc->r_gm_wm_ratio_, n_misc->r_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = VecDuplicate (p, &p_);                                 CHKERRQ (ierr);
    ierr = VecDuplicate (p, &p_true_);                            CHKERRQ (ierr);
    ierr = VecCopy (p, p_);                                       CHKERRQ (ierr);

    ierr = phi_->setValues (n_misc->user_cm_, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, mat_prop_, n_misc);
    ierr = phi_->apply(c_0_, p_);

    PetscFunctionReturn(0);
}

PetscErrorCode Tumor::setParams (Vec p, std::shared_ptr<NMisc> n_misc, bool npchanged) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    if (npchanged) {
      // re-create p vectors
      if (p_ != nullptr) {ierr = VecDestroy (&p_);                  CHKERRQ (ierr);}
      if (p_ != nullptr) {ierr = VecDestroy (&p_true_);             CHKERRQ (ierr);}
      ierr = VecDuplicate (p, &p_);                                 CHKERRQ (ierr);
      ierr = VecDuplicate (p, &p_true_);                            CHKERRQ (ierr);
      // re-create phi mesh (deletes old instance and creates new instance)
      phi_ = std::make_shared<Phi> (n_misc);
    }
    ierr = VecCopy (p, p_);                                       CHKERRQ (ierr);
    // ierr = VecCopy (p, p_true_);                                  CHKERRQ (ierr);

    // set new values
    ierr = k_->setValues (n_misc->k_, n_misc->k_gm_wm_ratio_, n_misc->k_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = rho_->setValues (n_misc->rho_, n_misc->r_gm_wm_ratio_, n_misc->r_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = phi_->setValues (n_misc->user_cm_, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, mat_prop_, n_misc);
    ierr = phi_->apply(c_0_, p_);

    PetscFunctionReturn(0);
}

PetscErrorCode Tumor::setTrueP (double p_scale) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ierr = VecSet (p_true_, p_scale);                               CHKERRQ (ierr);
    PetscFunctionReturn (0);
}


Tumor::~Tumor () {
    PetscErrorCode ierr;
    ierr = VecDestroy (&c_t_);
    ierr = VecDestroy (&c_0_);
    ierr = VecDestroy (&p_t_);
    ierr = VecDestroy (&p_0_);
    ierr = VecDestroy (&p_);
    ierr = VecDestroy (&p_true_);
}
