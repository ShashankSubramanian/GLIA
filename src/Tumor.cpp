#include "Tumor.h"

Tumor::Tumor (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    k_ = std::make_shared<DiffCoef> (n_misc);
    rho_ = std::make_shared<ReacCoef> (n_misc);
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

PetscErrorCode Tumor::initialize (Vec p, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Phi> phi, std::shared_ptr<MatProp> mat_prop) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (mat_prop == nullptr) {
        mat_prop_ = std::make_shared<MatProp> (n_misc);
        ierr = mat_prop_->setValues (n_misc);
    }
    else
        mat_prop_ = mat_prop;
    ierr = k_->setValues (n_misc->k_, n_misc->k_gm_wm_ratio_, n_misc->k_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = rho_->setValues (n_misc->rho_, n_misc->r_gm_wm_ratio_, n_misc->r_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = VecDuplicate (p, &p_);                                 CHKERRQ (ierr);
    ierr = VecDuplicate (p, &p_true_);                            CHKERRQ (ierr);
    ierr = VecCopy (p, p_);                                       CHKERRQ (ierr);
    ierr = setTrueP (n_misc);                                     CHKERRQ (ierr);

    if (phi == nullptr) {
        phi_ = std::make_shared<Phi> (n_misc);
        ierr = phi_->setGaussians (n_misc->user_cm_, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
        ierr = phi_->setValues (mat_prop_);
    }
    else
        phi_ = phi;
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
      // phi_ = std::make_shared<Phi> (n_misc);
    }
    ierr = VecCopy (p, p_);                                         CHKERRQ (ierr);
    // ierr = VecCopy (p, p_true_);                                  CHKERRQ (ierr);

    // set new values
    ierr = k_->setValues (n_misc->k_, n_misc->k_gm_wm_ratio_, n_misc->k_glm_wm_ratio_, mat_prop_, n_misc);
    ierr = rho_->setValues (n_misc->rho_, n_misc->r_gm_wm_ratio_, n_misc->r_glm_wm_ratio_, mat_prop_, n_misc);
    // ierr = phi_->setGaussians (n_misc->user_cm_, n_misc->phi_sigma_, n_misc->phi_spacing_factor_, n_misc->np_);
    ierr = phi_->setValues (mat_prop_);
    ierr = phi_->apply(c_0_, p_);

    PetscFunctionReturn(0);
}

PetscErrorCode Tumor::setTrueP (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    PetscScalar val[2] = {.9, .2};
    PetscInt center = (int) std::floor(n_misc->np_ / 2.);
    PetscInt idx[2] = {center-1, center};
    ierr = VecSetValues(p_true_, 2, idx, val, INSERT_VALUES );        CHKERRQ(ierr);
    ierr = VecAssemblyBegin(p_true_);                                 CHKERRQ(ierr);
    ierr = VecAssemblyEnd(p_true_);                                   CHKERRQ(ierr);
    PetscFunctionReturn (0);
}


PetscErrorCode Tumor::setTrueP (Vec p) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    ierr = VecCopy (p, p_true_);                                      CHKERRQ (ierr);
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
