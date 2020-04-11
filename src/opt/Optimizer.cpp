#include "Optimizer.h"

Optimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params,
  std::shared_ptr <Tumor> tumor)) {

    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    if (initialized_) PetscFunctionReturn(ierr);

    ctx_ = std::make_shared<CtxInv> ();
    ctx_->derivative_operators_ = derivative_operators;
    ctx_->pde_operators_ = pde_operators;
    ctx_->params_ = params;
    ctx_->tumor_ = tumor;

    ierr = allocateTaoObjects(); CHKERRQ(ierr);
    initialized_ = true;
    PetscFunctionReturn (ierr);
}

PetscErrorCode Optimizer::allocateTaoObjects() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: Optimizer not initialized."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }



  PetscFunctionReturn (ierr);
}
