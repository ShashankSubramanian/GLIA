#include "Optimizer.h"

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
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

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Optimizer::allocateTaoObjects() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: Optimizer not initialized."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }

  // allocate memory for xrec_ (n_inv_ is already set in the specialized function)
  ierr = VecCreateSeq (PETSC_COMM_SELF, n_inv_, &xrec_); CHKERRQ (ierr);
  ierr = setupVec (xrec_, SEQ); CHKERRQ (ierr);
  ierr = VecSet (xrec_, 0.0); CHKERRQ (ierr);

  // set up routine to compute the hessian matrix vector product
  if ((ctx_->params_->opt_->newton_solver_ == GAUSSNEWTON) && (H_ == nullptr)) {
    ierr = MatCreateShell (PETSC_COMM_SELF, n_inv_, n_inv_, n_inv_, n_inv_, (void*) ctx_.get(), &H_); CHKERRQ(ierr);
    ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))hessianMatVec); CHKERRQ(ierr);
    ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE); CHKERRQ(ierr);
  } // TODO(K): removed const Hessian approximation for quasi-Newton

  // create tao object
  if (tao_ == nullptr) {
    ierr = TaoCreate (PETSC_COMM_SELF, &tao_);
    tao_is_reset_ = true;  // triggers setTaoOptions TODO(K): check if we need this
  }
  PetscFunctionReturn (ierr);
}
