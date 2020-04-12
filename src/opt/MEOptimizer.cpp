#include "MEOptimizer.h"

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
MEOptimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params,
  std::shared_ptr <Tumor> tumor)) {

    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    std::stringstream ss;
    // number of dofs = {rho, kappa, gamma}
    n_inv_ = params_->get_nr() +  params_->get_nk() + 1;
    ss << " Initializing mass-effect optimizer with = " << n_inv_ << " = " <<  params_->get_nr() << " + " <<  params_->get_nk() << " + 1 dofs.";  
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    // initialize super class
    ierr = Optimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);

    PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode Optimizer::allocateTaoObjects() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = Optimizer::allocateTaoObjects(); CHKERRQ(ierr);
  // set initial guess TODO(K): move to solve()
  ScalarType *ptr;
  ierr = VecGetArray(xrec_, &ptr); CHKERRQ (ierr);
  ptr[0] = 1; ptr[1] = 6; ptr[2] = 0.5;
  ierr = VecRestoreArray(xrec_, &ptr); CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 10)
      ierr = MatShellSetOperation (H_, MATOP_CREATE_VECS, (void(*)(void)) operatorCreateVecsMassEffect);
  #endif

  PetscFunctionReturn (ierr);
}
