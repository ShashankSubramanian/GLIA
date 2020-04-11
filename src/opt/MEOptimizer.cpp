#include "MEOptimizer.h"

MEOptimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params,
  std::shared_ptr <Tumor> tumor)) {

    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    // initialize super class
    ierr = Optimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);
    // number of dofs = {rho, kappa, gamma}
    n_inv_ = params_->get_nr() +  params_->get_nk() + 1;

    // TODO(K): implement functinality of:
    // ierr = allocateTaoObjectsMassEffect(); CHKERRQ(ierr);

    // allocate TaoObjects
    // setTaoOptions

    PetscFunctionReturn(ierr);
}
