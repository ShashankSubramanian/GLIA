#include "SparseTILOptimizer.h"

SparseTILOptimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params,
  std::shared_ptr <Tumor> tumor)) {

    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    std:stringstream ss;

    // number of dofs = {p, kappa, rho}
    n_inv_ = params_->tu_->np_ + params_->get_nk() + params_->get_nr();
    ss << " Initializing sparseTIL optimizer with = " << n_inv_ << " = " << params_->tu_->np_ << " + " <<  params_->get_nr() << " + " <<  params_->get_nk() << " dofs.";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);

    // initialize super class
    ierr = Optimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);

    // initialize sub solvers
    til_opt_->initialize(derivative_operators, pde_operators, params, tumor);
    rd_opt_->initialize(derivative_operators, pde_operators, params, tumor);

    PetscFunctionReturn(ierr);
}
