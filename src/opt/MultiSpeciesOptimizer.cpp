#include <iostream>
#include <limits>

#include <petsc/private/vecimpl.h>
#include <libcmaes/esoptimizer.h>
#include <libcmaes/cmastrategy.h>



#include "Optimizer.h"
#include "MultiSpeciesOptimizer.h"



PetscErrorCode MultiSpeciesOptimizer::initialize(
  std::shared_ptr <DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params, 
  std::shared_ptr <Tumor> tumor) {

    PetscErrorCode ierr=0;
    PetscFunctionBegin;
    std::stringstream ss;

    n_inv_ = params->get_nr() + params->get_nk() + 5; // # of inverting parameters: kappa, rho, forcing factor, gamma(death rate), alpha_0, ox_consumption, ox 
    // number of dofs
    ss << " Initializing multi-species optimizer with =" << n_inv_ << " = " << params-> get_nr() << " + " << params->get_nk() << " + 5 dof."; 

    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ierr = Optimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);

PetscErrorCode MultiSpeciesOptimizer::setInitialGuess(Vec x_init) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initiliazed_) {
    ierr = tuMSGwarn("Error: Optimizer not initialized."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  std::stringstream ss;
  int nk = ctx_->params_->tu_->nk_;
  int nr = ctx_->params_->tu_->nr_;
  int np = ctx_->params_->tu_->np_;
  int in_size;
  ierr = VecGetSize(x_init, &in_size); CHKERRQ(ierr);
  ScalarType *init_ptr, *x_ptr;
  
  if(xin_ != nullptr) {ierr = VecDestroy(&xin_); CHKERRQ(ierr);}
  if(xout_ != nullptr) {ierr = VecDestroy(&xout_); CHKERRQ(ierr);}
  
  ss << " Setting initial guess: ";
  int off = 0, off_in = 0;
  PetscReal ic_max = 0.;
  // 1. TIL not given as parameterozation
  if (ctx_->params_->tu_->use_c0_) {
    ierr = VecCreateSeq (PETSC_COMM_SELF, nk + nr + n_g_, &xin_); CHKERRQ (ierr);
    ierr = VecMax(ctx_->tumor_->c_0_, NULL, &ic_max); CHKERRQ(ierr);
    ss << "TIL as d0 (max=" << ic_max << "); ";
    if (ctx_->params_->opt_->rescale_init_cond_){
      ScalarType scale = (1.0 / ic_max);
      if(ctx_->params_->opt_->multilevel_) scale = (1.0/4.0 * ctx_->params_->grid_->n_[0]/64.  / ic_max);
      ierr = VecScale(ctx_->tumor_->c_0_, scale); CHKERRQ(ierr);
      ss << " and rescaled with scale " << ic_max / scale;
      //ss << "rescaled; ";
    }
    // copy back to data
    ierr = VecCopy(ctx_->tumor_->c_0_, data_->dt0()); CHKERRQ(ierr);
    off = (in_size > n_g_ + nr + nk) ? np : 0; // offset in x_init vec
    off_in = 0;
  // 2. TIL given as parameterization 
  } else {

    ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &xin_); CHKERRQ(ierr);
    ierr = setupVec (xin_, SEQ); CHKERRQ(ierr);
    ierr = VecSet (xin_, 0.0); CHKERRQ(ierr);
    ierr = VecCopy(x_init, xin_); CHKERRQ(ierr);
    // === init TIL 
    ierr = ctx_->tumor_->phi_->apply (ctx_->tumor_->c_0_, x_init); CHKERRQ(ierr);
    ierr = VecMax(ctx_->tumor_->c_0_, NULL, &ic_max); CHKERRQ(ierr);
    ss << "TIL as Phi(p) (max =" << ic_max << "); ";
    if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(xin_, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_, std::string("til-init"));}
    off = off_in = np;
    
    // === kappa init guess 
  if (in_size > np) {

    ierr = VecGetArray(x_init, &init_ptr); CHKERRQ(ierr);
    ierr = VecGetArray(xin_, &x_ptr); CHKERRQ(ierr);
    x_ptr[off_int + 0] = init_ptr[off];
    k_init_ = x_ptr[off_in + 0];
    if (nk > 1) x_ptr[off_in + 1] = init_ptr[off + 1];
    if (nk > 2) x_ptr[off_in + 2] = init_ptr[off+2];
    ierr = VecRestoreArray(xin_, &x_ptr); CHKERRQ (ierr);
    ierr = VecRestoreArray(x_init, &init_ptr); CHKERRQ (ierr);
  }

  ss << "k_init=" << k_init_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // create xout_ vec
  ierr = VecDuplicate(xin_, &xout_) CHKERRQ(ierr);
  ierr = VecSet(xout_, 0.0) CHKERRQ (ierr);
  
  PetscFunctionReturn(ierr);

}

FitFunc runforward = [](const double *x, const int N){

  





} 



PetscErrorCode MultiSpeciesOptimizer:: solve() {

  








}    




