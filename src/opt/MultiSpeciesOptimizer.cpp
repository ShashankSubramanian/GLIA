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

  n_g_ = (params->opt_->invert_mass_effect_) ? 1 : 0; 
	n_inv_ = params->get_nr() + params->get_nk() + 5 + n_g_; // # of inverting parameters: kappa, rho, forcing factor, gamma(death rate), alpha_0, ox_consumption, ox 
	// number of dofs
	ss << " Initializing multi-species optimizer with =" << n_inv_ << " = " << params-> get_nr() << " + " << params->get_nk() << " + 5 dof."; 

	ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
	// initialize super class 
	ierr = Optimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);

	// set scales for inversion variables
	params->opt_->k_scale = (params->opt_->k_scale_ != 1) ? params->opt_->k_scale_ : 1E-2;
	params->opt_->rho_scale_ = 1;
	params->opt_->gamma_scale_ = (params->opt_->gamma_scale_ != 1) ? params->opt_->gamma_scale_ : 1E-4;
	params->opt_->deathrate_scale_ = 1;    
	params->opt_->ox_hypoxia_scale_ = 1;    
	params->opt_->alpha_0_scale_ = 1;    
	params->opt_->ox_consumption_scale_ = 1;    
	params->opt_->ox_source_scale_ = 1;    
	params->opt_->beta_0_scale_ = 1;    
 
  k_scale_ = params->opt_->k_scale_;
  rho_scale_ = params->opt_->rho_scale_;
  gamma_scale_ = params->opt_->gamma_scale_;
  deathrate_scale_ = params->opt_->deathrate_scale_;
  ox_hypoxia_scale_ = params->opt_->ox_hypoxia_scale_;
  alpha_0_scale_ = params->opt_->alpha_0_scale_;
  ox_consumption_scale_ = params->opt_->ox_consumption_scale_;
  ox_source_scale_ = params->opt_->ox_source_scale_;
  beta_0_scale_ = params->opt_->beta_0_scale_;     

	PetscFunctionReturn(ierr);

}

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
  } 

  // === gamma, rho, kappa, ox_hypoxia, death_rate, alpha_0, ox_consumption, ox_source, beta_0
  ierr = VecGetArray(x_init, &init_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(xin_, &x_ptr); CHKERRQ(ierr);

  if (params->opt_->invert_mass_effect_ == 1) {
    x_ptr[0] = init_ptr[0] / gamma_scale_;
    x_ptr[1] = init_ptr[1] / rho_scale_; 
    x_ptr[2] = init_ptr[2] / k_scale_;
    x_ptr[3] = init_ptr[3] / ox_hypoxia_scale_; 
    x_ptr[4] = init_ptr[4] / death_rate_scale_;
    x_ptr[5] = init_ptr[5] / alpha_0_scale_;
    x_ptr[6] = init_ptr[6] / ox_consumption_scale_;
    x_ptr[7] = init_ptr[7] / ox_source_scale_;
    x_ptr[8] = init_ptr[8] / beta_0_scale_;

    gamma_init_ = x_ptr[0];
    rho_init_ = x_ptr[1];
    k_init_ = x_ptr[2];
    ox_hypoxia_init_ = x_ptr[3];
    death_rate_init_ = x_ptr[4];
    alpha_0_init_ = x_ptr[5];
    ox_consumption_init_ = x_ptr[6];
    ox_source_init_ = x_ptr[7];
    beta_0_init_ = x_ptr[8];

    ierr = VecRestoreArray(xin_, &x_ptr); CHKERRQ(ierr);
    ierr = VecRestoreArray(x_init, &init_ptr); CHKERRQ(ierr);

    ss << " gamma_init = " << gamma_init_ << " ; rho_init = " << rho_init_ << " ;k_init = " << k_init_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    
    ss << " ox_hypoxia_init = " << ox_hypoxia_init_ << " ; deathrate_init = " << death_rate_init_ << " ;alpha_0_init = " << alpha_0_init_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    ss << " ox_consumption_init = " << ox_consumption_init_ << " ; ox_source_init = " << ox_source_init_ << " ;beta_0_init = " << beta_0_init_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

  } else {
    
    x_ptr[0] = init_ptr[0] / rho_scale_; 
    x_ptr[1] = init_ptr[1] / k_scale_;
    x_ptr[2] = init_ptr[2] / ox_hypoxia_scale_; 
    x_ptr[3] = init_ptr[3] / death_rate_scale_;
    x_ptr[4] = init_ptr[4] / alpha_0_scale_;
    x_ptr[5] = init_ptr[5] / ox_consumption_scale_;
    x_ptr[6] = init_ptr[6] / ox_source_scale_;
    x_ptr[7] = init_ptr[7] / beta_0_scale_;

    rho_init_ = x_ptr[1];
    k_init_ = x_ptr[2];
    ox_hypoxia_init_ = x_ptr[3];
    death_rate_init_ = x_ptr[4];
    alpha_0_init_ = x_ptr[5];
    ox_consumption_init_ = x_ptr[6];
    ox_source_init_ = x_ptr[7];
    beta_0_init_ = x_ptr[8];

    ierr = VecRestoreArray(xin_, &x_ptr); CHKERRQ(ierr);
    ierr = VecRestoreArray(x_init, &init_ptr); CHKERRQ(ierr);

    ss << " ; rho_init = " << rho_init_ << " ;k_init = " << k_init_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    
    ss << " ox_hypoxia_init = " << ox_hypoxia_init_ << " ; deathrate_init = " << death_rate_init_ << " ;alpha_0_init = " << alpha_0_init_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    ss << " ox_consumption_init = " << ox_consumption_init_ << " ; ox_source_init = " << ox_source_init_ << " ;beta_0_init = " << beta_0_init_;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  }

  // create xout_ vec
  ierr = VecDuplicate(xin_, &xout_); CHKERRQ(ierr);
  ierr = VecSet(xout_, 0.0); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);

}


  
PetscErrorCode MultiSpeciesOptimizer::solve() {

  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  TU_assert(initialized_, "MultiSpeciesOptimizer needs to be initialized.");  
  TU_assert(data_->dt1() != nullptr, "MultiSpeciesOptimizer requires non-null input data for inversion. ");
  TU_assert(xrec_ != nullptr, "MultiSpeciesOptimizer requires non-null xrec_ vector to be set. ");
  TU_assert(xin_ != nullptr, "MultiSpeciesOptimizer requires non-null xin_ vector to be set.");

  std::stringstream ss;
  ierr = tuMSGstd(""); CHKERRQ(ierr)
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
  ierr = tuMSG("###                                 MultiSpecies Inversion                                                 ###");CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

  // DOFs
  int nk = ctx_->params_->tu_->nk_;
  int nr = ctx_->params_->tu_->nr_;
  int np = ctx_->params_->tu_->np_;
  TU_assert(n_inv_ == nr + nk + n_g_ + 5, "MultiSpecies : n_inv is inconsistent.");
 
  if (ctx_->c0_old == nullptr) {
    ierr = VecDuplicate(data_->dt1(), &ctx_->c0_old); CHKERRQ(ierr);
  }

  ierr = VecSet (ctx_->c0_old, 0.0); CHKERRQ(ierr);
  if (ctx_-> tmp == nullptr) {
    ierr = VecDuplicate(data_->dt1(), &ctx_->tmp); CHKERRQ(ierr);
    ierr = VecSet(ctx_->tmp, 0.0); CHKERRQ(ierr);
  }
  if (ctx_->x_old != nullptr) { 
    ierr = VecDestroy(&ctx_->x_old); CHKERRQ(ierr);
    ctx_->x_old = nullptr;
  }
  ierr = VecDuplicate(xrec_, &ctx_->x_old); CHKERRQ(ierr);
  ierr = VecCopy(xrec_, ctx_->x_old); CHKERRQ(ierr);

  // initial guess 

  PetscReal *xin_ptr, *xout_ptr, *x_ptr; int in_size;
  ierr = VecGetSize(xin_, &in_size); CHKERRQ(ierr);
  ierr = VecGetArray(xrec_, &x_ptr); CHKERRQ(ierr);
 
  for(int i = 0; i < n_inv_; ++i)
    x_ptr[i] = xin_ptr[i]l
  
  ierr = VecRestoreArray(xin_, &xin_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(xrec_, &x_ptr); CHKERRQ(ierr);
  ierr = tuMSGstd(""); CHKERRQ(ierr);
  ierr = tuMSG ("### MultiSpecies inversion : initial guess ###"); CHKERRQ(ierr);
  ierr = tuMSGstd("### ----------------------- ###"); CHKERRQ(ierr);
  if (procid == 0) {ierr = VecView (xrec_, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);}
  ierr = tuMSGstd ("### ---------------------- ###"); CHKERRQ(ierr);
  
  // fitting function for cma-es
  FitFunc runforward = [](const double *xtest_ptr){
  
    // Assign the parameters 
    // including mass effect inversion
    ScalarType *xtest_ptr_;
    if (n_g_){
      ctx_->params_->tu_->forcing_factor_ = xtest_ptr[0];
      ctx_->params_->tu_->rho_ = xtest_ptr[1];
      ctx_->params_->tu_->k_ = xtest_ptr[2];
      ctx_->params_->tu_->ox_hypoxia_ = xtest_ptr[3];
      ctx_->params_->tu_->death_rate_ = xtest_ptr[4];
      ctx_->params_->tu_->alpha_0_ = xtest_ptr[5];
      ctx_->params_->tu_->ox_consumption_ = xtest_ptr[6];
      ctx_->params_->tu_->ox_source_ = xtest_ptr[7];
      ctx_->params_->tu_->beta_0_ = xtest_ptr[8];
      xtest_ptr_[0] = x_test_ptr[0];
      xtest_ptr_[1] = x_test_ptr[1];
      xtest_ptr_[2] = x_test_ptr[2];
      xtest_ptr_[3] = x_test_ptr[3];
      xtest_ptr_[4] = x_test_ptr[4];
      xtest_ptr_[5] = x_test_ptr[5];
      xtest_ptr_[6] = x_test_ptr[6];
      xtest_ptr_[7] = x_test_ptr[7];
      xtest_ptr_[8] = x_test_ptr[8];
      
    } else {
    // no mass effect inversion
      ctx_->params_->tu_->rho_ = xtest_ptr[0];
      ctx_->tu_->rho_->updateIsotropicCoefficients(xtest_ptr[0], 0., 0., ctx_->tu_->mat_prop, ctx_->params_); CHKERRQ(ierr);
      ctx_->params_->tu_->k_ = xtest_ptr[1];
      ctx_->tu_->k_->updateIsotropicCoefficients(xtest_ptr[1], 0., 0., ctx_->tu_->mat_prop, ctx_->params_); CHKERRQ(ierr);
      
      ctx_->params_->tu_->ox_hypoxia_scale_ = xtest_ptr[2];
      ctx_->params_->tu_->death_rate_ = xtest_ptr[3];
      ctx_->params_->tu_->alpha_0_ = xtest_ptr[4];
      ctx_->params_->tu_->ox_consumption_ = xtest_ptr[5];
      ctx_->params_->tu_->ox_source_ = xtest_ptr[6];
      ctx_->params_->tu_->beta_0_ = xtest_ptr[7];
      
      
      xtest_ptr_[0] = x_test_ptr[0];
      xtest_ptr_[1] = x_test_ptr[1];
      xtest_ptr_[2] = x_test_ptr[2];
      xtest_ptr_[3] = x_test_ptr[3];
      xtest_ptr_[4] = x_test_ptr[4];
      xtest_ptr_[5] = x_test_ptr[5];
      xtest_ptr_[6] = x_test_ptr[6];
      xtest_ptr_[7] = x_test_ptr[7];      
    }
    // TODO: add the inversion for the ic coeffs
   

    ScalarType dJ;
       

    Vec xtest_;
    ierr = VecSet(x_test_ptr_, 0); CHKERRQ(ierr);
    ierr = VecC


    ierr = ctx_->derivative_operators_->evaluateObjective(dJ, x_test_ptr_, ctx_->data); CHKERRQ(ierr);
     
    float J = dJ;

    return J   
  }
  
  std::vector<double> x0;
  
  double sigma = ctx_->params_->sigma_inv_;
  //int lambda = 100;
  
  CMAParameters<> cmaparams(x0, sigma);
  CMAsolutions cmasols = cmaes<>(runforward, cmaparams);
  std::cout << "best solution: " << cmasols << std::endl;
  std::cout << "optimization took " << cmasols.elapsed_time() / 1000.0 << " sec\n"; 
    
 
  PetscFunctionReturn(ierr)
}    




