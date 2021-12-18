#include <iostream>
#include <limits>
#include <memory>

#include <petsc/private/vecimpl.h>
#include <cmaes.h>
//#include <libcmaes/esoptimizer.h>
//#include <libcmaes/cmastrategy.h>
//#include <libcmaes/llogging.h>

#include "Optimizer.h"
#include "MultiSpeciesOptimizer.h"


PetscErrorCode MultiSpeciesOptimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr<PdeOperators> pde_operators,
  std::shared_ptr<Parameters> params, 
  std::shared_ptr<Tumor> tumor) {

	PetscErrorCode ierr=0;
	PetscFunctionBegin;
	std::stringstream ss;

  n_g_ = (params->opt_->invert_mass_effect_) ? 1 : 0; 
	n_inv_ = params->get_nr() + params->get_nk() + 6 + n_g_; // # of inverting parameters: kappa, rho, forcing factor, gamma(death rate), alpha_0, ox_consumption, ox 
	// number of dofs
	ss << " Initializing multi-species optimizer with =" << n_inv_ << " = " << params-> get_nr() << " + " << params->get_nk() << " + 5 dof."; 

	ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
	// initialize super class 
	ierr = CMAOptimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);

	// set scales for inversion variables
	params->opt_->k_scale_ = (params->opt_->k_scale_ != 1) ? params->opt_->k_scale_ : 1E-2;
	params->opt_->rho_scale_ = 1;
	params->opt_->gamma_scale_ = (params->opt_->gamma_scale_ != 1) ? params->opt_->gamma_scale_ : 1E-4;
	params->opt_->death_rate_scale_ = 1;    
	params->opt_->ox_hypoxia_scale_ = 1;    
	params->opt_->alpha_0_scale_ = 1;    
	params->opt_->ox_consumption_scale_ = 1;    
	params->opt_->ox_source_scale_ = 1;    
	params->opt_->beta_0_scale_ = 1;    
 
  k_scale_ = params->opt_->k_scale_;
  rho_scale_ = params->opt_->rho_scale_;
  gamma_scale_ = params->opt_->gamma_scale_;
  death_rate_scale_ = params->opt_->death_rate_scale_;
  ox_hypoxia_scale_ = params->opt_->ox_hypoxia_scale_;
  alpha_0_scale_ = params->opt_->alpha_0_scale_;
  ox_consumption_scale_ = params->opt_->ox_consumption_scale_;
  ox_source_scale_ = params->opt_->ox_source_scale_;
  beta_0_scale_ = params->opt_->beta_0_scale_;     

	PetscFunctionReturn(ierr);

}

PetscErrorCode MultiSpeciesOptimizer::allocateObjects() {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    ierr = CMAOptimizer::allocateObjects(); CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}


PetscErrorCode MultiSpeciesOptimizer::setInitialGuess(Vec x_init) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: Optimizer not initialized."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  std::stringstream ss;
  int nk = cma_ctx_->params_->tu_->nk_;
  int nr = cma_ctx_->params_->tu_->nr_;
  int np = cma_ctx_->params_->tu_->np_;
  int in_size;
  ierr = VecGetSize(x_init, &in_size); CHKERRQ(ierr);
  ScalarType *init_ptr, *x_ptr;
  
  if(xin_ != nullptr) {ierr = VecDestroy(&xin_); CHKERRQ(ierr);}
  if(xout_ != nullptr) {ierr = VecDestroy(&xout_); CHKERRQ(ierr);}
  
  ss << " Setting initial guess: ";
  int off = 0, off_in = 0;
  PetscReal ic_max = 0.;
  // 1. TIL not given as parameterozation
  if (cma_ctx_->params_->tu_->use_c0_) {
    ierr = VecCreateSeq (PETSC_COMM_SELF, nk + nr + n_g_ + 6, &xin_); CHKERRQ (ierr);
    ierr = VecMax(cma_ctx_->tumor_->c_0_, NULL, &ic_max); CHKERRQ(ierr);
    ss << "TIL as d0 (max=" << ic_max << "); ";
    if (cma_ctx_->params_->opt_->rescale_init_cond_){
      ScalarType scale = (1.0 / ic_max);
      if(cma_ctx_->params_->opt_->multilevel_) scale = (1.0/4.0 * cma_ctx_->params_->grid_->n_[0]/64.  / ic_max);
      ierr = VecScale(cma_ctx_->tumor_->c_0_, scale); CHKERRQ(ierr);
      ss << " and rescaled with scale " << ic_max / scale;
      //ss << "rescaled; ";
    }
    // copy back to data
    ierr = VecCopy(cma_ctx_->tumor_->c_0_, data_->dt0()); CHKERRQ(ierr);
    off = (in_size > n_g_ + nr + nk) ? np : 0; // offset in x_init vec
    off_in = 0;
  // 2. TIL given as parameterization 
  } else {

    ierr = VecCreateSeq (PETSC_COMM_SELF, nr + np + nk + n_g_ + 6, &xin_); CHKERRQ(ierr);
    std::cout << "# inv params = " << np+nk + n_g_ + 6 << " \n\n";
    std::cout << "# k params = " << nk << " \n\n";
    std::cout << "# p params = " << np << " \n\n";
    std::cout << "# r params = " << nr << " \n\n";
    ierr = setupVec (xin_, SEQ); CHKERRQ(ierr);
    ierr = VecSet (xin_, 0.0); CHKERRQ(ierr);
    ierr = VecCopy(x_init, xin_); CHKERRQ(ierr);
    // === init TIL 
    ierr = cma_ctx_->tumor_->phi_->apply (cma_ctx_->tumor_->c_0_, x_init); CHKERRQ(ierr);
    ierr = VecMax(cma_ctx_->tumor_->c_0_, NULL, &ic_max); CHKERRQ(ierr);
    ss << "TIL as Phi(p) (max =" << ic_max << "); ";
    if (cma_ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(xin_, cma_ctx_->tumor_->phi_, cma_ctx_->params_->tu_->writepath_, std::string("til-init"));}
    off = off_in = np;
  } 

  // === gamma, rho, kappa, ox_hypoxia, death_rate, alpha_0, ox_consumption, ox_source, beta_0
  ierr = VecGetArray(x_init, &init_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(xin_, &x_ptr); CHKERRQ(ierr);

  if (n_g_) {
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

PetscErrorCode MultiSpeciesOptimizer::runforward(const double *xtest_, double* J){

  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  std::cout << "----------------------------------- Running MultiSpecies Forward ---------------------------------------------- \n";
  PetscReal J_solve;
	Vec x_petsc;
	PetscReal *x_petsc_ptr;
  ierr = VecCreateSeq (PETSC_COMM_SELF, n_inv_, &x_petsc); CHKERRQ(ierr);
  ierr = setupVec (x_petsc, SEQ); CHKERRQ(ierr);
  ierr = VecSet (x_petsc, 0.0); CHKERRQ(ierr); 
  J_solve = 0.0;
  ierr = VecGetArray(x_petsc, &x_petsc_ptr); CHKERRQ(ierr);
  for (int i=0; i < n_inv_; i++) x_petsc_ptr[i] = xtest_[i];
  ierr = VecRestoreArray(x_petsc, &x_petsc_ptr); CHKERRQ(ierr);
  ierr = cma_ctx_->derivative_operators_->evaluateObjective(&J_solve, x_petsc, cma_ctx_->data); CHKERRQ(ierr); 
  *J = J_solve;
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

  using namespace libcmaes;
  std::stringstream ss;
  ierr = tuMSGstd(""); CHKERRQ(ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###"); CHKERRQ (ierr);
  ierr = tuMSG("###                                 MultiSpecies Inversion                                                 ###");CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

  // DOFs
  int nk = cma_ctx_->params_->tu_->nk_;
  int nr = cma_ctx_->params_->tu_->nr_;
  int np = cma_ctx_->params_->tu_->np_;
 
  cma_ctx_->data = data_; 
  n_g_ = (cma_ctx_->params_->opt_->invert_mass_effect_) ? 1 : 0;
  TU_assert(n_inv_ == nr + nk + n_g_ + 6, "MultiSpecies : n_inv is inconsistent.");
 
  if (cma_ctx_->c0_old == nullptr) {
    ierr = VecDuplicate(data_->dt1(), &cma_ctx_->c0_old); CHKERRQ(ierr);
  }

  ierr = VecSet (cma_ctx_->c0_old, 0.0); CHKERRQ(ierr);
  if (cma_ctx_-> tmp == nullptr) {
    ierr = VecDuplicate(data_->dt1(), &cma_ctx_->tmp); CHKERRQ(ierr);
    ierr = VecSet(cma_ctx_->tmp, 0.0); CHKERRQ(ierr);
  }
  if (cma_ctx_->x_old != nullptr) { 
    ierr = VecDestroy(&cma_ctx_->x_old); CHKERRQ(ierr);
    cma_ctx_->x_old = nullptr;
  }
  ierr = VecCreateSeq (PETSC_COMM_SELF, n_inv_, &xrec_); CHKERRQ(ierr);
  ierr = setupVec (xrec_, SEQ); CHKERRQ(ierr);
  ierr = VecSet (xrec_, 0.0); CHKERRQ(ierr); 



  //ierr = allocateObjects(); CHKERRQ(ierr);

  //ierr = VecDuplicate(xrec_, &cma_ctx_->x_old); CHKERRQ(ierr);
  //TU_assert(xrec_ != nullptr, "MultiSpeciesOptimizer requires non-null xrec_ vector to be set. ");
  //TU_assert(cma_ctx_->x_old != nullptr, "MultiSpeciesOptimizer requires non-null x_old vector to be set. ");
  //ierr = VecCopy(xrec_, cma_ctx_->x_old); CHKERRQ(ierr);

  // initial guess 

  PetscReal *xin_ptr, *xout_ptr, *x_ptr; int in_size;
  //ierr = VecGetSize(xin_, &in_size); CHKERRQ(ierr);
  ierr = VecGetArray(xrec_, &x_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(xin_, &xin_ptr); CHKERRQ(ierr);
 
  for(int i = 0; i < n_inv_; ++i) x_ptr[i] = xin_ptr[i];
  
  ierr = VecRestoreArray(xin_, &xin_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(xrec_, &x_ptr); CHKERRQ(ierr);
  ierr = tuMSGstd(""); CHKERRQ(ierr);
  ierr = tuMSG ("### MultiSpecies inversion : initial guess ###"); CHKERRQ(ierr);
  ierr = tuMSGstd("### ----------------------- ###"); CHKERRQ(ierr);
  if (procid == 0) {ierr = VecView (xin_, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);}
  ierr = tuMSGstd ("### ---------------------- ###"); CHKERRQ(ierr);
  /* 
  // fitting function for cma-es
  FitFunc runforward = [&](const double *xeval_ptr, const int N) -> double {
  
    // Assign the parameters 
    // including mass effect inversion
    // TODO: add the inversion for the ic coeffs
   

    dJ[0] = 0.0; 
    ierr = tuMSG ("-1 ### Step ###"); CHKERRQ(ierr);
    ierr = tuMSG ("0 ### Step ###"); CHKERRQ(ierr);
    ierr = tuMSG ("1 ### Step ###"); CHKERRQ(ierr);
		ierr = VecDuplicate(xin_, &x_petsc); CHKERRQ(ierr);
    ierr = tuMSG ("2 ### Step ###"); CHKERRQ(ierr);
		ierr = VecGetArray(x_petsc, &x_petsc_ptr); CHKERRQ(ierr);     
    ierr = tuMSG ("3 ### Step ###"); CHKERRQ(ierr);

		for (int i=0; i < n_inv_; i++) x_petsc_ptr[i] = xeval_ptr[i];
    ierr = tuMSG ("4 ### Step ###"); CHKERRQ(ierr);

		ierr = VecRestoreArray(x_petsc, &x_petsc_ptr); CHKERRQ(ierr);
    ierr = tuMSG ("5 ### Step ###"); CHKERRQ(ierr);
		
		ierr = cma_ctx_->derivative_operators_->evaluateObjective(dJ, x_petsc, cma_ctx_->data); CHKERRQ(ierr);
    ierr = tuMSG ("6 ### Step ###"); CHKERRQ(ierr);
    
    double J = dJ[0];

    return J; 
  };
  */

 
  //double *lbounds[n_inv_], *ubounds[n_inv_];
  //double *lbounds[n_inv_];
  //std::vector<double> lbounds[n_inv_], ubounds[n_inv_];
   
  std::vector<double> x0(n_inv_, 1.0);
  ierr = VecGetArray(xin_, &xin_ptr); CHKERRQ(ierr);

  xin_ptr[0] = cma_ctx_->params_->tu_->forcing_factor_;
  xin_ptr[1] = cma_ctx_->params_->tu_->rho_;
  xin_ptr[2] = cma_ctx_->params_->tu_->k_;
  xin_ptr[3] = cma_ctx_->params_->tu_->ox_hypoxia_;
  xin_ptr[4] = cma_ctx_->params_->tu_->death_rate_;
  xin_ptr[5] = cma_ctx_->params_->tu_->alpha_0_;
  xin_ptr[6] = cma_ctx_->params_->tu_->ox_consumption_;
  xin_ptr[7] = cma_ctx_->params_->tu_->ox_source_;
  xin_ptr[8] = cma_ctx_->params_->tu_->beta_0_;
 

  //std::array<double, n_inv_> ubounds; 
  //std::array<double, n_inv_> lbounds; 
  //double *ubounds = new double[n_inv_];
  //double *lbounds = new double[n_inv_];
 
  //double ubounds[n_inv_], lbounds[n_inv_];
 
  int dim = n_inv_;
  //std::vector<double> lbounds(dim, 0.0);
  //std::vector<double> ubounds(dim, 30000.0);
  
  double lbounds[dim], ubounds[dim];
  double tmp = 0.0;
  
  tmp = cma_ctx_->params_->opt_->gamma_ub_ ;
  ubounds[0] = tmp;
  tmp = cma_ctx_->params_->opt_->rho_ub_;
  ubounds[1] = tmp;
  tmp = cma_ctx_->params_->opt_->k_ub_;
  ubounds[2] = tmp;
  tmp = cma_ctx_->params_->opt_->ox_hypoxia_ub_ ;
  ubounds[3] = tmp; 
  tmp = cma_ctx_->params_->opt_->death_rate_ub_ ;
  ubounds[4] = tmp;
  tmp = cma_ctx_->params_->opt_->alpha_0_ub_ ;
  ubounds[5] = tmp;
  tmp = cma_ctx_->params_->opt_->ox_consumption_ub_;
  ubounds[6] = tmp;
  tmp = cma_ctx_->params_->opt_->ox_source_ub_;
  ubounds[7] = tmp;
  tmp = cma_ctx_->params_->opt_->beta_0_ub_;   
  ubounds[8] = tmp;   
 
  tmp = cma_ctx_->params_->opt_->gamma_lb_;
  lbounds[0] = tmp;
  tmp = cma_ctx_->params_->opt_->rho_lb_;
  lbounds[1] = tmp;
  tmp = cma_ctx_->params_->opt_->k_lb_;
  lbounds[2] = tmp;
  tmp = cma_ctx_->params_->opt_->ox_hypoxia_lb_;
  lbounds[3] = tmp;
  tmp = cma_ctx_->params_->opt_->death_rate_lb_;
  lbounds[4] = tmp;
  tmp = cma_ctx_->params_->opt_->alpha_0_lb_;
  lbounds[5] = tmp;
  tmp = cma_ctx_->params_->opt_->ox_consumption_lb_;
  lbounds[6] = tmp;
  tmp = cma_ctx_->params_->opt_->ox_source_lb_;
  lbounds[7] = tmp;
  tmp = cma_ctx_->params_->opt_->beta_0_lb_;
  lbounds[8] = tmp;

  std::cout << "---------------------- Initial Params and Bounds ----------------------- \n";
  for (int i = 0; i < n_inv_; i++) {
    std::cout << "\t\t\t " << lbounds[i] << " < " << xin_ptr[i] << " < " << ubounds[i] << " \n";
  }

  std::vector<double> xcma_ptr(n_inv_);
 
  for (int i=0; i<n_inv_; i++) { 
    xcma_ptr[i] = xin_ptr[i];
  }


  FitFunc runforward = [&](const double *xeval_ptr, const int N) -> double { 
    double err = 0.0;
    bool outofbounds = false;

    for (int i= 0; i < n_inv_; i++) {
      if (xeval_ptr[i] < lbounds[i] || xeval_ptr[i] > ubounds[i]) outofbounds= true;
      if (outofbounds) {
        std::cout << "----------------  Out of bounds for p[" << i << "] ----------------- \n";
        break;
      }
    }
    if (outofbounds) {
      err = std::numeric_limits<double>::infinity();
      
    } else {
      ierr = MultiSpeciesOptimizer::runforward(xeval_ptr, &err); CHKERRQ(ierr);
    } 
    return err;  
  };


 
  //const double sigma = cma_ctx_->params_->opt_->sigma_inv_;
  //const double sigma = 0.5;

  std::vector<double> sigma(n_inv_, 1.0);

  sigma[0] = cma_ctx_->params_->opt_->sigma_cma_gamma_;
  sigma[1] = cma_ctx_->params_->opt_->sigma_cma_rho_; 
  sigma[2] = cma_ctx_->params_->opt_->sigma_cma_k_;
  sigma[3] = cma_ctx_->params_->opt_->sigma_cma_ox_hypoxia_;
  sigma[4] = cma_ctx_->params_->opt_->sigma_cma_death_rate_;
  sigma[5] = cma_ctx_->params_->opt_->sigma_cma_alpha_0_;
  sigma[6] = cma_ctx_->params_->opt_->sigma_cma_ox_consumption_;
  sigma[7] = cma_ctx_->params_->opt_->sigma_cma_ox_source_;
  sigma[8] = cma_ctx_->params_->opt_->sigma_cma_beta_0_;
  int lambda = -1;

  

   /* 
  //GenoPheno<pwqBoundStrategy> gp(lbounds,ubounds,dim);

  //GenoPheno<pwqBoundStrategy,linScalingStrategy> gp(lbounds,ubounds,dim); 
  //GenoPheno<pwqBoundStrategy,linScalingStrategy> gp(&lbounds.at(0),&ubounds.at(0),dim); 
  //GenoPheno<pwqBoundStrategy> gp(lbounds,ubounds, dim); 
  std::cout << "best solution: \n";
  CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(xcma_ptr,sigma,-1,0,gp);
  std::cout << "best solution: \n";
  cmaparams.set_algo(aCMAES);
  std::cout << "best solution: \n";
  CMASolutions cmasols = cmaes<GenoPheno<pwqBoundStrategy>>(runforward,cmaparams);
  std::cout << "best solution: ";
  cmasols.print(std::cout,0,gp);
  std::cout << "best solution: \n";
  std::cout << std::endl;
  std::cout << "optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds\n";
  */

  /* 
  //CMAParameters<GenoPheno<pwqBoundStrategy,linScalingStrategy>> cmaparams(dim, &xcma_ptr.front(), sigma,-1,0,gp);
  std::cout << "best solution: \n";
  //ESOptimizer<CMAStrategy<CovarianceUpdate>,CMAParameters<>> cmaes<GenoPheno<pwqBoundStrategy>>(runtest,cmaparams);
  CMASolutions cmasols = cmaes<GenoPheno<pwqBoundStrategy,linScalingStrategy>>(runforward,cmaparams);
  std::cout << "best solution: \n";
  Eigen::VectorXd bestparameters = gp.pheno(cmasols.get_best_seen_candidate().get_x_dvec());
  std::cout << "bestparameters: \n" << bestparameters << std::endl;
  std::cout << "best solution: \n";
  cmasols.print(std::cout,0,gp);
  std::cout << std::endl;
  std::cout << "optimization took " << cmasols.elapsed_time() / 1000.0 << " seconds\n";
  */
   
  CMAParameters<> cmaparams(xcma_ptr, sigma);
  //CMASolutions cmasols = cmaes<>(runforward, cmaparams);
  ESOptimizer<CMAStrategy<CovarianceUpdate>,CMAParameters<>> cmaes(runforward,cmaparams);
  cmaes.optimize();
  
  const double *xcma = cmaes.get_solutions().best_candidate().get_x_ptr();

  
  ierr = VecCopy(xin_, xout_); CHKERRQ(ierr);
  ierr = VecGetArray(xout_, &xout_ptr); CHKERRQ(ierr);

  for (int i = 0; i < n_inv_; i++){
    std::cout << "------------ CMA Inv p[" << i << "] = " << xcma[i] << " ----------------- \n";
  
    xout_ptr[i] = xcma[i];
  }
  
  cma_ctx_->params_->tu_->forcing_factor_ = xcma[0];
  cma_ctx_->params_->tu_->rho_ = xcma[1];
  cma_ctx_->params_->tu_->k_ = xcma[2];
  cma_ctx_->params_->tu_->ox_hypoxia_ = xcma[3];
  cma_ctx_->params_->tu_->death_rate_ = xcma[4];
  cma_ctx_->params_->tu_->alpha_0_ = xcma[5];
  cma_ctx_->params_->tu_->ox_consumption_ = xcma[6];
  cma_ctx_->params_->tu_->ox_source_ = xcma[7];
  cma_ctx_->params_->tu_->beta_0_ = xcma[8];
  ierr = VecRestoreArray(xout_, &xout_ptr); CHKERRQ(ierr);
    

  PetscFunctionReturn(ierr);
} 



