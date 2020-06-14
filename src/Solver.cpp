#include <stdlib.h>
#include <iostream>
#include <memory>
#include <vector>

#include "Utils.h"
#include "SolverInterface.h"
#include "Solver.h"
#include "Optimizer.h"
#include "TILOptimizer.h"
#include "SparseTILOptimizer.h"
#include "RDOptimizer.h"
#include "MEOptimizer.h"


/* #### ------------------------------------------------------------------- #### */
/* #### ========                   ForwardSolver                   ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode ForwardSolver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ierr = tuMSGwarn(" Initializing Forward Solver."); CHKERRQ(ierr);
  // switch off time history
  params->tu_->time_history_off_ = true;
  ierr = tuMSGstd(" .. switching off time history."); CHKERRQ(ierr);
  // set and populate parameters; read material properties; read data
  params->tu_->np_ = 1;
  // transport mri if needed
  params->tu_->transport_mri_ = !app_settings->path_->mri_.empty();
  ierr = SolverInterface::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode ForwardSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // no-op
  std::stringstream ss;
  // === read data: generate synthetic or read real
  // data t1 and data t0 is generated synthetically using user given cm and tumor model
  ierr = createSynthetic(); CHKERRQ(ierr);

  ss << " Forward solve completed. Exiting.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  PetscFunctionReturn(ierr);
}

PetscErrorCode ForwardSolver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========                   InverseL2Solver                 ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL2Solver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ierr = tuMSGwarn(" Initializing Inversion for Non-Sparse TIL, and Diffusion."); CHKERRQ(ierr);
  // set and populate parameters; read material properties; read data
  ierr = SolverInterface::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);
  // reads or generates data, sets and applies observation operator
  ierr = setupData(); CHKERRQ(ierr);

  // === set Gaussians
  if (app_settings_->inject_solution_) {
    ss << " Error: injecting coarse level solution is not supported for L2 inversion. Ignoring input.";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  }
  // define number of additional inversion DOFs,
  // used in initializeGaussians to create p_rec_
  n_inv_ = params->get_nk();
  ierr = initializeGaussians(); CHKERRQ(ierr);

  // === create optimizer
  optimizer_ = std::make_shared<TILOptimizer>();
  ierr = optimizer_->initialize(derivative_operators_, pde_operators_, params_, tumor_); CHKERRQ(ierr);

  ierr = resetOperators(p_rec_); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
    ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL2Solver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Beginning Inversion for Non-Sparse TIL, and Diffusion."); CHKERRQ(ierr);
  SolverInterface::run(); CHKERRQ(ierr);

  // TODO(K) fix re-allocation of p-vector; allocation has to be moved in derived class initialize()
  // ---------
  // if(p_rec_ != nullptr) {ierr = VecDestroy(&p_rec_); CHKERRQ(ierr);}
  // ierr = VecCreateSeq(PETSC_COMM_SELF, params_->tu_->np_ + params_->get_nk(), &p_rec_); CHKERRQ(ierr);
  // ierr = setupVec(p_rec_, SEQ); CHKERRQ(ierr);
  // ierr = resetOperators(p_rec_); CHKERRQ(ierr);
  // ---------
  ScalarType *x_ptr;
  ierr = VecGetArray(p_rec_, &x_ptr); CHKERRQ (ierr);
  if (params_->get_nk() > 0) x_ptr[params_->tu_->np_] = params_->tu_->k_;
  ierr = VecRestoreArray (p_rec_, &x_ptr); CHKERRQ (ierr);

  optimizer_->setData(data_);// set data before initial guess
  ierr = optimizer_->setInitialGuess(p_rec_); CHKERRQ(ierr); // requires length: np + nk (nk=0 ok)
  ierr = optimizer_->solve(); CHKERRQ(ierr);
  ierr = VecCopy(optimizer_->getSolution(), p_rec_); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL2Solver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Finalizing Inversion for Non-Sparse TIL, and Diffusion."); CHKERRQ(ierr);
  ierr = SolverInterface::finalize(); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========                   InverseL1Solver                 ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL1Solver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ierr = tuMSGwarn(" Initializing Inversion for Sparse TIL, and Reaction/Diffusion."); CHKERRQ(ierr);
  // set and populate parameters; read material properties; read data
  ierr = SolverInterface::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);
  // reads or generates data, sets and applies observation operator
  ierr = setupData(); CHKERRQ(ierr);

  // read connected components; set sparsity level
  if (!app_settings_->path_->data_comps_data_.empty()) {
    readConCompDat(tumor_->phi_->component_weights_, tumor_->phi_->component_centers_, app_settings_->path_->data_comps_data_);
    int nnc = 0;
    for (auto w : tumor_->phi_->component_weights_)
      if (w >= 1E-3) nnc++;  // number of significant components
    ss << " Setting sparsity level to " << params_->tu_->sparsity_level_ << " x n_components (w > 1E-3) + n_components (w < 1E-3) = " << params_->tu_->sparsity_level_ << " x " << nnc << " + "
       << (tumor_->phi_->component_weights_.size() - nnc) << " = " << params_->tu_->sparsity_level_ * nnc + (tumor_->phi_->component_weights_.size() - nnc);
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    params_->tu_->sparsity_level_ = params_->tu_->sparsity_level_ * nnc + (tumor_->phi_->component_weights_.size() - nnc);
  }

  // === set Gaussians
  if (!warmstart_p_) {  // set component labels
    if (!app_settings_->path_->data_comps_.empty()) {
      ss << "  Setting component data from .nc image file.";
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      tumor_->phi_->setLabels(data_comps_); CHKERRQ(ierr);
    }
  }
  // define number of additional inversion DOFs,
  // used in initializeGaussians to create p_rec_
  n_inv_ = params_->get_nk() + params_->get_nr();
  ierr = initializeGaussians(); CHKERRQ(ierr);

  // === inject coarse level solution
  if(app_settings_->inject_solution_) {
    ss << " Injecting coarse level solution (adopting p_vec and Gaussians).";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    Vec coarse_sol = nullptr;
    int np_save = params_->tu_->np_, np_coarse = 0;  // save np, since overwritten in read function
    std::vector<ScalarType> coarse_sol_centers;
    ierr = readPhiMesh(coarse_sol_centers, params_, app_settings_->path_->phi_, false); CHKERRQ(ierr);
    ierr = readPVec(&coarse_sol, params_->tu_->np_ + params_->get_nk() + params_->get_nr(), params_->tu_->np_, app_settings_->path_->pvec_); CHKERRQ(ierr);
    np_coarse = params_->tu_->np_;
    params_->tu_->np_ = np_save;  // reset to correct value
    // find coarse centers in centers_ of current Phi
    int xc, yc, zc, xf, yf, zf;
    ScalarType *xf_ptr, *xc_ptr;
    ScalarType hx = 2.0 * M_PI / params_->grid_->n_[0];
    ScalarType hy = 2.0 * M_PI / params_->grid_->n_[1];
    ScalarType hz = 2.0 * M_PI / params_->grid_->n_[2];
    ierr = VecGetArray(p_rec_, &xf_ptr); CHKERRQ(ierr);
    ierr = VecGetArray(coarse_sol, &xc_ptr); CHKERRQ(ierr);
    for (int j = 0; j < np_coarse; ++j) {
      for (int i = 0; i < params_->tu_->np_; ++i) {
        xc = (int)std::round(coarse_sol_centers[3 * j + 0] / hx);
        yc = (int)std::round(coarse_sol_centers[3 * j + 1] / hy);
        zc = (int)std::round(coarse_sol_centers[3 * j + 2] / hz);
        xf = (int)std::round(tumor_->phi_->centers_[3 * i + 0] / hx);
        yf = (int)std::round(tumor_->phi_->centers_[3 * i + 1] / hy);
        zf = (int)std::round(tumor_->phi_->centers_[3 * i + 2] / hz);
        if (xc == xf && yc == yf && zc == zf) {
          xf_ptr[i] = 2 * xc_ptr[j];       // set initial guess (times 2 since sigma is halfed in every level)
          params_->tu_->support_.push_back(i);  // add to support
        }
      }
    }
    ierr = VecRestoreArray(p_rec_, &xf_ptr); CHKERRQ(ierr);
    ierr = VecRestoreArray(coarse_sol, &xc_ptr); CHKERRQ(ierr);
    if (coarse_sol != nullptr) {
      ierr = VecDestroy(&coarse_sol); CHKERRQ(ierr);
      coarse_sol = nullptr;
    }
  }

  // === create optimizer
  optimizer_ = std::make_shared<SparseTILOptimizer>();
  ierr = optimizer_->initialize(derivative_operators_, pde_operators_, params_, tumor_); CHKERRQ(ierr);

  ierr = resetOperators(p_rec_); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
    ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL1Solver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Beginning Inversion for Sparse TIL, and Diffusion/Reaction."); CHKERRQ(ierr);
  SolverInterface::run(); CHKERRQ(ierr);

  // set initial guess
  ScalarType *x_ptr;  // TODO(K): if read in vector has nonzero rho/kappa values, take those
  ierr = VecGetArray(p_rec_, &x_ptr); CHKERRQ (ierr);
  x_ptr[params_->tu_->np_] = params_->tu_->k_;
  if (params_->get_nr() > 0) 
    x_ptr[params_->tu_->np_ + params_->get_nk()] = params_->tu_->rho_;
  ierr = VecRestoreArray (p_rec_, &x_ptr); CHKERRQ (ierr);

  // set the reg norm as L2
  params_->opt_->regularization_norm_ = L2;
  // inv_solver_->getInverseSolverContext()->cosamp_->inexact_nits = params_->opt_->newton_maxit_; // TODO(K) restart version
  optimizer_->ctx_->cosamp_->maxit_newton = params_->opt_->newton_maxit_;
  optimizer_->setData(data_);
  ierr = optimizer_->setInitialGuess(p_rec_); CHKERRQ(ierr); // requires p_rec_ to be of lengt np + nk + nr
  ierr = optimizer_->solve(); CHKERRQ(ierr);
  ierr = VecCopy(optimizer_->getSolution(), p_rec_); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseL1Solver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Finalizing Inversion for Sparse TIL, and Reaction/Diffusion."); CHKERRQ(ierr);
  ierr = SolverInterface::finalize(); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========          InverseReactionDiffusionSolver           ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseReactionDiffusionSolver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;
  ierr = tuMSGwarn(" Initializing Reaction/Diffusion Inversion."); CHKERRQ(ierr);
  // set and populate parameters; read material properties; read data
  ierr = SolverInterface::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);
  // reads or generates data, sets and applies observation operator
  ierr = setupData(); CHKERRQ(ierr);

  // if TIL is given as parametrization Phi(p): set c(0), no phi apply in RD inversion
  if(!has_dt0_) {
    ierr = tumor_->phi_->apply(data_->dt0(), p_rec_); CHKERRQ(ierr);
    params_->tu_->use_c0_ = has_dt0_ = true;
  }

  // === create p_vec_ of correct length
  if (p_rec_ != nullptr) {
    ierr = VecDestroy(&p_rec_); CHKERRQ(ierr);
    p_rec_ = nullptr;
  }
  n_inv_ = params_->get_nk() + params_->get_nr();
  ierr = VecCreateSeq(PETSC_COMM_SELF, n_inv_, &p_rec_); CHKERRQ(ierr);
  ierr = setupVec(p_rec_, SEQ); CHKERRQ(ierr);
  ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  ss << "  .. creating p_vec of size " << n_inv_ << ", where nk = " << params_->get_nk() << " and nr = " << params_->get_nr();
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // reset p vec in tumor and pde_operators
  ierr = resetOperators(p_rec_); CHKERRQ(ierr);

  // === create optimizer
  optimizer_ = std::make_shared<RDOptimizer>();
  ierr = optimizer_->initialize(derivative_operators_, pde_operators_, params_, tumor_); CHKERRQ(ierr);

  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseReactionDiffusionSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  ierr = tuMSGwarn(" Beginning Reaction/Diffusion Inversion."); CHKERRQ(ierr);
  if (!warmstart_p_ && app_settings_->path_->data_t0_.empty()) {
    ss << " Error: c(0) needs to be set, read in p and Gaussians. Exiting.";
    ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    exit(1);
  }

  // set initial guess
  ScalarType *x_ptr;  // TODO(K): if read in vector has nonzero rho/kappa values, take those
  ierr = VecGetArray(p_rec_, &x_ptr); CHKERRQ (ierr);
  x_ptr[0] = params_->tu_->k_;
  x_ptr[params_->get_nk()] = params_->tu_->rho_;
  ierr = VecRestoreArray (p_rec_, &x_ptr); CHKERRQ (ierr);

  optimizer_->setData(data_); // set data before initial guess
  ierr = optimizer_->setInitialGuess(p_rec_); CHKERRQ(ierr); // p_vec_ has length nr + nk
  ierr = optimizer_->solve(); CHKERRQ(ierr);
  ierr = VecCopy(optimizer_->getSolution(), p_rec_); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseReactionDiffusionSolver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Finalizing Reaction/Diffusion Inversion."); CHKERRQ(ierr);
  ierr = SolverInterface::finalize(); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========               InverseMassEffectSolver             ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;
  ierr = tuMSGwarn(" Initializing Mass Effect Inversion."); CHKERRQ(ierr);

  // set and populate parameters; read material properties; read data
  ierr = SolverInterface::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);
  // reads or generates data, sets and applies observation operator
  // hack: for syn cases don't smooth data
//  params_->tu_->smoothing_factor_data_ = 0;
//  params_->tu_->smoothing_factor_patient_ = 0;
//  params_->tu_->smoothing_factor_data_t0_ = 0; // set to zero if c0 does not need to be smoothed
  ierr = setupData(); CHKERRQ(ierr);
  // read mass effect patient data
  ierr = readPatient(); CHKERRQ(ierr);
  // enable mass effect inversion in optimizer
  params_->opt_->invert_mass_effect_ = true;

  // if TIL is given as parametrization Phi(p): set c(0), no phi apply in RD inversion
  if(!has_dt0_) {
    // set gaussians from p_rec_ read and then apply phi
    ierr = initializeGaussians(); CHKERRQ(ierr);
    ierr = VecDuplicate(tmp_, &data_t0_); CHKERRQ(ierr);
    data_->setT0(data_t0_);
    ierr = tumor_->phi_->apply(data_->dt0(), p_rec_); CHKERRQ(ierr);
    params_->tu_->use_c0_ = has_dt0_ = true;
  }

  // === create p_vec_ of correct length
  if (p_rec_ != nullptr) {
    ierr = VecDestroy(&p_rec_); CHKERRQ(ierr);
    p_rec_ = nullptr;
  }
  n_inv_ = 1 + params_->get_nk() + params_->get_nr();
  ierr = VecCreateSeq(PETSC_COMM_SELF, n_inv_, &p_rec_); CHKERRQ(ierr);
  ierr = setupVec(p_rec_, SEQ); CHKERRQ(ierr);
  ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  ss << "  .. creating p_vec of size " << n_inv_ << ", where nk = " << params_->get_nk() << " and nr = " << params_->get_nr();
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // reset p vec in tumor and pde_operators
  ierr = resetOperators(p_rec_); CHKERRQ(ierr);


  // === create optimizer
  optimizer_ = std::make_shared<MEOptimizer>();
  ierr = optimizer_->initialize(derivative_operators_, pde_operators_, params_, tumor_); CHKERRQ(ierr);
  // set patient material properties
  ierr = derivative_operators_->setMaterialProperties(p_gm_, p_wm_, p_vt_, p_csf_); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  if (!warmstart_p_) {
    ierr = VecSet(p_rec_, 0); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Beginning Mass Effect Inversion."); CHKERRQ(ierr);

  // // TODO(K) fix re-allocation of p-vector; allocation has to be moved in derived class initialize()
  // // ---------
  // if(p_rec_ != nullptr) {ierr = VecDestroy(&p_rec_); CHKERRQ(ierr);}
  // ierr = VecCreateSeq(PETSC_COMM_SELF, params_->tu_->np_ + params_->get_nk() + params_->get_nr() + 1, &p_rec_); CHKERRQ(ierr);
  // ierr = setupVec(p_rec_, SEQ); CHKERRQ(ierr);
  // ierr = resetOperators(p_rec_); CHKERRQ(ierr);
  // // ---------
  // int np = params_->tu_->np_;

  // == set initial guess
  int nk = params_->get_nk();
  int nr = params_->get_nr();
  ScalarType *x_ptr; // TODO(K): if read in vector has nonzero gamma/rho/kappa values, take those
  ierr = VecGetArray(p_rec_, &x_ptr); CHKERRQ (ierr);
  std::random_device rd; // obtain a random number from hardware
  std::mt19937 eng(rd()); // seed the generator
  std::uniform_real_distribution<> distg(0.1, 1.0); // define the range
  std::uniform_real_distribution<> distr(5, 10); // define the range
  std::uniform_real_distribution<> distk(0.5, 5); // define the range
//  x_ptr[0] = params_->tu_->forcing_factor_;
//  x_ptr[1] = params_->tu_->rho_;
//  x_ptr[2] = params_->tu_->k_;
  x_ptr[0] = distg(eng) * 1E5;
  x_ptr[1] = distr(eng);
  x_ptr[2] = distk(eng) * 1E-2;
  ierr = VecRestoreArray(p_rec_, &x_ptr); CHKERRQ (ierr);

  optimizer_->setData(data_); // set data before initial guess
  ierr = optimizer_->setInitialGuess(p_rec_); CHKERRQ(ierr);

  
//  Vec x_scale; 
//  ierr = VecDuplicate(p_rec_, &x_scale); CHKERRQ(ierr);
//
//  ScalarType *scale_ptr;
//  ierr = VecGetArray(p_rec_, &x_ptr); CHKERRQ(ierr);
//  ierr = VecGetArray(x_scale, &scale_ptr); CHKERRQ(ierr);
//  scale_ptr[0] = 0*x_ptr[0] / 1E4;
//  scale_ptr[1] = 0*x_ptr[1];
//  scale_ptr[2] = 0*x_ptr[2] / 1E-2 + 0.5;
//  ierr = VecRestoreArray(p_rec_, &x_ptr); CHKERRQ(ierr);
//
//  Vec d1, d2;
//  ierr = VecDuplicate(x_scale, &d1); CHKERRQ(ierr);
//  ierr = VecDuplicate(x_scale, &d2); CHKERRQ(ierr);
//
//  ScalarType *d1_ptr, *d2_ptr;
//  ierr = VecGetArray(d1, &d1_ptr); CHKERRQ(ierr);
//  ierr = VecGetArray(d2, &d2_ptr); CHKERRQ(ierr);
// 
//  d1_ptr[0] = 0.402885 - scale_ptr[0];
//  d1_ptr[1] = 9.22808 - scale_ptr[1];
//  d1_ptr[2] = 2.4698 - scale_ptr[2];
//  d2_ptr[0] = 1.2 - scale_ptr[0];
//  d2_ptr[1] = 12 - scale_ptr[1];
//  d2_ptr[2] = 5 - scale_ptr[2];
//
//  ierr = VecRestoreArray(d1, &d1_ptr); CHKERRQ(ierr);
//  ierr = VecRestoreArray(d2, &d2_ptr); CHKERRQ(ierr);
//  ierr = VecRestoreArray(x_scale, &scale_ptr); CHKERRQ(ierr);
//
//  ierr = derivative_operators_->visLossLandscape(x_scale, d2, nullptr, data_, "obj-list"); CHKERRQ(ierr);
//  ierr = VecDestroy(&d1); CHKERRQ(ierr);
//  ierr = VecDestroy(&d2); CHKERRQ(ierr);

  // SNAFU: compute hessian
//  ierr = tuMSGstd("computing FD hessian at IC"); CHKERRQ(ierr);
//  Vec x_scale; 
//  ierr = VecDuplicate(p_rec_, &x_scale); CHKERRQ(ierr);
//  ScalarType *scale_ptr;
//  ierr = VecGetArray(p_rec_, &x_ptr); CHKERRQ(ierr);
//  ierr = VecGetArray(x_scale, &scale_ptr); CHKERRQ(ierr);
//  scale_ptr[0] = x_ptr[0] / 1E4;
//  scale_ptr[1] = x_ptr[1];
//  scale_ptr[2] = x_ptr[2] / 1E-2;
//  ierr = VecRestoreArray(p_rec_, &x_ptr); CHKERRQ(ierr);
//  ierr = VecRestoreArray(x_scale, &scale_ptr); CHKERRQ(ierr);
//  std::string ss_str = "IC";
//  ierr = derivative_operators_->computeFDHessian(x_scale, data_, ss_str); CHKERRQ(ierr); 
//  ierr = tuMSGstd("Hessian compute complete."); CHKERRQ(ierr);

  ierr = optimizer_->solve(); CHKERRQ(ierr);
  ierr = VecCopy(optimizer_->getSolution(), p_rec_); CHKERRQ(ierr);
//
 // SNAFU: compute hessian
// ierr = tuMSGstd("computing FD hessian at optimum"); CHKERRQ(ierr);
// ierr = VecCopy(p_rec_, x_scale); CHKERRQ(ierr);
// ss_str = "opt";
// ierr = derivative_operators_->computeFDHessian(x_scale, data_, ss_str); CHKERRQ(ierr); 
//
// ierr = VecDestroy(&x_scale); CHKERRQ(ierr);

  // Reset mat-props and diffusion and reaction operators, tumor IC does not change
  ierr = tumor_->mat_prop_->resetValues(); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);
  ierr = tumor_->velocity_->set(0.);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  std::stringstream ss;
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  ierr = tuMSGwarn(" Finalizing Mass Effect Inversion."); CHKERRQ(ierr);

  // === compute errors
  ScalarType *c0_ptr;
  if (params_->tu_->write_output_) {
    ierr = dataOut(tumor_->c_0_, params_, "c0_rec" + params_->tu_->ext_); CHKERRQ(ierr);
  }
  // transport mri
  params_->tu_->transport_mri_ = !app_settings_->path_->mri_.empty();
  if (params_->tu_->transport_mri_) {
    if (mri_ == nullptr) {
      ierr = VecDuplicate(tmp_, &mri_); CHKERRQ(ierr);
      ierr = dataIn(mri_, params_, app_settings_->path_->mri_); CHKERRQ(ierr);
      if (tumor_->mat_prop_->mri_ == nullptr) {
        tumor_->mat_prop_->mri_ = mri_;
      }
    }
  }

  if (procid == 0) {
    if (params_->tu_->verbosity_ >= 1) {
      // print reconstructed p_vec
      ss << " --------------  RECONST VEC -----------------";
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
      ierr = VecView(p_rec_, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
      ss << " --------------  -------------- -----------------";
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
    }
  }
  ierr = pde_operators_->solveState(0); CHKERRQ(ierr);
  ierr = VecCopy(tumor_->c_t_, tmp_); CHKERRQ(ierr);

  if (params_->tu_->write_output_) {
    ierr = tumor_->computeSegmentation(); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "seg_rec_final";
    ierr = dataOut(tumor_->seg_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "c_rec_final";
    ierr = dataOut(tumor_->c_t_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "vt_rec_final";
    ierr = dataOut(tumor_->mat_prop_->vt_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "csf_rec_final";
    ierr = dataOut(tumor_->mat_prop_->csf_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "wm_rec_final";
    ierr = dataOut(tumor_->mat_prop_->wm_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "gm_rec_final";
    ierr = dataOut(tumor_->mat_prop_->gm_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    Vec mag = nullptr;
    ierr = pde_operators_->getModelSpecificVector(&mag);
    ierr = tumor_->displacement_->computeMagnitude(mag);
    ss << "displacement_rec_final";
    ierr = dataOut(mag, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ScalarType mag_norm, mm;
    ierr = VecNorm(mag, NORM_2, &mag_norm); CHKERRQ(ierr);
    ierr = VecMax(mag, NULL, &mm); CHKERRQ(ierr);
    ss << " Norm of reconstructed displacement: " << mag_norm << "; max of reconstructed displacement: " << mm;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    if (tumor_->mat_prop_->mri_ != nullptr) {
      ss << "mri_rec_final";
      ierr = dataOut(tumor_->mat_prop_->mri_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
    }
  }

  ScalarType max, min;
  ierr = VecMax(tmp_, NULL, &max); CHKERRQ(ierr);
  ierr = VecMin(tmp_, NULL, &min); CHKERRQ(ierr);
  ss << " Reconstructed c(1) max and min : " << max << " " << min;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  if (params_->tu_->write_output_) ierr = dataOut(tmp_, params_, "c1_rec" + params_->tu_->ext_); CHKERRQ(ierr);

  // copy c(1)
  ScalarType data_norm, error_norm, error_norm_0;
  Vec c1_obs;
  ierr = VecDuplicate(tmp_, &c1_obs); CHKERRQ(ierr);
  ierr = VecCopy(tmp_, c1_obs); CHKERRQ(ierr);

  // c(1): error everywhere
  ierr = VecAXPY(tmp_, -1.0, data_->dt1()); CHKERRQ(ierr);
  ierr = VecNorm(data_->dt1(), NORM_2, &data_norm); CHKERRQ(ierr);
  ierr = VecNorm(tmp_, NORM_2, &error_norm); CHKERRQ(ierr);
  ss << " t=1: l2-error (everywhere): " << error_norm;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  error_norm /= data_norm;
  ss << " t=1: rel. l2-error (everywhere): " << error_norm;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();

  // c(1): error at observation points
  ierr = tumor_->obs_->apply(c1_obs, c1_obs, 1); CHKERRQ(ierr);
  ierr = tumor_->obs_->apply(tmp_, data_->dt1(), 1); CHKERRQ(ierr);
  ierr = VecAXPY(c1_obs, -1.0, tmp_); CHKERRQ(ierr);
  ierr = VecNorm(tmp_, NORM_2, &data_norm); CHKERRQ(ierr);
  ierr = VecNorm(c1_obs, NORM_2, &error_norm); CHKERRQ(ierr);
  ss << " t=1: l2-error (at observation points): " << error_norm;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  error_norm /= data_norm;
  ss << " t=1: rel. l2-error (at observation points): " << error_norm;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  ierr = VecDestroy(&c1_obs); CHKERRQ(ierr);

  // c(0): error everywhere
  if(data_t0_ != nullptr) {
    ierr = VecCopy(tumor_->c_0_, tmp_); CHKERRQ(ierr);
    ierr = VecAXPY(tmp_, -1.0, data_t0_); CHKERRQ(ierr);
    ierr = VecNorm(data_t0_, NORM_2, &data_norm); CHKERRQ(ierr);
    ierr = VecNorm(tmp_, NORM_2, &error_norm_0); CHKERRQ(ierr);
    ss << " t=0: l2-error (everywhere): " << error_norm_0;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    error_norm_0 /= data_norm;
    ss << " t=0: rel. l2-error (everywhere): " << error_norm_0;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  } else {
    ierr = tuMSGstd(" Cannot compute errors for TIL, since TIL is nullptr."); CHKERRQ(ierr);
  }

  // write file
  if (procid == 0) {
    std::ofstream opfile;
    opfile.open(params_->tu_->writepath_ + "reconstruction_info.dat");
    opfile << "rho k gamma c1_rel c0_rel \n";
    opfile << params_->tu_->rho_ << " " << params_->tu_->k_ << " " << params_->tu_->forcing_factor_ << " " << error_norm << " " << error_norm_0 << std::endl;
    opfile.flush();
    opfile.close();
  }

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InverseMassEffectSolver::readPatient() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ScalarType sigma_smooth = params_->tu_->smoothing_factor_ * 2 * M_PI / params_->grid_->n_[0];

  if (!app_settings_->path_->p_seg_.empty()) {
    ierr = dataIn(tmp_, params_, app_settings_->path_->p_seg_); CHKERRQ(ierr);
    if(app_settings_->patient_seg_[0] <= 0 || app_settings_->patient_seg_[1] <= 0 || app_settings_->patient_seg_[2] <= 0) {
      ierr = tuMSGwarn(" Error: Patient segmentation must at least have WM, GM, VT."); CHKERRQ(ierr);
      exit(0);
    } else {
      ierr = VecDuplicate(tmp_, &wm_); CHKERRQ(ierr);
      ierr = VecDuplicate(tmp_, &gm_); CHKERRQ(ierr);
      ierr = VecDuplicate(tmp_, &vt_); CHKERRQ(ierr);
      csf_ = nullptr; data_t1_ = nullptr; ed_ = nullptr;
      if (app_settings_->patient_seg_[3] > 0) {
        ierr = VecDuplicate(tmp_, &csf_); CHKERRQ(ierr);
      }
      // tc exists as label, or necrotic core + enhancing rim exist as labels
      if (app_settings_->patient_seg_[4] > 0 || (app_settings_->patient_seg_[5] > 0 && app_settings_->patient_seg_[6] > 0)) {
        ierr = VecDuplicate(tmp_, &data_t1_); CHKERRQ(ierr);
        data_t1_from_seg_ = true;
        // edema exists as label
        if (app_settings_->patient_seg_[7] > 0) {
          ierr = VecDuplicate(tmp_, &ed_); CHKERRQ(ierr);
        }
      }
    }
    ierr = splitSegmentation(tmp_, wm_, gm_, vt_, csf_, data_t1_, ed_, params_->grid_->nl_, app_settings_->patient_seg_); CHKERRQ(ierr);

    // TODO(K): populate to wm, gm, csf, ve
  } else {
    if (!app_settings_->path_->p_wm_.empty()) {
      ierr = VecDuplicate(tmp_, &p_wm_); CHKERRQ(ierr);
      ierr = dataIn(p_wm_, params_, app_settings_->path_->p_wm_); CHKERRQ(ierr);
    }
    if (!app_settings_->path_->p_gm_.empty()) {
      ierr = VecDuplicate(tmp_, &p_gm_); CHKERRQ(ierr);
      ierr = dataIn(p_gm_, params_, app_settings_->path_->p_gm_); CHKERRQ(ierr);
    }
    if (!app_settings_->path_->p_vt_.empty()) {
      ierr = VecDuplicate(tmp_, &p_vt_); CHKERRQ(ierr);
      ierr = dataIn(p_vt_, params_, app_settings_->path_->p_vt_); CHKERRQ(ierr);
    }
    if (!app_settings_->path_->p_csf_.empty()) {
      ierr = VecDuplicate(tmp_, &p_csf_); CHKERRQ(ierr);
      ierr = dataIn(p_csf_, params_, app_settings_->path_->p_csf_); CHKERRQ(ierr);
    }
  }
  // smooth
  if (params_->tu_->smoothing_factor_patient_ > 0) {
    if (p_gm_ != nullptr) {
      ierr = spec_ops_->weierstrassSmoother(p_gm_, p_gm_, params_, sigma_smooth); CHKERRQ(ierr);
    }
    if (p_wm_ != nullptr) {
      ierr = spec_ops_->weierstrassSmoother(p_wm_, p_wm_, params_, sigma_smooth); CHKERRQ(ierr);
    }
    if (p_vt_ != nullptr) {
      ierr = spec_ops_->weierstrassSmoother(p_vt_, p_vt_, params_, sigma_smooth); CHKERRQ(ierr);
    }
    if (p_csf_ != nullptr) {
      ierr = spec_ops_->weierstrassSmoother(p_csf_, p_csf_, params_, sigma_smooth); CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(ierr);
}

/* #### ------------------------------------------------------------------- #### */
/* #### ========             MultiSpeciesSolver             ======== #### */
/* #### ------------------------------------------------------------------- #### */

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode MultiSpeciesSolver::initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  ierr = tuMSGwarn(" Initializing Multi Species Forward Solver."); CHKERRQ(ierr);

  params->tu_->time_history_off_ = true;
  ierr = SolverInterface::initialize(spec_ops, params, app_settings); CHKERRQ(ierr);

  ierr = resetOperators(p_rec_); CHKERRQ(ierr);
  // ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_);
  // ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode MultiSpeciesSolver::run() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  if (has_dt0_) {
    ierr = VecCopy(data_->dt0(), tumor_->c_0_); CHKERRQ(ierr);
  } else {
    ierr = tumor_->phi_->apply(tumor_->c_0_, p_rec_); CHKERRQ(ierr);
  }

  ierr = tuMSGwarn(" Beginning Multi Species Forward Solve."); CHKERRQ(ierr);
  // TODO(K): call multi species solver

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode MultiSpeciesSolver::finalize() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = tuMSGwarn(" Finalizing Multi Species Forward Solve."); CHKERRQ(ierr);
  ierr = SolverInterface::finalize(); CHKERRQ(ierr);

  std::stringstream ss;
  if (params_->tu_->write_output_) {
    ierr = tumor_->computeSegmentation(); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "seg_rec_final";
    ierr = dataOut(tumor_->seg_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "c_rec_final";
    ierr = dataOut(tumor_->c_t_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "vt_rec_final";
    ierr = dataOut(tumor_->mat_prop_->vt_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "csf_rec_final";
    ierr = dataOut(tumor_->mat_prop_->csf_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "wm_rec_final";
    ierr = dataOut(tumor_->mat_prop_->wm_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ss << "gm_rec_final";
    ierr = dataOut(tumor_->mat_prop_->gm_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    Vec mag = nullptr;
    ierr = pde_operators_->getModelSpecificVector(&mag);
    ierr = tumor_->displacement_->computeMagnitude(mag);
    ss << "displacement_rec_final";
    ierr = dataOut(mag, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    ScalarType mag_norm, mm;
    ierr = VecNorm(mag, NORM_2, &mag_norm); CHKERRQ(ierr);
    ierr = VecMax(mag, NULL, &mm); CHKERRQ(ierr);
    ss << " Norm of displacement: " << mag_norm << "; max of displacement: " << mm;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str(std::string());
    ss.clear();
    if (tumor_->mat_prop_->mri_ != nullptr) {
      ss << "mri_rec_final";
      ierr = dataOut(tumor_->mat_prop_->mri_, params_, ss.str() + params_->tu_->ext_); CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
    }
  }

  PetscFunctionReturn(ierr);
}
