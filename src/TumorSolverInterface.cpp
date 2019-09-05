data_gradeval/*************************************************************************
 *  Copyright (c) 2016-2017.
 *  All rights reserved.
 *  This file is part of the SIBIA library.
 *
 *  SIBIA is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  SIBIA is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SIBIA. If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#include "TumorSolverInterface.h"
#include "EventTimings.hpp"


namespace pglistr {

// ### _____________________________________________________________________ ___
// ### ///////////////// constructor /////////////////////////////////////// ###
TumorSolverInterface::TumorSolverInterface (
    std::shared_ptr<NMisc> n_misc,
    std::shared_ptr<Phi> phi,
    std::shared_ptr<MatProp> mat_prop)
    :
  initialized_ (false)
, initializedFFT_(false)
, optimizer_settings_changed_ (false)
, regularization_norm_changed_(false)
, newton_solver_type_changed_(false)
, n_misc_ (n_misc)
, tumor_ ()
, pde_operators_ ()
, derivative_operators_ ()
, inv_solver_ ()
{
    if (n_misc != nullptr)
        initialize (n_misc, phi, mat_prop);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// finalize ////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::finalize ()
{
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    accfft_destroy_plan (ivars.plan);
    accfft_cleanup();
    MPI_Comm_free(&ivars.comm);
    initializedFFT_ = false;
    initialized_ = false;
    // ierr = PetscFinalize ();
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// initializeFFT ///////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::initializeFFT (
    DataDistributionParameters& ivars)
{
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    // don't do it twice
    if (initializeFFT_) PetscFunctionReturn (0);
    // initialize accfft, data distribution and comm plan
    accfft_init();
    accfft_create_comm(MPI_COMM_WORLD, ivars.cdims, &ivars.comm);
    ivars.alloc_max = accfft_local_size_dft_r2c (ivars.n, ivars.isize, ivars.istart, ivars.osize, ivars.ostart, ivars.comm);
    double *c_0 = (double*) accfft_alloc (ivars.alloc_max);
    Complex *c_hat = (Complex*) accfft_alloc (ivars.alloc_max);
    ivars.plan = accfft_plan_dft_3d_r2c (ivars.n, c_0, (double*) c_hat, ivars.comm, ACCFFT_MEASURE);
    accfft_free (c_0);
    accfft_free (c_hat);
    initializeFFT_ = true;
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// TODO ////////////////////////////////////////////// ###
// # TODO: rewrite initialization process
// #  - need to do all the setup stuff from app/inverse.cpp
// #  - function that reads in atlas (i.e., initial mat probs)
// #  - function that reads in data
// #  - possibly InitVars als otake np and Gaussian selection mode, and create PHi from that (so does't need to be recreated)
// ### _____________________________________________________________________ ___
// ### ///////////////// TODO ////////////////////////////////////////////// ###


// ### _____________________________________________________________________ ___
// ### ///////////////// initialize //////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::initialize (
    DataDistributionParameters& ivars)
{
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    // don't do it twice
    if (initialized_) PetscFunctionReturn (0);
    // FFT needs to be initialized
    if (!initializeFFT_) {ierr = tuMSGwarn("Error: FFT needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    // create n_misc
    n_misc_ =  std::make_shared<NMisc> (
        ivars.n, ivars.isize, ivars.osize, ivars.istart, ivars.ostart,
        ivars.plan, ivars.comm, ivars.cdims, ivars.testcase);

    // initialize tumor, initialize dummy phi, initialize mat probs
    initialize(n_misc_, nullptr, nullptr)
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// initialize //////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::initialize (
    std::shared_ptr<NMisc> n_misc,
    std::shared_ptr<Phi> phi,
    std::shared_ptr<MatProp> mat_prop)
{
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (initialized_) PetscFunctionReturn (0);

    tumor_ = std::make_shared<Tumor> (n_misc);
    n_misc_ = n_misc;
    // set up vector p (should also add option to pass a p vec, that is used to initialize tumor)
    Vec p;
    int np = n_misc_->np_;
    int nk = (n_misc_->diffusivity_inversion_) ? n_misc_->nk_ : 0;

    ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p);                             CHKERRQ (ierr);
    ierr = VecSet (p, n_misc->p_scale_);                                            CHKERRQ (ierr);
    ierr = tumor_->initialize (p, n_misc, phi, mat_prop);

    // create pde and derivative operators
    if (n_misc->model_ == 1) {
        pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc);
        derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc, tumor_);
    }
    if (n_misc->model_ == 2) {
        pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc);
        derivative_operators_ = std::make_shared<DerivativeOperatorsPos> (pde_operators_, n_misc, tumor_);
    }
    if (n_misc->model_ == 3) {
        pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc);
        derivative_operators_ = std::make_shared<DerivativeOperatorsRDObj> (pde_operators_, n_misc, tumor_);
    }
    if (n_misc->model_ == 4) {
        pde_operators_ = std::make_shared<PdeOperatorsMassEffect> (tumor_, n_misc);
        derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc, tumor_);
    }
    // create tumor inverse solver
    inv_solver_ = std::make_shared<InvSolver> (derivative_operators_, pde_operators_, n_misc, tumor_);
    ierr = inv_solver_->initialize (derivative_operators_, pde_operators_, n_misc, tumor_); CHKERRQ (ierr);
    initialized_ = true;
    // cleanup
    ierr = VecDestroy (&p);                                                         CHKERRQ (ierr);
    PetscFunctionReturn (ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setParams ///////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setParams (
    Vec p, s
    td::shared_ptr<TumorSettings> tumor_params = {})
{
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert (initialized_, "TumorSolverInterface::setParams(): TumorSolverInterface needs to be initialized.")
    bool npchanged = true, modelchanged = false, ntchanged = false;

    // ++ re-initialize nmisc ==
    if(tumor_params != nullptr) {
        // if one of these parameters has changed, we need to re-allocate maemory
        npchanged                       = (n_misc_->np_+n_misc_->nk_) != (tumor_params->np+tumor_params->nk);
        modelchanged                    = n_misc_->model_ != tumor_params->tumor_model;
        ntchanged                       = n_misc_->nt_ != tumor_params->time_steps;
        n_misc_->model_                 = tumor_params->tumor_model;
        n_misc_->dt_                    = tumor_params->time_step_size;
        n_misc_->nt_                    = tumor_params->time_steps;
        n_misc_->time_horizon_          = tumor_params->time_horizon;
        n_misc_->np_                    = tumor_params->np;
        n_misc_->beta_                  = tumor_params->betap;
        n_misc_->writeOutput_           = tumor_params->writeOutput;
        n_misc_->verbosity_             = tumor_params->verbosity;
        n_misc_->obs_threshold_         = tumor_params->obs_threshold;
        n_misc_->k_                     = tumor_params->diff_coeff_scale;
        n_misc_->kf_                    = tumor_params->diff_coeff_scale_anisotropic;
        n_misc_->rho_                   = tumor_params->reaction_coeff_scale;
        n_misc_->k_gm_wm_ratio_         = tumor_params->diffusion_ratio_gm_wm;
        n_misc_->k_glm_wm_ratio_        = tumor_params->diffusion_ratio_glm_wm;
        n_misc_->r_gm_wm_ratio_         = tumor_params->reaction_ratio_gm_wm;
        n_misc_->r_glm_wm_ratio_        = tumor_params->reaction_ratio_glm_wm;
        n_misc_->user_cm_               = tumor_params->phi_center_of_mass;
        n_misc_->phi_spacing_factor_    = tumor_params->phi_spacing_factor;
        n_misc_->phi_sigma_             = tumor_params->phi_sigma;
        n_misc_->phi_sigma_data_driven_ = tumor_params->phi_sigma_data_driven;
        n_misc_->gaussian_vol_frac_     = tumor_params->gaussian_volume_fraction;
        n_misc_->target_sparsity_       = tumor_params->target_sparsity;
        n_misc_->bounding_box_          = tumor_params->phi_selection_mode_bbox;
        n_misc_->diffusivity_inversion_ = tumor_params->diffusivity_inversion;
        n_misc_->reaction_inversion_    = tumor_params->reaction_inversion;
        n_misc_->nk_                    = tumor_params->nk;
        n_misc_->phi_store_             = tumor_params->phi_store;
        n_misc_->adjoint_store_         = tumor_params->adjoint_store;
        n_misc_->multilevel_            = tumor_params->multilevel;
        n_misc_->prune_components_      = tumor_params->prune_components;
        n_misc_->sparsity_level_        = tumor_params->sparsity_level;
    }
    // ++ re-initialize Tumor ++
    ierr = tumor_->setParams (p, n_misc_, npchanged);                           CHKERRQ (ierr);
    // ++ re-initialize pdeoperators and derivativeoperators ++
    // invcludes re-allocating time history for adjoint,
    if(modelchanged) {
        switch (n_misc_->model_) {
            case 1: pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc_);
                    derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc_, tumor_);
                    break;
            case 2: pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc_);
                    derivative_operators_ = std::make_shared<DerivativeOperatorsPos> (pde_operators_, n_misc_, tumor_);
                    break;
            case 3: pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc_);
                    derivative_operators_ = std::make_shared<DerivativeOperatorsRDObj> (pde_operators_, n_misc_, tumor_);
                    break;
            case 4: pde_operators_ = std::make_shared<PdeOperatorsMassEffect> (tumor_, n_misc_);
                    derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc_, tumor_);
                    break;
            default: break;
        }
    } else {
        // re-allocate vectors, same model
        if (npchanged) derivative_operators_->reset(p, pde_operators_, n_misc_, tumor_);
        if (ntchanged) pde_operators_->reset(n_misc_, tumor_);
    }
    // ++ re-initialize InvSolver ++, i.e. H matrix, p_rec vectores etc..
    inv_solver_->setParams(derivative_operators_, pde_operators_, n_misc_, tumor_, npchanged);  CHKERRQ (ierr);

    PetscFunctionReturn(0);
}

// TODO[MEMORY]: add switch, if we want to copy or take the pointer from incoming and outgoing data
// ### _____________________________________________________________________ ___
// ### ///////////////// solveForward ////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveForward (
    Vec c1, Vec c0)
{
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (!initialized_) {ierr = tuMSGwarn("Error: (solveForward) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    ierr = VecCopy (c0, tumor_->c_0_);         CHKERRQ (ierr); // set the initial condition
    ierr = pde_operators_->solveState (0);     CHKERRQ (ierr); // solve forward
    ierr = VecCopy (tumor_->c_t_, c1);         CHKERRQ (ierr); // get solution
    PetscFunctionReturn(0);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// solveInverseCoSaMp ////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveInverseCoSaMp (
    Vec prec,
    Vec data, Vec data_gradeval)
{
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    std::stringstream ss;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    if (!initialized_)  {ierr = tuMSGwarn("Error: (solveInverseCoSaMp) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (data == nullptr){ierr = tuMSGwarn("Error: (solveInverseCoSaMp) Variable data cannot be nullptr. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (prec == nullptr){ierr = tuMSGwarn("Error: (solveInverseCoSaMp) Variable prec cannot be nullptr. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (!optimizer_settings_changed_) {ierr = tuMSGwarn (" Tumor inverse solver running with default settings."); CHKERRQ (ierr);}

    // set target data for inversion (just sets the vector, no deep copy)
    inv_solver_->setData (data); if (data_gradeval == nullptr) data_gradeval = data;
    inv_solver_->setDataGradient (data_gradeval);

    // count the number of observed voxels
    if (n_misc_->verbosity > 2) {
      int sum = 0, global_sum = 0;
      double *pixel_ptr;
      ierr = VecGetArray (data, &pixel_ptr);                                        CHKERRQ (ierr);
      for (int i = 0; i < n_misc_->n_local_; i++)
          if (pixel_ptr[i] > n_misc_->obs_threshold_) sum++;
      ierr = VecRestoreArray (data, &pixel_ptr);                                    CHKERRQ (ierr);
      MPI_Reduce (&sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
      ss << " number of observed voxels: " << global_sum;
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    }

    // inexact Newton
    inv_solver_->getInverseSolverContext()->cosamp_->maxit_newton = n_misc_->newton_maxit_;
    // inv_solver_->getInverseSolverContext()->cosamp_->inexact_nits = n_misc_->newton_maxit_;

    // solve
    inv_solver_->solveInverseCoSaMp();
    // inv_solver_->solveInverseCoSaMpRS(false);
    ierr = VecCopy (inv_solver_->getPrec(), prec);                                  CHKERRQ (ierr);

    PetscFunctionReturn(ierr);
}

// TODO[MEMORY]: add switch, if we want to copy or take the pointer from incoming and outgoing data
// ### _____________________________________________________________________ ___
// ### ///////////////// solveInverse ////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveInverse (
    Vec prec,
    Vec data, Vec data_gradeval)
{
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    if (!initialized_)  {ierr = tuMSGwarn("Error: (solveInverse) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (data == nullptr){ierr = tuMSGwarn("Error: (solveInverse) Variable data cannot be nullptr. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (prec == nullptr){ierr = tuMSGwarn("Error: (solveInverse) Variable prec cannot be nullptr. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (!optimizer_settings_changed_) {ierr = tuMSGwarn (" Tumor inverse solver running with default settings."); CHKERRQ (ierr);}

    // reset TAO object if either the solver type changes or the regularization norm
    if(_newton_solver_type_changed || _regularization_norm_changed) {
        ierr = resetTaoSolver();                                                     CHKERRQ(ierr);
        _newton_solver_type_changed = false; _regularization_norm_changed = false;
    }

    // TODO[CHANGE]: set the observation operator filter : default filter
    ierr = tumor_->obs_->setDefaultFilter (data);                                   CHKERRQ (ierr);
    ierr = tumor_->obs_->apply (data, data);                                        CHKERRQ (ierr);

    // set target data for inversion (just sets the vector, no deep copy) and solve
    inv_solver_->setData (data); if (data_gradeval == nullptr) data_gradeval = data;
    inv_solver_->setDataGradient (data_gradeval);
    ierr = inv_solver_->solve ();                                                   CHKERRQ (ierr);
    // pass the reconstructed p vector to the caller (deep copy)
    ierr= VecCopy (inv_solver_->getPrec(), prec);                                   CHKERRQ (ierr);

    updateReferenceGradient(false);
    PetscFunctionReturn (0);
}


// ### _____________________________________________________________________ ___
// ### ///////////////// solveInverseReacDiff ////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveInverseReacDiff(
    Vec prec,
    Vec data, Vec data_gradeval)
{
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  double *ptr_pr_rec;
  int np = n_misc_->np_;
  int nk = (n_misc_->reaction_inversion_ || n_misc_->diffusivity_inversion_) ? n_misc_->nk_ : 0;
  int nr = (n_misc_->reaction_inversion_) ? n_misc_->nr_ : 0;

  // ------------- L2 solve to solve for p_i, given sparsity -------------------
  ierr = VecGetArray(prec, &ptr_pr_rec);                                        CHKERRQ (ierr);
  // set initial guess for k_inv (possibly != zero)
  if (n_misc_->diffusivity_inversion_) {
      ptr_pr_rec[np] = n_misc_->k_;
      if (nk > 1) ptr_pr_rec[np+1] = n_misc_->k_ * n_misc_->k_gm_wm_ratio_;
      if (nk > 2) ptr_pr_rec[np+2] = n_misc_->k_ * n_misc_->k_glm_wm_ratio_;
  } else {
      // set diff ops with this guess -- this will not change during the solve
      ierr = getTumor()->k_->setValues (n_misc_->k_, n_misc_->k_gm_wm_ratio_, n_misc_->k_glm_wm_ratio_, getTumor()->mat_prop_, n_misc_);  CHKERRQ (ierr);
  }
  // set initial guess for rho (possibly != zero)
  if (n_misc_->reaction_inversion_) {
      ptr_pr_rec[np + nk] = n_misc_->rho_;
      if (nr > 1) ptr_pr_rec[np + nk + 1] = n_misc_->rho_ * n_misc_->r_gm_wm_ratio_;
      if (nr > 2) ptr_pr_rec[np + nk + 2] = n_misc_->rho_ * n_misc_->r_glm_wm_ratio_;
  }
  ierr = VecRestoreArray(prec, &ptr_pr_rec);                                    CHKERRQ (ierr);

  // set target data for inversion (just sets the vector, no deep copy) and solve
  inv_solver_->setData (data); if (data_gradeval == nullptr) data_gradeval = data;
  inv_solver_->setDataGradient (data_gradeval);

  ierr = resetTaoSolver();                                                      CHKERRQ (ierr);
  ierr = setParams (prec, nullptr);                                             CHKERRQ (ierr);
  ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
  ierr = tuMSG    ("###                  Initial L2 Solve                 ###"); CHKERRQ (ierr);
  ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
  ierr = inv_solver_->solve ();
  ierr = VecCopy (inv_solver_->getPrec(), prec);                                               CHKERRQ (ierr);

  if (n_misc_->reaction_inversion_) {
    inv_solver_->itctx_->cosamp_->cosamp_stage = POST_RD;
    ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
    ierr = tuMSG    ("### rho/kappa inversion with scaled L2 solution guess ###"); CHKERRQ (ierr);
    ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
    inv_solver_->setData (data); if (data_gradeval == nullptr) data_gradeval = data;
    inv_solver_->setDataGradient (data_gradeval);
    ierr = inv_solver_->solveInverseReacDiff (prec);
    ierr = VecCopy (inv_solver_->getPrec(), prec);                            CHKERRQ (ierr);
  } else {
    ierr = tuMSGwarn ( " WARNING: Attempting to solve for reaction and diffusion, but reaction inversion is nor enabled."); CHKERRQ (ierr);
  }

  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// computeGradient /////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::computeGradient (
    Vec dJ, Vec p, Vec data_gradeval) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	if (!initialized_) {ierr = tuMSGwarn("Error: (solveForward) computeGradient needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    // TODO[SETTING]: set tumor regularization weight
    // compute gradient for given data 'data_gradeval' and control variable 'p'
    ierr = derivative_operators_->evaluateGradient (dJ, p, data_gradeval);      CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setOptimizerFeedback ////////////////////////////// ###
PetscErrorCode setOptimizerFeedback (std::shared_ptr<OptimizerFeedback> optfeed) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    ierr = inv_solver_->setOptFeedback(optfeed); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setOptimizerSettings ////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setOptimizerSettings (std::shared_ptr<OptimizerSettings> optset) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    TU_assert (inv_solver_->isInitialized(), "TumorSolverInterface::setOptimizerSettings(): InvSolver needs to be initialized.")
    TU_assert (optset != nullptr,            "TumorSolverInterface::setOptimizerSettings(): requires non-null input.");
    inv_solver_->getOptSettings ()->beta                  = optset->beta;
    inv_solver_->getOptSettings ()->opttolgrad            = optset->opttolgrad;
    inv_solver_->getOptSettings ()->gtolbound             = optset->gtolbound;
    inv_solver_->getOptSettings ()->grtol                 = optset->grtol;
    inv_solver_->getOptSettings ()->gatol                 = optset->gatol;
    inv_solver_->getOptSettings ()->newton_maxit          = optset->newton_maxit;
    inv_solver_->getOptSettings ()->krylov_maxit          = optset->krylov_maxit;
    inv_solver_->getOptSettings ()->gist_maxit            = optset->gist_maxit;
    inv_solver_->getOptSettings ()->iterbound             = optset->iterbound;
    inv_solver_->getOptSettings ()->fseqtype              = optset->fseqtype;
    inv_solver_->getOptSettings ()->newtonsolver          = optset->newtonsolver;
    inv_solver_->getOptSettings ()->lmvm_set_hessian      = optset->lmvm_set_hessian;
    inv_solver_->getOptSettings ()->verbosity             = optset->verbosity;
    inv_solver_->getOptSettings ()->regularization_norm   = optset->regularization_norm;
    inv_solver_->getOptSettings ()->diffusivity_inversion = optset->diffusivity_inversion;
    inv_solver_->getOptSettings ()->reaction_inversion    = optset->reaction_inversion;
    inv_solver_->getOptSettings ()->k_lb                  = optset->k_lb;
    inv_solver_->getOptSettings ()->k_ub                  = optset->k_ub;
    // overwrite n_misc params
    n_misc_->regularization_norm_                         = optset->regularization_norm;
    n_misc_->diffusivity_inversion_                       = optset->diffusivity_inversion;
    n_misc_->reaction_inversion_                          = optset->reaction_inversion;
    n_misc_->k_lb_                                        = optset->k_lb;
    n_misc_->k_ub_                                        = optset->k_ub;
    optimizer_settings_changed_ = true;
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setDistMeassureSimulationGeoImages //////////////// ###
PetscErrorCode TumorSolverInterface::setDistMeassureSimulationGeoImages(
    Vec wm, Vec gm, Vec csf)
{
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    if (!initialized_)  {ierr = tuMSGwarn("Error: (setDistMeassureSimulationGeoImages) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (wm  == nullptr) {ierr = tuMSGwarn("Warning: (setDistMeassureSimulationGeoImages) Vector wm is nullptr."); CHKERRQ(ierr); }
    if (gm  == nullptr) {ierr = tuMSGwarn("Warning: (setDistMeassureSimulationGeoImages) Vector gm is nullptr."); CHKERRQ(ierr); }
    if (csf == nullptr) {ierr = tuMSGwarn("Warning: (setDistMeassureSimulationGeoImages) Vector csf is nullptr."); CHKERRQ(ierr); }
    /** Sets the image vectors for the simulation geometry material properties
     *  - MOVING PATIENT: mA(0) (= initial helathy atlas)
     *  - MOVING ATLAS:   mA(1) (= initial helathy patient)
     */
    return derivative_operators_->setDistMeassureSimulationGeoImages(wm, gm, csf, nullptr, nullptr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setDistMeassureTargetDataImages /////////////////// ###
PetscErrorCode TumorSolverInterface::setDistMeassureTargetDataImages(
    Vec wm, Vec gm, Vec csf)
{
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    if (!initialized_)  {ierr = tuMSGwarn("Error: (setDistMeassureTargetDataImages) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (wm  == nullptr) {ierr = tuMSGwarn("Warning: (setDistMeassureTargetDataImages) Vector wm is nullptr."); CHKERRQ(ierr); }
    if (gm  == nullptr) {ierr = tuMSGwarn("Warning: (setDistMeassureTargetDataImages) Vector gm is nullptr."); CHKERRQ(ierr); }
    if (csf == nullptr) {ierr = tuMSGwarn("Warning: (setDistMeassureTargetDataImages) Vector csf is nullptr."); CHKERRQ(ierr); }
    /** Sets the image vectors for the simulation geometry material properties
    /** Sets the image vectors for the target (patient) geometry material properties
     *  - MOVING PATIENT: mP(1) (= advected patient)
     *  - MOVING ATLAS:   mR    (= patient data)
     */
    return derivative_operators_->setDistMeassureTargetDataImages(wm, gm, csf, nullptr, nullptr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setDistMeassureDiffImages ///////////////////////// ###
PetscErrorCode TumorSolverInterface::setDistMeassureDiffImages(
    Vec wm, Vec gm, Vec csf)
{
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    if (!initialized_)  {ierr = tuMSGwarn("Error: (setDistMeassureDiffImages) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (wm  == nullptr) {ierr = tuMSGwarn("Warning: (setDistMeassureDiffImages) Vector wm is nullptr."); CHKERRQ(ierr); }
    if (gm  == nullptr) {ierr = tuMSGwarn("Warning: (setDistMeassureDiffImages) Vector gm is nullptr."); CHKERRQ(ierr); }
    if (csf == nullptr) {ierr = tuMSGwarn("Warning: (setDistMeassureDiffImages) Vector csf is nullptr."); CHKERRQ(ierr); }
    /** Sets the image vectors for the simulation geometry material properties
    /** Sets the image vectors for the distance measure difference
     *  - MOVING PATIENT: || mA(0)(1-c(1)) - mP(1) ||^2
     *  - MOVING ATLAS:   || mA(1)(1-c(1)) - mR    ||^2
     */
    return derivative_operators_->setDistMeassureDiffImages(wm, gm, csf, nullptr, nullptr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// computeTumorContributionRegistration ////////////// ###
PetscErrorCode TumorSolverInterface::computeTumorContributionRegistration(Vec q1, Vec q2, Vec q4) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    PetscFunctionBegin;
    if (pde_operators_ != nullptr) {
      ierr = pde_operators_->computeTumorContributionRegistration(q1, q2, nullptr, q4); CHKERRQ(ierr);
    } else {
        ierr = tuMSGwarn ("Error: (in computeTumorContributionRegistration()) PdeOperators not initialized. Exiting .."); CHKERRQ (ierr);
        PetscFunctionReturn(ierr);
    }
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// computeTumorContributionRegistration ////////////// ###
PetscErrorCode TumorSolverInterface::setGaussians (Vec data) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    ierr = tumor_->phi_->setGaussians(data); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// computeTumorContributionRegistration ////////////// ###
PetscErrorCode TumorSolverInterface::setGaussians (std::array<double, 3> cm, double sigma, double spacing, int np) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    _tuparams->user_cm_             = cm;
    _tuparams->phi_spacing_factor_  = spacing;
    _tuparams->phi_sigma_           = sigma;
    ierr = tumor_->phi_->setGaussians(cm, sigma, spacing, np); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setTumorSolverType //////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setTumorSolverType (int type) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    _newton_solver_type_changed = (type != inv_solver_->_optsettings->newtonsolver);
    inv_solver_->_optsettings->newtonsolver = type;
    PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setTumorRegularizationNorm //////////////////////// ###
PetscErrorCode TumorSolverInterface::setTumorRegularizationNorm (int type) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
  _regularization_norm_changed = (type != inv_solver->_optsettings->regularization_norm);
  inv_solver_->_optsettings->regularization_norm = type;
  PetscFunctionReturn(ierr);
}


// ### _____________________________________________________________________ ___
// ### ///////////////// resetTaoSolver //////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::resetTaoSolver() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  ierr = inv_solver_->resetTao(n_misc_);                            CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setInitialGuess /////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setInitialGuess(Vec p) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  TU_assert (p != nullptr, "TumorSolverInterface::setInitialGuess(): requires non-null input.");
  ierr = VecCopy (p, tumor_->p_);                                  CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setInitialGuess /////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setInitialGuess (double d) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  ierr = VecSet (tumor_->p_, d);                                   CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// TODO[MEMORY]: switch to not allocate mat probs in tumor code.
// ### _____________________________________________________________________ ___
// ### ///////////////// updateTumorCoefficients /////////////////////////// ###
PetscErrorCode TumorSolverInterface::updateTumorCoefficients (
    Vec wm, Vec gm, Vec csf,
    Vec filter,
    std::shared_ptr<TumorSettings> tumor_params,
    bool use_nmisc)
{
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;
    TU_assert(initialized_, "TumorSolverInterface::updateTumorCoefficients(): TumorSolverInterface needs to be initialized.")
    if (!initialized_)  {ierr = tuMSGwarn("Error: (updateTumorCoefficients) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr); PetscFunctionReturn(ierr); }
    if (wm  == nullptr) {ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector wm is nullptr."); CHKERRQ(ierr); }
    if (gm  == nullptr) {ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector gm is nullptr."); CHKERRQ(ierr); }
    if (csf == nullptr) {ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector csf is nullptr."); CHKERRQ(ierr); }
    // timing
    Event e("update-tumor-coefficients");
    std::array<double, 7> t = {0}; double self_exec_time = -MPI_Wtime ();
    std::stringstream s;

    // TODO[SETTINGS]:
    n_misc_->k_ = tumor_params->diff_coeff_scalel;
    n_misc_->k_gm_wm_ratio_ = tumor_params->diffusion_ratio_gm_wm;
    n_misc_->rho_ = tumor_params->reaction_coeff_Scale;
    n_misc_->rho_gm_wm_ratio_ = tumor_params->reaction_ratio_gm_wm;

    s << " updating tumor material properties; update values of kappa = " <<n_misc_->k_ << ", and rho = " << n_misc_->rho_;
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);

    if (!use_nmisc) {
        // update matprob, deep copy of probability maps
    		if(wm != nullptr)     {ierr = VecCopy (wm, tumor_->mat_prop_->wm_);         CHKERRQ(ierr); }
    		else                  {ierr = VecSet (tumor_->mat_prop_->wm_, 0.0);         CHKERRQ(ierr); }
    		if(gm != nullptr)     {ierr = VecCopy (gm, tumor_->mat_prop_->gm_);         CHKERRQ(ierr); }
    		else                  {ierr = VecSet (tumor_->mat_prop_->gm_, 0.0);         CHKERRQ(ierr); }
    		if(csf != nullptr)    {ierr = VecCopy (csf, tumor_->mat_prop_->csf_);       CHKERRQ(ierr); }
    		else                  {ierr = VecSet (tumor_->mat_prop_->csf_, 0.0);        CHKERRQ(ierr); }
    		if(filter != nullptr) {ierr = VecCopy (filter, tumor_->mat_prop_->filter_); CHKERRQ(ierr); }
    		else                  {ierr = VecSet (tumor_->mat_prop_->filter_, 0.0);     CHKERRQ(ierr); }

        // don't apply k_i values coming from outside if we invert for diffusivity
        if(n_misc_->diffusivity_inversion_) {
          tumor_params->diffusion_ratio_gm_wm  = tumor_->k_->k_gm_wm_ratio_;
          tumor_params->diffusion_ratio_glm_wm = tumor_->k_->k_glm_wm_ratio_;
          tumor_params->diff_coeff_scale       = tumor_->k_->k_scale_;
        }
        // update diffusion coefficient
        tumor_->k_->setValues (
            tumor_params->diff_coeff_scale,
            tumor_params->diffusion_ratio_gm_wm,
            tumor_params->diffusion_ratio_glm_wm,
            tumor_->mat_prop_, n_misc_); CHKERRQ (ierr);
        // update reaction coefficient
        tumor_->rho_->setValues (
            tumor_params->reaction_coeff_scale,
            tumor_params->reaction_ratio_gm_wm,
            tumor_params->reaction_ratio_glm_wm,
            tumor_->mat_prop_, n_misc_); CHKERRQ (ierr);
        // update the phi values, i.e., update the filter
        tumor_->phi_->setValues (tumor_->mat_prop_);
        // need to update prefactors for diffusion KSP preconditioner, as k changed
        pde_operators_->diff_solver_->precFactor();
    // use n_misc to update tumor coefficients. Needed if matprop is changed from tumor solver application
    } else {
        ierr = tumor_->k_->setValues (n_misc_->k_, n_misc_->k_gm_wm_ratio_, n_misc_->k_glm_wm_ratio_, tumor_->mat_prop_, n_misc_);
        ierr = tumor_->rho_->setValues (n_misc_->rho_, n_misc_->r_gm_wm_ratio_, n_misc_->r_glm_wm_ratio_, tumor_->mat_prop_, n_misc_);
        tumor_->phi_->setValues (tumor_->mat_prop_);  // update the phi values, i.e., update the filter
        pde_operators_->diff_solver_->precFactor();   // need to update prefactors for diffusion KSP preconditioner, as k changed
    }

    // timing
    self_exec_time += MPI_Wtime ();
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

} // namespace pglistr
