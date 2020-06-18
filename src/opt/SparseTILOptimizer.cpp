#include <iostream>
#include <limits>

#include "petsctao.h"
// #include <petsc/private/vecimpl.h>

#include "Optimizer.h"
#include "TaoInterface.h"
#include "SparseTILOptimizer.h"

PetscErrorCode SparseTILOptimizer::initialize(
  std::shared_ptr<DerivativeOperators> derivative_operators,
  std::shared_ptr <PdeOperators> pde_operators,
  std::shared_ptr <Parameters> params,
  std::shared_ptr <Tumor> tumor) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::stringstream ss;

  // number of dofs = {p, kappa, rho}
  n_inv_ = params->tu_->np_ + params->get_nk() + params->get_nr();
  ss << " Initializing sparseTIL optimizer with = " << n_inv_ << " = " << params->tu_->np_ << " + " <<  params->get_nr() << " + " <<  params->get_nk() << " dofs.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);

  // initialize super class
  ierr = Optimizer::initialize(derivative_operators, pde_operators, params, tumor); CHKERRQ(ierr);

  // initialize sub solvers
  til_opt_->initialize(derivative_operators, pde_operators, params, tumor);
  rd_opt_->initialize(derivative_operators, pde_operators, params, tumor);
  cosamp_ = std::make_shared<CtxCoSaMp>();
  ctx_->cosamp_ = cosamp_;
  til_opt_->ctx_->cosamp_ = cosamp_;
  rd_opt_->ctx_->cosamp_ = cosamp_;

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::allocateTaoObjects() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  ierr = VecCreateSeq (PETSC_COMM_SELF, n_inv_, &xrec_); CHKERRQ (ierr);
  ierr = setupVec (xrec_, SEQ); CHKERRQ (ierr);
  ierr = VecSet (xrec_, 0.0); CHKERRQ (ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::setTaoOptions() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // this function is empty since no tao object is allocated and no options are set.
  // this is done in the sub solvers RD and TIL
  ierr = tuMSGstd(" Sparse TIL optimizer does not set options for tao (done in RD and TIL subsolvers)."); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::setVariableBounds() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // this function is empty since no tao objects should be allocated
  // this is done in the sub solvers RD and TIL
  PetscFunctionReturn(ierr);
}
// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::restrictSubspace (Vec *x_restricted, Vec x_full, std::shared_ptr<CtxInv> ctx, bool create_rho_dofs = false) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  PetscReal *x_restricted_ptr, *x_full_ptr;
  int np = ctx->params_->tu_->support_.size(); // size of restricted subspace (not necessarily 2s, since merged)
  int nk = (ctx->params_->opt_->diffusivity_inversion_ || (create_rho_dofs &&  ctx->params_->opt_->reaction_inversion_)) ? ctx->params_->tu_->nk_ : 0;
  int nr = (ctx->params_->opt_->reaction_inversion_ && create_rho_dofs) ? ctx->params_->tu_->nr_ : 0;

  ctx->params_->tu_->np_ = np; // change np to solve the restricted subsystem
  ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk + nr, x_restricted); CHKERRQ (ierr);
  ierr = setupVec (*x_restricted, SEQ); CHKERRQ (ierr);
  ierr = VecSet (*x_restricted, 0); CHKERRQ (ierr);
  ierr = VecGetArray (*x_restricted, &x_restricted_ptr); CHKERRQ (ierr);
  ierr = VecGetArray (x_full, &x_full_ptr); CHKERRQ (ierr);
  for (int i = 0; i < np; i++)
      x_restricted_ptr[i] = x_full_ptr[ctx->params_->tu_->support_[i]];
  // initial guess diffusivity
  if (ctx->params_->opt_->diffusivity_inversion_) {
    x_restricted_ptr[np] = ctx->params_->tu_->k_;                                                    // equals x_full_ptr[np_full];
    if (nk > 1) x_restricted_ptr[np+1] = ctx->params_->tu_->k_ * ctx->params_->tu_->k_gm_wm_ratio_;  // equals x_full_ptr[np_full+1];
    if (nk > 2) x_restricted_ptr[np+2] = ctx->params_->tu_->k_ * ctx->params_->tu_->k_glm_wm_ratio_; // equals x_full_ptr[np_full+2];
  }
  // initial guess reaction
  if (create_rho_dofs && ctx->params_->opt_->reaction_inversion_) {
    x_restricted_ptr[np + nk] = ctx->params_->tu_->rho_;
    if (nr > 1) x_restricted_ptr[np + nk + 1] = ctx->params_->tu_->rho_ * ctx->params_->tu_->r_gm_wm_ratio_;
    if (nr > 2) x_restricted_ptr[np + nk + 2] = ctx->params_->tu_->rho_ * ctx->params_->tu_->r_glm_wm_ratio_;
  }
  ierr = VecRestoreArray (*x_restricted, &x_restricted_ptr); CHKERRQ (ierr);
  ierr = VecRestoreArray (x_full, &x_full_ptr); CHKERRQ (ierr);
  // Modifies the centers
  ctx->tumor_->phi_->modifyCenters (ctx->params_->tu_->support_);
  // resets the phis and other operators, x_restricted is copied into tumor->p_ and is used as init cond for
  // the L2 solver (needs to be done in every iteration, since location of basis functions updated)
  ierr = til_opt_->resetOperators(*x_restricted); CHKERRQ (ierr); // reset phis and other operators

  PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::prolongateSubspace (Vec x_full, Vec *x_restricted, std::shared_ptr<CtxInv> ctx, int np_full, bool reset_operators = true) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  PetscReal *x_restricted_ptr, *x_full_ptr;
  int np_r = ctx->params_->tu_->support_.size(); // size of restricted subspace (not necessarily 2s, since merged)
  int nk   = (ctx->params_->opt_->diffusivity_inversion_) ? ctx->params_->tu_->nk_ : 0;

  ierr = VecSet (x_full, 0.); CHKERRQ (ierr);
  ierr = VecGetArray (*x_restricted, &x_restricted_ptr); CHKERRQ (ierr);
  ierr = VecGetArray (x_full, &x_full_ptr); CHKERRQ (ierr);
  // correct L1 guess
  for (int i = 0; i < np_r; i++)
    x_full_ptr[ctx->params_->tu_->support_[i]] = x_restricted_ptr[i];
  // correct diffusivity
  if (ctx->params_->opt_->diffusivity_inversion_) {
    ctx->params_->tu_->k_ = x_restricted_ptr[np_r];
    x_full_ptr[np_full] = ctx->params_->tu_->k_;
    if (nk > 1) x_full_ptr[np_full+1] = ctx->params_->tu_->k_ * ctx->params_->tu_->k_gm_wm_ratio_;
    if (nk > 2) x_full_ptr[np_full+2] = ctx->params_->tu_->k_ * ctx->params_->tu_->k_glm_wm_ratio_;
  }
  ierr = VecRestoreArray (*x_restricted, &x_restricted_ptr); CHKERRQ (ierr);
  ierr = VecRestoreArray (x_full, &x_full_ptr); CHKERRQ (ierr);

  ctx->params_->tu_->np_ = np_full;    /* reset to full space         */
  ctx->tumor_->phi_->resetCenters ();  /* reset all the basis centers */
  if (reset_operators) {
    ierr = til_opt_->resetOperators(x_full); CHKERRQ (ierr);} /* reset phis and other ops    */
  /* destroy, size will change   */
  if (*x_restricted != nullptr) {
    ierr = VecDestroy(x_restricted); CHKERRQ (ierr);
    x_restricted = nullptr;
  }
  PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::setInitialGuess(Vec x_init) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: Optimizer not initialized."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
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
  // 1. TIL not given as parametrization
  if(ctx_->params_->tu_->use_c0_) {
    ierr = tuMSGwarn(" Error: cannot invert for TIL if c(0) parametrization is disabled (set use_c0 = false)."); CHKERRQ(ierr);
    exit(0);

  // 2. TIL given as parametrization
  } else {
    TU_assert(in_size == np + nk + nr, "SparseTILOptimizer::setInitialGuess: Size of input vector not correct."); CHKERRQ(ierr);
    ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk + nr, &xin_); CHKERRQ (ierr);
    ierr = setupVec (xin_, SEQ); CHKERRQ (ierr);
    ierr = VecSet (xin_, 0.0); CHKERRQ (ierr);
    ierr = VecCopy(x_init, xin_); CHKERRQ(ierr);
    // === init TIL
    ierr = ctx_->tumor_->phi_->apply (ctx_->tumor_->c_0_, x_init); CHKERRQ(ierr);
    ierr = VecMax(ctx_->tumor_->c_0_, NULL, &ic_max); CHKERRQ (ierr);
    ss << "TIL as Phi(p) (max=" << ic_max<<"); ";
    if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(xin_, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_, std::string("til-init"));}
    off = off_in = np;
  }
  // === kappa and rho init guess
  ierr = VecGetArray(x_init, &init_ptr); CHKERRQ (ierr);
  ierr = VecGetArray(xin_, &x_ptr); CHKERRQ (ierr);
  x_ptr[off_in + 0] = init_ptr[off];                           // kappa
  k_init_ = x_ptr[off_in + 0];
  if (nk > 1) x_ptr[off_in + 1] = init_ptr[off + 1];           // k2
  if (nk > 2) x_ptr[off_in + 2] = init_ptr[off + 2];           // k3
  x_ptr[off_in + nk] = init_ptr[off + nk];                     // rho
  rho_init_ = x_ptr[off_in + nk];
  if (nr > 1) x_ptr[off_in + nk + 1] = init_ptr[off + nk + 1]; // r2
  if (nr > 2) x_ptr[off_in + nk + 2] = init_ptr[off + nk + 2]; // r3
  ierr = VecRestoreArray(xin_, &x_ptr); CHKERRQ (ierr);
  ierr = VecRestoreArray(x_init, &init_ptr); CHKERRQ (ierr);

  ss << " rho_init="<<rho_init_<<"; k_init="<<k_init_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  // create xout_ vec
  ierr = VecDuplicate(xin_, &xout_); CHKERRQ (ierr);
  ierr = VecSet(xout_, 0.0); CHKERRQ (ierr);

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::solve() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  TU_assert (initialized_, "SparseTILOptimizer needs to be initialized.")
  TU_assert (data_->dt1() != nullptr, "SparseTILOptimizer requires non-null input data for inversion.");
  TU_assert (xrec_ != nullptr, "SparseTILOptimizer requires non-null xrec_ vector to be set.");
  TU_assert (xin_ != nullptr, "SparseTILOptimizer requires non-null xin_ vector to be set.");

  std::stringstream ss;
  ierr = tuMSGstd (""); CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
  ierr = tuMSG("###                                Sparse TIL + RD Inversion                                              ###");CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

  Vec g, x_L2, x_L1, x_L1_old, temp, all_phis;
  PetscReal *x_L2_ptr, *x_L1_ptr, *temp_ptr, *grad_ptr;
  PetscReal J, J_ref, J_old;   // objective
  PetscReal ftol = 1E-5;
  PetscReal norm_rel, norm, norm_g, beta_store;
  std::vector<int> idx;        // idx list of support after thresholding
  int its = 0, nnz = 0;
  int flag_convergence = 0;

  // DOFs
  int nk = ctx_->params_->get_nk();
  int nr = ctx_->params_->tu_->nr_;
  int np_full = ctx_->params_->tu_->np_; // store np of unrestricted ansatz space
  TU_assert(n_inv_ == nr + nk + np_full, "SparseTILOptimizer: n_inv is inconsistent.");

  // legacy: cosamp function assumes that xrec is np + nk,
  // so we're creating a new vec with appropriate size
  // TODO: change restric/prolongatSubspace and solver such that full np+nk+nr vec can be used.
  ierr = VecCreateSeq (PETSC_COMM_SELF, np_full + nk, &x_L1); CHKERRQ (ierr);
  ierr = setupVec (x_L1, SEQ); CHKERRQ (ierr);
  ierr = VecDuplicate(x_L1, &g); CHKERRQ (ierr);
  ierr = VecDuplicate(x_L1, &x_L1_old); CHKERRQ (ierr);
  ierr = VecDuplicate(x_L1, &temp); CHKERRQ (ierr);
  ierr = VecSet(g, 0); CHKERRQ (ierr);
  ierr = VecSet(x_L1_old, 0); CHKERRQ (ierr);
  ierr = VecSet (temp, 0); CHKERRQ (ierr);

  // defines whether or not initial guess for rho should be estimated or taken from input
  if(ctx_->params_->opt_->multilevel_ && ctx_->params_->grid_->n_[0] > 64) {
    ctx_->params_->opt_->estimate_rho_init_guess_ = false;
  } else {
    ctx_->params_->opt_->estimate_rho_init_guess_ = true;
  }
  // re-scales c0 to max(c0) = 1 for RD solve                                                                                                         
  ctx_->params_->opt_->rescale_init_cond_ = true;

  // === initial guess
  PetscReal *xin_ptr;
  ctx_->params_->tu_->k_ = k_init_;     // set in setInitialGuess, used in restrictSubspace
  ctx_->params_->tu_->rho_ = rho_init_;
  ierr = VecCopy(xin_, xrec_); CHKERRQ(ierr);
  ierr = VecGetArray(xrec_, &xin_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(x_L1, &x_L1_ptr); CHKERRQ(ierr);
  for(int i = 0; i < np_full + nk; ++i)
    x_L1_ptr[i] = xin_ptr[i]; // copy p and nk
  ierr = VecRestoreArray(xrec_, &xin_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(x_L1, &x_L1_ptr); CHKERRQ(ierr);

  /* ------------------------------------------------------------------------ */
  // ### (0) (pre-)reaction/diffusion inversion ###
  if (ctx_->params_->opt_->pre_reacdiff_solve_ && ctx_->params_->grid_->n_[0] > 64) {
    if (ctx_->params_->opt_->reaction_inversion_) {
      // restrict to new L2 subspace, holding p_i, kappa, and rho
      ierr = restrictSubspace(&x_L2, x_L1, ctx_, true); CHKERRQ (ierr); // x_L2 <-- R(x_L1)
      // solve
      cosamp_->cosamp_stage = PRE_RD;
      rd_opt_->setData(data_);
      ierr = rd_opt_->setInitialGuess(x_L2); CHKERRQ (ierr); // with current guess as init cond.
      ierr = tuMSGstd(""); CHKERRQ (ierr);
      ierr = tuMSG   ("### scaled init guess w/ incorrect reaction coefficient  ###"); CHKERRQ (ierr);
      ierr = tuMSGstd("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
      if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 1) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}
      ierr = tuMSGstd("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
      if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(x_L2, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_, std::string("scaled-pre-l1"));}
      ierr = tuMSGstd(""); CHKERRQ (ierr);
      ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
      ierr = tuMSG("###                     (PRE) rho/kappa inversion with scaled L2 solution guess                           ###");CHKERRQ (ierr);
      ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

      ierr = rd_opt_->solve(); CHKERRQ (ierr);
      ierr = VecCopy(rd_opt_->getSolution(), x_L2); CHKERRQ (ierr); // get solution (length: np_r + nk + nr)
      // update full space solution
      ierr = prolongateSubspace(x_L1, &x_L2, ctx_, np_full); CHKERRQ (ierr); // x_L1 <-- P(x_L2)
    }
  }

  /* ------------------------------------------------------------------------ */
  // === (1) L1 CoSaMp solver ===
  // if no inversion for kappa, set coefficients  -- this will not change during the solve
  if(!ctx_->params_->opt_->diffusivity_inversion_) {
    ierr = ctx_->tumor_->k_->setValues (ctx_->params_->tu_->k_, ctx_->params_->tu_->k_gm_wm_ratio_, ctx_->params_->tu_->k_glm_wm_ratio_, ctx_->tumor_->mat_prop_, ctx_->params_);  CHKERRQ (ierr);
  } else {
    //set initial guess for k_inv (possibly != zero)
    ierr = VecGetArray(x_L1, &x_L1_ptr); CHKERRQ (ierr);
    if(ctx_->params_->opt_->diffusivity_inversion_)
      x_L1_ptr[np_full] = ctx_->params_->tu_->k_;
    ierr = VecRestoreArray(x_L1, &x_L1_ptr); CHKERRQ (ierr);
    ierr = VecCopy        (x_L1, x_L1_old); CHKERRQ (ierr);
  }
  // compute reference value for  objective
  // set beta to zero for gradient thresholding
  beta_store = ctx_->params_->opt_->beta_; ctx_->params_->opt_->beta_ = 0.;
  ierr = evalObjectiveAndGradient(x_L1, &J_ref, g); CHKERRQ (ierr); // length: np_full + nk
  ctx_->params_->opt_->beta_ = beta_store;
  ierr = VecNorm(g, NORM_2, &norm_g); CHKERRQ (ierr);
  J = J_ref;
  // print monitor
  ierr = cosampMonitor (its, J_ref, 1, norm_g, 1, x_L1); CHKERRQ(ierr);
  // number of connected components
  ctx_->tumor_->phi_->num_components_ = ctx_->tumor_->phi_->component_weights_.size ();
  // output warmstart (injection) support
  ss << "starting CoSaMP solver with initial support: ["; for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
  ss << "component label of initial support : [";         for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

  // === L1 solver ===
  while (true) {
    its++;
    /* === hard threshold abs gradient === */
    ierr = VecCopy(g, temp); CHKERRQ (ierr);
    ierr = VecAbs(temp); CHKERRQ (ierr);
    // print gradient to file
    if (ctx_->params_->tu_->verbosity_ >= 2) {
        ierr = VecGetArray(temp, &grad_ptr); CHKERRQ(ierr);
        for (int i = 0; i < np_full-1; i++) {
          if(procid == 0) ctx_->params_->tu_->outfile_glob_grad_ << grad_ptr[i] << ", ";
        }
        if(procid == 0) ctx_->params_->tu_->outfile_glob_grad_ << grad_ptr[np_full-1] << ";\n" <<std::endl;
        ierr = VecRestoreArray(temp, &grad_ptr); CHKERRQ(ierr);
    }
    // threshold gradient
    idx.clear();
    ierr = hardThreshold(temp, 2 * ctx_->params_->tu_->sparsity_level_,
                         np_full, idx, ctx_->tumor_->phi_->gaussian_labels_,
                         ctx_->tumor_->phi_->component_weights_,
                         nnz, ctx_->tumor_->phi_->num_components_, ctx_->params_->tu_->thresh_component_weight_, true);
    CHKERRQ(ierr);

    /* === update support of prev. solution with new support === */
    ctx_->params_->tu_->support_.insert (ctx_->params_->tu_->support_.end(), idx.begin(), idx.end());
    std::sort (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end()); // sort and remove duplicates
    ctx_->params_->tu_->support_.erase (std::unique (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end()), ctx_->params_->tu_->support_.end());
    // print out
    ss << "support for corrective L2 solve : [";
    for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << "component label of support : [";
    for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    /* === corrective L2 solver === */
    ierr = tuMSGstd (""); CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    ierr = tuMSG("###                                corrective L2 solver in restricted subspace                            ###");CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

    // x_L2 <-- R(x_L1)
    ierr = restrictSubspace(&x_L2, x_L1, ctx_); CHKERRQ (ierr);
    // print vec
    if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) {
      ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);
    }

    // update reference gradient and referenc objective (commented in the solve function)
    til_opt_->updateReferenceGradient(true);
    til_opt_->updateReferenceObjective(true);
    til_opt_->setData(data_);
    ierr = til_opt_->setInitialGuess(x_L2); CHKERRQ(ierr);
    ierr = til_opt_->solve(); CHKERRQ(ierr);
    ierr = VecCopy(til_opt_->getSolution(), x_L2); CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------- L2 solver end --------------------------------------------- ###");CHKERRQ (ierr);
    ierr = tuMSGstd(""); CHKERRQ (ierr);
    ierr = VecCopy(x_L1, x_L1_old); CHKERRQ (ierr);

    // print support
    ierr = VecDuplicate(ctx_->tumor_->phi_->phi_vec_[0], &all_phis); CHKERRQ (ierr);
    ierr = VecSet(all_phis, 0.); CHKERRQ (ierr);
    for (int i = 0; i < ctx_->params_->tu_->np_; i++) {
      ierr = VecAXPY (all_phis, 1.0, ctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);
    }
    ss << "phiSupport_csitr-" << its << ".nc";
    if (ctx_->params_->tu_->write_output_) dataOut (all_phis, ctx_->params_, ss.str().c_str()); ss.str(""); ss.clear();
    if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}
    // print vec
    if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) {
      ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);
    }
    // x_L1 <-- P(x_L2)
    ierr = prolongateSubspace(x_L1, &x_L2, ctx_, np_full); CHKERRQ (ierr);

    /* === hard threshold solution to sparsity level === */
    idx.clear();
    if (ctx_->params_->opt_->prune_components_) {
      ierr = hardThreshold(x_L1, ctx_->params_->tu_->sparsity_level_,
        np_full, idx, ctx_->tumor_->phi_->gaussian_labels_,
        ctx_->tumor_->phi_->component_weights_,
        nnz, ctx_->tumor_->phi_->num_components_, ctx_->params_->tu_->thresh_component_weight_);
      CHKERRQ(ierr);
    } else {
      ierr = hardThreshold(x_L1, ctx_->params_->tu_->sparsity_level_, np_full, idx, nnz); CHKERRQ(ierr);
    }
    ctx_->params_->tu_->support_.clear();
    ctx_->params_->tu_->support_ = idx;
    // sort and remove duplicates
    std::sort(ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end());
    ctx_->params_->tu_->support_.erase(std::unique (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end()), ctx_->params_->tu_->support_.end());
    // print out
    ss << "support after hard thresholding the solution : [";
    for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << "component label of support : [";
    for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    // set only support values in x_L1 (rest hard thresholded to zero)
    ierr = VecCopy(x_L1, temp); CHKERRQ (ierr);
    ierr = VecSet(x_L1, 0.0); CHKERRQ (ierr);
    ierr = VecGetArray(temp, &temp_ptr); CHKERRQ (ierr);
    ierr = VecGetArray(x_L1, &x_L1_ptr); CHKERRQ (ierr);
    for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++)
      x_L1_ptr[ctx_->params_->tu_->support_[i]] = temp_ptr[ctx_->params_->tu_->support_[i]];
    if (ctx_->params_->opt_->diffusivity_inversion_) {
      x_L1_ptr[np_full] = temp_ptr[np_full];
      if (ctx_->params_->tu_->nk_ > 1) x_L1_ptr[np_full+1] = temp_ptr[np_full+1];
      if (ctx_->params_->tu_->nk_ > 2) x_L1_ptr[np_full+2] = temp_ptr[np_full+2];
    }
    ierr = VecRestoreArray (x_L1, &x_L1_ptr); CHKERRQ (ierr);
    ierr = VecRestoreArray (temp, &temp_ptr); CHKERRQ (ierr);
    ierr = VecCopy (x_L1, ctx_->tumor_->p_); CHKERRQ (ierr); // copy initial guess for p
    // print initial guess to file
    if (ctx_->params_->tu_->write_output_) {
      ss << "c0guess_csitr-" << its << ".nc";  dataOut (ctx_->tumor_->c_0_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
      ss << "c1guess_csitr-" << its << ".nc"; if (ctx_->params_->tu_->verbosity_ >= 4) dataOut (ctx_->tumor_->c_t_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
    }

    /* === convergence check === */
    J_old = J;
    // compute objective (only mismatch term)
    beta_store = ctx_->params_->opt_->beta_; ctx_->params_->opt_->beta_ = 0.;
    ierr = evalObjectiveAndGradient(x_L1, &J, g); CHKERRQ (ierr); // length: np_full + nk
    ctx_->params_->opt_->beta_ = beta_store;
    ierr = VecNorm(x_L1, NORM_INFINITY, &norm); CHKERRQ (ierr);
    ierr = VecAXPY(temp, -1.0, x_L1_old); CHKERRQ (ierr);             // holds x_L1
    ierr = VecNorm(temp, NORM_INFINITY, &norm_rel); CHKERRQ (ierr);   // norm change in sol
    ierr = VecNorm(g, NORM_2, &norm_g); CHKERRQ (ierr);
    // solver status
    ierr = tuMSGstd(""); CHKERRQ(ierr); ierr = tuMSGstd(""); CHKERRQ(ierr);
    ierr = tuMSGstd("--------------------------------------------- L1 solver statistics -------------------------------------------"); CHKERRQ(ierr);
    ierr = cosampMonitor(its, J, PetscAbsReal (J_old - J) / PetscAbsReal (1 + J_ref), norm_g, norm_rel / (1 + norm), x_L1); CHKERRQ(ierr);
    ierr = tuMSGstd("--------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGstd(""); CHKERRQ(ierr);
    if (its >= ctx_->params_->opt_->gist_maxit_) {ierr = tuMSGwarn (" L1 maxiter reached."); CHKERRQ(ierr); flag_convergence = 1; break;}
    else if (PetscAbsReal (J) < 1E-5)  {ierr = tuMSGwarn (" L1 absolute objective tolerance reached."); CHKERRQ(ierr); flag_convergence = 1; break;}
    else if (PetscAbsReal (J_old - J) < ftol * PetscAbsReal (1 + J_ref)) {ierr = tuMSGwarn (" L1 relative objective tolerance reached."); CHKERRQ(ierr); flag_convergence = 1; break;}
    else { flag_convergence = 0; }  // continue iterating
  } // end while


  /* === (3) if converged: corrective L2 solver === */
  ierr = tuMSGstd(""); CHKERRQ(ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ(ierr);
  ierr = tuMSG("###                                              final L2 solve                                           ###");CHKERRQ(ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ(ierr);

  // x_L2 <-- R(x_L1)
  ierr = restrictSubspace(&x_L2, x_L1, ctx_); CHKERRQ (ierr);
  if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) {
    ierr = VecView(x_L2, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);
  }

  // update reference gradient and referenc objective (commented in the solve function)
  til_opt_->updateReferenceGradient(true);
  til_opt_->updateReferenceObjective(true);
  til_opt_->setData(data_);
  ierr = til_opt_->setInitialGuess(x_L2); CHKERRQ(ierr);
  ierr = til_opt_->solve(); CHKERRQ(ierr);
  ierr = VecCopy(til_opt_->getSolution(), x_L2); CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------- L2 solver end --------------------------------------------- ###");CHKERRQ (ierr);
  ierr = tuMSGstd(""); CHKERRQ (ierr);
  // print vec
  if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) {
    ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);
  }
  // print phi's to file
  if (ctx_->params_->tu_->write_output_) {
    ierr = VecDuplicate(ctx_->tumor_->phi_->phi_vec_[0], &all_phis); CHKERRQ (ierr);
    ierr = VecSet(all_phis, 0.); CHKERRQ (ierr);
    for (int i = 0; i < ctx_->params_->tu_->np_; i++) {
      ierr = VecAXPY(all_phis, 1.0, ctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);
    }
    ss << "phiSupportFinal.nc";  {dataOut(all_phis, ctx_->params_, ss.str().c_str());} ss.str(std::string()); ss.clear();
    ss << "c0FinalGuess.nc";  dataOut(ctx_->tumor_->c_0_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
    ss << "c1FinalGuess.nc"; if (ctx_->params_->tu_->verbosity_ >= 4) { dataOut(ctx_->tumor_->c_t_, ctx_->params_, ss.str().c_str()); } ss.str(std::string()); ss.clear();
    if (all_phis != nullptr) {ierr = VecDestroy(&all_phis); CHKERRQ(ierr); all_phis = nullptr;}
  }
  // write out p vector after sparse TIL and kappa inversion (unscaled)
  if (ctx_->params_->tu_->write_p_checkpoint_) {
    writeCheckpoint(x_L2, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_, std::string("unscaled"));
  }
  // TODO(K): not sure if this is correct; check if runtime error
  // prolongate restricted x_L2 to full x_L1, but do not resize vectors, i.e., call resetOperators
  // if inversion for reaction disabled, also reset operators
  ierr = prolongateSubspace(x_L1, &x_L2, ctx_, np_full, !ctx_->params_->opt_->reaction_inversion_); CHKERRQ (ierr); // x_L1 <-- P(x_L2)

  // === (4) reaction/diffusion inversion ===
  if (ctx_->params_->opt_->reaction_inversion_) {
    // restrict to new L2 subspace, holding p_i, kappa, and rho
    ierr = restrictSubspace(&x_L2, x_L1, ctx_, true); CHKERRQ (ierr); // x_L2 <-- R(x_L1)
    // solve
    cosamp_->cosamp_stage = POST_RD;
    rd_opt_->setData(data_);
    ierr = rd_opt_->setInitialGuess(x_L2); CHKERRQ (ierr); // with current guess as init cond.

    ierr = tuMSGstd(""); CHKERRQ (ierr);
    ierr = tuMSG   ("### scaled L2 sol. w/ incorrect reaction coefficient     ###"); CHKERRQ (ierr);
    ierr = tuMSGstd("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
    if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 1) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}
    ierr = tuMSGstd("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
    if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(x_L2, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_, std::string("scaled"));}
    ierr = tuMSGstd(""); CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    ierr = tuMSG("###                    (POST) rho/kappa inversion with scaled L2 solution guess                           ###");CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

    ierr = rd_opt_->solve(); CHKERRQ (ierr);
    ierr = VecCopy(rd_opt_->getSolution(), x_L2); CHKERRQ (ierr); // get solution (length: np_r + nk + nr)
    // update full space solution
    ierr = prolongateSubspace(x_L1, &x_L2, ctx_, np_full); CHKERRQ (ierr); // x_L1 <-- P(x_L2)
  }

  // === populate solution to xout_
  PetscReal *x_ptr;
  ierr = VecGetArray(xrec_, &x_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(x_L1, &x_L1_ptr); CHKERRQ(ierr);
  for(int i = 0; i < np_full + nk; ++i)
    x_ptr[i] = x_L1_ptr[i];                          // copy p and nk
  if (ctx_->params_->opt_->reaction_inversion_)
    x_ptr[np_full + nk] = ctx_->params_->tu_->rho_;  // copy rho
  ierr = VecRestoreArray(xrec_, &x_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(x_L1, &x_L1_ptr); CHKERRQ(ierr);
  ierr = VecCopy(xrec_, xout_); CHKERRQ (ierr);

  // clean-up
  if (g != nullptr) {ierr = VecDestroy(&g); CHKERRQ (ierr); g = nullptr; }
  if (x_L1 != nullptr) {ierr = VecDestroy(&x_L1); CHKERRQ (ierr); x_L1 = nullptr;}
  if (x_L1_old != nullptr) {ierr = VecDestroy(&x_L1_old);CHKERRQ (ierr); x_L1_old = nullptr;}
  if (temp != nullptr) {ierr = VecDestroy(&temp); CHKERRQ (ierr); temp = nullptr;}

  PetscFunctionReturn(ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::solve_rs(bool rs_mode_active) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  TU_assert (initialized_, "MEOptimizer needs to be initialized.")
  TU_assert (data_->dt1() != nullptr, "MEOptimizer requires non-null input data for inversion.");
  TU_assert (xrec_ != nullptr, "MEOptimizer requires non-null xrec_ vector to be set.");
  TU_assert (xin_ != nullptr, "MEOptimizer requires non-null xin_ vector to be set.");

  std::stringstream ss;
  ierr = tuMSGstd (""); CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
  ierr = tuMSG("###                                Sparse TIL + RD Inversion                                              ###");CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

  std::vector<int> idx;        // idx list of support after thresholding
  PetscReal *x_full_ptr, *x_work_ptr, *grad_ptr;
  PetscReal beta_store, norm_rel, norm;
  int nnz = 0;
  bool conv_maxit = false;
  bool finalize = false, contiterating = true;
  Vec all_phis;

  // abbrev
  int np_full = ctx_->cosamp_->np_full;

  ierr = tuMSG(" >> entering inverse CoSaMp"); CHKERRQ(ierr);
  switch(ctx_->cosamp_->cosamp_stage) {
    // ================
    case INIT: {
      ierr = tuMSG(" >> entering stage INIT"); CHKERRQ(ierr); ss.str(""); ss.clear();
      ctx_->params_->tu_->k_ = k_init_;     // set in setInitialGuess, used in restrictSubspace
      ctx_->params_->tu_->rho_ = rho_init_;
      ctx_->cosamp_->np_full = ctx_->params_->tu_->np_; // store np of unrestricted ansatz space
      np_full = ctx_->cosamp_->np_full;
      ctx_->cosamp_->converged_l1 = false;
      ctx_->cosamp_->converged_l2 = false;
      ctx_->cosamp_->f_tol = 1E-5;
      ierr = ctx_->cosamp_->cleanup(); CHKERRQ (ierr);
      /* allocate vecs and copy initial guess for p */
      ierr = VecCopy(xin_, xrec_); CHKERRQ(ierr);
      ierr = ctx_->cosamp_->initialize(xrec_, np_full + ctx_->params_->get_nk()); CHKERRQ (ierr);
      // no break; go into next case
      ctx_->cosamp_->cosamp_stage = PRE_RD;
      // defines whether or not initial guess for rho should be estimated or taken from input
      ctx_->params_->opt_->estimate_rho_init_guess_ = !(ctx_->params_->opt_->multilevel_ && ctx_->params_->grid_->n_[0] > 64);
      ierr = tuMSG(" << leaving stage INIT"); CHKERRQ(ierr);
    }
    // ================
    // this case is executed at once without going back to caller in between
    case PRE_RD: {
        /* ------------------------------------------------------------------------ */
        // ### (0) (pre-)reaction/diffusion inversion ###
        ierr = tuMSG(" >> entering stage PRE_RD"); CHKERRQ(ierr);
        if (ctx_->params_->opt_->pre_reacdiff_solve_ && ctx_->params_->grid_->n_[0] > 64) {
          if (ctx_->params_->opt_->reaction_inversion_) {
            // == restrict == to new L2 subspace, holding p_i, kappa, and rho
            ierr = restrictSubspace(&ctx_->cosamp_->x_sub, ctx_->cosamp_->x_full, ctx_, true); CHKERRQ (ierr); // x_sub <-- R(x_full)
            ctx_->cosamp_->cosamp_stage = PRE_RD;
            ctx_->params_->opt_->newton_maxit_ = ctx_->cosamp_->maxit_newton;
            // == solve ==
            rd_opt_->setData(data_);
            ierr = rd_opt_->setInitialGuess(ctx_->cosamp_->x_sub); CHKERRQ (ierr); // with current guess as init cond.
            ierr = rd_opt_->solve(); CHKERRQ (ierr);
            ierr = VecCopy(rd_opt_->getSolution(), ctx_->cosamp_->x_sub); CHKERRQ (ierr); // get solution (length: np_r + nk + nr)
            // == prolongate ==
            ierr = prolongateSubspace(ctx_->cosamp_->x_full, &ctx_->cosamp_->x_sub, ctx_, np_full); CHKERRQ (ierr); // x_full <-- P(x_sub)
          }
      } else {ierr = tuMSGstd("    ... skipping stage, reaction diffusion disabled."); CHKERRQ(ierr);}
        // no break; go into next case
        ctx_->cosamp_->cosamp_stage = COSAMP_L1_INIT;
        ierr = tuMSG(" << leaving stage PRE_RD"); CHKERRQ(ierr);
      }
    // ================
    // setting up L1-pahse, computing reference gradeint, and print statistics
    case COSAMP_L1_INIT: {
        ierr = tuMSG(" >> entering stage COSAMP_L1_INIT"); CHKERRQ(ierr);
        // set initial guess for k_inv (possibly != zero)
        ierr = VecGetArray(ctx_->cosamp_->x_full, &x_full_ptr); CHKERRQ (ierr);
        if (ctx_->params_->opt_->diffusivity_inversion_) x_full_ptr[np_full] = ctx_->params_->tu_->k_;
        else { // set diff ops with this guess -- this will not change during the solve
          ierr = ctx_->tumor_->k_->setValues (ctx_->params_->tu_->k_, ctx_->params_->tu_->k_gm_wm_ratio_, ctx_->params_->tu_->k_glm_wm_ratio_, ctx_->tumor_->mat_prop_, ctx_->params_); CHKERRQ (ierr);
        }
        ierr = VecRestoreArray(ctx_->cosamp_->x_full, &x_full_ptr); CHKERRQ (ierr);
        ierr = VecCopy        (ctx_->cosamp_->x_full, ctx_->cosamp_->x_full_prev); CHKERRQ (ierr);

        // compute reference value for  objective
        beta_store = ctx_->params_->opt_->beta_; ctx_->params_->opt_->beta_ = 0.; // set beta to zero for gradient thresholding
        ierr = evalObjectiveAndGradient(ctx_->cosamp_->x_full, &ctx_->cosamp_->J_ref, ctx_->cosamp_->g); CHKERRQ (ierr); // length: np_full + nk
        ctx_->params_->opt_->beta_ = beta_store;
        ierr = VecNorm (ctx_->cosamp_->g, NORM_2, &ctx_->cosamp_->g_norm); CHKERRQ (ierr);
        ctx_->cosamp_->J = ctx_->cosamp_->J_ref;

        // print statistics
        ierr = cosampMonitor(ctx_->cosamp_->its_l1, ctx_->cosamp_->J_ref, 1, ctx_->cosamp_->g_norm, 1, ctx_->cosamp_->x_full); CHKERRQ(ierr);
        // number of connected components
        ctx_->tumor_->phi_->num_components_ = ctx_->tumor_->phi_->component_weights_.size ();
        // output warmstart (injection) support
        ss << "starting CoSaMP solver with initial support: ["; for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ss << "component label of initial support : [";         for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

        // no break; go into next case
        ctx_->cosamp_->cosamp_stage = COSAMP_L1_THRES_GRAD;
        ierr = tuMSG(" << leaving stage COSAMP_L1_INIT"); CHKERRQ(ierr);
      }
    // ================
    // thresholding the gradient, restrict subspace
    case COSAMP_L1_THRES_GRAD: {
        ierr = tuMSG(" >> entering stage COSAMP_L1_THRES_GRAD"); CHKERRQ(ierr);
        ctx_->cosamp_->its_l1++;
        /* === hard threshold abs gradient === */
        ierr = VecCopy(ctx_->cosamp_->g, ctx_->cosamp_->work); CHKERRQ (ierr);
        ierr = VecAbs(ctx_->cosamp_->work); CHKERRQ (ierr);
        // print gradient to file
        if (ctx_->params_->tu_->verbosity_ >= 2) {
          ierr = VecGetArray(ctx_->cosamp_->work, &grad_ptr); CHKERRQ(ierr);
          for (int i = 0; i < np_full-1; i++) if(procid == 0) ctx_->params_->tu_->outfile_glob_grad_ << grad_ptr[i] << ", ";
          if(procid == 0)                                     ctx_->params_->tu_->outfile_glob_grad_ << grad_ptr[np_full-1] << ";\n" <<std::endl;
          ierr = VecRestoreArray(ctx_->cosamp_->work, &grad_ptr); CHKERRQ(ierr);
        }
        idx.clear();
        ierr = hardThreshold(ctx_->cosamp_->work, 2 * ctx_->params_->tu_->sparsity_level_, np_full, idx, ctx_->tumor_->phi_->gaussian_labels_, ctx_->tumor_->phi_->component_weights_, nnz, ctx_->tumor_->phi_->num_components_, ctx_->params_->tu_->thresh_component_weight_, true); CHKERRQ(ierr);

        /* === update support of prev. solution with new support === */
        ctx_->params_->tu_->support_.insert (ctx_->params_->tu_->support_.end(), idx.begin(), idx.end());
        // sort and remove duplicates
        std::sort (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end());
        ctx_->params_->tu_->support_.erase (std::unique (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end()), ctx_->params_->tu_->support_.end());
        // print out
        ss << "support for corrective L2 solve : ["; for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ss << "component label of support : [";      for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

        // no break; go into next case
        ctx_->cosamp_->cosamp_stage = COSAMP_L1_SOLVE_SUBSPACE;
        ierr = tuMSG(" << leaving stage COSAMP_L1_THRES_GRAD"); CHKERRQ(ierr);
      }
    // ================
    // this case may be executed in parts, i.e., going back to caller after inexact_nit Newton iterations
    case COSAMP_L1_SOLVE_SUBSPACE: {
        ierr = tuMSG(" >> entering stage COSAMP_L1_SOLVE_SUBSPACE"); CHKERRQ(ierr);
        /* === corrective L2 solver === */
        ierr = tuMSGstd (""); CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSG("###                                corrective L2 solver in restricted subspace                            ###");CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

        // == restrict ==
        ierr = restrictSubspace(&ctx_->cosamp_->x_sub, ctx_->cosamp_->x_full, ctx_); CHKERRQ (ierr); // x_L2 <-- R(x_L1)
        // print vec
        if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) {ierr = VecView (ctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}

        // solve interpolation
        // ierr = solveInterpolation (data_);                                   CHKERRQ (ierr);

        ctx_->params_->opt_->newton_maxit_ = ctx_->cosamp_->inexact_nits;
        // only update reference gradient and referenc objective if this is the first inexact solve for this subspace, otherwise don't
        til_opt_->updateReferenceGradient(ctx_->cosamp_->nits < ctx_->cosamp_->inexact_nits);
        til_opt_->updateReferenceObjective(ctx_->cosamp_->nits < ctx_->cosamp_->inexact_nits);
        // == solve ==
        til_opt_->setData(data_);
        ierr = til_opt_->setInitialGuess(ctx_->cosamp_->x_sub); CHKERRQ(ierr);
        ierr = til_opt_->solve(); CHKERRQ(ierr);
        ierr = VecCopy(til_opt_->getSolution(), ctx_->cosamp_->x_sub); CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------- L2 solver end --------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSGstd (""); CHKERRQ (ierr);
        ierr = VecCopy (ctx_->cosamp_->x_full, ctx_->cosamp_->x_full_prev); CHKERRQ (ierr);
        // print support
        ierr = VecDuplicate (ctx_->tumor_->phi_->phi_vec_[0], &all_phis); CHKERRQ (ierr);
        ierr = VecSet (all_phis, 0.); CHKERRQ (ierr);
        for (int i = 0; i < ctx_->params_->tu_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, ctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
        ss << "phiSupport_csitr-" << ctx_->cosamp_->its_l1 << ".nc";
        if (ctx_->params_->tu_->write_output_) dataOut (all_phis, ctx_->params_, ss.str().c_str()); ss.str(""); ss.clear();
        if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}
        if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (ctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}

        // == prolongate ==
        ierr = prolongateSubspace(ctx_->cosamp_->x_full, &ctx_->cosamp_->x_sub, ctx_, np_full); CHKERRQ (ierr); // x_L1 <-- P(x_L2)

        // == convergence test ==
        // neither gradient sufficiently small nor ls-failure (i.e., inexact_nit hit)
        // if(!ctx_->cosamp_->converged_l2 && !ctx_->cosamp_->converged_l2) {ctx_->cosamp_->nits += ctx_->cosamp_->inexact_nits;}
        ctx_->cosamp_->nits += ctx_->params_->optf_->nb_newton_it_;
        conv_maxit = ctx_->cosamp_->nits >= ctx_->cosamp_->maxit_newton;
        // check if L2 solver converged
        if(!ctx_->cosamp_->converged_l2 && !ctx_->cosamp_->converged_error_l2 && !conv_maxit) {
          ss << "    ... inexact solve terminated (L2 solver not converged, will be continued; its "<< ctx_->cosamp_->nits <<"/"<< ctx_->cosamp_->maxit_newton <<").";
          ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();
          ierr = tuMSG(" << leaving stage COSAMP_L1_SOLVE_SUBSPACE");
          break;
        } else {
          // if L2 solver converged
          if(ctx_->cosamp_->converged_l2)        {ss << "    ... L2 solver converged; its "<< ctx_->cosamp_->nits <<"/"<< ctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
          // if L2 solver ran into ls-failure
          if(ctx_->cosamp_->converged_error_l2)  {ss << "    ... L2 solver terminated (ls-failure); its "<< ctx_->cosamp_->nits <<"/"<< ctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
          // if L2 solver hit maxit
          if(conv_maxit)                           {ss << "    ... L2 solver terminated (maxit); its "<< ctx_->cosamp_->nits <<"/"<< ctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
          ctx_->cosamp_->cosamp_stage = COSAMP_L1_THRES_SOL;
          conv_maxit = ctx_->cosamp_->nits = 0;
          ierr = tuMSG(" << leaving stage COSAMP_L1_SOLVE_SUBSPACE"); CHKERRQ(ierr);
        }
      }
    // ================
    // thresholding the gradient, restrict subspace
    case COSAMP_L1_THRES_SOL: {
        ierr = tuMSG(" >> entering stage COSAMP_L1_THRES_SOL"); CHKERRQ(ierr);
        /* === hard threshold solution to sparsity level === */
        idx.clear();
        if (ctx_->params_->opt_->prune_components_) hardThreshold (ctx_->cosamp_->x_full, ctx_->params_->tu_->sparsity_level_, np_full, idx, ctx_->tumor_->phi_->gaussian_labels_, ctx_->tumor_->phi_->component_weights_, nnz, ctx_->tumor_->phi_->num_components_, ctx_->params_->tu_->thresh_component_weight_);
        else                                        hardThreshold (ctx_->cosamp_->x_full, ctx_->params_->tu_->sparsity_level_, np_full, idx, nnz);
        ctx_->params_->tu_->support_.clear();
        ctx_->params_->tu_->support_ = idx;
        // sort and remove duplicates
        std::sort(ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end());
        ctx_->params_->tu_->support_.erase (std::unique (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end()), ctx_->params_->tu_->support_.end());
        // print out
        ss << "support after hard thresholding the solution : ["; for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ss << "component label of support : ["; for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

        // set only support values in x_L1 (rest hard thresholded to zero)
        ierr = VecCopy (ctx_->cosamp_->x_full, ctx_->cosamp_->work); CHKERRQ (ierr);
        ierr = VecSet  (ctx_->cosamp_->x_full, 0.0); CHKERRQ (ierr);
        ierr = VecGetArray (ctx_->cosamp_->work, &x_work_ptr); CHKERRQ (ierr);
        ierr = VecGetArray (ctx_->cosamp_->x_full, &x_full_ptr); CHKERRQ (ierr);
        for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++)
          x_full_ptr[ctx_->params_->tu_->support_[i]] = x_work_ptr[ctx_->params_->tu_->support_[i]];
        if (ctx_->params_->opt_->diffusivity_inversion_) {
          x_full_ptr[np_full] = x_work_ptr[np_full];
          if (ctx_->params_->tu_->nk_ > 1) x_full_ptr[np_full+1] = x_work_ptr[np_full+1];
          if (ctx_->params_->tu_->nk_ > 2) x_full_ptr[np_full+2] = x_work_ptr[np_full+2];
        }
        ierr = VecRestoreArray (ctx_->cosamp_->x_full, &x_full_ptr); CHKERRQ (ierr);
        ierr = VecRestoreArray (ctx_->cosamp_->work, &x_work_ptr); CHKERRQ (ierr);
        /* copy initial guess for p */
        ierr = VecCopy (ctx_->cosamp_->x_full, ctx_->tumor_->p_); CHKERRQ (ierr);

        // print initial guess to file
        if (ctx_->params_->tu_->write_output_) {
          ss << "c0guess_csitr-" << ctx_->cosamp_->its_l1 << ".nc";  dataOut (ctx_->tumor_->c_0_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
          ss << "c1guess_csitr-" << ctx_->cosamp_->its_l1 << ".nc"; if (ctx_->params_->tu_->verbosity_ >= 4) dataOut (ctx_->tumor_->c_t_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
        }

        /* === convergence check === */
        ctx_->cosamp_->J_prev = ctx_->cosamp_->J;
        // compute objective (only mismatch term)
        beta_store = ctx_->params_->opt_->beta_; ctx_->params_->opt_->beta_ = 0.; // set beta to zero for gradient thresholding
        ierr = evalObjectiveAndGradient(ctx_->cosamp_->x_full, &ctx_->cosamp_->J, ctx_->cosamp_->g); CHKERRQ (ierr);
        ctx_->params_->opt_->beta_ = beta_store;
        ierr = VecNorm (ctx_->cosamp_->x_full, NORM_INFINITY, &norm); CHKERRQ (ierr);
        ierr = VecAXPY (ctx_->cosamp_->work, -1.0, ctx_->cosamp_->x_full_prev); CHKERRQ (ierr); /* holds x_L1 */
        ierr = VecNorm (ctx_->cosamp_->work, NORM_INFINITY, &norm_rel); CHKERRQ (ierr); /*norm change in sol */
        ierr = VecNorm (ctx_->cosamp_->g, NORM_2, &ctx_->cosamp_->g_norm); CHKERRQ (ierr);
        // solver status
        ierr = tuMSGstd (""); CHKERRQ(ierr); ierr = tuMSGstd (""); CHKERRQ(ierr);
        ierr = tuMSGstd ("--------------------------------------------- L1 solver statistics -------------------------------------------"); CHKERRQ(ierr);
        ierr = cosampMonitor(ctx_->cosamp_->its_l1, ctx_->cosamp_->J, PetscAbsReal (ctx_->cosamp_->J_prev - ctx_->cosamp_->J) / PetscAbsReal (1 + ctx_->cosamp_->J_ref), ctx_->cosamp_->g_norm, norm_rel / (1 + norm), ctx_->cosamp_->x_full); CHKERRQ(ierr);
        ierr = tuMSGstd ("--------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGstd (""); CHKERRQ(ierr);
        if (ctx_->cosamp_->its_l1 >= ctx_->params_->opt_->gist_maxit_) {ierr = tuMSGwarn (" L1 maxiter reached"); CHKERRQ(ierr); ctx_->cosamp_->converged_l1 = true;}
        else if (PetscAbsReal (ctx_->cosamp_->J) < 1E-5)       {ierr = tuMSGwarn (" L1 absolute objective tolerance reached."); CHKERRQ(ierr); ctx_->cosamp_->converged_l1 = true;}
        else if (PetscAbsReal (ctx_->cosamp_->J_prev - ctx_->cosamp_->J) < ctx_->cosamp_->f_tol * PetscAbsReal (1 + ctx_->cosamp_->J_ref)) {ierr = tuMSGwarn (" L1 relative objective tolerance reached."); CHKERRQ(ierr); ctx_->cosamp_->converged_l1 = true;}
        else { ctx_->cosamp_->converged_l1 = false; }  // continue iterating

        ierr = tuMSG(" << leaving stage COSAMP_L1_THRES_SOL"); CHKERRQ(ierr);
        if(ctx_->cosamp_->converged_l1) {              // no break; go into next case
          ctx_->cosamp_->cosamp_stage = FINAL_L2;
        } else{                                          // break; continue iterating
          ctx_->cosamp_->cosamp_stage = COSAMP_L1_THRES_GRAD; break;
        }
      }
    // ================
    // this case may be executed in parts, i.e., going back to caller after inexact_nit Newton iterations
    case FINAL_L2: {
        ierr = tuMSG(" >> entering stage FINAL_L2"); CHKERRQ(ierr);
        /* === (3) if converged: corrective L2 solver === */
        ierr = tuMSGstd (""); CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSG("###                                              final L2 solve                                           ###");CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

        ierr = restrictSubspace(&ctx_->cosamp_->x_sub, ctx_->cosamp_->x_full, ctx_); CHKERRQ (ierr); // x_sub <-- R(x_full)

        // print vec
        if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (ctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}

        // solve interpolation
        // ierr = solveInterpolation (data_);                                        CHKERRQ (ierr);
        ctx_->params_->opt_->newton_maxit_ = ctx_->cosamp_->inexact_nits;
        // only update reference gradient and referenc objective if this is the first inexact solve for this subspace, otherwise don't
        til_opt_->updateReferenceGradient(ctx_->cosamp_->nits < ctx_->cosamp_->inexact_nits);
        til_opt_->updateReferenceObjective(ctx_->cosamp_->nits < ctx_->cosamp_->inexact_nits);
        // == solve ==
        til_opt_->setData(data_);
        ierr = til_opt_->setInitialGuess(ctx_->cosamp_->x_sub); CHKERRQ(ierr);
        ierr = til_opt_->solve(); CHKERRQ(ierr);
        ierr = VecCopy(til_opt_->getSolution(), ctx_->cosamp_->x_sub); CHKERRQ (ierr);
        ierr = tuMSG("### -------------------------------------------- L2 solver end ------------------------------------------ ###");CHKERRQ (ierr);
        ierr = tuMSGstd (""); CHKERRQ (ierr);
        // print phi's to file
        if (ctx_->params_->tu_->write_output_) {
          ierr = VecDuplicate(ctx_->tumor_->phi_->phi_vec_[0], &all_phis); CHKERRQ (ierr);
          ierr = VecSet(all_phis, 0.); CHKERRQ (ierr);
          for (int i = 0; i < ctx_->params_->tu_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, ctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
          ss << "phiSupportFinal.nc";  {dataOut (all_phis, ctx_->params_, ss.str().c_str());} ss.str(std::string()); ss.clear();
          ss << "c0FinalGuess.nc";      dataOut (ctx_->tumor_->c_0_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
          ss << "c1FinalGuess.nc"; if (ctx_->params_->tu_->verbosity_ >= 4) { dataOut (ctx_->tumor_->c_t_, ctx_->params_, ss.str().c_str()); } ss.str(std::string()); ss.clear();
          if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}
        }
        // write out p vector after IC, k inversion (unscaled)
        if (ctx_->params_->tu_->write_p_checkpoint_) { writeCheckpoint(ctx_->cosamp_->x_sub, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_, std::string("unscaled"));}
        if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (ctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}

        // == convergence test ==
        // neither gradient sufficiently small nor ls-failure (i.e., inexact_nit hit)
        // if(!ctx_->cosamp_->converged_l2 && !ctx_->cosamp_->converged_error_l2) {ctx_->cosamp_->nits += ctx_->cosamp_->inexact_nits;}
        ctx_->cosamp_->nits += ctx_->params_->optf_->nb_newton_it_;
        conv_maxit = ctx_->cosamp_->nits >= ctx_->cosamp_->maxit_newton;

        // == prolongate ==
        // prolongate restricted x_L2 to full x_L1, but do not resize vectors, i.e., call resetOperators
        // if inversion for reaction disabled, also reset operators
        finalize = !ctx_->params_->opt_->reaction_inversion_;
        contiterating = !ctx_->cosamp_->converged_l2 && !ctx_->cosamp_->converged_error_l2 && !conv_maxit;
        // TODO(K): not sure if this is correct; check if runtime error
        ierr = prolongateSubspace(ctx_->cosamp_->x_full, &ctx_->cosamp_->x_sub, ctx_, np_full, (finalize || contiterating)); CHKERRQ (ierr); // x_full <-- P(x_sub)

        // check if L2 solver converged
        if(contiterating) {
          ss << "    ... inexact solve terminated (L2 solver not converged, will be continued; its "<< ctx_->cosamp_->nits <<"/"<< ctx_->cosamp_->maxit_newton <<").";
          ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();
          ierr = tuMSG(" << leaving stage FINAL_L2"); CHKERRQ(ierr); ss.str(""); ss.clear();
          break;
        } else {
          // if L2 solver converged
          if(ctx_->cosamp_->converged_l2)        {ss << "    ... L2 solver converged; its "<< ctx_->cosamp_->nits <<"/"<< ctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
          // if L2 solver ran into ls-failure
          if(ctx_->cosamp_->converged_error_l2)  {ss << "    ... L2 solver terminated (ls-failure); its "<< ctx_->cosamp_->nits <<"/"<< ctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
          // if L2 solver hit maxit
          if(conv_maxit)                           {ss << "    ... L2 solver terminated (maxit); its "<< ctx_->cosamp_->nits <<"/"<< ctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
          ctx_->cosamp_->cosamp_stage = finalize ?  FINALIZE : POST_RD;
          conv_maxit = ctx_->cosamp_->nits = 0;
          ierr = tuMSG(" << leaving stage FINAL_L2"); CHKERRQ(ierr); ss.str(""); ss.clear();
        }
      }
    // ================
    // this case is executed at once without going back to caller in between
    case POST_RD: {
        ierr = tuMSG(" >> entering stage POST_RD"); CHKERRQ(ierr);
        // === (4) reaction/diffusion inversion ===
        if (ctx_->params_->opt_->reaction_inversion_) {
          // restrict to new L2 subspace, holding p_i, kappa, and rho
          ierr = restrictSubspace(&ctx_->cosamp_->x_sub, ctx_->cosamp_->x_full, ctx_, true); CHKERRQ (ierr); // x_sub <-- R(x_full)

          ctx_->cosamp_->cosamp_stage = POST_RD;
          ctx_->params_->opt_->newton_maxit_ = ctx_->cosamp_->maxit_newton;
          // == solve ==
          rd_opt_->setData(data_);
          ierr = rd_opt_->setInitialGuess(ctx_->cosamp_->x_sub); CHKERRQ (ierr); // with current guess as init cond.

          ierr = tuMSGstd(""); CHKERRQ (ierr);
          ierr = tuMSG   ("### scaled L2 sol. w/ incorrect reaction coefficient     ###"); CHKERRQ (ierr);
          ierr = tuMSGstd("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
          if (procid == 0) { ierr = VecView (ctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF); CHKERRQ (ierr);}
          ierr = tuMSGstd("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
          if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(ctx_->cosamp_->x_sub, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_, std::string("scaled"));}
          ierr = tuMSGstd(""); CHKERRQ (ierr);
          ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
          ierr = tuMSG("###                    (POST) rho/kappa inversion with scaled L2 solution guess                           ###");CHKERRQ (ierr);
          ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
          ierr = rd_opt_->solve(); CHKERRQ (ierr); // solve reaction/diffusion inversion
          ierr = VecCopy(rd_opt_->getSolution(), ctx_->cosamp_->x_sub); CHKERRQ (ierr); // get solution (length: np_r + nk + nr)
          // update full space solution
          ierr = prolongateSubspace(ctx_->cosamp_->x_full, &ctx_->cosamp_->x_sub, ctx_, np_full); CHKERRQ (ierr); // x_full <-- P(x_sub)
        } else {ierr = tuMSGstd("    ... skipping stage, reaction diffusion disabled."); CHKERRQ(ierr);}

        // break; go to finalize
        ctx_->cosamp_->cosamp_stage = FINALIZE;
        ierr = tuMSG(" << leaving stage POST_RD"); CHKERRQ(ierr);
        break;
      }
  }

  // prolongate (in case we are in a subuspace solve, we still need the solution to be prolongated)
  // if (ctx_->cosamp_->cosamp_stage != FINALIZE) {
      // ierr = tuMSG(" >> entering stage FINALIZE"); CHKERRQ(ierr); ss.str(""); ss.clear();
      // ierr = prolongateSubspace(ctx_->cosamp_->x_full, &ctx_->cosamp_->x_sub, ctx_, np_full); CHKERRQ (ierr); // x_full <-- P(x_sub)
  // }

  // === populate solution to xout_
  // TODO should be a copy but not the same length
  // ierr = VecCopy (ctx_->cosamp_->x_full, xrec_); CHKERRQ (ierr);
  PetscReal *x_ptr, *xout_ptr;
  ierr = VecGetArray(xrec_, &xout_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(ctx_->cosamp_->x_full, &xout_ptr); CHKERRQ(ierr);
  for(int i = 0; i < np_full + ctx_->params_->get_nk(); ++i)
    xout_ptr[i] = xout_ptr[i];                          // copy p and nk
  if (ctx_->params_->opt_->reaction_inversion_)
    xout_ptr[np_full + ctx_->params_->get_nk()] = ctx_->params_->tu_->rho_;  // copy rho
  ierr = VecRestoreArray(xrec_, &xout_ptr); CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx_->cosamp_->x_full, &xout_ptr); CHKERRQ(ierr);
  ierr = VecCopy(xrec_, xout_); CHKERRQ (ierr);

  // if not in restart mode, call function again
  if (!rs_mode_active && ctx_->cosamp_->cosamp_stage != FINALIZE) {
    solve_rs(false);
  } else {ierr = tuMSG(" << leaving inverse CoSaMp"); CHKERRQ(ierr);}
  PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::cosampMonitor(int its, PetscReal J, PetscReal jrel, PetscReal g_norm, PetscReal p_relnorm, Vec x_l1) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  PetscReal *x_ptr;
  std::stringstream s;

  s << std::setw(4)  << " iter"               << "   " << std::setw(18) << "objective (abs)" << "   "
  << std::setw(18) << "||objective||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
  << "   "  << std::setw(18) << "||dp||_rel"<< std::setw(18) << "k";
  ierr = tuMSGstd ("--------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  ierr = tuMSGwarn (s.str()); CHKERRQ(ierr); s.str (""); s.clear ();
  ierr = tuMSGstd ("--------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
  s << " "   << std::scientific << std::setprecision(5)  << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
   << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
   << "   " << std::scientific << std::setprecision(12) << std::setw(18) << jrel
   << "   " << std::scientific << std::setprecision(12) << std::setw(18) << g_norm
   << "   " << std::scientific << std::setprecision(12) << std::setw(18) << p_relnorm;
  ierr = VecGetArray(x_l1, &x_ptr); CHKERRQ(ierr);
  if (ctx_->params_->opt_->diffusivity_inversion_) {
     s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[ctx_->params_->tu_->np_];
     if (ctx_->params_->tu_->nk_ > 1) { s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[ctx_->params_->tu_->np_ + 1]; }
  } else {
   s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << ctx_->params_->tu_->k_;
  }
  ierr = VecRestoreArray(x_l1, &x_ptr); CHKERRQ(ierr);
  ierr = tuMSGwarn (s.str()); CHKERRQ(ierr); s.str (""); s.clear ();
  PetscFunctionReturn (ierr);
}
