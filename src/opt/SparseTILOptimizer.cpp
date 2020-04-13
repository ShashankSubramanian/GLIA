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
      x_restricted_ptr[np] = ctx->params_->tu_->k_;                                                 // equals x_full_ptr[np_full];
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
  ierr = resetOperators (*x_restricted); CHKERRQ (ierr); // reset phis and other operators

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
      ierr = resetOperators (x_full); CHKERRQ (ierr);} /* reset phis and other ops    */
  /* destroy, size will change   */
  if (*x_restricted != nullptr) {
      ierr = VecDestroy (x_restricted); CHKERRQ (ierr);
      x_restricted = nullptr;
  }
  PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode SparseTILOptimizer::solve() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  std::stringstream ss;

  // ctx_->params_->opt_->estimate_rho_init_guess_ = !(ctx_->params_->opt_->multilevel_ && ctx_->params_->grid_->n_[0] > 64);

  PetscFunctionReturn(ierr);
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
