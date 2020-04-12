#include "InvSolver.h"
#include "petsctao.h"
#include <iostream>
#include <limits>
#include "DerivativeOperators.h"
#include "PdeOperators.h"
#include "Parameters.h"
#include "TaoL1Solver.h"
#include <petsc/private/vecimpl.h>


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
// InvSolver::InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <Parameters> params, std::shared_ptr <Tumor> tumor)
// :
//   initialized_(false),
//   tao_is_reset_(true),
//   data_(),
//   ctx_() {
//     PetscFunctionBegin;
//     PetscErrorCode ierr = 0;
//     tao_      = nullptr;
//     H_        = nullptr;
//     xrec_     = nullptr;
//     xrec_rd_ = nullptr;
//     if( derivative_operators != nullptr && pde_operators !=nullptr && params != nullptr && tumor != nullptr) {
//         initialize (derivative_operators, pde_operators, params, tumor);
//     }
// }

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
// PetscErrorCode InvSolver::initialize (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor) {
//     PetscFunctionBegin;
//     PetscErrorCode ierr = 0;
//     if (initialized_)
//         PetscFunctionReturn (ierr);
//     ctx_ = std::make_shared<CtxInv> ();
//     ctx_->derivative_operators_ = derivative_operators;
//     ctx_->pde_operators_ = pde_operators;
//     ctx_->params_ = params;
//     ctx_->tumor_ = tumor;
//
//     if (params->opt_->invert_mass_effect_) {
//         ierr = allocateTaoObjectsMassEffect(); CHKERRQ(ierr);
//     } else {
//         // allocate memory for H, x_rec and TAO
//         ierr = allocateTaoObjects(); CHKERRQ(ierr);
//     }
//
//
//     initialized_ = true;
//     PetscFunctionReturn (ierr);
// }

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
// PetscErrorCode InvSolver::allocateTaoObjectsMassEffect (bool initialize_tao) {
//     PetscFunctionBegin;
//     PetscErrorCode ierr = 0;
//
//     // For mass-effect; invert for rho, kappa, and gamma
//     int n_inv = 3;
//     ScalarType *xrec_ptr;
//     // allocate memory for xrec_
//     ierr = VecCreateSeq (PETSC_COMM_SELF, n_inv, &xrec_);          CHKERRQ (ierr);
//     ierr = setupVec (xrec_, SEQ);                                  CHKERRQ (ierr);
//     ierr = VecSet (xrec_, 0.0);                                    CHKERRQ (ierr);
//     ierr = VecGetArray (xrec_, &xrec_ptr);                         CHKERRQ (ierr);
//     xrec_ptr[0] = 1; xrec_ptr[1] = 6; xrec_ptr[2] = 0.5;
//     // xrec_ptr[0] = 0.4; xrec_ptr[1] = 0.08;
//     ierr = VecRestoreArray (xrec_, &xrec_ptr);                     CHKERRQ (ierr);
//
//     // set up routine to compute the hessian matrix vector product
//     if (H_ == nullptr) {
//       ierr = MatCreateShell (PETSC_COMM_SELF, n_inv, n_inv, n_inv, n_inv, (void*) ctx_.get(), &H_); CHKERRQ(ierr);
//     }
//     // create TAO solver object
//     if ( tao_ == nullptr && initialize_tao) {
//       ierr = TaoCreate (PETSC_COMM_SELF, &tao_); tao_is_reset_ = true;  // triggers setTaoOptions
//     }
//
//     ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))hessianMatVec);         CHKERRQ(ierr);
//     #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 10)
//         ierr = MatShellSetOperation (H_, MATOP_CREATE_VECS, (void(*)(void)) operatorCreateVecsMassEffect);
//     #endif
//     ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                                 CHKERRQ(ierr);
//
//     PetscFunctionReturn (ierr);
// }

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
// PetscErrorCode InvSolver::allocateTaoObjects (bool initialize_tao) {
//   PetscFunctionBegin;
//   PetscErrorCode ierr = 0;
//
//   int np = ctx_->params_->tu_->np_;
//   int nk = (ctx_->params_opt_->diffusivity_inversion_) ?  ctx_->params_->tu_->nk_ : 0;
//   int nr = 0;
//
//   // allocate memory for xrec_
//   ierr = VecDuplicate (ctx_->tumor_->p_, &xrec_);                         CHKERRQ(ierr);
//   // set up routine to compute the hessian matrix vector product
//   if (H_ == nullptr) {
//     ierr = MatCreateShell (PETSC_COMM_SELF, np + nk + nr, np + nk + nr, np + nk + nr, np + nk + nr, (void*) ctx_.get(), &H_); CHKERRQ(ierr);
//   }
//   // create TAO solver object
//   if ( tao_ == nullptr && initialize_tao) {
//     ierr = TaoCreate (PETSC_COMM_SELF, &tao_); tao_is_reset_ = true;  // triggers setTaoOptions
//   }
//   ierr = VecSet (xrec_, 0.0);                                                   CHKERRQ(ierr);
//
//   // if tao's lmvm (l-bfgs) method is used and the initial hessian approximation is explicitly set
//   if ((ctx_->params_->opt_->newton_solver_ == QUASINEWTON) && ctx_->params_->opt_->lmvm_set_hessian_) {
//     ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))constApxHessianMatVec); CHKERRQ(ierr);
//     ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                                 CHKERRQ(ierr);
//     // if tao's nls (gauss-newton) method is used, define hessian matvec
//   }
//   else {
//     ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))hessianMatVec);         CHKERRQ(ierr);
//     ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                                 CHKERRQ(ierr);
//   }
//
//   PetscFunctionReturn (ierr);
// }

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::resetTao (std::shared_ptr<Parameters> params) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (tao_  != nullptr) {ierr = TaoDestroy (&tao_);  CHKERRQ(ierr); tao_  = nullptr;}
    if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
    if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}

    // allocate memory for H, x_rec and TAO
    ierr = allocateTaoObjects (); CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
// PetscErrorCode InvSolver::setParams (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor, bool npchanged) {
//     PetscFunctionBegin;
//     PetscErrorCode ierr = 0;
//     ctx_->derivative_operators_ = derivative_operators;
//     ctx_->pde_operators_ = pde_operators;
//     ctx_->params_ = params;
//     ctx_->tumor_ = tumor;
//     // re-allocate memory
//     if (npchanged && !params->opt_->invert_mass_effect_){                              // re-allocate memory for xrec_
//       // allocate memory for H, x_rec and TAO
//       ctx_->x_old = nullptr; // Will be set accordingly in the solver
//       if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
//       if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}
//       ierr = allocateTaoObjects (false); CHKERRQ(ierr);
//     }
//
//     if (params->opt_->invert_mass_effect_) {
//         // allocate memory for H, x_rec and TAO
//       ctx_->x_old = nullptr; // Will be set accordingly in the solver
//       if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
//       if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}
//       ierr = allocateTaoObjectsMassEffect (false); CHKERRQ(ierr);
//     }
//
//     tao_is_reset_ = true;                        // triggers setTaoOptions
//     PetscFunctionReturn (ierr);
// }

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::resetOperators (Vec p) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    // reset tumor_ object, re-size solution vector and copy p into tumor_->p_
    ierr = ctx_->tumor_->setParams (p, ctx_->params_, true);                CHKERRQ (ierr);
    // reset derivative operators, re-size vectors
    ctx_->derivative_operators_->reset(p, ctx_->pde_operators_, ctx_->params_, ctx_->tumor_);
    // re-allocate memory
    ctx_->x_old = nullptr; // Will be set accordingly in the solver
    if (tao_  != nullptr) {ierr = TaoDestroy (&tao_);  CHKERRQ(ierr); tao_  = nullptr;}
    if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
    if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}
    ierr = allocateTaoObjects (true); CHKERRQ(ierr);
    tao_is_reset_ = true;                        // triggers setTaoOptions
    PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::restrictSubspace (Vec *x_restricted, Vec x_full, std::shared_ptr<CtxInv> itctx, bool create_rho_dofs = false) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscReal *x_restricted_ptr, *x_full_ptr;
    int np = itctx->params_->tu_->support_.size();        // size of restricted subspace (not necessarily 2s, since merged)
    int nk = (itctx->params_->opt_->diffusivity_inversion_ || (create_rho_dofs &&  itctx->params_->opt_->reaction_inversion_)) ? itctx->params_->tu_->nk_ : 0;
    int nr = (itctx->params_->opt_->reaction_inversion_ && create_rho_dofs) ? itctx->params_->tu_->nr_ : 0;

    itctx->params_->tu_->np_ = np;                    // change np to solve the restricted subsystem
    ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk + nr, x_restricted);          CHKERRQ (ierr);
    ierr = setupVec (*x_restricted, SEQ);                                       CHKERRQ (ierr);
    ierr = VecSet (*x_restricted, 0);                                           CHKERRQ (ierr);
    ierr = VecGetArray (*x_restricted, &x_restricted_ptr);                      CHKERRQ (ierr);
    ierr = VecGetArray (x_full, &x_full_ptr);                                   CHKERRQ (ierr);
    for (int i = 0; i < np; i++)
        x_restricted_ptr[i] = x_full_ptr[itctx->params_->tu_->support_[i]];
    // initial guess diffusivity
    if (itctx->params_->opt_->diffusivity_inversion_) {
        x_restricted_ptr[np] = itctx->params_->tu_->k_;                                                 // equals x_full_ptr[np_full];
        if (nk > 1) x_restricted_ptr[np+1] = itctx->params_->tu_->k_ * itctx->params_->tu_->k_gm_wm_ratio_;  // equals x_full_ptr[np_full+1];
        if (nk > 2) x_restricted_ptr[np+2] = itctx->params_->tu_->k_ * itctx->params_->tu_->k_glm_wm_ratio_; // equals x_full_ptr[np_full+2];
    }
    // initial guess reaction
    if (create_rho_dofs && itctx->params_->opt_->reaction_inversion_) {
        x_restricted_ptr[np + nk] = itctx->params_->tu_->rho_;
        if (nr > 1) x_restricted_ptr[np + nk + 1] = itctx->params_->tu_->rho_ * itctx->params_->tu_->r_gm_wm_ratio_;
        if (nr > 2) x_restricted_ptr[np + nk + 2] = itctx->params_->tu_->rho_ * itctx->params_->tu_->r_glm_wm_ratio_;
    }
    ierr = VecRestoreArray (*x_restricted, &x_restricted_ptr);                  CHKERRQ (ierr);
    ierr = VecRestoreArray (x_full, &x_full_ptr);                               CHKERRQ (ierr);
    // Modifies the centers
    itctx->tumor_->phi_->modifyCenters (itctx->params_->tu_->support_);
    // resets the phis and other operators, x_restricted is copied into tumor->p_ and is used as init cond for
    // the L2 solver (needs to be done in every iteration, since location of basis functions updated)
    ierr = resetOperators (*x_restricted); /* reset phis and other operators */ CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::prolongateSubspace (Vec x_full, Vec *x_restricted, std::shared_ptr<CtxInv> itctx, int np_full, bool reset_operators = true) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscReal *x_restricted_ptr, *x_full_ptr;
    int np_r = itctx->params_->tu_->support_.size();        // size of restricted subspace (not necessarily 2s, since merged)
    int nk   = (itctx->params_->opt_->diffusivity_inversion_) ? itctx->params_->tu_->nk_ : 0;

    ierr = VecSet (x_full, 0.);                                                 CHKERRQ (ierr);
    ierr = VecGetArray (*x_restricted, &x_restricted_ptr);                      CHKERRQ (ierr);
    ierr = VecGetArray (x_full, &x_full_ptr);                                   CHKERRQ (ierr);
    // correct L1 guess
    for (int i = 0; i < np_r; i++)
      x_full_ptr[itctx->params_->tu_->support_[i]] = x_restricted_ptr[i];
    // correct diffusivity
    if (itctx->params_->opt_->diffusivity_inversion_) {
        itctx->params_->tu_->k_ = x_restricted_ptr[np_r];
        x_full_ptr[np_full] = itctx->params_->tu_->k_;
        if (nk > 1) x_full_ptr[np_full+1] = itctx->params_->tu_->k_ * itctx->params_->tu_->k_gm_wm_ratio_;
        if (nk > 2) x_full_ptr[np_full+2] = itctx->params_->tu_->k_ * itctx->params_->tu_->k_glm_wm_ratio_;
    }
    ierr = VecRestoreArray (*x_restricted, &x_restricted_ptr);                  CHKERRQ (ierr);
    ierr = VecRestoreArray (x_full, &x_full_ptr);                               CHKERRQ (ierr);

    itctx->params_->tu_->np_ = np_full;         /* reset to full space         */
    itctx->tumor_->phi_->resetCenters ();  /* reset all the basis centers */
    if (reset_operators) {
        ierr = resetOperators (x_full);    /* reset phis and other ops    */    CHKERRQ (ierr);}
    /* destroy, size will change   */
    if (*x_restricted != nullptr) {
        ierr = VecDestroy (x_restricted);            CHKERRQ (ierr);
        x_restricted = nullptr;
    }
    PetscFunctionReturn (ierr);
}


/* ------------------------------------------------------------------- */
/*
 solveInverseReacDiff - solves tumor inversion for rho and kappa, given c(0)
 Input Parameters:
 .  Vec x_in - initial guess for c(0) and rho, kappa
 Output Parameters:
 .  none
 .  (implicitly writes solution in x_rec, i.e., tumor_->p_)
 Assumptions:
 .  observation operator is set
 .  data for objective and gradient is set (InvSolver::setData(Vec d))
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::solveInverseReacDiff (Vec x_in) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    TU_assert (initialized_,              "InvSolver::solveInverseReacDiff (): InvSolver needs to be initialized.")
    TU_assert (data_ != nullptr,          "InvSolver::solveInverseReacDiff (): requires non-null input data for inversion.");
    TU_assert (xrec_ != nullptr,          "InvSolver::solveInverseReacDiff (): requires non-null p_rec vector to be set");

    PetscReal beta_p = ctx_->params_->opt_->beta_;     // set beta to zero here as the params are rho and kappa
    ctx_->params_->opt_->flag_reaction_inv_ = true;    // enables derivative operators to compute the gradient w.r.t rho
    ctx_->params_->opt_->beta_ = 0.;
    PetscReal *d_ptr, *x_in_ptr, *x_ptr, *ub_ptr, *lb_ptr, *x_full_ptr;
    PetscReal d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0., max, min, xdiff;
    PetscReal upper_bound_kappa, lower_bound_kappa, minstep;
    std::string msg;
    std::stringstream ss;
    int nk, nr, np, x_sz;
    Vec lower_bound, upper_bound, p;
    CtxInv *ctx = ctx_.get();
    TaoLineSearch linesearch;
    TaoConvergedReason reason;

    // ls ministep
    minstep = std::pow (2.0, 18.0);
    minstep = 1.0 / minstep;

    // DOFs
    nk = ctx_->params_->tu_->nk_;
    nr = ctx_->params_->tu_->nr_;
    np = ctx_->params_->tu_->np_;
    x_sz = nk + nr;

    // rescale init cond. and invert for rho/kappa
    PetscReal ic_max = 0., g_norm_ref = 0.;
    ierr = ctx_->tumor_->phi_->apply(ctx_->tumor_->c_0_, x_in);             CHKERRQ (ierr);
    ierr = VecMax (ctx_->tumor_->c_0_, NULL, &ic_max);                        CHKERRQ (ierr);
    ierr = VecGetArray (x_in, &x_in_ptr);                                       CHKERRQ (ierr);
    /* scale p to one according to our modeling assumptions:
     * scales INT_Omega phi(x) dx = const across levels, factor in between levels: 2
     * scales nx=256 to max {Phi p} = 1, nx=128 to max {Phi p} = 0.5, nx=64 to max {Phi p} = 0.25 */
    for (int i = 0; i < np ; i++){
        if(ctx_->params_->tu_->multilevel_) { x_in_ptr[i] *= (1.0/4.0 * ctx_->params_->grid_->n_[0]/64.  / ic_max);}
        else                             { x_in_ptr[i] *= (1.0 / ic_max); }
      }
    ierr = VecRestoreArray (x_in, &x_in_ptr);                                   CHKERRQ (ierr);

    // write out p vector after IC, k inversion (scaled)
    ierr = tuMSGstd ("");                                                       CHKERRQ (ierr);
    if (ctx_->cosamp_->cosamp_stage == PRE_RD) {
        ierr = tuMSG    ("### scaled init guess w/ incorrect reaction coefficient  ###"); CHKERRQ (ierr);
        ierr = tuMSGstd ("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
        if (procid == 0) { ierr = VecView (x_in, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}
        ierr = tuMSGstd ("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
        if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(x_in, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_ .str(), std::string("scaled-pre-l1"));}
        ierr = tuMSGstd ("");                                                             CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSG("###                     (PRE) rho/kappa inversion with scaled L2 solution guess                           ###");CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    } else if (ctx_->cosamp_->cosamp_stage == POST_RD) {
        ierr = tuMSG    ("### scaled L2 sol. w/ incorrect reaction coefficient     ###"); CHKERRQ (ierr);
        ierr = tuMSGstd ("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
        if (procid == 0) { ierr = VecView (x_in, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}
        ierr = tuMSGstd ("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
        if (ctx_->params_->tu_->write_p_checkpoint_) {writeCheckpoint(x_in, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_ .str(), std::string("scaled"));}
        ierr = tuMSGstd ("");                                                             CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSG("###                          rho/kappa inversion with scaled L2 solution guess                            ###");CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    }

    // Reset tao
    if (tao_      != nullptr)         {ierr = TaoDestroy (&tao_);                 CHKERRQ(ierr); tao_  = nullptr;}
    if (H_        != nullptr)         {ierr = MatDestroy (&H_);                   CHKERRQ(ierr); H_    = nullptr;}
    if (xrec_     != nullptr)         {ierr = VecDestroy (&xrec_);                CHKERRQ(ierr); xrec_ = nullptr;}
    if (xrec_rd_ != nullptr)          {ierr = VecDestroy (&xrec_rd_);             CHKERRQ(ierr); xrec_rd_ = nullptr;}
    if (ctx_->x_old != nullptr)     {ierr = VecDestroy (&ctx_->x_old);        CHKERRQ(ierr); ctx_->x_old = nullptr;}
    // TODO: x_old here is used to store the full solution vector and not old guess. (Maybe change to new vector to avoid confusion?)
    // re-allocate
    ierr = VecDuplicate (x_in, &ctx_->x_old);                                   CHKERRQ (ierr);
    ierr = VecCopy      (x_in,  ctx_->x_old);   /* stores full solution vec */  CHKERRQ (ierr);
    ierr = VecDuplicate (x_in, &xrec_);                                           CHKERRQ(ierr);
    ierr = VecSet       (xrec_, 0.0);                                             CHKERRQ(ierr);
    ierr = TaoCreate    (PETSC_COMM_SELF, &tao_);                                 CHKERRQ (ierr);
    ierr = TaoSetType   (tao_, "blmvm");                                          CHKERRQ (ierr);
    ierr = VecCreateSeq (PETSC_COMM_SELF, x_sz, &xrec_rd_); /* inv rho and k */   CHKERRQ (ierr);
    ierr = setupVec     (xrec_rd_, SEQ);                                          CHKERRQ (ierr);
    ierr = VecSet       (xrec_rd_, 0.);                                           CHKERRQ (ierr);
    ierr = MatCreateShell (PETSC_COMM_SELF, np + nk, np + nk, np + nk, np + nk, (void*) ctx_.get(), &H_); CHKERRQ(ierr);

    // initial guess kappa
    ierr = VecGetArray (x_in, &x_in_ptr);                                         CHKERRQ (ierr);
    ierr = VecGetArray (xrec_rd_, &x_ptr);                                        CHKERRQ (ierr);
    x_ptr[0] = (nk > 0) ? x_in_ptr[ctx_->params_->tu_->np_] : 0;   // k1
    if (nk > 1) x_ptr[1] = x_in_ptr[ctx_->params_->tu_->np_ + 1];  // k2
    if (nk > 2) x_ptr[2] = x_in_ptr[ctx_->params_->tu_->np_ + 2];  // k3
    ss << " initial guess for diffusion coefficient: " << x_ptr[0]; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    // initial guess rho
    if (ctx_->params_->tu_->multilevel_ && ctx_->params_->grid_->n_[0] > 64) {
    x_ptr[nk] = x_in_ptr[ctx_->params_->tu_->np_ + nk];                      // r1
    if (nr > 1) x_ptr[nk + 1] = x_in_ptr[ctx_->params_->tu_->np_ + nk + 1];  // r2
    if (nr > 2) x_ptr[nk + 2] = x_in_ptr[ctx_->params_->tu_->np_ + nk + 2];  // r3
    } else {
    ss<<" computing rough approximation to rho.."; ierr = tuMSGstd(ss.str());   CHKERRQ(ierr); ss.str(""); ss.clear();
    std::array<ScalarType, 7> rho_guess = {0, 3, 6, 9, 10, 12, 15};
    ScalarType min_norm = 1E15, norm = 0.;

    int idx = 0;
    for (int i = 0; i < rho_guess.size(); i++) {
      // update the tumor with this rho
      ierr = ctx_->tumor_->rho_->updateIsotropicCoefficients (rho_guess[i], 0., 0., ctx_->tumor_->mat_prop_, ctx_->params_);
      ierr = ctx_->tumor_->phi_->apply (ctx_->tumor_->c_0_, x_in);          CHKERRQ (ierr);   // apply scaled p to IC
      ierr = ctx_->derivative_operators_->pde_operators_->solveState (0);    // solve state with guess reaction and inverted diffusivity
      ierr = ctx_->tumor_->obs_->apply (ctx_->derivative_operators_->temp_, ctx_->tumor_->c_t_, 1);               CHKERRQ (ierr);
      // mismatch between data and c
      ierr = VecAXPY (ctx_->derivative_operators_->temp_, -1.0, data_);       CHKERRQ (ierr);    // Oc(1) - d
      ierr = VecNorm (ctx_->derivative_operators_->temp_, NORM_2, &norm);     CHKERRQ (ierr);
      if (norm < min_norm) { min_norm = norm; idx = i; }
    }
    x_ptr[nk] = rho_guess[idx];  // rho
    if (nr > 1) x_ptr[nk + 1] = 0;  // r2
    if (nr > 2) x_ptr[nk + 2] = 0;  // r3
    }
    ss << " initial guess for reaction coefficient: " << x_ptr[nk]; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    ierr = VecRestoreArray     (x_in, &x_in_ptr);                                 CHKERRQ (ierr);
    ierr = VecRestoreArray     (xrec_rd_, &x_ptr);                                CHKERRQ (ierr);
    ierr = TaoSetInitialVector (tao_, xrec_rd_);                                  CHKERRQ (ierr);

    // TAO type from user input
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    TaoType taotype = NULL;
    ierr = TaoGetType (tao_, &taotype);                                         CHKERRQ(ierr);
    #else
    const TaoType taotype;
    ierr = TaoGetType (tao_, &taotype);                                         CHKERRQ(ierr);
    #endif
    if (strcmp(taotype, "nls") == 0) {
      msg = " limited memory variable metric method (unconstrained) selected";
    } else if (strcmp(taotype, "ntr") == 0) {
      msg = " Newton's method with trust region for unconstrained minimization";
    } else if (strcmp(taotype, "ntl") == 0) {
      msg = " Newton's method with trust region, line search for unconstrained minimization";
    } else if (strcmp(taotype, "nls") == 0) {
      msg = " Newton's method (line search; unconstrained) selected";
    } else if (strcmp(taotype, "bnls") == 0) {
      msg = " Newton's method (line search; bound constraints) selected";
    } else if (strcmp(taotype, "bqnls") == 0) {
      msg = " Quasi-Newton's method (line search; bound constraints) selected";
    } else if (strcmp(taotype, "ntr") == 0) {
      msg = " Newton's method (trust region; unconstrained) selected";
    } else if (strcmp(taotype, "fd_test") == 0) {
      msg = " gradient test selected";
    } else if (strcmp(taotype, "cg") == 0) {
      msg = " CG selected";
    } else if (strcmp(taotype, "tron") == 0) {
      msg = " Newton Trust Region method chosen";
    } else if (strcmp(taotype, "blmvm") == 0) {
      msg = "  Bounded limited memory variable metric method chosen";
    } else if (strcmp(taotype, "tao_blmvm_m") == 0) {
      msg = " user modified bounded limited memory variable metric method chosen";
    } else if (strcmp(taotype, "lmvm") == 0) {
      msg = " Limited memory variable metric method chosen";
    } else if (strcmp(taotype, "gpcg") == 0) {
      msg = " Newton Trust Region method for quadratic bound constrained minimization";
    } else if (strcmp(taotype, "tao_L1") == 0) {
      msg = " user defined solver for L1 minimization";
    } else {
      msg = " numerical optimization method not supported (setting default: BLMVM)";
      ierr = TaoSetType (tao_, "blmvm");                                        CHKERRQ(ierr);
    }
    ierr = tuMSGstd(msg);                                                         CHKERRQ(ierr);

    // lower and upper bounds
    upper_bound_kappa = ctx_->params_->opt_->k_ub_;
    lower_bound_kappa = ctx_->params_->opt_->k_lb_;
    ierr = VecDuplicate (xrec_rd_, &lower_bound);                                CHKERRQ (ierr);
    ierr = VecSet       (lower_bound, 0.);                                        CHKERRQ (ierr);
    ierr = VecDuplicate (xrec_rd_, &upper_bound);                                CHKERRQ (ierr);
    ierr = VecSet       (upper_bound, PETSC_INFINITY);                            CHKERRQ (ierr);
    ierr = VecGetArray  (upper_bound, &ub_ptr);                                   CHKERRQ (ierr);
    ub_ptr[0] = upper_bound_kappa;
    if (nk > 1) ub_ptr[1] = upper_bound_kappa;
    if (nk > 2) ub_ptr[2] = upper_bound_kappa;
    ierr = VecRestoreArray (upper_bound, &ub_ptr);                                CHKERRQ (ierr);
    ierr = VecGetArray     (lower_bound, &lb_ptr);                                CHKERRQ (ierr);
    lb_ptr[0] = lower_bound_kappa;
    if (nk > 1) lb_ptr[1] = lower_bound_kappa;
    if (nk > 2) lb_ptr[2] = lower_bound_kappa;
    ierr = VecRestoreArray (lower_bound, &lb_ptr);                                CHKERRQ (ierr);
    ierr = TaoSetVariableBounds(tao_, lower_bound, upper_bound);                  CHKERRQ (ierr);
    if (lower_bound != nullptr) {ierr = VecDestroy (&lower_bound); CHKERRQ (ierr); lower_bound = nullptr;}
    if (upper_bound != nullptr) {ierr = VecDestroy (&upper_bound); CHKERRQ (ierr); upper_bound = nullptr;}

    ierr = TaoSetObjectiveRoutine (tao_, evaluateObjectiveReacDiff, (void*) ctx);                                      CHKERRQ(ierr);
    ierr = TaoSetGradientRoutine (tao_, evaluateGradientReacDiff, (void*) ctx);                                        CHKERRQ(ierr);
    ierr = TaoSetObjectiveAndGradientRoutine (tao_, evaluateObjectiveAndGradientReacDiff, (void*) ctx);                CHKERRQ (ierr);
    ierr = TaoSetMonitor (tao_, optimizationMonitorReacDiff, (void *) ctx, NULL);                                      CHKERRQ(ierr);
    ierr = TaoSetTolerances (tao_, ctx->params_->opt_->gatol_, ctx->params_->opt_->grtol_, ctx->params_->opt_->opttolgrad_); CHKERRQ(ierr);
    ierr = TaoSetMaximumIterations (tao_, ctx->params_->opt_->newton_maxit_);                                            CHKERRQ(ierr);
    ierr = TaoSetConvergenceTest (tao_, checkConvergenceGradReacDiff, ctx);                                            CHKERRQ(ierr);

    ctx_->update_reference_gradient = true;    // compute ref gradient
    ctx_->params_->opt_->ls_minstep_ = minstep;  // overwrite linesearch objects
    ierr = TaoGetLineSearch (tao_, &linesearch);                                  CHKERRQ(ierr);
    linesearch->stepmin = minstep;
    if (ctx->params_->opt_->linesearch_ == ARMIJO) {
    ierr = TaoLineSearchSetType (linesearch, "armijo");                         CHKERRQ(ierr);
    ierr = tuMSGstd(" using line-search type: armijo");                         CHKERRQ(ierr);
    } else { ierr = tuMSGstd(" using line-search type: more-thuene"); CHKERRQ(ierr);}
    ierr = TaoLineSearchSetOptionsPrefix (linesearch,"tumor_");                   CHKERRQ(ierr);


    ierr = tuMSGstd(" parameters (optimizer):");                                  CHKERRQ(ierr);
    ierr = tuMSGstd(" tolerances (stopping conditions):");                        CHKERRQ(ierr);
    ss << "   gatol: "<< ctx->params_->opt_->gatol_;      tuMSGstd(ss.str()); ss.str(""); ss.clear();
    ss << "   grtol: "<< ctx->params_->opt_->grtol_;      tuMSGstd(ss.str()); ss.str(""); ss.clear();
    ss << "   gttol: "<< ctx->params_->opt_->opttolgrad_; tuMSGstd(ss.str()); ss.str(""); ss.clear();
    ierr = TaoSetFromOptions(tao_);                                               CHKERRQ(ierr);
    // reset feedback variables, reset data
    ctx_->update_reference_gradient_hessian_ksp        = true;
    ctx_->params_->optf_->converged_    = false;
    ctx_->params_->optf_->nb_krylov_it_->solverstatus_ = "";
    ctx_->params_->optf_->nb_newton_it_ = 0;
    ctx_->params_->optf_->nb_krylov_it_ = 0;
    ctx_->params_->optf_->nb_matvecs_   = 0;
    ctx_->params_->optf_->nb_objevals_  = 0;
    ctx_->params_->optf_->nb_gradevals_ = 0;
    ctx_->data                       = data_;
    ctx_->params_->tu_->statistics_.reset();
    ss << " tumor regularization = "<< ctx_->params_->opt_->beta_ << " type: " << ctx_->params_->opt_->regularization_norm_;  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    double self_exec_time_tuninv = -MPI_Wtime(); double invtime = 0;
    // ====== solve ======
    ierr = TaoSolve (tao_);                                                       CHKERRQ(ierr);
    self_exec_time_tuninv += MPI_Wtime();
    MPI_Reduce(&self_exec_time_tuninv, &invtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    ierr = TaoGetSolutionVector (tao_, &p);                                       CHKERRQ(ierr);
    ierr = VecGetArray          (p, &x_ptr);                                      CHKERRQ (ierr);
    ierr = VecGetArray          (ctx_->x_old, &x_full_ptr);                     CHKERRQ (ierr);
    x_full_ptr[ctx_->params_->tu_->np_] = x_ptr[0];                                    // k1
    if (ctx_->params_->tu_->nk_ > 1) x_full_ptr[ctx_->params_->tu_->np_ + 1] = x_ptr[1];  // k2
    if (ctx_->params_->tu_->nk_ > 2) x_full_ptr[ctx_->params_->tu_->np_ + 2] = x_ptr[2];  // k3
    x_full_ptr[ctx_->params_->tu_->np_ + ctx_->params_->tu_->nk_] = x_ptr[ctx_->params_->tu_->nk_];                                        // r1
    if (ctx_->params_->tu_->nr_ > 1) x_full_ptr[ctx_->params_->tu_->np_ + ctx_->params_->tu_->nk_ + 1] = x_ptr[ctx_->params_->tu_->nk_ + 1];  // r2
    if (ctx_->params_->tu_->nr_ > 2) x_full_ptr[ctx_->params_->tu_->np_ + ctx_->params_->tu_->nk_ + 2] = x_ptr[ctx_->params_->tu_->nk_ + 2];  // r2
    ierr = VecRestoreArray      (p, &x_ptr);                                      CHKERRQ (ierr);
    ierr = VecRestoreArray      (ctx_->x_old, &x_full_ptr);                     CHKERRQ (ierr);
    // store sol in xrec_
    ierr = VecCopy (ctx_->x_old, xrec_);                                        CHKERRQ(ierr);

    /* Get information on termination */
    ierr = TaoGetConvergedReason (tao_, &reason);                                 CHKERRQ(ierr);
    /* get solution status */
    ierr = TaoGetSolutionStatus (tao_, NULL, &ctx_->params_->optf_->jval_, &ctx_->params_->optf_->gradnorm_, NULL, &xdiff, NULL);         CHKERRQ(ierr);
    /* display convergence reason: */
    ierr = dispTaoConvReason (reason, ctx_->params_->optf_->nb_krylov_it_->solverstatus_);        CHKERRQ(ierr);
    ctx_->params_->optf_->nb_newton_it_--;
    ss << " optimization done: #N-it: " << ctx_->params_->optf_->nb_newton_it_    << ", #K-it: " << ctx_->params_->optf_->nb_krylov_it_
                      << ", #matvec: " << ctx_->params_->optf_->nb_matvecs_    << ", #evalJ: " << ctx_->params_->optf_->nb_objevals_
                      << ", #evaldJ: " << ctx_->params_->optf_->nb_gradevals_  << ", exec time: " << invtime;
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGstd (ss.str());                                                                                           CHKERRQ(ierr);  ss.str(""); ss.clear();
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ctx_->params_->tu_->statistics_.print();
    ctx_->params_->tu_->statistics_.reset();

    tao_is_reset_ = false;
    ctx_->params_->opt_->beta_ = beta_p;              // restore beta value
    ctx_->params_->opt_->flag_reaction_inv_ = false;  // disables derivative operators to compute the gradient w.r.t rho

    // get diffusivity and reaction
    ierr = VecGetArray (xrec_, &x_ptr);                                         CHKERRQ (ierr);
    ctx_->params_->tu_->rho_ = x_ptr[np + nk];
    ctx_->params_->tu_->k_   = x_ptr[np];
    ierr = VecRestoreArray (xrec_, &x_ptr);                                     CHKERRQ (ierr);
    PetscReal r1, r2, r3, k1, k2, k3;
    r1 = ctx_->params_->tu_->rho_;                                                                      // equals x_in_ptr[np + nk]
    r2 = (ctx_->params_->tu_->nr_ > 1) ? ctx_->params_->tu_->rho_ * ctx_->params_->tu_->r_gm_wm_ratio_  : 0;  // equals x_in_ptr[np + nk + 1]
    r3 = (ctx_->params_->tu_->nr_ > 2) ? ctx_->params_->tu_->rho_ * ctx_->params_->tu_->r_glm_wm_ratio_ : 0;  // equals x_in_ptr[np + nk + 2]
    k1 = ctx_->params_->tu_->k_;                                                                        // equals x_in_ptr[np];
    k2 = (ctx_->params_->tu_->nk_ > 1) ? ctx_->params_->tu_->k_   * ctx_->params_->tu_->k_gm_wm_ratio_  : 0;  // equals x_in_ptr[np+1];
    k3 = (ctx_->params_->tu_->nk_ > 2) ? ctx_->params_->tu_->k_   * ctx_->params_->tu_->k_glm_wm_ratio_ : 0;  // equals x_in_ptr[np+2];

    ierr = ctx_->tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, ctx_->tumor_->mat_prop_, ctx_->params_);    CHKERRQ (ierr);
    ierr = ctx_->tumor_->rho_->updateIsotropicCoefficients (r1, r2, r3, ctx_->tumor_->mat_prop_, ctx_->params_);  CHKERRQ (ierr);

    if (ctx_->cosamp_->cosamp_stage == PRE_RD) {
        ierr = tuMSG("### -------------------------------------- (PRE) rho/kappa solver end ----------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSGstd (""); CHKERRQ (ierr);
        ierr = tuMSGstd ("");                                                   CHKERRQ (ierr);
        ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
        ierr = tuMSG    ("### estimated reaction coefficients (pre L1):         ###"); CHKERRQ (ierr);
        ss << "    r1: "<< r1 << ", r2: " << r2 << ", r3: "<< r3;
        ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ierr = tuMSG    ("### estimated diffusion coefficients (pre L1):        ###"); CHKERRQ (ierr);
        ss << "    k1: "<< k1 << ", k2: " << k2 << ", k3: "<< k3;
        ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
        ierr = tuMSGstd ("");                                                   CHKERRQ (ierr);
        ierr = tuMSGstd ("");                                                   CHKERRQ (ierr);
    } else if (ctx_->cosamp_->cosamp_stage == POST_RD) {
        ierr = tuMSG("### ---------------------------------------- rho/kappa solver end --------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSGstd (""); CHKERRQ (ierr);
        ierr = tuMSGstd ("");                                                     CHKERRQ (ierr);
        ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
        ierr = tuMSG    ("### estimated reaction coefficients:                  ###"); CHKERRQ (ierr);
        ss << "    r1: "<< r1 << ", r2: " << r2 << ", r3: "<< r3;
        ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ierr = tuMSG    ("### estimated diffusion coefficients:                 ###"); CHKERRQ (ierr);
        ss << "    k1: "<< k1 << ", k2: " << k2 << ", k3: "<< k3;
        ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
    }
    // cleanup
    if (ctx_->x_old != nullptr) {ierr = VecDestroy (&ctx_->x_old);  CHKERRQ (ierr); ctx_->x_old = nullptr;}
    if (noise != nullptr)         {ierr = VecDestroy (&noise);          CHKERRQ (ierr);         noise = nullptr;}
    // go home
    PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::solveForMassEffect () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    TU_assert (initialized_,              "InvSolver::solve (): InvSolver needs to be initialized.")
    TU_assert (data_ != nullptr,          "InvSolver::solve (): requires non-null input data for inversion.");
    TU_assert (xrec_ != nullptr,          "InvSolver::solve (): requires non-null p_rec vector to be set");

    std::stringstream s;
    PetscScalar max, min, w = 1, p_max, xdiff;
    PetscScalar d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0.;
    PetscScalar *d_ptr, *noise_ptr, *p_ptr, *w_ptr;
    TaoConvergedReason reason;
    Vec noise;

    if (ctx_->params_->tu_->write_output__) { dataOut (data_, ctx_->params_, "data.nc"); }

    /* === initialize inverse tumor context === */
    if (ctx_->c0_old == nullptr) {
      // ierr = VecDuplicate (data_, &ctx_->c0_old);                              CHKERRQ(ierr);
      // ierr = VecSet (ctx_->c0_old, 0.0);                                       CHKERRQ(ierr);
    }
    if (ctx_->tmp == nullptr) {
      // ierr = VecDuplicate (data_, &ctx_->tmp);                                CHKERRQ(ierr);
      // ierr = VecSet (ctx_->tmp, 0.0);                                         CHKERRQ(ierr);
    }
    if (ctx_->x_old == nullptr)  {
      ierr = VecDuplicate (xrec_, &ctx_->x_old);                              CHKERRQ (ierr);
      ierr = VecCopy (xrec_, ctx_->x_old);                                    CHKERRQ (ierr);
    }
    // reset opt solver statistics
    ctx_->update_reference_gradient_hessian_ksp        = true;
    ctx_->params_->optf_->converged_    = false;
    ctx_->params_->optf_->solverstatus_ = "";
    ctx_->params_->optf_->nb_newton_it_ = 0;
    ctx_->params_->optf_->nb_krylov_it_ = 0;
    ctx_->params_->optf_->nb_matvecs_   = 0;
    ctx_->params_->optf_->nb_objevals_  = 0;
    ctx_->params_->optf_->nb_gradevals_ = 0;
    ctx_->data                       = data_;
    // reset tao, if we want virgin TAO for every inverse solve
    if (ctx_->params_->opt_->reset_tao_) {
        ierr = resetTao(ctx_->params_);                                       CHKERRQ(ierr);
    }

    /* === set TAO options === */
    if (tao_is_reset_) {
        ierr = setTaoOptionsMassEffect (tao_, ctx_.get());                                          CHKERRQ(ierr);
        ierr = TaoSetHessianRoutine (tao_, H_, H_, matfreeHessian, (void *) ctx_.get());  CHKERRQ(ierr);
    }

    s << " using tumor regularization = "<< ctx_->params_->opt_->beta_ << " type: " << ctx_->params_->opt_->regularization_norm_;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    if (ctx_->params_->tu_->verbosity_ >= 2) { ctx_->params_->tu_->outfile_sol_  << "\n ## ----- ##" << std::endl << std::flush; ctx_->params_->tu_->outfile_grad_ << "\n ## ----- ## "<< std::endl << std::flush; }
    //Gradient check begin
    //    ierr = ctx_->derivative_operators_->checkGradient (ctx_->tumor_->p_, ctx_->data);
    //Gradient check end

    /* === solve === */
    ctx_->params_->tu_->statistics_.reset();
    double self_exec_time_tuninv = -MPI_Wtime(); double invtime = 0;
    ierr = TaoSolve (tao_);                                                       CHKERRQ(ierr);
    self_exec_time_tuninv += MPI_Wtime();
    MPI_Reduce(&self_exec_time_tuninv, &invtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* === get solution === */
    Vec p; ierr = TaoGetSolutionVector (tao_, &p);                                CHKERRQ(ierr);
    ierr = VecCopy (p, xrec_);                                                    CHKERRQ(ierr);

    PetscScalar *x_ptr;
    ierr = VecGetArray (xrec_, &x_ptr);                                 CHKERRQ (ierr);
    ctx_->params_->tu_->forcing_factor_ = 1E4 * x_ptr[0]; // re-scaling parameter scales
    ctx_->params_->tu_->rho_ = 1 * x_ptr[1];                  // rho
    ctx_->params_->tu_->k_   = 1E-2 * x_ptr[2];                  // kappa
    ierr = VecRestoreArray (xrec_, &x_ptr);                             CHKERRQ (ierr);

    s << " Forcing factor at final guess = " << ctx_->params_->tu_->forcing_factor_; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    s << " Reaction at final guess       = " << ctx_->params_->tu_->rho_; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    s << " Diffusivity at final guess    = " << ctx_->params_->tu_->k_; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    /* === get termination info === */
    TaoGetConvergedReason (tao_, &reason);

    /* === get last line-search step used ==== */
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    TaoType taotype = NULL; ierr = TaoGetType (tao_, &taotype);                 CHKERRQ(ierr);
    #else
    const TaoType taotype; ierr = TaoGetType (tao_, &taotype);                  CHKERRQ(ierr);
    #endif
    ierr = TaoGetType (tao_, &taotype); CHKERRQ(ierr);

    /* === get solution status === */
    ierr = TaoGetSolutionStatus (tao_, NULL, &ctx_->params_->optf_->jval_, &ctx_->params_->optf_->gradnorm_, NULL, &xdiff, NULL); CHKERRQ(ierr);
    /* display convergence reason: */
    ierr = dispTaoConvReason (reason, ctx_->params_->optf_->solverstatus_);        CHKERRQ(ierr);
    ctx_->params_->optf_->nb_newton_it_--;
    s << " optimization done: #N-it: " << ctx_->params_->optf_->nb_newton_it_    << ", #K-it: " << ctx_->params_->optf_->nb_krylov_it_
                      << ", #matvec: " << ctx_->params_->optf_->nb_matvecs_    << ", #evalJ: " << ctx_->params_->optf_->nb_objevals_
                      << ", #evaldJ: " << ctx_->params_->optf_->nb_gradevals_  << ", exec time: " << invtime;
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGstd (s.str());                                                                                            CHKERRQ(ierr);  s.str(""); s.clear();
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ctx_->params_->tu_->statistics_.print();
    ctx_->params_->tu_->statistics_.reset();
    // only update if triggered from outside, i.e., if new information to the ITP solver is present
    ctx_->update_reference_gradient = false;
    tao_is_reset_ = false;

    if (ctx_->x_old != nullptr) {ierr = VecDestroy (&ctx_->x_old);  CHKERRQ (ierr); ctx_->x_old = nullptr;}
    if (noise != nullptr)         {ierr = VecDestroy (&noise); CHKERRQ (ierr);                  noise = nullptr;}
    PetscFunctionReturn (ierr);
}


/* ------------------------------------------------------------------- */
/*
 solve - solves tumor inversion with an L2 regularizer.
 Input Parameters:
 .  none
 .  (implicitly takes x_rec, i.e., tumor_->p_ as initial guess)
 Output Parameters:
 .  none
 .  (implicitly writes solution in x_rec, i.e., tumor_->p_)
 Assumptions:
 .  observation operator is set
 .  initial guess is set (TumorSolverInterface::setInitialGuess(Vec p)); stored in tumor->p_
 .  data for objective and gradient is set (InvSolver::setData(Vec d))
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::solve () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    TU_assert (initialized_,              "InvSolver::solve (): InvSolver needs to be initialized.")
    TU_assert (data_ != nullptr,          "InvSolver::solve (): requires non-null input data for inversion.");
    TU_assert (xrec_ != nullptr,          "InvSolver::solve (): requires non-null p_rec vector to be set");

    std::stringstream s;
    PetscScalar max, min, w = 1, p_max, xdiff;
    PetscScalar d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0.;
    PetscScalar *d_ptr, *noise_ptr, *p_ptr, *w_ptr;
    TaoConvergedReason reason;
    Vec noise;


    if (ctx_->params_->tu_->write_output__) { dataOut (data_, ctx_->params_, "data.nc"); }

    /* === initialize inverse tumor context === */
    if (ctx_->c0_old == nullptr) {
      ierr = VecDuplicate (data_, &ctx_->c0_old);                              CHKERRQ(ierr);
      ierr = VecSet (ctx_->c0_old, 0.0);                                       CHKERRQ(ierr);
    }
    if (ctx_->tmp == nullptr) {
      ierr = VecDuplicate (data_, &ctx_->tmp);                                CHKERRQ(ierr);
      ierr = VecSet (ctx_->tmp, 0.0);                                         CHKERRQ(ierr);
    }
    if (ctx_->x_old == nullptr)  {
      ierr = VecDuplicate (ctx_->tumor_->p_, &ctx_->x_old);                 CHKERRQ (ierr);
      ierr = VecCopy (ctx_->tumor_->p_, ctx_->x_old);                       CHKERRQ (ierr);
    }
    // initialize with zero; fresh solve
    ierr = VecSet (ctx_->c0_old, 0.0);                                         CHKERRQ(ierr);
    // reset opt solver statistics
    ctx_->update_reference_gradient_hessian_ksp        = true;
    ctx_->params_->optf_->converged_    = false;
    ctx_->params_->optf_->solverstatus_ = "";
    ctx_->params_->optf_->nb_newton_it_ = 0;
    ctx_->params_->optf_->nb_krylov_it_ = 0;
    ctx_->params_->optf_->nb_matvecs_   = 0;
    ctx_->params_->optf_->nb_objevals_  = 0;
    ctx_->params_->optf_->nb_gradevals_ = 0;
    ctx_->data                       = data_;
    // reset tao, if we want virgin TAO for every inverse solve
    if (ctx_->params_->opt_->reset_tao) {
        ierr = resetTao(ctx_->params_);                                       CHKERRQ(ierr);
    }

    /* === set TAO options === */
    if (tao_is_reset_) {
        // ctx_->update_reference_gradient = true;   // TODO: K: I commented this; for CoSaMp_RS we don't want to re-compute reference gradient between inexact blocks (if coupled with sibia, the data will change)
        // ctx_->update_reference_objective = true;  // TODO: K: I commented this; for CoSaMp_RS we don't want to re-compute reference gradient between inexact blocks (if coupled with sibia, the data will change)
        ierr = setTaoOptions (tao_, ctx_.get());                              CHKERRQ(ierr);
        if ((ctx_->params_->opt_->newton_solver_ == QUASINEWTON) &&
            ctx_->params_->opt_->lmvm_set_hessian_) {
            ierr = TaoLMVMSetH0 (tao_, H_);                                     CHKERRQ(ierr);
        } else {
            ierr = TaoSetHessianRoutine (tao_, H_, H_, matfreeHessian, (void *) ctx_.get());CHKERRQ(ierr);
        }
    }

    s << " using tumor regularization = "<< ctx_->params_->opt_->beta_ << " type: " << ctx_->params_->opt_->regularization_norm_;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    if (ctx_->params_->tu_->verbosity_ >= 2) { ctx_->params_->tu_->outfile_sol_  << "\n ## ----- ##" << std::endl << std::flush; ctx_->params_->tu_->outfile_grad_ << "\n ## ----- ## "<< std::endl << std::flush; }
    //Gradient check begin
    //    ierr = ctx_->derivative_operators_->checkGradient (ctx_->tumor_->p_, ctx_->data);
    //Gradient check end

    /* === solve === */
    ctx_->params_->tu_->statistics_.reset();
    double self_exec_time_tuninv = -MPI_Wtime(); double invtime = 0;
    ierr = TaoSolve (tao_);                                                       CHKERRQ(ierr);
    self_exec_time_tuninv += MPI_Wtime();
    MPI_Reduce(&self_exec_time_tuninv, &invtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* === get solution === */
    Vec p; ierr = TaoGetSolutionVector (tao_, &p);                                CHKERRQ(ierr);
    ierr = VecCopy (p, xrec_);                                                    CHKERRQ(ierr);

    /* === get termination info === */
    TaoGetConvergedReason (tao_, &reason);

    /* === get last line-search step used ==== */
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    TaoType taotype = NULL; ierr = TaoGetType (tao_, &taotype);                 CHKERRQ(ierr);
    #else
    const TaoType taotype; ierr = TaoGetType (tao_, &taotype);                  CHKERRQ(ierr);
    #endif
    ierr = TaoGetType (tao_, &taotype); CHKERRQ(ierr);

    /* === get solution status === */
    ierr = TaoGetSolutionStatus (tao_, NULL, &ctx_->params_->optf_->jval_, &ctx_->params_->optf_->gradnorm_, NULL, &xdiff, NULL); CHKERRQ(ierr);
    /* display convergence reason: */
    ierr = dispTaoConvReason (reason, ctx_->params_->optf_->solverstatus_);        CHKERRQ(ierr);
    ctx_->params_->optf_->nb_newton_it_--;
    s << " optimization done: #N-it: " << ctx_->params_->optf_->nb_newton_it_    << ", #K-it: " << ctx_->params_->optf_->nb_krylov_it_
                      << ", #matvec: " << ctx_->params_->optf_->nb_matvecs_    << ", #evalJ: " << ctx_->params_->optf_->nb_objevals_
                      << ", #evaldJ: " << ctx_->params_->optf_->nb_gradevals_  << ", exec time: " << invtime;
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGstd (s.str());                                                                                            CHKERRQ(ierr);  s.str(""); s.clear();
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ctx_->params_->tu_->statistics_.print();
    ctx_->params_->tu_->statistics_.reset();
    // only update if triggered from outside, i.e., if new information to the ITP solver is present
    ctx_->update_reference_gradient = false;
    tao_is_reset_ = false;

    if (ctx_->x_old != nullptr) {ierr = VecDestroy (&ctx_->x_old);  CHKERRQ (ierr); ctx_->x_old = nullptr;}
    if (noise != nullptr)         {ierr = VecDestroy (&noise); CHKERRQ (ierr);                  noise = nullptr;}
    PetscFunctionReturn (ierr);
}


/* ------------------------------------------------------------------- */
/*
 solveInverseCoSaMp - solves tumor inversion with sparse localization of tumor
                      initial condition using the CoSaMp algorithm.
 Input Parameters:
 .  none
 .  (implicitly takes x_rec, i.e., tumor_->p_ as initial guess)
 Output Parameters:
 .  none
 .  (implicitly writes solution in x_rec, i.e., tumor_->p_)
 Assumptions:
 .  observation operator is set
 .  initial guess is set (TumorSolverInterface::setInitialGuess(Vec p)); stored in tumor->p_
 .  data for objective and gradient is set (InvSolver::setData(Vec d))
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::solveInverseCoSaMpRS(bool rs_mode_active = true) {

    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    std::stringstream ss;
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
        case INIT:
            ierr = tuMSG(" >> entering stage INIT"); CHKERRQ(ierr); ss.str(""); ss.clear();
            ctx_->cosamp_->np_full = ctx_->params_->tu_->np_; // store np of unrestricted ansatz space
            np_full = ctx_->cosamp_->np_full;
            ctx_->cosamp_->converged_l1 = false;
            ctx_->cosamp_->converged_l2 = false;
            ctx_->cosamp_->f_tol = 1E-5;
            ierr = ctx_->cosamp_->cleanup();                                  CHKERRQ (ierr);
            /* allocate vecs and copy initial guess for p */
            ierr = ctx_->cosamp_->initialize(ctx_->tumor_->p_);             CHKERRQ (ierr);
            // no break; go into next case
            ctx_->cosamp_->cosamp_stage = PRE_RD;
            ierr = tuMSG(" << leaving stage INIT"); CHKERRQ(ierr);

        // ================
        // this case is executed at once without going back to caller in between
        case PRE_RD:
            /* ------------------------------------------------------------------------ */
            // ### (0) (pre-)reaction/diffusion inversion ###
            ierr = tuMSG(" >> entering stage PRE_RD"); CHKERRQ(ierr);
            if (ctx_->params_->opt_->pre_reacdiff_solve_ && ctx_->params_->grid_->n_[0] > 64) {
              if (ctx_->params_->opt_->reaction_inversion_) {
                  // == restrict == to new L2 subspace, holding p_i, kappa, and rho
                  ierr = restrictSubspace(&ctx_->cosamp_->x_sub, ctx_->cosamp_->x_full, ctx_, true);      CHKERRQ (ierr); // x_sub <-- R(x_full)
                  ctx_->cosamp_->cosamp_stage = PRE_RD;
                  ctx_->params_->opt_->newton_maxit_ = ctx_->cosamp_->maxit_newton;
                  // == solve ==
                  ierr = solveInverseReacDiff (ctx_->cosamp_->x_sub);          /* with current guess as init cond. */
                  ierr = VecCopy (getPrec(), ctx_->cosamp_->x_sub);            /* get solution */             CHKERRQ (ierr);
                  // == prolongate ==
                  ierr = prolongateSubspace(ctx_->cosamp_->x_full, &ctx_->cosamp_->x_sub, ctx_, np_full); CHKERRQ (ierr); // x_full <-- P(x_sub)
              }
          } else {ierr = tuMSGstd("    ... skipping stage, reaction diffusion disabled."); CHKERRQ(ierr);}
            // no break; go into next case
            ctx_->cosamp_->cosamp_stage = COSAMP_L1_INIT;
            ierr = tuMSG(" << leaving stage PRE_RD"); CHKERRQ(ierr);

        // ================
        // setting up L1-pahse, computing reference gradeint, and print statistics
        case COSAMP_L1_INIT:
            ierr = tuMSG(" >> entering stage COSAMP_L1_INIT"); CHKERRQ(ierr);
            // set initial guess for k_inv (possibly != zero)
            ierr = VecGetArray(ctx_->cosamp_->x_full, &x_full_ptr);                                            CHKERRQ (ierr);
            if (ctx_->params_->opt_->diffusivity_inversion_) x_full_ptr[np_full] = ctx_->params_->tu_->k_;
            else { // set diff ops with this guess -- this will not change during the solve
                ierr = ctx_->tumor_->k_->setValues (ctx_->params_->tu_->k_, ctx_->params_->tu_->k_gm_wm_ratio_, ctx_->params_->tu_->k_glm_wm_ratio_, ctx_->tumor_->mat_prop_, ctx_->params_);  CHKERRQ (ierr);
            }
            ierr = VecRestoreArray(ctx_->cosamp_->x_full, &x_full_ptr);                                        CHKERRQ (ierr);
            ierr = VecCopy        (ctx_->cosamp_->x_full, ctx_->cosamp_->x_full_prev);                       CHKERRQ (ierr);

            // compute reference value for  objective
            beta_store = ctx_->params_->opt_->beta_; ctx_->params_->opt_->beta_ = 0.; // set beta to zero for gradient thresholding
            ierr = getObjectiveAndGradient (ctx_->cosamp_->x_full, &ctx_->cosamp_->J_ref, ctx_->cosamp_->g);CHKERRQ (ierr);
            ctx_->params_->opt_->beta_ = beta_store;
            ierr = VecNorm (ctx_->cosamp_->g, NORM_2, &ctx_->cosamp_->g_norm);                                CHKERRQ (ierr);
            ctx_->cosamp_->J = ctx_->cosamp_->J_ref;

            // print statistics
            ierr = printStatistics (ctx_->cosamp_->its_l1, ctx_->cosamp_->J_ref, 1, ctx_->cosamp_->g_norm, 1, ctx_->cosamp_->x_full); CHKERRQ(ierr);
            // number of connected components
            ctx_->tumor_->phi_->num_components_ = ctx_->tumor_->phi_->component_weights_.size ();
            // output warmstart (injection) support
            ss << "starting CoSaMP solver with initial support: ["; for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            ss << "component label of initial support : [";         for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

            // no break; go into next case
            ctx_->cosamp_->cosamp_stage = COSAMP_L1_THRES_GRAD;
            ierr = tuMSG(" << leaving stage COSAMP_L1_INIT"); CHKERRQ(ierr);

        // ================
        // thresholding the gradient, restrict subspace
        case COSAMP_L1_THRES_GRAD:
            ierr = tuMSG(" >> entering stage COSAMP_L1_THRES_GRAD"); CHKERRQ(ierr);
            ctx_->cosamp_->its_l1++;
            /* === hard threshold abs gradient === */
            ierr = VecCopy (ctx_->cosamp_->g, ctx_->cosamp_->work);                                            CHKERRQ (ierr);
            ierr = VecAbs  (ctx_->cosamp_->work);                                                                CHKERRQ (ierr);
            // print gradient to file
            if (ctx_->params_->tu_->verbosity_ >= 2) {
              ierr = VecGetArray(ctx_->cosamp_->work, &grad_ptr);                                                CHKERRQ(ierr);
              for (int i = 0; i < np_full-1; i++) if(procid == 0) ctx_->params_->tu_->outfile_glob_grad_ << grad_ptr[i] << ", ";
              if(procid == 0)                                     ctx_->params_->tu_->outfile_glob_grad_ << grad_ptr[np_full-1] << ";\n" <<std::endl;
              ierr = VecRestoreArray(ctx_->cosamp_->work, &grad_ptr);                                            CHKERRQ(ierr);
            }
            idx.clear();
            ierr = hardThreshold (ctx_->cosamp_->work, 2 * ctx_->params_->tu_->sparsity_level_, np_full, idx, ctx_->tumor_->phi_->gaussian_labels_, ctx_->tumor_->phi_->component_weights_, nnz, ctx_->tumor_->phi_->num_components_);

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

        // ================
        // this case may be executed in parts, i.e., going back to caller after inexact_nit Newton iterations
        case COSAMP_L1_SOLVE_SUBSPACE:
            ierr = tuMSG(" >> entering stage COSAMP_L1_SOLVE_SUBSPACE"); CHKERRQ(ierr);
            /* === corrective L2 solver === */
            ierr = tuMSGstd (""); CHKERRQ (ierr);
            ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
            ierr = tuMSG("###                                corrective L2 solver in restricted subspace                            ###");CHKERRQ (ierr);
            ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

            // == restrict ==
            ierr = restrictSubspace(&ctx_->cosamp_->x_sub, ctx_->cosamp_->x_full, ctx_); CHKERRQ (ierr); // x_L2 <-- R(x_L1)
            // print vec
            if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (ctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

            // solve interpolation
            // ierr = solveInterpolation (data_);                                   CHKERRQ (ierr);

            ctx_->params_->opt_->newton_maxit_ = ctx_->cosamp_->inexact_nits;
            // only update reference gradient and referenc objective if this is the first inexact solve for this subspace, otherwise don't
            ctx_->update_reference_gradient  = (ctx_->cosamp_->nits < ctx_->cosamp_->inexact_nits);
            ctx_->update_reference_objective = (ctx_->cosamp_->nits < ctx_->cosamp_->inexact_nits);
            // == solve ==
            ierr = solve ();                                                        CHKERRQ (ierr);
            ierr = VecCopy (getPrec(), ctx_->cosamp_->x_sub);                     CHKERRQ (ierr);
            ierr = tuMSG("### ----------------------------------------- L2 solver end --------------------------------------------- ###");CHKERRQ (ierr);
            ierr = tuMSGstd ("");                                                   CHKERRQ (ierr);
            ierr = VecCopy (ctx_->cosamp_->x_full, ctx_->cosamp_->x_full_prev); CHKERRQ (ierr);
            // print support
            ierr = VecDuplicate (ctx_->tumor_->phi_->phi_vec_[0], &all_phis);     CHKERRQ (ierr);
            ierr = VecSet (all_phis, 0.);                                           CHKERRQ (ierr);
            for (int i = 0; i < ctx_->params_->tu_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, ctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
            ss << "phiSupport_csitr-" << ctx_->cosamp_->its_l1 << ".nc";
            if (ctx_->params_->tu_->write_output__) dataOut (all_phis, ctx_->params_, ss.str().c_str()); ss.str(""); ss.clear();
            if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}
            if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (ctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

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


        // ================
        // thresholding the gradient, restrict subspace
        case COSAMP_L1_THRES_SOL:
            ierr = tuMSG(" >> entering stage COSAMP_L1_THRES_SOL"); CHKERRQ(ierr);
            /* === hard threshold solution to sparsity level === */
            idx.clear();
            if (ctx_->params_->opt_->prune_components_) hardThreshold (ctx_->cosamp_->x_full, ctx_->params_->tu_->sparsity_level_, np_full, idx, ctx_->tumor_->phi_->gaussian_labels_, ctx_->tumor_->phi_->component_weights_, nnz, ctx_->tumor_->phi_->num_components_);
            else                                    hardThreshold (ctx_->cosamp_->x_full, ctx_->params_->tu_->sparsity_level_, np_full, idx, nnz);
            ctx_->params_->tu_->support_.clear ();
            ctx_->params_->tu_->support_ = idx;
            // sort and remove duplicates
            std::sort (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end());
            ctx_->params_->tu_->support_.erase (std::unique (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end()), ctx_->params_->tu_->support_.end());
            // print out
            ss << "support after hard thresholding the solution : ["; for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            ss << "component label of support : ["; for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

            // set only support values in x_L1 (rest hard thresholded to zero)
            ierr = VecCopy (ctx_->cosamp_->x_full, ctx_->cosamp_->work);        CHKERRQ (ierr);
            ierr = VecSet  (ctx_->cosamp_->x_full, 0.0);                          CHKERRQ (ierr);
            ierr = VecGetArray (ctx_->cosamp_->work, &x_work_ptr);                CHKERRQ (ierr);
            ierr = VecGetArray (ctx_->cosamp_->x_full, &x_full_ptr);              CHKERRQ (ierr);
            for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++)
                x_full_ptr[ctx_->params_->tu_->support_[i]] = x_work_ptr[ctx_->params_->tu_->support_[i]];
            if (ctx_->params_->opt_->diffusivity_inversion_) {
                x_full_ptr[np_full] = x_work_ptr[np_full];
                if (ctx_->params_->tu_->nk_ > 1) x_full_ptr[np_full+1] = x_work_ptr[np_full+1];
                if (ctx_->params_->tu_->nk_ > 2) x_full_ptr[np_full+2] = x_work_ptr[np_full+2];
            }
            ierr = VecRestoreArray (ctx_->cosamp_->x_full, &x_full_ptr);          CHKERRQ (ierr);
            ierr = VecRestoreArray (ctx_->cosamp_->work, &x_work_ptr);              CHKERRQ (ierr);
            /* copy initial guess for p */
            ierr = VecCopy (ctx_->cosamp_->x_full, ctx_->tumor_->p_);           CHKERRQ (ierr);

            // print initial guess to file
            if (ctx_->params_->tu_->write_output__) {
              ss << "c0guess_csitr-" << ctx_->cosamp_->its_l1 << ".nc";  dataOut (ctx_->tumor_->c_0_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
              ss << "c1guess_csitr-" << ctx_->cosamp_->its_l1 << ".nc"; if (ctx_->params_->tu_->verbosity_ >= 4) dataOut (ctx_->tumor_->c_t_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
            }

            /* === convergence check === */
            ctx_->cosamp_->J_prev = ctx_->cosamp_->J;
            // compute objective (only mismatch term)
            beta_store = ctx_->params_->opt_->beta_; ctx_->params_->opt_->beta_ = 0.; // set beta to zero for gradient thresholding
            ierr = getObjectiveAndGradient (ctx_->cosamp_->x_full, &ctx_->cosamp_->J, ctx_->cosamp_->g);   CHKERRQ (ierr);
            ctx_->params_->opt_->beta_ = beta_store;
            ierr = VecNorm (ctx_->cosamp_->x_full, NORM_INFINITY, &norm);                                      CHKERRQ (ierr);
            ierr = VecAXPY (ctx_->cosamp_->work, -1.0, ctx_->cosamp_->x_full_prev);  /* holds x_L1 */        CHKERRQ (ierr);
            ierr = VecNorm (ctx_->cosamp_->work, NORM_INFINITY, &norm_rel);            /*norm change in sol */ CHKERRQ (ierr);
            ierr = VecNorm (ctx_->cosamp_->g, NORM_2, &ctx_->cosamp_->g_norm);                               CHKERRQ (ierr);
            // solver status
            ierr = tuMSGstd (""); CHKERRQ(ierr); ierr = tuMSGstd (""); CHKERRQ(ierr);
            ierr = tuMSGstd ("--------------------------------------------- L1 solver statistics -------------------------------------------"); CHKERRQ(ierr);
            ierr = printStatistics (ctx_->cosamp_->its_l1, ctx_->cosamp_->J, PetscAbsReal (ctx_->cosamp_->J_prev - ctx_->cosamp_->J) / PetscAbsReal (1 + ctx_->cosamp_->J_ref), ctx_->cosamp_->g_norm, norm_rel / (1 + norm), ctx_->cosamp_->x_full); CHKERRQ(ierr);
            ierr = tuMSGstd ("--------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
            ierr = tuMSGstd (""); CHKERRQ(ierr);
            if (ctx_->cosamp_->its_l1 >= params_->opt_->gist_maxit_) {ierr = tuMSGwarn (" L1 maxiter reached"); CHKERRQ(ierr); ctx_->cosamp_->converged_l1 = true;}
            else if (PetscAbsReal (ctx_->cosamp_->J) < 1E-5)       {ierr = tuMSGwarn (" L1 absolute objective tolerance reached."); CHKERRQ(ierr); ctx_->cosamp_->converged_l1 = true;}
            else if (PetscAbsReal (ctx_->cosamp_->J_prev - ctx_->cosamp_->J) < ctx_->cosamp_->f_tol * PetscAbsReal (1 + ctx_->cosamp_->J_ref)) {ierr = tuMSGwarn (" L1 relative objective tolerance reached."); CHKERRQ(ierr); ctx_->cosamp_->converged_l1 = true;}
            else { ctx_->cosamp_->converged_l1 = false; }  // continue iterating

            ierr = tuMSG(" << leaving stage COSAMP_L1_THRES_SOL"); CHKERRQ(ierr);
            if(ctx_->cosamp_->converged_l1) {              // no break; go into next case
                ctx_->cosamp_->cosamp_stage = FINAL_L2;
            } else{                                          // break; continue iterating
                ctx_->cosamp_->cosamp_stage = COSAMP_L1_THRES_GRAD; break;
            }

        // ================
        // this case may be executed in parts, i.e., going back to caller after inexact_nit Newton iterations
        case FINAL_L2:
            ierr = tuMSG(" >> entering stage FINAL_L2"); CHKERRQ(ierr);
            /* === (3) if converged: corrective L2 solver === */
            ierr = tuMSGstd ("");                                                       CHKERRQ (ierr);
            ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
            ierr = tuMSG("###                                              final L2 solve                                           ###");CHKERRQ (ierr);
            ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

            ierr = restrictSubspace(&ctx_->cosamp_->x_sub, ctx_->cosamp_->x_full, ctx_); CHKERRQ (ierr); // x_sub <-- R(x_full)

            // print vec
            if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (ctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

            // solve interpolation
            // ierr = solveInterpolation (data_);                                        CHKERRQ (ierr);
            ctx_->params_->opt_->newton_maxit_ = ctx_->cosamp_->inexact_nits;
            // only update reference gradient and referenc objective if this is the first inexact solve for this subspace, otherwise don't
            ctx_->update_reference_gradient  = (ctx_->cosamp_->nits < ctx_->cosamp_->inexact_nits);
            ctx_->update_reference_objective = (ctx_->cosamp_->nits < ctx_->cosamp_->inexact_nits);
            // == solve ==
            ierr = solve ();                                    /* L2 solver    */
            ierr = VecCopy (getPrec(), ctx_->cosamp_->x_sub); /* get solution */       CHKERRQ (ierr);
            ierr = tuMSG("### -------------------------------------------- L2 solver end ------------------------------------------ ###");CHKERRQ (ierr);
            ierr = tuMSGstd (""); CHKERRQ (ierr);
            // print phi's to file
            if (ctx_->params_->tu_->write_output__) {
                ierr = VecDuplicate (ctx_->tumor_->phi_->phi_vec_[0], &all_phis);     CHKERRQ (ierr);
                ierr = VecSet       (all_phis, 0.);                                     CHKERRQ (ierr);
                for (int i = 0; i < ctx_->params_->tu_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, ctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
                ss << "phiSupportFinal.nc";  {dataOut (all_phis, ctx_->params_, ss.str().c_str());} ss.str(std::string()); ss.clear();
                ss << "c0FinalGuess.nc";      dataOut (ctx_->tumor_->c_0_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
                ss << "c1FinalGuess.nc"; if (ctx_->params_->tu_->verbosity_ >= 4) { dataOut (ctx_->tumor_->c_t_, ctx_->params_, ss.str().c_str()); } ss.str(std::string()); ss.clear();
                if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}
            }
            // write out p vector after IC, k inversion (unscaled)
            if (ctx_->params_->tu_->write_p_checkpoint_) { writeCheckpoint(ctx_->cosamp_->x_sub, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_ .str(), std::string("unscaled"));}
            if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (ctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

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
            ierr = prolongateSubspace(ctx_->cosamp_->x_full, &ctx_->cosamp_->x_sub, ctx_, np_full, (finalize || contiterating));  CHKERRQ (ierr); // x_full <-- P(x_sub)

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

        // ================
        // this case is executed at once without going back to caller in between
        case POST_RD:
            ierr = tuMSG(" >> entering stage POST_RD"); CHKERRQ(ierr);
            // === (4) reaction/diffusion inversion ===
            if (ctx_->params_->opt_->reaction_inversion_) {
                // restrict to new L2 subspace, holding p_i, kappa, and rho
                ierr = restrictSubspace(&ctx_->cosamp_->x_sub, ctx_->cosamp_->x_full, ctx_, true);     CHKERRQ (ierr); // x_sub <-- R(x_full)

                ctx_->cosamp_->cosamp_stage = POST_RD;
                ctx_->params_->opt_->newton_maxit_ = ctx_->cosamp_->maxit_newton;
                // == solve ==
                ierr = solveInverseReacDiff (ctx_->cosamp_->x_sub); /* with current guess as init cond. */  CHKERRQ (ierr);
                ierr = VecCopy (getPrec(), ctx_->cosamp_->x_sub);   /* get solution */                      CHKERRQ (ierr);
                // update full space solution
                ierr = prolongateSubspace(ctx_->cosamp_->x_full, &ctx_->cosamp_->x_sub, ctx_, np_full); CHKERRQ (ierr); // x_full <-- P(x_sub)
            } else {ierr = tuMSGstd("    ... skipping stage, reaction diffusion disabled."); CHKERRQ(ierr);}

            // break; go to finalize
            ctx_->cosamp_->cosamp_stage = FINALIZE;
            ierr = tuMSG(" << leaving stage POST_RD"); CHKERRQ(ierr);
            break;
    }


    // prolongate (in case we are in a subuspace solve, we still need the solution to be prolongated)
    // if (ctx_->cosamp_->cosamp_stage != FINALIZE) {
        // ierr = tuMSG(" >> entering stage FINALIZE"); CHKERRQ(ierr); ss.str(""); ss.clear();
        // ierr = prolongateSubspace(ctx_->cosamp_->x_full, &ctx_->cosamp_->x_sub, ctx_, np_full); CHKERRQ (ierr); // x_full <-- P(x_sub)
    // }
    // pass the reconstructed p vector to the caller (deep copy)
    ierr = VecCopy (ctx_->cosamp_->x_full, xrec_);                                                  CHKERRQ (ierr);
    if (!rs_mode_active && ctx_->cosamp_->cosamp_stage != FINALIZE)
        {solveInverseCoSaMpRS(false);
    } else {ierr = tuMSG(" << leaving inverse CoSaMp"); CHKERRQ(ierr);}

    // go home
    PetscFunctionReturn (ierr);
}



/* ------------------------------------------------------------------- */
/*
 solveInverseCoSaMp - solves tumor inversion with sparse localization of tumor
                      initial condition using the CoSaMp algorithm.
 Input Parameters:
 .  none
 .  (implicitly takes x_rec, i.e., tumor_->p_ as initial guess)
 Output Parameters:
 .  none
 .  (implicitly writes solution in x_rec, i.e., tumor_->p_)
 Assumptions:
 .  observation operator is set
 .  initial guess is set (TumorSolverInterface::setInitialGuess(Vec p)); stored in tumor->p_
 .  data for objective and gradient is set (InvSolver::setData(Vec d))
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::solveInverseCoSaMp() {

    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    std::stringstream ss;
    Vec g, x_L2, x_L1, x_L1_old, temp, all_phis;
    PetscReal *x_L2_ptr, *x_L1_ptr, *temp_ptr, *grad_ptr;
    PetscReal J, J_ref, J_old;   // objective
    PetscReal ftol = 1E-5;
    PetscReal norm_rel, norm, norm_g, beta_store;
    std::vector<int> idx;        // idx list of support after thresholding
    int np_full, its = 0, nnz = 0;
    int flag_convergence = 0;

    np_full = ctx_->params_->tu_->np_; // store np of unrestricted ansatz space
    ierr = VecDuplicate (ctx_->tumor_->p_, &g);                                 CHKERRQ (ierr);
    ierr = VecDuplicate (ctx_->tumor_->p_, &x_L1);                              CHKERRQ (ierr);
    ierr = VecDuplicate (ctx_->tumor_->p_, &x_L1_old);                          CHKERRQ (ierr);
    ierr = VecDuplicate (ctx_->tumor_->p_, &temp);                              CHKERRQ (ierr);
    ierr = VecSet  (g, 0);                                                        CHKERRQ (ierr);
    ierr = VecCopy (ctx_->tumor_->p_, x_L1); /* copy initial guess for p */     CHKERRQ (ierr);
    ierr = VecSet  (x_L1_old, 0);                                                 CHKERRQ (ierr);
    ierr = VecSet  (temp, 0);                                                     CHKERRQ (ierr);


    /* ------------------------------------------------------------------------ */
    // ### (0) (pre-)reaction/diffusion inversion ###
    if (ctx_->params_->opt_->pre_reacdiff_solve_ && ctx_->params_->grid_->n_[0] > 64) {
    if (ctx_->params_->opt_->reaction_inversion_) {
        // restrict to new L2 subspace, holding p_i, kappa, and rho
        ierr = restrictSubspace(&x_L2, x_L1, ctx_, true);                     CHKERRQ (ierr); // x_L2 <-- R(x_L1)
        // solve
        ctx_->cosamp_->cosamp_stage = PRE_RD;
        ierr = solveInverseReacDiff (x_L2);          /* with current guess as init cond. */
        ierr = VecCopy (getPrec(), x_L2);            /* get solution */         CHKERRQ (ierr);
        // update full space solution
        ierr = prolongateSubspace(x_L1, &x_L2, ctx_, np_full);                CHKERRQ (ierr); // x_L1 <-- P(x_L2)
    }
    }

    /* ------------------------------------------------------------------------ */
    // === (1) L1 CoSaMp solver ===
    // set initial guess for k_inv (possibly != zero)
    ierr = VecGetArray(x_L1, &x_L1_ptr);                                          CHKERRQ (ierr);
    if (ctx_->params_->opt_->diffusivity_inversion_) x_L1_ptr[np_full] = ctx_->params_->tu_->k_;
    else { // set diff ops with this guess -- this will not change during the solve
      ierr = ctx_->tumor_->k_->setValues (ctx_->params_->tu_->k_, ctx_->params_->tu_->k_gm_wm_ratio_, ctx_->params_->tu_->k_glm_wm_ratio_, ctx_->tumor_->mat_prop_, ctx_->params_);  CHKERRQ (ierr);
    }
    ierr = VecRestoreArray(x_L1, &x_L1_ptr);                                      CHKERRQ (ierr);
    ierr = VecCopy        (x_L1, x_L1_old);                                       CHKERRQ (ierr);

    // compute reference value for  objective
    beta_store = ctx_->params_->opt_->beta_; ctx_->params_->opt_->beta_ = 0.; // set beta to zero for gradient thresholding
    ierr = getObjectiveAndGradient (x_L1, &J_ref, g);                             CHKERRQ (ierr);
    ctx_->params_->opt_->beta_ = beta_store;
    ierr = VecNorm (g, NORM_2, &norm_g);                                          CHKERRQ (ierr);
    J = J_ref;

    // print statistics
    printStatistics (its, J_ref, 1, norm_g, 1, x_L1);

    // number of connected components
    ctx_->tumor_->phi_->num_components_ = ctx_->tumor_->phi_->component_weights_.size ();
    // output warmstart (injection) support
    ss << "starting CoSaMP solver with initial support: ["; for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << "component label of initial support : [";         for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    // === L1 solver ===
    while (true) {
        its++;

        /* === hard threshold abs gradient === */
        ierr = VecCopy (g, temp);                                                 CHKERRQ (ierr);
        ierr = VecAbs (temp);                                                     CHKERRQ (ierr);
        // print gradient to file
        if (ctx_->params_->tu_->verbosity_ >= 2) {
            ierr = VecGetArray(temp, &grad_ptr);                                    CHKERRQ(ierr);
            for (int i = 0; i < np_full-1; i++) if(procid == 0) ctx_->params_->tu_->outfile_glob_grad_ << grad_ptr[i] << ", ";
            if(procid == 0)                                     ctx_->params_->tu_->outfile_glob_grad_ << grad_ptr[np_full-1] << ";\n" <<std::endl;
            ierr = VecRestoreArray(temp, &grad_ptr);                                CHKERRQ(ierr);
        }
        // threshold gradient
        idx.clear();
        ierr = hardThreshold (temp, 2 * ctx_->params_->tu_->sparsity_level_, np_full, idx, ctx_->tumor_->phi_->gaussian_labels_, ctx_->tumor_->phi_->component_weights_, nnz, ctx_->tumor_->phi_->num_components_);

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

        ierr = restrictSubspace(&x_L2, x_L1, ctx_);                             CHKERRQ (ierr); // x_L2 <-- R(x_L1)

        // print vec
        if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}


        // solve interpolation
        // ierr = solveInterpolation (data_);                                        CHKERRQ (ierr);

        // update reference gradient and referenc objective (commented in the solve function)
        ctx_->update_reference_gradient  = true;
        ctx_->update_reference_objective = true;
        ierr = solve ();                                                          CHKERRQ (ierr);
        ierr = VecCopy (getPrec(), x_L2);                                         CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------- L2 solver end --------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSGstd ("");                                                     CHKERRQ (ierr);
        ierr = VecCopy (x_L1, x_L1_old);                                          CHKERRQ (ierr);

        // print support
        ierr = VecDuplicate (ctx_->tumor_->phi_->phi_vec_[0], &all_phis);       CHKERRQ (ierr);
        ierr = VecSet (all_phis, 0.);                                             CHKERRQ (ierr);
        for (int i = 0; i < ctx_->params_->tu_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, ctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
        ss << "phiSupport_csitr-" << its << ".nc";
        if (ctx_->params_->tu_->write_output__) dataOut (all_phis, ctx_->params_, ss.str().c_str()); ss.str(""); ss.clear();
        if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}

        // print vec
        if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}


        ierr = prolongateSubspace(x_L1, &x_L2, ctx_, np_full);                  CHKERRQ (ierr); // x_L1 <-- P(x_L2)

        /* === hard threshold solution to sparsity level === */
        idx.clear();
        if (ctx_->params_->opt_->prune_components_) hardThreshold (x_L1, ctx_->params_->tu_->sparsity_level_, np_full, idx, ctx_->tumor_->phi_->gaussian_labels_, ctx_->tumor_->phi_->component_weights_, nnz, ctx_->tumor_->phi_->num_components_);
        else                                    hardThreshold (x_L1, ctx_->params_->tu_->sparsity_level_, np_full, idx, nnz);
        ctx_->params_->tu_->support_.clear ();
        ctx_->params_->tu_->support_ = idx;
        // sort and remove duplicates
        std::sort (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end());
        ctx_->params_->tu_->support_.erase (std::unique (ctx_->params_->tu_->support_.begin(), ctx_->params_->tu_->support_.end()), ctx_->params_->tu_->support_.end());
        // print out
        ss << "support after hard thresholding the solution : [";
        for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->params_->tu_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ss << "component label of support : [";
        for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++) ss << ctx_->tumor_->phi_->gaussian_labels_[ctx_->params_->tu_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

        // set only support values in x_L1 (rest hard thresholded to zero)
        ierr = VecCopy (x_L1, temp);                                              CHKERRQ (ierr);
        ierr = VecSet  (x_L1, 0.0);                                               CHKERRQ (ierr);
        ierr = VecGetArray (temp, &temp_ptr);                                     CHKERRQ (ierr);
        ierr = VecGetArray (x_L1, &x_L1_ptr);                                     CHKERRQ (ierr);
        for (int i = 0; i < ctx_->params_->tu_->support_.size(); i++)
          x_L1_ptr[ctx_->params_->tu_->support_[i]] = temp_ptr[ctx_->params_->tu_->support_[i]];
        if (ctx_->params_->opt_->diffusivity_inversion_) {
          x_L1_ptr[np_full] = temp_ptr[np_full];
          if (ctx_->params_->tu_->nk_ > 1) x_L1_ptr[np_full+1] = temp_ptr[np_full+1];
          if (ctx_->params_->tu_->nk_ > 2) x_L1_ptr[np_full+2] = temp_ptr[np_full+2];
        }
        ierr = VecRestoreArray (x_L1, &x_L1_ptr);                                 CHKERRQ (ierr);
        ierr = VecRestoreArray (temp, &temp_ptr);                                 CHKERRQ (ierr);
        ierr = VecCopy (x_L1, ctx_->tumor_->p_); /* copy initial guess for p */ CHKERRQ (ierr);

        // print initial guess to file
        if (ctx_->params_->tu_->write_output__) {
        ss << "c0guess_csitr-" << its << ".nc";  dataOut (ctx_->tumor_->c_0_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
        ss << "c1guess_csitr-" << its << ".nc"; if (ctx_->params_->tu_->verbosity_ >= 4) dataOut (ctx_->tumor_->c_t_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
        }

        /* === convergence check === */
        J_old = J;

        // compute objective (only mismatch term)
        beta_store = ctx_->params_->opt_->beta_; ctx_->params_->opt_->beta_ = 0.; // set beta to zero for gradient thresholding
        ierr = getObjectiveAndGradient (x_L1, &J, g);                             CHKERRQ (ierr);
        ctx_->params_->opt_->beta_ = beta_store;
        ierr = VecNorm (x_L1, NORM_INFINITY, &norm);                              CHKERRQ (ierr);
        ierr = VecAXPY (temp, -1.0, x_L1_old);            /* holds x_L1 */        CHKERRQ (ierr);
        ierr = VecNorm (temp, NORM_INFINITY, &norm_rel);  /*norm change in sol */ CHKERRQ (ierr);
        ierr = VecNorm (g, NORM_2, &norm_g);                                      CHKERRQ (ierr);
        // solver status
        ierr = tuMSGstd (""); CHKERRQ(ierr); ierr = tuMSGstd (""); CHKERRQ(ierr);
        ierr = tuMSGstd ("--------------------------------------------- L1 solver statistics -------------------------------------------"); CHKERRQ(ierr);
        printStatistics (its, J, PetscAbsReal (J_old - J) / PetscAbsReal (1 + J_ref), norm_g, norm_rel / (1 + norm), x_L1);
        ierr = tuMSGstd ("--------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGstd (""); CHKERRQ(ierr);
        if (its >= params_->opt_->gist_maxit_) {ierr = tuMSGwarn (" L1 maxiter reached."); CHKERRQ(ierr); flag_convergence = 1; break;}
        else if (PetscAbsReal (J) < 1E-5)  {ierr = tuMSGwarn (" L1 absolute objective tolerance reached."); CHKERRQ(ierr); flag_convergence = 1; break;}
        else if (PetscAbsReal (J_old - J) < ftol * PetscAbsReal (1 + J_ref)) {ierr = tuMSGwarn (" L1 relative objective tolerance reached."); CHKERRQ(ierr); flag_convergence = 1; break;}
        else { flag_convergence = 0; }  // continue iterating
    } // end while

    /* === (3) if converged: corrective L2 solver === */
    ierr = tuMSGstd ("");                                                       CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    ierr = tuMSG("###                                              final L2 solve                                           ###");CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

    ierr = restrictSubspace(&x_L2, x_L1, ctx_);                               CHKERRQ (ierr); // x_L2 <-- R(x_L1)
    if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

    // solve interpolation
    // ierr = solveInterpolation (data_);                                        CHKERRQ (ierr);
    // update reference gradient and referenc objective (commented in the solve function)
    ctx_->update_reference_gradient  = true;
    ctx_->update_reference_objective = true;
    ierr = solve ();                                   /* L2 solver    */
    ierr = VecCopy (getPrec(), x_L2);                  /* get solution */       CHKERRQ (ierr);

    // print vec
    if (procid == 0 && ctx_->params_->tu_->verbosity_ >= 4) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}


    ierr = tuMSG("### -------------------------------------------- L2 solver end ------------------------------------------ ###");CHKERRQ (ierr);
    ierr = tuMSGstd (""); CHKERRQ (ierr);

    // print phi's to file
    if (ctx_->params_->write_output_) {
        ierr = VecDuplicate (ctx_->tumor_->phi_->phi_vec_[0], &all_phis);     CHKERRQ (ierr);
        ierr = VecSet       (all_phis, 0.);                                     CHKERRQ (ierr);
        for (int i = 0; i < ctx_->params_->tu_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, ctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
        ss << "phiSupportFinal.nc";  {dataOut (all_phis, ctx_->params_, ss.str().c_str());} ss.str(std::string()); ss.clear();
        ss << "c0FinalGuess.nc";  dataOut (ctx_->tumor_->c_0_, ctx_->params_, ss.str().c_str()); ss.str(std::string()); ss.clear();
        ss << "c1FinalGuess.nc"; if (ctx_->params_->tu_->verbosity_ >= 4) { dataOut (ctx_->tumor_->c_t_, ctx_->params_, ss.str().c_str()); } ss.str(std::string()); ss.clear();
        if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}
    }

    // write out p vector after IC, k inversion (unscaled)
    if (ctx_->params_->tu_->write_p_checkpoint_) {
      writeCheckpoint(x_L2, ctx_->tumor_->phi_, ctx_->params_->tu_->writepath_ .str(), std::string("unscaled"));
    }
    // prolongate restricted x_L2 to full x_L1, but do not resize vectors, i.e., call resetOperators
    // if inversion for reaction disabled, also reset operators
    ierr = prolongateSubspace(x_L1, &x_L2, ctx_, np_full, !ctx_->params_->opt_->reaction_inversion_);  CHKERRQ (ierr); // x_L1 <-- P(x_L2)

    // === (4) reaction/diffusion inversion ===
    if (ctx_->params_->opt_->reaction_inversion_) {
        // restrict to new L2 subspace, holding p_i, kappa, and rho
        ierr = restrictSubspace(&x_L2, x_L1, ctx_, true);                     CHKERRQ (ierr); // x_L2 <-- R(x_L1)
        // solve
        ctx_->cosamp_->cosamp_stage = POST_RD;
        ierr = solveInverseReacDiff (x_L2);          /* with current guess as init cond. */
        ierr = VecCopy (getPrec(), x_L2);            /* get solution */         CHKERRQ (ierr);
        // update full space solution
        ierr = prolongateSubspace(x_L1, &x_L2, ctx_, np_full);                CHKERRQ (ierr); // x_L1 <-- P(x_L2)
    }

    // pass the reconstructed p vector to the caller (deep copy)
    ierr = VecCopy (x_L1, xrec_);                                               CHKERRQ (ierr);

    // clean-up
    if (g != nullptr)        { ierr = VecDestroy (&g);        CHKERRQ (ierr); g        = nullptr; }
    if (x_L1 != nullptr)     { ierr = VecDestroy (&x_L1);     CHKERRQ (ierr); x_L1     = nullptr;}
    if (x_L1_old != nullptr) { ierr = VecDestroy (&x_L1_old); CHKERRQ (ierr); x_L1_old = nullptr;}
    if (temp != nullptr)     { ierr = VecDestroy (&temp);     CHKERRQ (ierr); temp     = nullptr;}
    PetscFunctionReturn (ierr);
}


/* ------------------------------------------------------------------- */
/*
 printStatistics - prints solver status of L1 cosamp solver.
 Input Parameters:
 .  its        - current L1 iteration
 .  J          - objective function value (only mismatch term)
 .  J_rel      - rel. objective function value
 .  g_norm     - norm of gradient
 .  p_rel_norm - relative norm of p
 .  x_L1       - current sparse solution
 Output Parameters:
 .  none
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::printStatistics (int its, PetscReal J, PetscReal J_rel, PetscReal g_norm, PetscReal p_rel_norm, Vec x_L1) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    PetscReal *x_ptr;
    std::stringstream s;

    s << std::setw(4)  << " iter"               << "   " << std::setw(18) << "objective (abs)" << "   "
      << std::setw(18) << "||objective||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
      << "   "  << std::setw(18) << "||dp||_rel"
      << std::setw(18) << "k";
    ierr = tuMSGstd ("--------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
    ierr = tuMSGstd ("--------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    s.str ("");
    s.clear ();
    s << " "   << std::scientific << std::setprecision(5)  << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J_rel
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << g_norm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << p_rel_norm;

    ierr = VecGetArray(x_L1, &x_ptr);                                           CHKERRQ(ierr);
    if (ctx_->params_->opt_->diffusivity_inversion_) {
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[ctx_->params_->tu_->np_];
        if (ctx_->params_->tu_->nk_ > 1) { s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[ctx_->params_->tu_->np_ + 1]; }
    } else {
      s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << ctx_->params_->tu_->k_;
    }
    ierr = VecRestoreArray(x_L1, &x_ptr);                                       CHKERRQ(ierr);
    ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
    s.str (""); s.clear ();
    PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
InvSolver::~InvSolver () {
    PetscErrorCode ierr = 0;
    if (tao_     != nullptr) {TaoDestroy (&tao_);      tao_  = nullptr;}
    if (H_       != nullptr) {MatDestroy (&H_);        H_    = nullptr;}
    if (xrec_    != nullptr) {VecDestroy (&xrec_);     xrec_ = nullptr;}
    if (xrec_rd_ != nullptr) {VecDestroy (&xrec_rd_);  xrec_rd_ = nullptr;}
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::setTaoOptionsMassEffect (Tao tao, CtxInv *ctx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    TaoLineSearch linesearch;        // line-search object
    std::string msg;

    PetscReal minstep;
//    minstep = std::pow (2.0, 20.0);
//    minstep = 1.0 / minstep;
    minstep = PETSC_MACHINE_EPSILON;
    ctx_->params_->opt_->ls_minstep_ = minstep;

    if (ctx_->params_->opt_->newton_solver_ == QUASINEWTON)  {
        ierr = TaoSetType (tao_, "blmvm");                                          CHKERRQ (ierr);
    } else {
        ierr = TaoSetType (tao, "bnls");                                            CHKERRQ(ierr);  // set TAO solver type
    }

    PetscBool flag = PETSC_FALSE;

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
      PetscOptionsHasName (NULL, NULL, "-tao_nls_pc_type", &flag);
      if (flag == PETSC_FALSE)
          PetscOptionsSetValue (NULL, "-tao_nls_pc_type", "none");
    #else
      PetscOptionsHasName (NULL, "-tao_nls_pc_type", &flag);
      if (flag == PETSC_FALSE)
          PetscOptionsSetValue ("-tao_nls_pc_type", "none");
    #endif
    flag = PETSC_FALSE;
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
      PetscOptionsHasName (NULL, NULL, "-tao_nls_ksp_type", &flag);
      if (flag == PETSC_FALSE)
          PetscOptionsSetValue (NULL, "-tao_nls_ksp_type", "cg");
    #else
      PetscOptionsHasName (NULL, "-tao_nls_ksp_type", &flag);
      if (flag == PETSC_FALSE)
         PetscOptionsSetValue ("-tao_nls_ksp_type", "cg");
    #endif
    flag = PETSC_FALSE;
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
      PetscOptionsHasName (NULL, NULL, "-tao_ntr_pc_type", &flag);
      if (flag == PETSC_FALSE)
          PetscOptionsSetValue (NULL, "-tao_ntr_pc_type", "none");
    #else
      PetscOptionsHasName (NULL, "-tao_ntr_pc_type", &flag);
      if (flag == PETSC_FALSE)
          PetscOptionsSetValue ("-tao_ntr_pc_type", "none");
    #endif

    // parse options user has set
    ierr = TaoSetFromOptions (tao);                                                 CHKERRQ(ierr);
    // set the initial vector
    ierr = TaoSetInitialVector (tao, xrec_);                                        CHKERRQ(ierr);
    // set routine for evaluating the objective
    ierr = TaoSetObjectiveRoutine (tao, evaluateObjectiveFunction, (void*) ctx);    CHKERRQ(ierr);
    // set routine for evaluating the Gradient
    ierr = TaoSetGradientRoutine (tao, evaluateGradient, (void*) ctx);              CHKERRQ(ierr);
    // set the routine to evaluate the objective and compute the gradient
    ierr = TaoSetObjectiveAndGradientRoutine (tao, evaluateObjectiveFunctionAndGradient, (void*) ctx);  CHKERRQ(ierr);
    // set monitor function
    ierr = TaoSetMonitor (tao, optimizationMonitorMassEffect, (void *) ctx, NULL);            CHKERRQ(ierr);

    // Lower and Upper Bounds
    Vec lower_bound;
    ierr = VecDuplicate (xrec_, &lower_bound);                            CHKERRQ (ierr);
    ierr = VecSet (lower_bound, 0.);                                                CHKERRQ (ierr);
    Vec upper_bound;
    ierr = VecDuplicate (xrec_, &upper_bound);                            CHKERRQ (ierr);
    ierr = VecSet (upper_bound, PETSC_INFINITY);                                             CHKERRQ (ierr);

    ScalarType *ub_ptr;
    ierr = VecGetArray (upper_bound, &ub_ptr);                            CHKERRQ (ierr);
    ub_ptr[0] = ctx_->params_->opt_->gamma_ub_;
    ub_ptr[1] = ctx_->params_->opt_->rho_ub_;
    ub_ptr[2] = ctx_->params_->opt_->k_ub_;
    ctx_->params_->opt_->bounds_array_[0] = ub_ptr[0];
    ctx_->params_->opt_->bounds_array_[1] = ub_ptr[1];
    ctx_->params_->opt_->bounds_array_[2] = ub_ptr[2];
    ierr = VecRestoreArray (upper_bound, &ub_ptr);                        CHKERRQ (ierr);


    ScalarType *lb_ptr;
    ierr = VecGetArray (lower_bound, &lb_ptr);                            CHKERRQ (ierr);
    lb_ptr[2] = ctx_->params_->opt_->k_lb_;
    ierr = VecRestoreArray (lower_bound, &lb_ptr);                        CHKERRQ (ierr);

    ierr = TaoSetVariableBounds(tao, lower_bound, upper_bound);                     CHKERRQ (ierr);
    if (lower_bound != nullptr) {ierr = VecDestroy(&lower_bound); CHKERRQ(ierr); lower_bound = nullptr;}
    if (upper_bound != nullptr) {ierr = VecDestroy(&upper_bound); CHKERRQ(ierr); upper_bound = nullptr;}

    // TAO type from user input
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
      TaoType taotype = NULL;
      ierr = TaoGetType (tao, &taotype);                                            CHKERRQ(ierr);
    #else
      const TaoType taotype;
      ierr = TaoGetType (tao, &taotype);                                            CHKERRQ(ierr);
    #endif
    if (strcmp(taotype, "nls") == 0) {
        msg = " limited memory variable metric method (unconstrained) selected";
    } else if (strcmp(taotype, "ntr") == 0) {
        msg = " Newton's method with trust region for unconstrained minimization";
    } else if (strcmp(taotype, "ntl") == 0) {
        msg = " Newton's method with trust region, line search for unconstrained minimization";
    } else if (strcmp(taotype, "nls") == 0) {
        msg = " Newton's method (line search; unconstrained) selected";
    } else if (strcmp(taotype, "bnls") == 0) {
        msg = " Newton's method (line search; bound constraints) selected";
    } else if (strcmp(taotype, "bqnls") == 0) {
        msg = " Quasi-Newton's method (line search; bound constraints) selected";
    } else if (strcmp(taotype, "ntr") == 0) {
        msg = " Newton's method (trust region; unconstrained) selected";
    } else if (strcmp(taotype, "fd_test") == 0) {
        msg = " gradient test selected";
    } else if (strcmp(taotype, "cg") == 0) {
        msg = " CG selected\n";
    } else if (strcmp(taotype, "tron") == 0) {
        msg = " Newton Trust Region method chosen";
    } else if (strcmp(taotype, "blmvm") == 0) {
        msg = " bounded limited memory variable metric method chosen";
    } else if (strcmp(taotype, "lmvm") == 0) {
        msg = " limited memory variable metric method chosen\n";
    } else if (strcmp(taotype, "tao_blmvm_m") == 0) {
        msg = " user modified limited memory variable metric method chosen";
    } else if (strcmp(taotype, "gpcg") == 0) {
        msg = " Newton Trust Region method for quadratic bound constrained minimization";
    } else if (strcmp(taotype, "tao_L1") == 0) {
        msg = " User defined solver for L1 minimization";
    } else {
        msg = " numerical optimization method not supported (setting default: LMVM)";
        ierr = TaoSetType (tao, "lmvm");                                          CHKERRQ(ierr);
    }
    ierr = tuMSGstd(msg); CHKERRQ(ierr);
    // set tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoSetTolerances (tao, ctx->params_->opt_->gatol_, ctx->params_->opt_->grtol_, ctx->params_->opt_->opttolgrad_); CHKERRQ(ierr);
    #else
        ierr = TaoSetTolerances (tao, 1E-12, 1E-12, ctx->params_->opt_->gatol_, ctx->params_->opt_->grtol_, ctx->params_->opt_->opttolgrad_); CHKERRQ(ierr);
    #endif

    ierr = TaoSetMaximumIterations (tao, ctx->params_->opt_->newton_maxit_); CHKERRQ(ierr);
    ierr = TaoSetConvergenceTest (tao, checkConvergenceGradMassEffect, ctx);                 CHKERRQ(ierr);

    // set linesearch (only for Gauss-Newton, lmvm uses more-thuente type line-search automatically)
    ierr = TaoGetLineSearch (tao, &linesearch);                                   CHKERRQ(ierr);
    linesearch->stepmin = minstep;

    if (ctx->params_->opt_->linesearch_ == ARMIJO) {
        ierr = TaoLineSearchSetType (linesearch, "armijo");                          CHKERRQ(ierr);
        tuMSGstd(" using line-search type: armijo");
    } else {
        tuMSGstd(" using line-search type: more-thuene");
    }

    ierr = TaoLineSearchSetInitialStepLength (linesearch, 1.0);                    CHKERRQ(ierr);
    ierr = TaoLineSearchSetOptionsPrefix (linesearch,"tumor_");                    CHKERRQ(ierr);

    std::stringstream s;
    tuMSGstd(" parameters (optimizer):");
    tuMSGstd(" tolerances (stopping conditions):");
    s << "   gatol: "<< ctx->params_->opt_->gatol_;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
    s << "   grtol: "<< ctx->params_->opt_->grtol_;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
    s << "   gttol: "<< ctx->params_->opt_->opttolgrad_;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();

    ierr = TaoSetFromOptions(tao);                                                CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode InvSolver::setTaoOptions (Tao tao, CtxInv *ctx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    TaoLineSearch linesearch;        // line-search object
    std::string msg;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PetscReal minstep;
    minstep = std::pow (2.0, 30.0);
    minstep = 1.0 / minstep;
    ctx_->params_->opt_->ls_minstep_ = minstep;

    if (ctx_->params_->opt_->newton_solver_ == QUASINEWTON)  {
      ierr = TaoSetType   (tao_, "blmvm");                                          CHKERRQ (ierr);
    } else {
      ierr = TaoSetType (tao, "bnls");    CHKERRQ(ierr);  // set TAO solver type
    }

    PetscBool flag = PETSC_FALSE;

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        PetscOptionsHasName (NULL, NULL, "-tao_nls_pc_type", &flag);
        if (flag == PETSC_FALSE)
            PetscOptionsSetValue (NULL, "-tao_nls_pc_type", "none");
    #else
        PetscOptionsHasName (NULL, "-tao_nls_pc_type", &flag);
        if (flag == PETSC_FALSE)
            PetscOptionsSetValue ("-tao_nls_pc_type", "none");
    #endif
    flag = PETSC_FALSE;
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        PetscOptionsHasName (NULL, NULL, "-tao_nls_ksp_type", &flag);
        if (flag == PETSC_FALSE)
            PetscOptionsSetValue (NULL, "-tao_nls_ksp_type", "cg");
    #else
        PetscOptionsHasName (NULL, "-tao_nls_ksp_type", &flag);
        if (flag == PETSC_FALSE)
           PetscOptionsSetValue ("-tao_nls_ksp_type", "cg");
    #endif
    flag = PETSC_FALSE;
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        PetscOptionsHasName (NULL, NULL, "-tao_ntr_pc_type", &flag);
        if (flag == PETSC_FALSE)
            PetscOptionsSetValue (NULL, "-tao_ntr_pc_type", "none");
    #else
        PetscOptionsHasName (NULL, "-tao_ntr_pc_type", &flag);
        if (flag == PETSC_FALSE)
            PetscOptionsSetValue ("-tao_ntr_pc_type", "none");
    #endif


    // parse options user has set
    ierr = TaoSetFromOptions (tao);                                                 CHKERRQ(ierr);
    // set the initial vector
    ierr = TaoSetInitialVector (tao, ctx->tumor_->p_);                              CHKERRQ(ierr);
    // set routine for evaluating the objective
    ierr = TaoSetObjectiveRoutine (tao, evaluateObjectiveFunction, (void*) ctx);    CHKERRQ(ierr);
    // set routine for evaluating the Gradient
    ierr = TaoSetGradientRoutine (tao, evaluateGradient, (void*) ctx);              CHKERRQ(ierr);
    // set the routine to evaluate the objective and compute the gradient
    ierr = TaoSetObjectiveAndGradientRoutine (tao, evaluateObjectiveFunctionAndGradient, (void*) ctx);  CHKERRQ(ierr);
    // set monitor function
    ierr = TaoSetMonitor (tao, optimizationMonitor, (void *) ctx, NULL);          CHKERRQ(ierr);

    // Lower and Upper Bounds
    Vec lower_bound;
    ierr = VecDuplicate (ctx->tumor_->p_, &lower_bound);                            CHKERRQ (ierr);
    ierr = VecSet (lower_bound, 0.);                                                CHKERRQ (ierr);
    Vec upper_bound;
    ierr = VecDuplicate (ctx->tumor_->p_, &upper_bound);                            CHKERRQ (ierr);
    ierr = VecSet (upper_bound, PETSC_INFINITY);                                    CHKERRQ (ierr);

    ScalarType *ub_ptr, *lb_ptr;
    ScalarType upper_bound_kappa = ctx_->params_->opt_->k_ub_, lower_bound_kappa = ctx_->params_->opt_->k_lb_;
    if (ctx_->params_->opt_->diffusivity_inversion_) {
      ierr = VecGetArray (upper_bound, &ub_ptr);                                    CHKERRQ (ierr);
      ub_ptr[ctx_->params_->tu_->np_] = upper_bound_kappa;
      if (ctx_->params_->tu_->nk_ > 1) ub_ptr[ctx_->params_->tu_->np_ + 1] = upper_bound_kappa;
      if (ctx_->params_->tu_->nk_ > 2) ub_ptr[ctx_->params_->tu_->np_ + 2] = upper_bound_kappa;
      ierr = VecRestoreArray (upper_bound, &ub_ptr);                                CHKERRQ (ierr);

      ierr = VecGetArray (lower_bound, &lb_ptr);                                    CHKERRQ (ierr);
      lb_ptr[ctx_->params_->tu_->np_] = lower_bound_kappa;
      if (ctx_->params_->tu_->nk_ > 1) lb_ptr[ctx_->params_->tu_->np_ + 1] = lower_bound_kappa;
      if (ctx_->params_->tu_->nk_ > 2) lb_ptr[ctx_->params_->tu_->np_ + 2] = lower_bound_kappa;
      ierr = VecRestoreArray (lower_bound, &lb_ptr);                                CHKERRQ (ierr);
    }

    ierr = TaoSetVariableBounds(tao, lower_bound, upper_bound);                     CHKERRQ (ierr);
    if (lower_bound != nullptr) {ierr = VecDestroy(&lower_bound); CHKERRQ(ierr); lower_bound = nullptr;}
    if (upper_bound != nullptr) {ierr = VecDestroy(&upper_bound); CHKERRQ(ierr); upper_bound = nullptr;}

    // TAO type from user input
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
      TaoType taotype = NULL;
      ierr = TaoGetType (tao, &taotype);                                            CHKERRQ(ierr);
    #else
      const TaoType taotype;
      ierr = TaoGetType (tao, &taotype);                                            CHKERRQ(ierr);
    #endif
    if (strcmp(taotype, "nls") == 0) {
        msg = " limited memory variable metric method (unconstrained) selected";
    } else if (strcmp(taotype, "ntr") == 0) {
        msg = " Newton's method with trust region for unconstrained minimization";
    } else if (strcmp(taotype, "ntl") == 0) {
        msg = " Newton's method with trust region, line search for unconstrained minimization";
    } else if (strcmp(taotype, "nls") == 0) {
        msg = " Newton's method (line search; unconstrained) selected";
    } else if (strcmp(taotype, "bnls") == 0) {
        msg = " Newton's method (line search; bound constraints) selected";
    } else if (strcmp(taotype, "bqnls") == 0) {
        msg = " Quasi-Newton's method (line search; bound constraints) selected";
    } else if (strcmp(taotype, "ntr") == 0) {
        msg = " Newton's method (trust region; unconstrained) selected";
    } else if (strcmp(taotype, "fd_test") == 0) {
        msg = " gradient test selected";
    } else if (strcmp(taotype, "cg") == 0) {
        msg = " CG selected\n";
    } else if (strcmp(taotype, "tron") == 0) {
        msg = " Newton Trust Region method chosen";
    } else if (strcmp(taotype, "blmvm") == 0) {
        msg = " bounded limited memory variable metric method chosen";
    } else if (strcmp(taotype, "lmvm") == 0) {
        msg = " limited memory variable metric method chosen\n";
    } else if (strcmp(taotype, "tao_blmvm_m") == 0) {
        msg = " user modified limited memory variable metric method chosen";
    } else if (strcmp(taotype, "gpcg") == 0) {
        msg = " Newton Trust Region method for quadratic bound constrained minimization";
    } else if (strcmp(taotype, "tao_L1") == 0) {
        msg = " User defined solver for L1 minimization";
    } else {
        msg = " numerical optimization method not supported (setting default: LMVM)";
        ierr = TaoSetType (tao, "lmvm");                                          CHKERRQ(ierr);
    }
    ierr = tuMSGstd(msg); CHKERRQ(ierr);
    // set tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoSetTolerances (tao, ctx->params_->opt_->gatol_, ctx->params_->opt_->grtol_, ctx->params_->opt_->opttolgrad_); CHKERRQ(ierr);
    #else
        ierr = TaoSetTolerances (tao, 1E-12, 1E-12, ctx->params_->opt_->gatol_, ctx->params_->opt_->grtol_, ctx->params_->opt_->opttolgrad_); CHKERRQ(ierr);
    #endif
    ierr = TaoSetMaximumIterations (tao, ctx->params_->opt_->newton_maxit_); CHKERRQ(ierr);

    ierr = TaoSetConvergenceTest (tao, checkConvergenceGrad, ctx);                 CHKERRQ(ierr);
    // ierr = TaoSetConvergenceTest(tao, checkConvergenceGradObj, ctx);              CHKERRQ(ierr);

    // set linesearch (only for Gauss-Newton, lmvm uses more-thuente type line-search automatically)
    ierr = TaoGetLineSearch (tao, &linesearch);                                   CHKERRQ(ierr);
    linesearch->stepmin = minstep;

    if (ctx->params_->opt_->linesearch_ == ARMIJO) {
      ierr = TaoLineSearchSetType (linesearch, "armijo");                          CHKERRQ(ierr);
      tuMSGstd(" using line-search type: armijo");
    } else {
      tuMSGstd(" using line-search type: more-thuene");
    }

    //if(ctx_->params_->opt_->newton_solver_ == GAUSSNEWTON) {
    //  ierr = TaoLineSearchSetType (linesearch, "armijo");                         CHKERRQ(ierr);
    //}
    ierr = TaoLineSearchSetOptionsPrefix (linesearch,"tumor_");                    CHKERRQ(ierr);

    std::stringstream s;
    tuMSGstd(" parameters (optimizer):");
    tuMSGstd(" tolerances (stopping conditions):");
    s << "   gatol: "<< ctx->params_->opt_->gatol_;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
    s << "   grtol: "<< ctx->params_->opt_->grtol_;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
    s << "   gttol: "<< ctx->params_->opt_->opttolgrad_;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();

    ierr = TaoSetFromOptions(tao);                                                CHKERRQ(ierr);
    /* === set the KSP Krylov solver settings === */
    KSP ksp = PETSC_NULL;

    if (ctx_->params_->opt_->newton_solver_ == QUASINEWTON)  {
      // if (use_intial_hessian_lmvm_) {
      //   // get the ksp of H0 initial matrix
      //   ierr = TaoLMVMGetH0KSP(tao, &ksp);                                        CHKERRQ(ierr);
      //   if (ksp != PETSC_NULL) {
      //       ierr = KSPSetOptionsPrefix(ksp, "init-hessian_");                     CHKERRQ(ierr);
      //       // set default tolerance to 1E-6
      //       ierr = KSPSetTolerances(ksp, 1E-6, PETSC_DEFAULT, PETSC_DEFAULT, ctx->params_->opt_->krylov_maxit); CHKERRQ(ierr);
      //       ierr = KSPMonitorSet(ksp, constHessianKSPMonitor,ctx, 0);              CHKERRQ(ierr);
      //   }
      //}
    } else {
      // get the ksp of the optimizer (use gauss-newton-krylov)
      ierr = TaoGetKSP(tao, &ksp);                                                CHKERRQ(ierr);
      if (ksp != PETSC_NULL) {
          ierr = KSPSetOptionsPrefix(ksp, "hessian_");                            CHKERRQ(ierr);
          // set default tolerance to 1E-6
          ierr = KSPSetTolerances(ksp, 1E-6, PETSC_DEFAULT, PETSC_DEFAULT, ctx->params_->opt_->krylov_maxit_); CHKERRQ(ierr);
          // to use Eisenstat/Walker convergence crit.
          KSPSetPreSolve (ksp, preKrylovSolve, ctx);                              CHKERRQ(ierr);
          ierr = KSPMonitorSet(ksp, hessianKSPMonitor,ctx, 0);                    CHKERRQ(ierr);
          // ierr = KSPSetComputeSingularValues(ksp, PETSC_TRUE);                CHKERRQ (ierr);  // To compute the condition number
          ierr = KSPSetFromOptions (ksp);                                     CHKERRQ (ierr);
      }
      // set the preconditioner (we check if KSP exists, as there are also
      // solvers that do not require a KSP solve (BFGS and friends))
      if (ksp != PETSC_NULL) {
          PC pc;
          ierr = KSPGetPC(ksp, &pc);                                                CHKERRQ(ierr);
          ierr = PCSetType (pc, PCSHELL);                                           CHKERRQ(ierr);
          ierr = PCShellSetApply(pc, preconditionerMatVec);                         CHKERRQ(ierr);
          ierr = PCShellSetContext(pc, ctx);                                        CHKERRQ(ierr);
      }
    }

    PetscFunctionReturn (ierr);
}
