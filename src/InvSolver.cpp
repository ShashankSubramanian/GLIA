#include "InvSolver.h"
#include "petsctao.h"
#include <iostream>
#include <limits>
#include "DerivativeOperators.h"
#include "PdeOperators.h"
#include "Utils.h"
#include "TaoL1Solver.h"
#include <petsc/private/vecimpl.h>


InvSolver::InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc, std::shared_ptr <Tumor> tumor)
:
  initialized_(false),
  tao_is_reset_(true),
  data_(),
  data_gradeval_(),
  optsettings_(),
  optfeedback_(),
  itctx_() {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    tao_      = nullptr;
    H_        = nullptr;
    xrec_     = nullptr;
    xrec_rd_ = nullptr;
    if( derivative_operators != nullptr && pde_operators !=nullptr && n_misc != nullptr && tumor != nullptr) {
        initialize (derivative_operators, pde_operators, n_misc, tumor);
    }
}

PetscErrorCode InvSolver::initialize (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (initialized_)
        PetscFunctionReturn (ierr);
    optsettings_ = std::make_shared<OptimizerSettings> ();
    optfeedback_ = std::make_shared<OptimizerFeedback> ();
    itctx_ = std::make_shared<CtxInv> ();
    itctx_->derivative_operators_ = derivative_operators;
    itctx_->pde_operators_ = pde_operators;
    itctx_->n_misc_ = n_misc;
    itctx_->tumor_ = tumor;
    itctx_->optsettings_ = this->optsettings_;
    itctx_->optfeedback_ = this->optfeedback_;

    if (n_misc->invert_mass_effect_) {
        ierr = allocateTaoObjectsMassEffect(); CHKERRQ(ierr);
    } else {
        // allocate memory for H, x_rec and TAO
        ierr = allocateTaoObjects(); CHKERRQ(ierr);
    }


    initialized_ = true;
    PetscFunctionReturn (ierr);
}

PetscErrorCode InvSolver::allocateTaoObjectsMassEffect (bool initialize_tao) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    // For mass-effect; invert for rho, kappa, and gamma
    int n_inv = 3;
    ScalarType *xrec_ptr;
    // allocate memory for xrec_
    ierr = VecCreateSeq (PETSC_COMM_SELF, n_inv, &xrec_);          CHKERRQ (ierr);
    ierr = setupVec (xrec_, SEQ);                                  CHKERRQ (ierr);
    ierr = VecSet (xrec_, 0.0);                                    CHKERRQ (ierr);    
    ierr = VecGetArray (xrec_, &xrec_ptr);                         CHKERRQ (ierr);
    xrec_ptr[0] = 0.6; xrec_ptr[1] = 0.6; xrec_ptr[2] = 0.05;
    // xrec_ptr[0] = 0.4; xrec_ptr[1] = 0.08;
    ierr = VecRestoreArray (xrec_, &xrec_ptr);                     CHKERRQ (ierr);

    // set up routine to compute the hessian matrix vector product
    if (H_ == nullptr) {
      ierr = MatCreateShell (PETSC_COMM_SELF, n_inv, n_inv, n_inv, n_inv, (void*) itctx_.get(), &H_); CHKERRQ(ierr);
    }
    // create TAO solver object
    if ( tao_ == nullptr && initialize_tao) {
      ierr = TaoCreate (PETSC_COMM_SELF, &tao_); tao_is_reset_ = true;  // triggers setTaoOptions
    }

    ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))hessianMatVec);         CHKERRQ(ierr);
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 10)
        ierr = MatShellSetOperation (H_, MATOP_CREATE_VECS, (void(*)(void)) operatorCreateVecsMassEffect);
    #endif
    ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                                 CHKERRQ(ierr);

    PetscFunctionReturn (ierr);
}

PetscErrorCode operatorCreateVecsMassEffect (Mat A, Vec *left, Vec *right) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    CtxInv *ctx;
    ierr = MatShellGetContext (A, &ctx);                        CHKERRQ (ierr);

    if (right) {
        ierr = VecDuplicate (ctx->x_old, right);             CHKERRQ(ierr);
    }
    if (left) {
        ierr = VecDuplicate (ctx->x_old, left);              CHKERRQ(ierr);
    }

    PetscFunctionReturn (ierr);
}

PetscErrorCode InvSolver::allocateTaoObjects (bool initialize_tao) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int np = itctx_->n_misc_->np_;
    int nk = (itctx_->n_misc_->diffusivity_inversion_) ?  itctx_->n_misc_->nk_ : 0;
    int nr = 0;

    if (itctx_->n_misc_->regularization_norm_ == L1) {//Register new Tao solver and initialize variables for parameter continuation
    ierr = TaoRegister ("tao_L1", TaoCreate_ISTA);                              CHKERRQ (ierr);
    itctx_->lam_right = itctx_->n_misc_->lambda_;
    itctx_->lam_left = 0;
    }

    if (itctx_->n_misc_->regularization_norm_ == wL2) {
    if (itctx_->weights != nullptr) {ierr = VecDestroy(&itctx_->weights); CHKERRQ(ierr); itctx_->weights = nullptr;}
    ierr = VecDuplicate (itctx_->tumor_->p_, &itctx_->weights);                 CHKERRQ (ierr);
    }

    #ifdef SERIAL
        // allocate memory for xrec_
        ierr = VecDuplicate (itctx_->tumor_->p_, &xrec_);                         CHKERRQ(ierr);
        // set up routine to compute the hessian matrix vector product
        if (H_ == nullptr) {
          ierr = MatCreateShell (PETSC_COMM_SELF, np + nk + nr, np + nk + nr, np + nk + nr, np + nk + nr, (void*) itctx_.get(), &H_); CHKERRQ(ierr);
        }
        // create TAO solver object
        if ( tao_ == nullptr && initialize_tao) {
          ierr = TaoCreate (PETSC_COMM_SELF, &tao_); tao_is_reset_ = true;  // triggers setTaoOptions
        }
    #else
        TU_assert (!itctx_->n_misc_->diffusivity_inversion_, "Inversion for diffusifity is only implemented for SERIAL p");
        // allocate memory for xrec_
        ierr = VecDuplicate (itctx_->tumor_->p_, &xrec_);                           CHKERRQ(ierr);
        // set up routine to compute the hessian matrix vector product
        if (H_ == nullptr) {
          ierr = MatCreateShell (MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, np + nk, np + nk, (void*) itctx_.get(), &H_); CHKERRQ(ierr);
        }
        // create TAO solver object
        if( tao_ == nullptr && initialize_tao) {
          ierr = TaoCreate (MPI_COMM_WORLD, &tao_); tao_is_reset_ = true;  // triggers setTaoOptions
        }
    #endif
    ierr = VecSet (xrec_, 0.0);                                                   CHKERRQ(ierr);

    // if tao's lmvm (l-bfgs) method is used and the initial hessian approximation is explicitly set
    if ((itctx_->optsettings_->newtonsolver == QUASINEWTON) && itctx_->optsettings_->lmvm_set_hessian) {
    ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))constApxHessianMatVec); CHKERRQ(ierr);
    ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                                 CHKERRQ(ierr);
    // if tao's nls (gauss-newton) method is used, define hessian matvec
    }
    else {
    ierr = MatShellSetOperation (H_, MATOP_MULT, (void (*)(void))hessianMatVec);         CHKERRQ(ierr);
    ierr = MatSetOption (H_, MAT_SYMMETRIC, PETSC_TRUE);                                 CHKERRQ(ierr);
    }

    PetscFunctionReturn (ierr);
}

PetscErrorCode InvSolver::resetTao (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (tao_  != nullptr) {ierr = TaoDestroy (&tao_);  CHKERRQ(ierr); tao_  = nullptr;}
    if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
    if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}

    // allocate memory for H, x_rec and TAO
    ierr = allocateTaoObjects (); CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode InvSolver::setParams (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, bool npchanged) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    itctx_->derivative_operators_ = derivative_operators;
    itctx_->pde_operators_ = pde_operators;
    itctx_->n_misc_ = n_misc;
    itctx_->tumor_ = tumor;
    // re-allocate memory
    if (npchanged && !n_misc->invert_mass_effect_){                              // re-allocate memory for xrec_
      // allocate memory for H, x_rec and TAO
      itctx_->x_old = nullptr; // Will be set accordingly in the solver
      if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
      if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}
      ierr = allocateTaoObjects (false); CHKERRQ(ierr);
    } 

    if (n_misc->invert_mass_effect_) {
        // allocate memory for H, x_rec and TAO
      itctx_->x_old = nullptr; // Will be set accordingly in the solver
      if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
      if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}
      ierr = allocateTaoObjectsMassEffect (false); CHKERRQ(ierr);
    }

    tao_is_reset_ = true;                        // triggers setTaoOptions
    PetscFunctionReturn (ierr);
}

PetscErrorCode InvSolver::resetOperators (Vec p) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    // reset tumor_ object, re-size solution vector and copy p into tumor_->p_
    ierr = itctx_->tumor_->setParams (p, itctx_->n_misc_, true);                CHKERRQ (ierr);
    // reset derivative operators, re-size vectors
    itctx_->derivative_operators_->reset(p, itctx_->pde_operators_, itctx_->n_misc_, itctx_->tumor_);
    // re-allocate memory
    itctx_->x_old = nullptr; // Will be set accordingly in the solver
    if (tao_  != nullptr) {ierr = TaoDestroy (&tao_);  CHKERRQ(ierr); tao_  = nullptr;}
    if (H_    != nullptr) {ierr = MatDestroy (&H_);    CHKERRQ(ierr); H_    = nullptr;}
    if (xrec_ != nullptr) {ierr = VecDestroy (&xrec_); CHKERRQ(ierr); xrec_ = nullptr;}
    ierr = allocateTaoObjects (true); CHKERRQ(ierr);
    tao_is_reset_ = true;                        // triggers setTaoOptions
    PetscFunctionReturn (ierr);
}

PetscErrorCode phiMult (Mat A, Vec x, Vec y) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    //A = phiT * phi
    InterpolationContext *ctx;
    ierr = MatShellGetContext (A, &ctx);                                        CHKERRQ (ierr);
    ierr = ctx->tumor_->phi_->apply (ctx->temp_, x);                            CHKERRQ (ierr);
    ierr = ctx->tumor_->phi_->applyTranspose (y, ctx->temp_);                   CHKERRQ (ierr);

    // Regularization
    ScalarType beta = 0e-3;
    ierr = VecAXPY (y, beta, x);                                                CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}


PetscErrorCode interpolationKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec x; int maxit; PetscScalar divtol, abstol, reltol;
    ierr = KSPBuildSolution (ksp, NULL, &x);                                       CHKERRQ(ierr);
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);              CHKERRQ(ierr);
    InterpolationContext *itctx = reinterpret_cast<InterpolationContext*>(ptr);     // get user context

    std::stringstream s;
    if (its == 0) {
      s << std::setw(3)  << " PCG:" << " computing solution of interpolation system (tol="
        << std::scientific << std::setprecision(5) << reltol << ")";
      ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
      s.str (""); s.clear ();
    }
    s << std::setw(3)  << " PCG:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
    << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
    ierr = tuMSGstd (s.str());                                                  CHKERRQ(ierr);
    s.str (""); s.clear ();

    PetscFunctionReturn (ierr);
}

PetscErrorCode InvSolver::solveInterpolation (Vec data) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    KSP ksp;
    Mat A;
    Vec rhs, phi_p, p_out;

    std::shared_ptr<NMisc> n_misc = itctx_->n_misc_;
    std::shared_ptr<Tumor> tumor = itctx_->tumor_;
    std::shared_ptr<InterpolationContext> ctx = std::make_shared<InterpolationContext> (n_misc);
    ctx->tumor_ = tumor;
    int sz = n_misc->np_;
    ierr = VecCreateSeq (PETSC_COMM_SELF, sz, &p_out);                          CHKERRQ (ierr);
    ierr = setupVec (p_out, SEQ);                                               CHKERRQ (ierr);

    std::stringstream ss;
    ss << " ------- Interpolating within subspace of size = " << sz << " -------";
    ierr = tuMSGstd(ss.str());                                                  CHKERRQ(ierr);
    ss.str(""); ss.clear();

    ierr = VecDuplicate (data, &phi_p);                                         CHKERRQ (ierr);
    ierr = VecDuplicate (p_out, &rhs);                                          CHKERRQ (ierr);
    ierr = VecSet (rhs, 0);                                                     CHKERRQ (ierr);
    ierr = VecSet (p_out, 0);                                                   CHKERRQ (ierr);

    ierr = KSPCreate (PETSC_COMM_SELF, &ksp);                                   CHKERRQ (ierr);
    ierr = MatCreateShell (PETSC_COMM_SELF, sz, sz, sz, sz, ctx.get(), &A);     CHKERRQ (ierr);
    ierr = MatShellSetOperation (A, MATOP_MULT, (void (*) (void))phiMult);      CHKERRQ (ierr);
    ierr = KSPSetOperators (ksp, A, A);                                         CHKERRQ (ierr);
    ierr = KSPSetTolerances (ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 500);     CHKERRQ (ierr);
    ierr = KSPSetType (ksp, KSPCG);                                             CHKERRQ (ierr);
    ierr = KSPSetOptionsPrefix (ksp, "phieq_");                                 CHKERRQ (ierr);
    ierr = KSPSetComputeSingularValues(ksp, PETSC_TRUE);                        CHKERRQ (ierr);  // To compute the condition number
    ierr = KSPSetFromOptions (ksp);                                             CHKERRQ (ierr);
    ierr = KSPSetUp (ksp);                                                      CHKERRQ (ierr);
    ierr = KSPMonitorSet (ksp, interpolationKSPMonitor, ctx.get(), 0);          CHKERRQ (ierr);

    //RHS -- phiT * d
    ierr = tumor->phi_->applyTranspose (rhs, data);                             CHKERRQ (ierr);

    ierr = KSPSolve (ksp, rhs, p_out);                                          CHKERRQ (ierr);
    // Compute extreme singular values for condition number
    PetscReal e_max, e_min;
    ierr = KSPComputeExtremeSingularValues (ksp, &e_max, &e_min);               CHKERRQ (ierr);
    ss << "Condition number of PhiTPhi is: " << e_max / e_min << " | largest singular values is: " << e_max << ", smallest singular values is: " << e_min;
    ierr = tuMSGstd(ss.str());                                                  CHKERRQ(ierr);
    ss.str(""); ss.clear();

    //Compute reconstruction error
    PetscReal error_norm, p_norm, d_norm;
    ierr = tumor->phi_->apply (phi_p, p_out);                                   CHKERRQ (ierr);
    // dataOut (phi_p, ctx->n_misc_, "CInterp.nc");

    ierr = VecNorm (data, NORM_2, &d_norm);                                     CHKERRQ (ierr);
    ierr = VecAXPY (phi_p, -1.0, data);                                   CHKERRQ (ierr);
    ierr = VecNorm (phi_p, NORM_2, &error_norm);                                CHKERRQ (ierr);
    ierr = VecNorm (p_out, NORM_2, &p_norm);                                    CHKERRQ (ierr);

    ss << "Data mismatch using interpolated basis is: " << error_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << "Data mismatch using zero vector is: " << d_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << "Rel error in interpolation reconstruction: " << error_norm / d_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << "Norm of reconstructed p: " << p_norm; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << " ------- Interpolation end -------"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    ierr = VecDestroy (&p_out);                                                 CHKERRQ (ierr);
    ierr = VecDestroy (&rhs);                                                   CHKERRQ (ierr);
    ierr = VecDestroy (&phi_p);                                                 CHKERRQ (ierr);
    ierr = MatDestroy (&A);                                                     CHKERRQ (ierr);
    ierr = KSPDestroy (&ksp);                                                   CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}


PetscErrorCode InvSolver::restrictSubspace (Vec *x_restricted, Vec x_full, std::shared_ptr<CtxInv> itctx, bool create_rho_dofs = false) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscReal *x_restricted_ptr, *x_full_ptr;
    int np = itctx->n_misc_->support_.size();        // size of restricted subspace (not necessarily 2s, since merged)
    int nk = (itctx->n_misc_->diffusivity_inversion_ || (create_rho_dofs &&  itctx->n_misc_->reaction_inversion_)) ? itctx->n_misc_->nk_ : 0;
    int nr = (itctx->n_misc_->reaction_inversion_ && create_rho_dofs) ? itctx->n_misc_->nr_ : 0;

    itctx->n_misc_->np_ = np;                    // change np to solve the restricted subsystem
    ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk + nr, x_restricted);          CHKERRQ (ierr);
    ierr = setupVec (*x_restricted, SEQ);                                       CHKERRQ (ierr);
    ierr = VecSet (*x_restricted, 0);                                           CHKERRQ (ierr);
    ierr = VecGetArray (*x_restricted, &x_restricted_ptr);                      CHKERRQ (ierr);
    ierr = VecGetArray (x_full, &x_full_ptr);                                   CHKERRQ (ierr);
    for (int i = 0; i < np; i++)
        x_restricted_ptr[i] = x_full_ptr[itctx->n_misc_->support_[i]];
    // initial guess diffusivity
    if (itctx->n_misc_->diffusivity_inversion_) {
        x_restricted_ptr[np] = itctx->n_misc_->k_;                                                 // equals x_full_ptr[np_full];
        if (nk > 1) x_restricted_ptr[np+1] = itctx->n_misc_->k_ * itctx->n_misc_->k_gm_wm_ratio_;  // equals x_full_ptr[np_full+1];
        if (nk > 2) x_restricted_ptr[np+2] = itctx->n_misc_->k_ * itctx->n_misc_->k_glm_wm_ratio_; // equals x_full_ptr[np_full+2];
    }
    // initial guess reaction
    if (create_rho_dofs && itctx->n_misc_->reaction_inversion_) {
        x_restricted_ptr[np + nk] = itctx->n_misc_->rho_;
        if (nr > 1) x_restricted_ptr[np + nk + 1] = itctx->n_misc_->rho_ * itctx->n_misc_->r_gm_wm_ratio_;
        if (nr > 2) x_restricted_ptr[np + nk + 2] = itctx->n_misc_->rho_ * itctx->n_misc_->r_glm_wm_ratio_;
    }
    ierr = VecRestoreArray (*x_restricted, &x_restricted_ptr);                  CHKERRQ (ierr);
    ierr = VecRestoreArray (x_full, &x_full_ptr);                               CHKERRQ (ierr);
    // Modifies the centers
    itctx->tumor_->phi_->modifyCenters (itctx->n_misc_->support_);
    // resets the phis and other operators, x_restricted is copied into tumor->p_ and is used as init cond for
    // the L2 solver (needs to be done in every iteration, since location of basis functions updated)
    ierr = resetOperators (*x_restricted); /* reset phis and other operators */ CHKERRQ (ierr);

    PetscFunctionReturn (ierr);
}


PetscErrorCode InvSolver::prolongateSubspace (Vec x_full, Vec *x_restricted, std::shared_ptr<CtxInv> itctx, int np_full, bool reset_operators = true) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscReal *x_restricted_ptr, *x_full_ptr;
    int np_r = itctx->n_misc_->support_.size();        // size of restricted subspace (not necessarily 2s, since merged)
    int nk   = (itctx->n_misc_->diffusivity_inversion_) ? itctx->n_misc_->nk_ : 0;

    ierr = VecSet (x_full, 0.);                                                 CHKERRQ (ierr);
    ierr = VecGetArray (*x_restricted, &x_restricted_ptr);                      CHKERRQ (ierr);
    ierr = VecGetArray (x_full, &x_full_ptr);                                   CHKERRQ (ierr);
    // correct L1 guess
    for (int i = 0; i < np_r; i++)
      x_full_ptr[itctx->n_misc_->support_[i]] = x_restricted_ptr[i];
    // correct diffusivity
    if (itctx->n_misc_->diffusivity_inversion_) {
        x_full_ptr[np_full] = itctx->n_misc_->k_;
        if (nk > 1) x_full_ptr[np_full+1] = itctx->n_misc_->k_ * itctx->n_misc_->k_gm_wm_ratio_;
        if (nk > 2) x_full_ptr[np_full+2] = itctx->n_misc_->k_ * itctx->n_misc_->k_glm_wm_ratio_;
    }
    ierr = VecRestoreArray (*x_restricted, &x_restricted_ptr);                  CHKERRQ (ierr);
    ierr = VecRestoreArray (x_full, &x_full_ptr);                               CHKERRQ (ierr);

    itctx->n_misc_->np_ = np_full;         /* reset to full space         */
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
PetscErrorCode InvSolver::solveInverseReacDiff (Vec x_in) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    TU_assert (initialized_,              "InvSolver::solveInverseReacDiff (): InvSolver needs to be initialized.")
    TU_assert (data_ != nullptr,          "InvSolver::solveInverseReacDiff (): requires non-null input data for inversion.");
    TU_assert (data_gradeval_ != nullptr, "InvSolver::solveInverseReacDiff (): requires non-null input data for gradient evaluation.");
    TU_assert (xrec_ != nullptr,          "InvSolver::solveInverseReacDiff (): requires non-null p_rec vector to be set");
    TU_assert (optsettings_ != nullptr,   "InvSolver::solveInverseReacDiff (): requires non-null optimizer settings to be passed.");

    PetscReal beta_p = itctx_->n_misc_->beta_;     // set beta to zero here as the params are rho and kappa
    itctx_->n_misc_->flag_reaction_inv_ = true;    // enables derivative operators to compute the gradient w.r.t rho
    itctx_->n_misc_->beta_ = 0.;
    PetscReal *d_ptr, *x_in_ptr, *x_ptr, *ub_ptr, *lb_ptr, *x_full_ptr;
    PetscReal d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0., max, min, xdiff;
    PetscReal upper_bound_kappa, lower_bound_kappa, minstep;
    std::string msg;
    std::stringstream ss;
    int nk, nr, np, x_sz;
    Vec lower_bound, upper_bound, p;
    CtxInv *ctx = itctx_.get();
    TaoLineSearch linesearch;
    TaoConvergedReason reason;

    // ls ministep
    minstep = std::pow (2.0, 18.0);
    minstep = 1.0 / minstep;

    // DOFs
    nk = itctx_->n_misc_->nk_;
    nr = itctx_->n_misc_->nr_;
    np = itctx_->n_misc_->np_;
    x_sz = nk + nr;

    // rescale init cond. and invert for rho/kappa
    PetscReal ic_max = 0., g_norm_ref = 0.;
    ierr = itctx_->tumor_->phi_->apply(itctx_->tumor_->c_0_, x_in);             CHKERRQ (ierr);
    ierr = VecMax (itctx_->tumor_->c_0_, NULL, &ic_max);                        CHKERRQ (ierr);
    ierr = VecGetArray (x_in, &x_in_ptr);                                       CHKERRQ (ierr);
    /* scale p to one according to our modeling assumptions:
     * scales INT_Omega phi(x) dx = const across levels, factor in between levels: 2
     * scales nx=256 to max {Phi p} = 1, nx=128 to max {Phi p} = 0.5, nx=64 to max {Phi p} = 0.25 */
    for (int i = 0; i < np ; i++){
        if(itctx_->n_misc_->multilevel_) { x_in_ptr[i] *= (1.0/4.0 * itctx_->n_misc_->n_[0]/64.  / ic_max);}
        else                             { x_in_ptr[i] *= (1.0 / ic_max); }
      }
    ierr = VecRestoreArray (x_in, &x_in_ptr);                                   CHKERRQ (ierr);

    // write out p vector after IC, k inversion (scaled)
    ierr = tuMSGstd ("");                                                       CHKERRQ (ierr);
    if (itctx_->cosamp_->cosamp_stage == PRE_RD) {
        ierr = tuMSG    ("### scaled init guess w/ incorrect reaction coefficient  ###"); CHKERRQ (ierr);
        ierr = tuMSGstd ("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
        if (procid == 0) { ierr = VecView (x_in, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}
        ierr = tuMSGstd ("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
        if (itctx_->n_misc_->write_p_checkpoint_) {writeCheckpoint(x_in, itctx_->tumor_->phi_, itctx_->n_misc_->writepath_ .str(), std::string("scaled-pre-l1"));}
        ierr = tuMSGstd ("");                                                             CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSG("###                     (PRE) rho/kappa inversion with scaled L2 solution guess                           ###");CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    } else if (itctx_->cosamp_->cosamp_stage == POST_RD) {
        ierr = tuMSG    ("### scaled L2 sol. w/ incorrect reaction coefficient     ###"); CHKERRQ (ierr);
        ierr = tuMSGstd ("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
        if (procid == 0) { ierr = VecView (x_in, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}
        ierr = tuMSGstd ("### ---------------------------------------------------- ###"); CHKERRQ (ierr);
        if (itctx_->n_misc_->write_p_checkpoint_) {writeCheckpoint(x_in, itctx_->tumor_->phi_, itctx_->n_misc_->writepath_ .str(), std::string("scaled"));}
        ierr = tuMSGstd ("");                                                             CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSG("###                          rho/kappa inversion with scaled L2 solution guess                            ###");CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    }

    /* === Add Noise === */
    Vec noise; ScalarType *noise_ptr;
    ierr = VecCreate (PETSC_COMM_WORLD, &noise);                                  CHKERRQ(ierr);
    ierr = setupVec (noise);                                                      CHKERRQ (ierr);
    ierr = VecSetSizes(noise, itctx_->n_misc_->n_local_, itctx_->n_misc_->n_global_); CHKERRQ(ierr);
    ierr = VecSetFromOptions(noise);                                              CHKERRQ(ierr);
    ierr = VecSetRandom(noise, NULL);                                             CHKERRQ(ierr);
    ierr = VecGetArray (noise, &noise_ptr);                                       CHKERRQ(ierr);
    ierr = VecGetArray (data_, &d_ptr);                                           CHKERRQ(ierr);
    for (int i = 0; i < itctx_->n_misc_->n_local_; i++) {
      d_ptr[i] += noise_ptr[i] * itctx_->n_misc_->noise_scale_;
      noise_ptr[i] = d_ptr[i];                                                  // just to measure d norm
    }
    ierr = VecRestoreArray (noise, &noise_ptr);                                   CHKERRQ(ierr);
    ierr = VecRestoreArray (data_, &d_ptr);                                       CHKERRQ(ierr);
    #ifdef POSITIVITY
    ierr = enforcePositivity (data_, itctx_->n_misc_);
    ierr = enforcePositivity (noise, itctx_->n_misc_);
    #endif
    ierr = VecNorm (noise, NORM_2, &d_norm);                                      CHKERRQ(ierr);
    ierr = VecMax  (noise, NULL, &max);                                           CHKERRQ(ierr);
    ierr = VecMin  (noise, NULL, &min);                                           CHKERRQ(ierr);
    ierr = VecAXPY (noise, -1.0, data_);                                          CHKERRQ(ierr);
    ierr = VecNorm (noise, NORM_2, &d_errorl2norm);                               CHKERRQ(ierr);
    ierr = VecNorm (noise, NORM_INFINITY, &d_errorInfnorm);                       CHKERRQ(ierr);
    ss << " tumor inversion target data (with noise): l2norm = "<< d_norm <<" [max: "<<max<<", min: "<<min<<"]";  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << " tumor inversion target data error (due to thresholding and smoothing): l2norm = "<< d_errorl2norm <<", inf-norm = " <<d_errorInfnorm;  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    // Reset tao
    if (tao_      != nullptr)         {ierr = TaoDestroy (&tao_);                 CHKERRQ(ierr); tao_  = nullptr;}
    if (H_        != nullptr)         {ierr = MatDestroy (&H_);                   CHKERRQ(ierr); H_    = nullptr;}
    if (xrec_     != nullptr)         {ierr = VecDestroy (&xrec_);                CHKERRQ(ierr); xrec_ = nullptr;}
    if (xrec_rd_ != nullptr)          {ierr = VecDestroy (&xrec_rd_);             CHKERRQ(ierr); xrec_rd_ = nullptr;}
    if (itctx_->x_old != nullptr)     {ierr = VecDestroy (&itctx_->x_old);        CHKERRQ(ierr); itctx_->x_old = nullptr;}
    // TODO: x_old here is used to store the full solution vector and not old guess. (Maybe change to new vector to avoid confusion?)
    // re-allocate
    ierr = VecDuplicate (x_in, &itctx_->x_old);                                   CHKERRQ (ierr);
    ierr = VecCopy      (x_in,  itctx_->x_old);   /* stores full solution vec */  CHKERRQ (ierr);
    ierr = VecDuplicate (x_in, &xrec_);                                           CHKERRQ(ierr);
    ierr = VecSet       (xrec_, 0.0);                                             CHKERRQ(ierr);
    ierr = TaoCreate    (PETSC_COMM_SELF, &tao_);                                 CHKERRQ (ierr);
    ierr = TaoSetType   (tao_, "blmvm");                                          CHKERRQ (ierr);
    ierr = VecCreateSeq (PETSC_COMM_SELF, x_sz, &xrec_rd_); /* inv rho and k */   CHKERRQ (ierr);
    ierr = setupVec     (xrec_rd_, SEQ);                                          CHKERRQ (ierr);
    ierr = VecSet       (xrec_rd_, 0.);                                           CHKERRQ (ierr);
    ierr = MatCreateShell (PETSC_COMM_SELF, np + nk, np + nk, np + nk, np + nk, (void*) itctx_.get(), &H_); CHKERRQ(ierr);

    // initial guess kappa
    ierr = VecGetArray (x_in, &x_in_ptr);                                         CHKERRQ (ierr);
    ierr = VecGetArray (xrec_rd_, &x_ptr);                                        CHKERRQ (ierr);
    x_ptr[0] = (nk > 0) ? x_in_ptr[itctx_->n_misc_->np_] : 0;   // k1
    if (nk > 1) x_ptr[1] = x_in_ptr[itctx_->n_misc_->np_ + 1];  // k2
    if (nk > 2) x_ptr[2] = x_in_ptr[itctx_->n_misc_->np_ + 2];  // k3
    ss << " initial guess for diffusion coefficient: " << x_ptr[0]; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    // initial guess rho
    if (itctx_->n_misc_->multilevel_ && itctx_->n_misc_->n_[0] > 64) {
    x_ptr[nk] = x_in_ptr[itctx_->n_misc_->np_ + nk];                      // r1
    if (nr > 1) x_ptr[nk + 1] = x_in_ptr[itctx_->n_misc_->np_ + nk + 1];  // r2
    if (nr > 2) x_ptr[nk + 2] = x_in_ptr[itctx_->n_misc_->np_ + nk + 2];  // r3
    } else {
    ss<<" computing rough approximation to rho.."; ierr = tuMSGstd(ss.str());   CHKERRQ(ierr); ss.str(""); ss.clear();
    std::array<ScalarType, 7> rho_guess = {0, 3, 6, 9, 10, 12, 15};
    ScalarType min_norm = 1E15, norm = 0.;

    int idx = 0;
    for (int i = 0; i < rho_guess.size(); i++) {
      // update the tumor with this rho
      ierr = itctx_->tumor_->rho_->updateIsotropicCoefficients (rho_guess[i], 0., 0., itctx_->tumor_->mat_prop_, itctx_->n_misc_);
      ierr = itctx_->tumor_->phi_->apply (itctx_->tumor_->c_0_, x_in);          CHKERRQ (ierr);   // apply scaled p to IC
      ierr = itctx_->derivative_operators_->pde_operators_->solveState (0);    // solve state with guess reaction and inverted diffusivity
      ierr = itctx_->tumor_->obs_->apply (itctx_->derivative_operators_->temp_, itctx_->tumor_->c_t_);               CHKERRQ (ierr);
      // mismatch between data and c
      ierr = VecAXPY (itctx_->derivative_operators_->temp_, -1.0, data_);       CHKERRQ (ierr);    // Oc(1) - d
      ierr = VecNorm (itctx_->derivative_operators_->temp_, NORM_2, &norm);     CHKERRQ (ierr);
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
    upper_bound_kappa = itctx_->n_misc_->k_ub_;
    lower_bound_kappa = itctx_->n_misc_->k_lb_;
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
    ierr = TaoSetTolerances (tao_, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad); CHKERRQ(ierr);
    ierr = TaoSetMaximumIterations (tao_, ctx->optsettings_->newton_maxit);                                            CHKERRQ(ierr);
    ierr = TaoSetConvergenceTest (tao_, checkConvergenceGradReacDiff, ctx);                                            CHKERRQ(ierr);

    itctx_->update_reference_gradient = true;    // compute ref gradient
    itctx_->optsettings_->ls_minstep = minstep;  // overwrite linesearch objects
    ierr = TaoGetLineSearch (tao_, &linesearch);                                  CHKERRQ(ierr);
    linesearch->stepmin = minstep;
    if (ctx->optsettings_->linesearch == ARMIJO) {
    ierr = TaoLineSearchSetType (linesearch, "armijo");                         CHKERRQ(ierr);
    ierr = tuMSGstd(" using line-search type: armijo");                         CHKERRQ(ierr);
    } else { ierr = tuMSGstd(" using line-search type: more-thuene"); CHKERRQ(ierr);}
    ierr = TaoLineSearchSetOptionsPrefix (linesearch,"tumor_");                   CHKERRQ(ierr);


    ierr = tuMSGstd(" parameters (optimizer):");                                  CHKERRQ(ierr);
    ierr = tuMSGstd(" tolerances (stopping conditions):");                        CHKERRQ(ierr);
    ss << "   gatol: "<< ctx->optsettings_->gatol;      tuMSGstd(ss.str()); ss.str(""); ss.clear();
    ss << "   grtol: "<< ctx->optsettings_->grtol;      tuMSGstd(ss.str()); ss.str(""); ss.clear();
    ss << "   gttol: "<< ctx->optsettings_->opttolgrad; tuMSGstd(ss.str()); ss.str(""); ss.clear();
    ierr = TaoSetFromOptions(tao_);                                               CHKERRQ(ierr);
    // reset feedback variables, reset data
    itctx_->is_ksp_gradnorm_set        = false;
    itctx_->optfeedback_->converged    = false;
    itctx_->optfeedback_->solverstatus = "";
    itctx_->optfeedback_->nb_newton_it = 0;
    itctx_->optfeedback_->nb_krylov_it = 0;
    itctx_->optfeedback_->nb_matvecs   = 0;
    itctx_->optfeedback_->nb_objevals  = 0;
    itctx_->optfeedback_->nb_gradevals = 0;
    itctx_->data                       = data_;
    itctx_->data_gradeval              = data_gradeval_;
    itctx_->n_misc_->statistics_.reset();
    ss << " tumor regularization = "<< itctx_->n_misc_->beta_ << " type: " << itctx_->n_misc_->regularization_norm_;  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    double self_exec_time_tuninv = -MPI_Wtime(); double invtime = 0;
    // ====== solve ======
    ierr = TaoSolve (tao_);                                                       CHKERRQ(ierr);
    self_exec_time_tuninv += MPI_Wtime();
    MPI_Reduce(&self_exec_time_tuninv, &invtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    ierr = TaoGetSolutionVector (tao_, &p);                                       CHKERRQ(ierr);
    ierr = VecGetArray          (p, &x_ptr);                                      CHKERRQ (ierr);
    ierr = VecGetArray          (itctx_->x_old, &x_full_ptr);                     CHKERRQ (ierr);
    x_full_ptr[itctx_->n_misc_->np_] = x_ptr[0];                                    // k1
    if (itctx_->n_misc_->nk_ > 1) x_full_ptr[itctx_->n_misc_->np_ + 1] = x_ptr[1];  // k2
    if (itctx_->n_misc_->nk_ > 2) x_full_ptr[itctx_->n_misc_->np_ + 2] = x_ptr[2];  // k3
    x_full_ptr[itctx_->n_misc_->np_ + itctx_->n_misc_->nk_] = x_ptr[itctx_->n_misc_->nk_];                                        // r1
    if (itctx_->n_misc_->nr_ > 1) x_full_ptr[itctx_->n_misc_->np_ + itctx_->n_misc_->nk_ + 1] = x_ptr[itctx_->n_misc_->nk_ + 1];  // r2
    if (itctx_->n_misc_->nr_ > 2) x_full_ptr[itctx_->n_misc_->np_ + itctx_->n_misc_->nk_ + 2] = x_ptr[itctx_->n_misc_->nk_ + 2];  // r2
    ierr = VecRestoreArray      (p, &x_ptr);                                      CHKERRQ (ierr);
    ierr = VecRestoreArray      (itctx_->x_old, &x_full_ptr);                     CHKERRQ (ierr);
    // store sol in xrec_
    ierr = VecCopy (itctx_->x_old, xrec_);                                        CHKERRQ(ierr);

    /* Get information on termination */
    ierr = TaoGetConvergedReason (tao_, &reason);                                 CHKERRQ(ierr);
    /* get solution status */
    ierr = TaoGetSolutionStatus (tao_, NULL, &itctx_->optfeedback_->jval, &itctx_->optfeedback_->gradnorm, NULL, &xdiff, NULL);         CHKERRQ(ierr);
    /* display convergence reason: */
    ierr = dispTaoConvReason (reason, itctx_->optfeedback_->solverstatus);        CHKERRQ(ierr);
    itctx_->optfeedback_->nb_newton_it--;
    ss << " optimization done: #N-it: " << itctx_->optfeedback_->nb_newton_it    << ", #K-it: " << itctx_->optfeedback_->nb_krylov_it
                      << ", #matvec: " << itctx_->optfeedback_->nb_matvecs    << ", #evalJ: " << itctx_->optfeedback_->nb_objevals
                      << ", #evaldJ: " << itctx_->optfeedback_->nb_gradevals  << ", exec time: " << invtime;
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGstd (ss.str());                                                                                           CHKERRQ(ierr);  ss.str(""); ss.clear();
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    itctx_->n_misc_->statistics_.print();
    itctx_->n_misc_->statistics_.reset();

    tao_is_reset_ = false;
    itctx_->n_misc_->beta_ = beta_p;              // restore beta value
    itctx_->n_misc_->flag_reaction_inv_ = false;  // disables derivative operators to compute the gradient w.r.t rho

    // get diffusivity and reaction
    ierr = VecGetArray (xrec_, &x_ptr);                                         CHKERRQ (ierr);
    itctx_->n_misc_->rho_ = x_ptr[np + nk];
    itctx_->n_misc_->k_   = x_ptr[np];
    ierr = VecRestoreArray (xrec_, &x_ptr);                                     CHKERRQ (ierr);
    PetscReal r1, r2, r3, k1, k2, k3;
    r1 = itctx_->n_misc_->rho_;                                                                      // equals x_in_ptr[np + nk]
    r2 = (itctx_->n_misc_->nr_ > 1) ? itctx_->n_misc_->rho_ * itctx_->n_misc_->r_gm_wm_ratio_  : 0;  // equals x_in_ptr[np + nk + 1]
    r3 = (itctx_->n_misc_->nr_ > 2) ? itctx_->n_misc_->rho_ * itctx_->n_misc_->r_glm_wm_ratio_ : 0;  // equals x_in_ptr[np + nk + 2]
    k1 = itctx_->n_misc_->k_;                                                                        // equals x_in_ptr[np];
    k2 = (itctx_->n_misc_->nk_ > 1) ? itctx_->n_misc_->k_   * itctx_->n_misc_->k_gm_wm_ratio_  : 0;  // equals x_in_ptr[np+1];
    k3 = (itctx_->n_misc_->nk_ > 2) ? itctx_->n_misc_->k_   * itctx_->n_misc_->k_glm_wm_ratio_ : 0;  // equals x_in_ptr[np+2];

    ierr = itctx_->tumor_->k_->updateIsotropicCoefficients (k1, k2, k3, itctx_->tumor_->mat_prop_, itctx_->n_misc_);    CHKERRQ (ierr);
    ierr = itctx_->tumor_->rho_->updateIsotropicCoefficients (r1, r2, r3, itctx_->tumor_->mat_prop_, itctx_->n_misc_);  CHKERRQ (ierr);

    if (itctx_->cosamp_->cosamp_stage == PRE_RD) {
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
    } else if (itctx_->cosamp_->cosamp_stage == POST_RD) {
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
    if (itctx_->x_old != nullptr) {ierr = VecDestroy (&itctx_->x_old);  CHKERRQ (ierr); itctx_->x_old = nullptr;}
    if (noise != nullptr)         {ierr = VecDestroy (&noise);          CHKERRQ (ierr);         noise = nullptr;}
    // go home
    PetscFunctionReturn (ierr);
}




PetscErrorCode InvSolver::solveForMassEffect () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    TU_assert (initialized_,              "InvSolver::solve (): InvSolver needs to be initialized.")
    TU_assert (data_ != nullptr,          "InvSolver::solve (): requires non-null input data for inversion.");
    TU_assert (data_gradeval_ != nullptr, "InvSolver::solve (): requires non-null input data for gradient evaluation.");
    TU_assert (xrec_ != nullptr,          "InvSolver::solve (): requires non-null p_rec vector to be set");
    TU_assert (optsettings_ != nullptr,   "InvSolver::solve (): requires non-null optimizer settings to be passed.");

    std::stringstream s;
    PetscScalar max, min, w = 1, p_max, xdiff;
    PetscScalar d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0.;
    PetscScalar *d_ptr, *noise_ptr, *p_ptr, *w_ptr;
    TaoConvergedReason reason;
    Vec noise;

    /* === Add Noise === */
    ierr = VecCreate (PETSC_COMM_WORLD, &noise);                                CHKERRQ(ierr);
    ierr = setupVec  (noise);                                                   CHKERRQ (ierr);
    ierr = VecSetSizes(noise, itctx_->n_misc_->n_local_, itctx_->n_misc_->n_global_);CHKERRQ(ierr);
    ierr = VecSetFromOptions(noise);                                            CHKERRQ(ierr);
    ierr = VecSetRandom(noise, NULL);                                           CHKERRQ(ierr);
    ierr = VecGetArray (noise, &noise_ptr);                                     CHKERRQ(ierr);
    ierr = VecGetArray (data_, &d_ptr);                                         CHKERRQ(ierr);
    for (int i = 0; i < itctx_->n_misc_->n_local_; i++) {
      d_ptr[i] += noise_ptr[i] * itctx_->n_misc_->noise_scale_;
      noise_ptr[i] = d_ptr[i];     //just to measure d norm
    }
    ierr = VecRestoreArray (noise, &noise_ptr);                                 CHKERRQ(ierr);
    ierr = VecRestoreArray (data_, &d_ptr);                                     CHKERRQ(ierr);
    #ifdef POSITIVITY
    ierr = enforcePositivity (data_, itctx_->n_misc_);
    ierr = enforcePositivity (noise, itctx_->n_misc_);
    #endif
    ierr = VecNorm (noise, NORM_2, &d_norm);                                    CHKERRQ(ierr);
    ierr = VecMax  (noise, NULL, &max);                                         CHKERRQ(ierr);
    ierr = VecMin  (noise, NULL, &min);                                         CHKERRQ(ierr);
    ierr = VecAXPY (noise, -1.0, data_);                                        CHKERRQ(ierr);
    ierr = VecNorm (noise, NORM_2, &d_errorl2norm);                             CHKERRQ(ierr);
    ierr = VecNorm (noise, NORM_INFINITY, &d_errorInfnorm);                     CHKERRQ(ierr);
    s << " tumor inversion target data (with noise): l2norm = "<< d_norm <<" [max: "<<max<<", min: "<<min<<"]";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    s << " tumor inversion target data error (due to thresholding and smoothing): l2norm = "<< d_errorl2norm <<", inf-norm = " <<d_errorInfnorm;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    if (itctx_->n_misc_->writeOutput_) { dataOut (data_, itctx_->n_misc_, "data.nc"); }

    /* === initialize inverse tumor context === */
    if (itctx_->c0old == nullptr) {
      // ierr = VecDuplicate (data_, &itctx_->c0old);                              CHKERRQ(ierr);
      // ierr = VecSet (itctx_->c0old, 0.0);                                       CHKERRQ(ierr);
    }
    if (itctx_->tmp == nullptr) {
      // ierr = VecDuplicate (data_, &itctx_->tmp);                                CHKERRQ(ierr);
      // ierr = VecSet (itctx_->tmp, 0.0);                                         CHKERRQ(ierr);
    }
    if (itctx_->x_old == nullptr)  {
      ierr = VecDuplicate (xrec_, &itctx_->x_old);                              CHKERRQ (ierr);
      ierr = VecCopy (xrec_, itctx_->x_old);                                    CHKERRQ (ierr);
    }
    // initialize with zero; fresh solve
    // ierr = VecSet (itctx_->c0old, 0.0);                                         CHKERRQ(ierr);
    // set beta for this inverse solver call
    if (itctx_->n_misc_->beta_changed_)
        itctx_->optsettings_->beta   = itctx_->n_misc_->beta_;
    else
        itctx_->n_misc_->beta_       = itctx_->optsettings_->beta;
    // reset opt solver statistics
    itctx_->is_ksp_gradnorm_set        = false;
    itctx_->optfeedback_->converged    = false;
    itctx_->optfeedback_->solverstatus = "";
    itctx_->optfeedback_->nb_newton_it = 0;
    itctx_->optfeedback_->nb_krylov_it = 0;
    itctx_->optfeedback_->nb_matvecs   = 0;
    itctx_->optfeedback_->nb_objevals  = 0;
    itctx_->optfeedback_->nb_gradevals = 0;
    itctx_->data                       = data_;
    itctx_->data_gradeval              = data_gradeval_;
    // reset tao, if we want virgin TAO for every inverse solve
    if (itctx_->optsettings_->reset_tao) {
        ierr = resetTao(itctx_->n_misc_);                                       CHKERRQ(ierr);
    }

    /* === set TAO options === */
    if (tao_is_reset_) {
        ierr = setTaoOptionsMassEffect (tao_, itctx_.get());                                          CHKERRQ(ierr);
        ierr = TaoSetHessianRoutine (tao_, H_, H_, matfreeHessian, (void *) itctx_.get());  CHKERRQ(ierr);
    }

    s << " using tumor regularization = "<< itctx_->n_misc_->beta_ << " type: " << itctx_->n_misc_->regularization_norm_;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    if (itctx_->n_misc_->verbosity_ >= 2) { itctx_->n_misc_->outfile_sol_  << "\n ## ----- ##" << std::endl << std::flush; itctx_->n_misc_->outfile_grad_ << "\n ## ----- ## "<< std::endl << std::flush; }
    //Gradient check begin
    //    ierr = itctx_->derivative_operators_->checkGradient (itctx_->tumor_->p_, itctx_->data);
    //Gradient check end

    /* === solve === */
    itctx_->n_misc_->statistics_.reset();
    double self_exec_time_tuninv = -MPI_Wtime(); double invtime = 0;
    ierr = TaoSolve (tao_);                                                       CHKERRQ(ierr);
    self_exec_time_tuninv += MPI_Wtime();
    MPI_Reduce(&self_exec_time_tuninv, &invtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* === get solution === */
    Vec p; ierr = TaoGetSolutionVector (tao_, &p);                                CHKERRQ(ierr);
    ierr = VecCopy (p, xrec_);                                                    CHKERRQ(ierr);
    
    PetscScalar *x_ptr;
    ierr = VecGetArray (xrec_, &x_ptr);                                 CHKERRQ (ierr);
    itctx_->n_misc_->forcing_factor_ = 1E5 * x_ptr[0]; // re-scaling parameter scales
    itctx_->n_misc_->rho_ = 10 * x_ptr[1];                  // rho
    itctx_->n_misc_->k_   = 1E-1 * x_ptr[2];                  // kappa
    ierr = VecRestoreArray (xrec_, &x_ptr);                             CHKERRQ (ierr);

    s << " Forcing factor at final guess = " << itctx_->n_misc_->forcing_factor_; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    s << " Reaction at final guess       = " << itctx_->n_misc_->rho_; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    s << " Diffusivity at final guess    = " << itctx_->n_misc_->k_; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
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
    ierr = TaoGetSolutionStatus (tao_, NULL, &itctx_->optfeedback_->jval, &itctx_->optfeedback_->gradnorm, NULL, &xdiff, NULL); CHKERRQ(ierr);
    /* display convergence reason: */
    ierr = dispTaoConvReason (reason, itctx_->optfeedback_->solverstatus);        CHKERRQ(ierr);
    itctx_->optfeedback_->nb_newton_it--;
    s << " optimization done: #N-it: " << itctx_->optfeedback_->nb_newton_it    << ", #K-it: " << itctx_->optfeedback_->nb_krylov_it
                      << ", #matvec: " << itctx_->optfeedback_->nb_matvecs    << ", #evalJ: " << itctx_->optfeedback_->nb_objevals
                      << ", #evaldJ: " << itctx_->optfeedback_->nb_gradevals  << ", exec time: " << invtime;
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGstd (s.str());                                                                                            CHKERRQ(ierr);  s.str(""); s.clear();
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    itctx_->n_misc_->statistics_.print();
    itctx_->n_misc_->statistics_.reset();
    out_params_.push_back (itctx_->optfeedback_->nb_newton_it);
    // only update if triggered from outside, i.e., if new information to the ITP solver is present
    itctx_->update_reference_gradient = false;
    // reset vectors (remember, memory managed on caller side):
    // data_ = nullptr;
    // data_gradeval_ = nullptr;
    tao_is_reset_ = false;

    if (itctx_->x_old != nullptr) {ierr = VecDestroy (&itctx_->x_old);  CHKERRQ (ierr); itctx_->x_old = nullptr;}
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
PetscErrorCode InvSolver::solve () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    TU_assert (initialized_,              "InvSolver::solve (): InvSolver needs to be initialized.")
    TU_assert (data_ != nullptr,          "InvSolver::solve (): requires non-null input data for inversion.");
    TU_assert (data_gradeval_ != nullptr, "InvSolver::solve (): requires non-null input data for gradient evaluation.");
    TU_assert (xrec_ != nullptr,          "InvSolver::solve (): requires non-null p_rec vector to be set");
    TU_assert (optsettings_ != nullptr,   "InvSolver::solve (): requires non-null optimizer settings to be passed.");

    std::stringstream s;
    PetscScalar max, min, w = 1, p_max, xdiff;
    PetscScalar d_norm = 0., d_errorl2norm = 0., d_errorInfnorm = 0.;
    PetscScalar *d_ptr, *noise_ptr, *p_ptr, *w_ptr;
    TaoConvergedReason reason;
    Vec noise;

    /* === Add Noise === */
    ierr = VecCreate (PETSC_COMM_WORLD, &noise);                                CHKERRQ(ierr);
    ierr = setupVec  (noise);                                                   CHKERRQ (ierr);
    ierr = VecSetSizes(noise, itctx_->n_misc_->n_local_, itctx_->n_misc_->n_global_);CHKERRQ(ierr);
    ierr = VecSetFromOptions(noise);                                            CHKERRQ(ierr);
    ierr = VecSetRandom(noise, NULL);                                           CHKERRQ(ierr);
    ierr = VecGetArray (noise, &noise_ptr);                                     CHKERRQ(ierr);
    ierr = VecGetArray (data_, &d_ptr);                                         CHKERRQ(ierr);
    for (int i = 0; i < itctx_->n_misc_->n_local_; i++) {
      d_ptr[i] += noise_ptr[i] * itctx_->n_misc_->noise_scale_;
      noise_ptr[i] = d_ptr[i];     //just to measure d norm
    }
    ierr = VecRestoreArray (noise, &noise_ptr);                                 CHKERRQ(ierr);
    ierr = VecRestoreArray (data_, &d_ptr);                                     CHKERRQ(ierr);
    #ifdef POSITIVITY
    ierr = enforcePositivity (data_, itctx_->n_misc_);
    ierr = enforcePositivity (noise, itctx_->n_misc_);
    #endif
    ierr = VecNorm (noise, NORM_2, &d_norm);                                    CHKERRQ(ierr);
    ierr = VecMax  (noise, NULL, &max);                                         CHKERRQ(ierr);
    ierr = VecMin  (noise, NULL, &min);                                         CHKERRQ(ierr);
    ierr = VecAXPY (noise, -1.0, data_);                                        CHKERRQ(ierr);
    ierr = VecNorm (noise, NORM_2, &d_errorl2norm);                             CHKERRQ(ierr);
    ierr = VecNorm (noise, NORM_INFINITY, &d_errorInfnorm);                     CHKERRQ(ierr);
    s << " tumor inversion target data (with noise): l2norm = "<< d_norm <<" [max: "<<max<<", min: "<<min<<"]";  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    s << " tumor inversion target data error (due to thresholding and smoothing): l2norm = "<< d_errorl2norm <<", inf-norm = " <<d_errorInfnorm;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    if (itctx_->n_misc_->writeOutput_) { dataOut (data_, itctx_->n_misc_, "data.nc"); }

    /* === initialize inverse tumor context === */
    if (itctx_->c0old == nullptr) {
      ierr = VecDuplicate (data_, &itctx_->c0old);                              CHKERRQ(ierr);
      ierr = VecSet (itctx_->c0old, 0.0);                                       CHKERRQ(ierr);
    }
    if (itctx_->tmp == nullptr) {
      ierr = VecDuplicate (data_, &itctx_->tmp);                                CHKERRQ(ierr);
      ierr = VecSet (itctx_->tmp, 0.0);                                         CHKERRQ(ierr);
    }
    if (itctx_->x_old == nullptr)  {
      ierr = VecDuplicate (itctx_->tumor_->p_, &itctx_->x_old);                 CHKERRQ (ierr);
      ierr = VecCopy (itctx_->tumor_->p_, itctx_->x_old);                       CHKERRQ (ierr);
    }
    // initialize with zero; fresh solve
    ierr = VecSet (itctx_->c0old, 0.0);                                         CHKERRQ(ierr);
    // set beta for this inverse solver call
    if (itctx_->n_misc_->beta_changed_)
        itctx_->optsettings_->beta   = itctx_->n_misc_->beta_;
    else
        itctx_->n_misc_->beta_       = itctx_->optsettings_->beta;
    // reset opt solver statistics
    itctx_->is_ksp_gradnorm_set        = false;
    itctx_->optfeedback_->converged    = false;
    itctx_->optfeedback_->solverstatus = "";
    itctx_->optfeedback_->nb_newton_it = 0;
    itctx_->optfeedback_->nb_krylov_it = 0;
    itctx_->optfeedback_->nb_matvecs   = 0;
    itctx_->optfeedback_->nb_objevals  = 0;
    itctx_->optfeedback_->nb_gradevals = 0;
    itctx_->data                       = data_;
    itctx_->data_gradeval              = data_gradeval_;
    // reset tao, if we want virgin TAO for every inverse solve
    if (itctx_->optsettings_->reset_tao) {
        ierr = resetTao(itctx_->n_misc_);                                       CHKERRQ(ierr);
    }

    // if weighted L2 solve, set the weights
    if (itctx_->n_misc_->regularization_norm_ == wL2) {
        itctx_->n_misc_->beta_ = 1.;
        ierr = VecGetArray (itctx_->tumor_->p_, &p_ptr);                        CHKERRQ (ierr);
        ierr = VecSet (itctx_->weights, 0.0);                                   CHKERRQ (ierr);
        ierr = VecGetArray (itctx_->weights, &w_ptr);                           CHKERRQ (ierr);
        ierr = VecMax (itctx_->tumor_->p_, NULL, &p_max);                       CHKERRQ (ierr);
        for (int i = 0; i < itctx_->n_misc_->np_; i++)
            w_ptr[i] =  (p_ptr[i] > 0.1 * p_max) ? 0 : w;
        ierr = VecRestoreArray (itctx_->tumor_->p_, &p_ptr);                    CHKERRQ (ierr);
        ierr = VecRestoreArray (itctx_->weights, &w_ptr);                       CHKERRQ (ierr);
        ierr = VecCopy (itctx_->weights, itctx_->tumor_->weights_);             CHKERRQ (ierr);
    }

    /* === set TAO options === */
    if (tao_is_reset_) {
        // itctx_->update_reference_gradient = true;   // TODO: K: I commented this; for CoSaMp_RS we don't want t ore-compute reference gradient between inexact blocks (if coupled with sibia, the data will change)
        // itctx_->update_reference_objective = true;  // TODO: K: I commented this; for CoSaMp_RS we don't want t ore-compute reference gradient between inexact blocks (if coupled with sibia, the data will change)
        ierr = setTaoOptions (tao_, itctx_.get());                              CHKERRQ(ierr);
        if ((itctx_->optsettings_->newtonsolver == QUASINEWTON) &&
            itctx_->optsettings_->lmvm_set_hessian) {
            ierr = TaoLMVMSetH0 (tao_, H_);                                     CHKERRQ(ierr);
        } else {
            ierr = TaoSetHessianRoutine (tao_, H_, H_, matfreeHessian, (void *) itctx_.get());CHKERRQ(ierr);
        }
    }

    s << " using tumor regularization = "<< itctx_->n_misc_->beta_ << " type: " << itctx_->n_misc_->regularization_norm_;  ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    if (itctx_->n_misc_->verbosity_ >= 2) { itctx_->n_misc_->outfile_sol_  << "\n ## ----- ##" << std::endl << std::flush; itctx_->n_misc_->outfile_grad_ << "\n ## ----- ## "<< std::endl << std::flush; }
    //Gradient check begin
    //    ierr = itctx_->derivative_operators_->checkGradient (itctx_->tumor_->p_, itctx_->data);
    //Gradient check end

    /* === solve === */
    itctx_->n_misc_->statistics_.reset();
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
    ierr = TaoGetSolutionStatus (tao_, NULL, &itctx_->optfeedback_->jval, &itctx_->optfeedback_->gradnorm, NULL, &xdiff, NULL); CHKERRQ(ierr);
    /* display convergence reason: */
    ierr = dispTaoConvReason (reason, itctx_->optfeedback_->solverstatus);        CHKERRQ(ierr);
    itctx_->optfeedback_->nb_newton_it--;
    s << " optimization done: #N-it: " << itctx_->optfeedback_->nb_newton_it    << ", #K-it: " << itctx_->optfeedback_->nb_krylov_it
                      << ", #matvec: " << itctx_->optfeedback_->nb_matvecs    << ", #evalJ: " << itctx_->optfeedback_->nb_objevals
                      << ", #evaldJ: " << itctx_->optfeedback_->nb_gradevals  << ", exec time: " << invtime;
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGstd (s.str());                                                                                            CHKERRQ(ierr);  s.str(""); s.clear();
    ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    itctx_->n_misc_->statistics_.print();
    itctx_->n_misc_->statistics_.reset();
    out_params_.push_back (itctx_->optfeedback_->nb_newton_it);
    // only update if triggered from outside, i.e., if new information to the ITP solver is present
    itctx_->update_reference_gradient = false;
    // reset vectors (remember, memory managed on caller side):
    // data_ = nullptr;
    // data_gradeval_ = nullptr;
    tao_is_reset_ = false;

    if (itctx_->x_old != nullptr) {ierr = VecDestroy (&itctx_->x_old);  CHKERRQ (ierr); itctx_->x_old = nullptr;}
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
    int np_full = itctx_->cosamp_->np_full;

    ierr = tuMSG(" >> entering inverse CoSaMp"); CHKERRQ(ierr);
    switch(itctx_->cosamp_->cosamp_stage) {
        // ================
        case INIT:
            ierr = tuMSG(" >> entering stage INIT"); CHKERRQ(ierr); ss.str(""); ss.clear();
            itctx_->cosamp_->np_full = itctx_->n_misc_->np_; // store np of unrestricted ansatz space
            np_full = itctx_->cosamp_->np_full;
            itctx_->cosamp_->converged_l1 = false;
            itctx_->cosamp_->converged_l2 = false;
            itctx_->cosamp_->f_tol = 1E-5;
            ierr = itctx_->cosamp_->cleanup();                                  CHKERRQ (ierr);
            /* allocate vecs and copy initial guess for p */
            ierr = itctx_->cosamp_->initialize(itctx_->tumor_->p_);             CHKERRQ (ierr);
            // no break; go into next case
            itctx_->cosamp_->cosamp_stage = PRE_RD;
            ierr = tuMSG(" << leaving stage INIT"); CHKERRQ(ierr);

        // ================
        // this case is executed at once without going back to caller in between
        case PRE_RD:
            /* ------------------------------------------------------------------------ */
            // ### (0) (pre-)reaction/diffusion inversion ###
            ierr = tuMSG(" >> entering stage PRE_RD"); CHKERRQ(ierr);
            if (itctx_->n_misc_->pre_reacdiff_solve_ && itctx_->n_misc_->n_[0] > 64) {
              if (itctx_->n_misc_->reaction_inversion_) {
                  // == restrict == to new L2 subspace, holding p_i, kappa, and rho
                  ierr = restrictSubspace(&itctx_->cosamp_->x_sub, itctx_->cosamp_->x_full, itctx_, true);      CHKERRQ (ierr); // x_sub <-- R(x_full)
                  itctx_->cosamp_->cosamp_stage = PRE_RD;
                  itctx_->optsettings_->newton_maxit = itctx_->cosamp_->maxit_newton;
                  // == solve ==
                  ierr = solveInverseReacDiff (itctx_->cosamp_->x_sub);          /* with current guess as init cond. */
                  ierr = VecCopy (getPrec(), itctx_->cosamp_->x_sub);            /* get solution */             CHKERRQ (ierr);
                  // == prolongate ==
                  ierr = prolongateSubspace(itctx_->cosamp_->x_full, &itctx_->cosamp_->x_sub, itctx_, np_full); CHKERRQ (ierr); // x_full <-- P(x_sub)
              }
          } else {ierr = tuMSGstd("    ... skipping stage, reaction diffusion disabled."); CHKERRQ(ierr);}
            // no break; go into next case
            itctx_->cosamp_->cosamp_stage = COSAMP_L1_INIT;
            ierr = tuMSG(" << leaving stage PRE_RD"); CHKERRQ(ierr);

        // ================
        // setting up L1-pahse, computing reference gradeint, and print statistics
        case COSAMP_L1_INIT:
            ierr = tuMSG(" >> entering stage COSAMP_L1_INIT"); CHKERRQ(ierr);
            // set initial guess for k_inv (possibly != zero)
            ierr = VecGetArray(itctx_->cosamp_->x_full, &x_full_ptr);                                            CHKERRQ (ierr);
            if (itctx_->n_misc_->diffusivity_inversion_) x_full_ptr[np_full] = itctx_->n_misc_->k_;
            else { // set diff ops with this guess -- this will not change during the solve
                ierr = itctx_->tumor_->k_->setValues (itctx_->n_misc_->k_, itctx_->n_misc_->k_gm_wm_ratio_, itctx_->n_misc_->k_glm_wm_ratio_, itctx_->tumor_->mat_prop_, itctx_->n_misc_);  CHKERRQ (ierr);
            }
            ierr = VecRestoreArray(itctx_->cosamp_->x_full, &x_full_ptr);                                        CHKERRQ (ierr);
            ierr = VecCopy        (itctx_->cosamp_->x_full, itctx_->cosamp_->x_full_prev);                       CHKERRQ (ierr);

            // compute reference value for  objective
            beta_store = itctx_->n_misc_->beta_; itctx_->n_misc_->beta_ = 0.; // set beta to zero for gradient thresholding
            ierr = getObjectiveAndGradient (itctx_->cosamp_->x_full, &itctx_->cosamp_->J_ref, itctx_->cosamp_->g);CHKERRQ (ierr);
            itctx_->n_misc_->beta_ = beta_store;
            ierr = VecNorm (itctx_->cosamp_->g, NORM_2, &itctx_->cosamp_->g_norm);                                CHKERRQ (ierr);
            itctx_->cosamp_->J = itctx_->cosamp_->J_ref;

            // print statistics
            ierr = printStatistics (itctx_->cosamp_->its_l1, itctx_->cosamp_->J_ref, 1, itctx_->cosamp_->g_norm, 1, itctx_->cosamp_->x_full); CHKERRQ(ierr);
            // number of connected components
            itctx_->tumor_->phi_->num_components_ = itctx_->tumor_->phi_->component_weights_.size ();
            // output warmstart (injection) support
            ss << "starting CoSaMP solver with initial support: ["; for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->n_misc_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            ss << "component label of initial support : [";         for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->tumor_->phi_->gaussian_labels_[itctx_->n_misc_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

            // no break; go into next case
            itctx_->cosamp_->cosamp_stage = COSAMP_L1_THRES_GRAD;
            ierr = tuMSG(" << leaving stage COSAMP_L1_INIT"); CHKERRQ(ierr);

        // ================
        // thresholding the gradient, restrict subspace
        case COSAMP_L1_THRES_GRAD:
            ierr = tuMSG(" >> entering stage COSAMP_L1_THRES_GRAD"); CHKERRQ(ierr);
            itctx_->cosamp_->its_l1++;
            /* === hard threshold abs gradient === */
            ierr = VecCopy (itctx_->cosamp_->g, itctx_->cosamp_->work);                                            CHKERRQ (ierr);
            ierr = VecAbs  (itctx_->cosamp_->work);                                                                CHKERRQ (ierr);
            // print gradient to file
            if (itctx_->n_misc_->verbosity_ >= 2) {
              ierr = VecGetArray(itctx_->cosamp_->work, &grad_ptr);                                                CHKERRQ(ierr);
              for (int i = 0; i < np_full-1; i++) if(procid == 0) itctx_->n_misc_->outfile_glob_grad_ << grad_ptr[i] << ", ";
              if(procid == 0)                                     itctx_->n_misc_->outfile_glob_grad_ << grad_ptr[np_full-1] << ";\n" <<std::endl;
              ierr = VecRestoreArray(itctx_->cosamp_->work, &grad_ptr);                                            CHKERRQ(ierr);
            }
            idx.clear();
            ierr = hardThreshold (itctx_->cosamp_->work, 2 * itctx_->n_misc_->sparsity_level_, np_full, idx, itctx_->tumor_->phi_->gaussian_labels_, itctx_->tumor_->phi_->component_weights_, nnz, itctx_->tumor_->phi_->num_components_);

            /* === update support of prev. solution with new support === */
            itctx_->n_misc_->support_.insert (itctx_->n_misc_->support_.end(), idx.begin(), idx.end());
            // sort and remove duplicates
            std::sort (itctx_->n_misc_->support_.begin(), itctx_->n_misc_->support_.end());
            itctx_->n_misc_->support_.erase (std::unique (itctx_->n_misc_->support_.begin(), itctx_->n_misc_->support_.end()), itctx_->n_misc_->support_.end());
            // print out
            ss << "support for corrective L2 solve : ["; for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->n_misc_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            ss << "component label of support : [";      for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->tumor_->phi_->gaussian_labels_[itctx_->n_misc_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

            // no break; go into next case
            itctx_->cosamp_->cosamp_stage = COSAMP_L1_SOLVE_SUBSPACE;
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
            ierr = restrictSubspace(&itctx_->cosamp_->x_sub, itctx_->cosamp_->x_full, itctx_); CHKERRQ (ierr); // x_L2 <-- R(x_L1)
            // print vec
            if (procid == 0 && itctx_->n_misc_->verbosity_ >= 4) { ierr = VecView (itctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

            // solve interpolation
            // ierr = solveInterpolation (data_);                                   CHKERRQ (ierr);

            itctx_->optsettings_->newton_maxit = itctx_->cosamp_->inexact_nits;
            // only update reference gradient and referenc objective if this is the first inexact solve for this subspace, otherwise don't
            itctx_->update_reference_gradient  = (itctx_->cosamp_->nits < itctx_->cosamp_->inexact_nits);
            itctx_->update_reference_objective = (itctx_->cosamp_->nits < itctx_->cosamp_->inexact_nits);
            // == solve ==
            ierr = solve ();                                                        CHKERRQ (ierr);
            ierr = VecCopy (getPrec(), itctx_->cosamp_->x_sub);                     CHKERRQ (ierr);
            ierr = tuMSG("### ----------------------------------------- L2 solver end --------------------------------------------- ###");CHKERRQ (ierr);
            ierr = tuMSGstd ("");                                                   CHKERRQ (ierr);
            ierr = VecCopy (itctx_->cosamp_->x_full, itctx_->cosamp_->x_full_prev); CHKERRQ (ierr);
            // print support
            ierr = VecDuplicate (itctx_->tumor_->phi_->phi_vec_[0], &all_phis);     CHKERRQ (ierr);
            ierr = VecSet (all_phis, 0.);                                           CHKERRQ (ierr);
            for (int i = 0; i < itctx_->n_misc_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, itctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
            ss << "phiSupport_csitr-" << itctx_->cosamp_->its_l1 << ".nc";
            if (itctx_->n_misc_->writeOutput_) dataOut (all_phis, itctx_->n_misc_, ss.str().c_str()); ss.str(""); ss.clear();
            if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}
            if (procid == 0 && itctx_->n_misc_->verbosity_ >= 4) { ierr = VecView (itctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

            // == prolongate ==
            ierr = prolongateSubspace(itctx_->cosamp_->x_full, &itctx_->cosamp_->x_sub, itctx_, np_full); CHKERRQ (ierr); // x_L1 <-- P(x_L2)

            // == convergence test ==
            // neither gradient sufficiently small nor ls-failure (i.e., inexact_nit hit)
            // if(!itctx_->cosamp_->converged_l2 && !itctx_->cosamp_->converged_l2) {itctx_->cosamp_->nits += itctx_->cosamp_->inexact_nits;}
            itctx_->cosamp_->nits += itctx_->optfeedback_->nb_newton_it;
            conv_maxit = itctx_->cosamp_->nits >= itctx_->cosamp_->maxit_newton;
            // check if L2 solver converged
            if(!itctx_->cosamp_->converged_l2 && !itctx_->cosamp_->converged_error_l2 && !conv_maxit) {
                ss << "    ... inexact solve terminated (L2 solver not converged, will be continued; its "<< itctx_->cosamp_->nits <<"/"<< itctx_->cosamp_->maxit_newton <<").";
                ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();
                ierr = tuMSG(" << leaving stage COSAMP_L1_SOLVE_SUBSPACE");
                break;
            } else {
                // if L2 solver converged
                if(itctx_->cosamp_->converged_l2)        {ss << "    ... L2 solver converged; its "<< itctx_->cosamp_->nits <<"/"<< itctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
                // if L2 solver ran into ls-failure
                if(itctx_->cosamp_->converged_error_l2)  {ss << "    ... L2 solver terminated (ls-failure); its "<< itctx_->cosamp_->nits <<"/"<< itctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
                // if L2 solver hit maxit
                if(conv_maxit)                           {ss << "    ... L2 solver terminated (maxit); its "<< itctx_->cosamp_->nits <<"/"<< itctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
                itctx_->cosamp_->cosamp_stage = COSAMP_L1_THRES_SOL;
                conv_maxit = itctx_->cosamp_->nits = 0;
                ierr = tuMSG(" << leaving stage COSAMP_L1_SOLVE_SUBSPACE"); CHKERRQ(ierr);
            }


        // ================
        // thresholding the gradient, restrict subspace
        case COSAMP_L1_THRES_SOL:
            ierr = tuMSG(" >> entering stage COSAMP_L1_THRES_SOL"); CHKERRQ(ierr);
            /* === hard threshold solution to sparsity level === */
            idx.clear();
            if (itctx_->n_misc_->prune_components_) hardThreshold (itctx_->cosamp_->x_full, itctx_->n_misc_->sparsity_level_, np_full, idx, itctx_->tumor_->phi_->gaussian_labels_, itctx_->tumor_->phi_->component_weights_, nnz, itctx_->tumor_->phi_->num_components_);
            else                                    hardThreshold (itctx_->cosamp_->x_full, itctx_->n_misc_->sparsity_level_, np_full, idx, nnz);
            itctx_->n_misc_->support_.clear ();
            itctx_->n_misc_->support_ = idx;
            // sort and remove duplicates
            std::sort (itctx_->n_misc_->support_.begin(), itctx_->n_misc_->support_.end());
            itctx_->n_misc_->support_.erase (std::unique (itctx_->n_misc_->support_.begin(), itctx_->n_misc_->support_.end()), itctx_->n_misc_->support_.end());
            // print out
            ss << "support after hard thresholding the solution : ["; for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->n_misc_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
            ss << "component label of support : ["; for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->tumor_->phi_->gaussian_labels_[itctx_->n_misc_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

            // set only support values in x_L1 (rest hard thresholded to zero)
            ierr = VecCopy (itctx_->cosamp_->x_full, itctx_->cosamp_->work);        CHKERRQ (ierr);
            ierr = VecSet  (itctx_->cosamp_->x_full, 0.0);                          CHKERRQ (ierr);
            ierr = VecGetArray (itctx_->cosamp_->work, &x_work_ptr);                CHKERRQ (ierr);
            ierr = VecGetArray (itctx_->cosamp_->x_full, &x_full_ptr);              CHKERRQ (ierr);
            for (int i = 0; i < itctx_->n_misc_->support_.size(); i++)
                x_full_ptr[itctx_->n_misc_->support_[i]] = x_work_ptr[itctx_->n_misc_->support_[i]];
            if (itctx_->n_misc_->diffusivity_inversion_) {
                x_full_ptr[np_full] = x_work_ptr[np_full];
                if (itctx_->n_misc_->nk_ > 1) x_full_ptr[np_full+1] = x_work_ptr[np_full+1];
                if (itctx_->n_misc_->nk_ > 2) x_full_ptr[np_full+2] = x_work_ptr[np_full+2];
            }
            ierr = VecRestoreArray (itctx_->cosamp_->x_full, &x_full_ptr);          CHKERRQ (ierr);
            ierr = VecRestoreArray (itctx_->cosamp_->work, &x_work_ptr);              CHKERRQ (ierr);
            /* copy initial guess for p */
            ierr = VecCopy (itctx_->cosamp_->x_full, itctx_->tumor_->p_);           CHKERRQ (ierr);

            // print initial guess to file
            if (itctx_->n_misc_->writeOutput_) {
              ss << "c0guess_csitr-" << itctx_->cosamp_->its_l1 << ".nc";  dataOut (itctx_->tumor_->c_0_, itctx_->n_misc_, ss.str().c_str()); ss.str(std::string()); ss.clear();
              ss << "c1guess_csitr-" << itctx_->cosamp_->its_l1 << ".nc"; if (itctx_->n_misc_->verbosity_ >= 4) dataOut (itctx_->tumor_->c_t_, itctx_->n_misc_, ss.str().c_str()); ss.str(std::string()); ss.clear();
            }

            /* === convergence check === */
            itctx_->cosamp_->J_prev = itctx_->cosamp_->J;
            // compute objective (only mismatch term)
            beta_store = itctx_->n_misc_->beta_; itctx_->n_misc_->beta_ = 0.; // set beta to zero for gradient thresholding
            ierr = getObjectiveAndGradient (itctx_->cosamp_->x_full, &itctx_->cosamp_->J, itctx_->cosamp_->g);   CHKERRQ (ierr);
            itctx_->n_misc_->beta_ = beta_store;
            ierr = VecNorm (itctx_->cosamp_->x_full, NORM_INFINITY, &norm);                                      CHKERRQ (ierr);
            ierr = VecAXPY (itctx_->cosamp_->work, -1.0, itctx_->cosamp_->x_full_prev);  /* holds x_L1 */        CHKERRQ (ierr);
            ierr = VecNorm (itctx_->cosamp_->work, NORM_INFINITY, &norm_rel);            /*norm change in sol */ CHKERRQ (ierr);
            ierr = VecNorm (itctx_->cosamp_->g, NORM_2, &itctx_->cosamp_->g_norm);                               CHKERRQ (ierr);
            // solver status
            ierr = tuMSGstd (""); CHKERRQ(ierr); ierr = tuMSGstd (""); CHKERRQ(ierr);
            ierr = tuMSGstd ("--------------------------------------------- L1 solver statistics -------------------------------------------"); CHKERRQ(ierr);
            ierr = printStatistics (itctx_->cosamp_->its_l1, itctx_->cosamp_->J, PetscAbsReal (itctx_->cosamp_->J_prev - itctx_->cosamp_->J) / PetscAbsReal (1 + itctx_->cosamp_->J_ref), itctx_->cosamp_->g_norm, norm_rel / (1 + norm), itctx_->cosamp_->x_full); CHKERRQ(ierr);
            ierr = tuMSGstd ("--------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
            ierr = tuMSGstd (""); CHKERRQ(ierr);
            if (itctx_->cosamp_->its_l1 >= optsettings_->gist_maxit) {ierr = tuMSGwarn (" L1 maxiter reached"); CHKERRQ(ierr); itctx_->cosamp_->converged_l1 = true;}
            else if (PetscAbsReal (itctx_->cosamp_->J) < 1E-5)       {ierr = tuMSGwarn (" L1 absolute objective tolerance reached."); CHKERRQ(ierr); itctx_->cosamp_->converged_l1 = true;}
            else if (PetscAbsReal (itctx_->cosamp_->J_prev - itctx_->cosamp_->J) < itctx_->cosamp_->f_tol * PetscAbsReal (1 + itctx_->cosamp_->J_ref)) {ierr = tuMSGwarn (" L1 relative objective tolerance reached."); CHKERRQ(ierr); itctx_->cosamp_->converged_l1 = true;}
            else { itctx_->cosamp_->converged_l1 = false; }  // continue iterating

            ierr = tuMSG(" << leaving stage COSAMP_L1_THRES_SOL"); CHKERRQ(ierr);
            if(itctx_->cosamp_->converged_l1) {              // no break; go into next case
                itctx_->cosamp_->cosamp_stage = FINAL_L2;
            } else{                                          // break; continue iterating
                itctx_->cosamp_->cosamp_stage = COSAMP_L1_THRES_GRAD; break;
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

            ierr = restrictSubspace(&itctx_->cosamp_->x_sub, itctx_->cosamp_->x_full, itctx_); CHKERRQ (ierr); // x_sub <-- R(x_full)

            // print vec
            if (procid == 0 && itctx_->n_misc_->verbosity_ >= 4) { ierr = VecView (itctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

            // solve interpolation
            // ierr = solveInterpolation (data_);                                        CHKERRQ (ierr);
            itctx_->optsettings_->newton_maxit = itctx_->cosamp_->inexact_nits;
            // only update reference gradient and referenc objective if this is the first inexact solve for this subspace, otherwise don't
            itctx_->update_reference_gradient  = (itctx_->cosamp_->nits < itctx_->cosamp_->inexact_nits);
            itctx_->update_reference_objective = (itctx_->cosamp_->nits < itctx_->cosamp_->inexact_nits);
            // == solve ==
            ierr = solve ();                                    /* L2 solver    */
            ierr = VecCopy (getPrec(), itctx_->cosamp_->x_sub); /* get solution */       CHKERRQ (ierr);
            ierr = tuMSG("### -------------------------------------------- L2 solver end ------------------------------------------ ###");CHKERRQ (ierr);
            ierr = tuMSGstd (""); CHKERRQ (ierr);
            // print phi's to file
            if (itctx_->n_misc_->writeOutput_) {
                ierr = VecDuplicate (itctx_->tumor_->phi_->phi_vec_[0], &all_phis);     CHKERRQ (ierr);
                ierr = VecSet       (all_phis, 0.);                                     CHKERRQ (ierr);
                for (int i = 0; i < itctx_->n_misc_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, itctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
                ss << "phiSupportFinal.nc";  {dataOut (all_phis, itctx_->n_misc_, ss.str().c_str());} ss.str(std::string()); ss.clear();
                ss << "c0FinalGuess.nc";      dataOut (itctx_->tumor_->c_0_, itctx_->n_misc_, ss.str().c_str()); ss.str(std::string()); ss.clear();
                ss << "c1FinalGuess.nc"; if (itctx_->n_misc_->verbosity_ >= 4) { dataOut (itctx_->tumor_->c_t_, itctx_->n_misc_, ss.str().c_str()); } ss.str(std::string()); ss.clear();
                if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}
            }
            // write out p vector after IC, k inversion (unscaled)
            if (itctx_->n_misc_->write_p_checkpoint_) { writeCheckpoint(itctx_->cosamp_->x_sub, itctx_->tumor_->phi_, itctx_->n_misc_->writepath_ .str(), std::string("unscaled"));}
            if (procid == 0 && itctx_->n_misc_->verbosity_ >= 4) { ierr = VecView (itctx_->cosamp_->x_sub, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

            // == convergence test ==
            // neither gradient sufficiently small nor ls-failure (i.e., inexact_nit hit)
            // if(!itctx_->cosamp_->converged_l2 && !itctx_->cosamp_->converged_error_l2) {itctx_->cosamp_->nits += itctx_->cosamp_->inexact_nits;}
            itctx_->cosamp_->nits += itctx_->optfeedback_->nb_newton_it;
            conv_maxit = itctx_->cosamp_->nits >= itctx_->cosamp_->maxit_newton;

            // == prolongate ==
            // prolongate restricted x_L2 to full x_L1, but do not resize vectors, i.e., call resetOperators
            // if inversion for reaction disabled, also reset operators
            finalize = !itctx_->n_misc_->reaction_inversion_;
            contiterating = !itctx_->cosamp_->converged_l2 && !itctx_->cosamp_->converged_error_l2 && !conv_maxit;
            ierr = prolongateSubspace(itctx_->cosamp_->x_full, &itctx_->cosamp_->x_sub, itctx_, np_full, (finalize || contiterating));  CHKERRQ (ierr); // x_full <-- P(x_sub)

            // check if L2 solver converged
            if(contiterating) {
                ss << "    ... inexact solve terminated (L2 solver not converged, will be continued; its "<< itctx_->cosamp_->nits <<"/"<< itctx_->cosamp_->maxit_newton <<").";
                ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();
                ierr = tuMSG(" << leaving stage FINAL_L2"); CHKERRQ(ierr); ss.str(""); ss.clear();
                break;
            } else {
                // if L2 solver converged
                if(itctx_->cosamp_->converged_l2)        {ss << "    ... L2 solver converged; its "<< itctx_->cosamp_->nits <<"/"<< itctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
                // if L2 solver ran into ls-failure
                if(itctx_->cosamp_->converged_error_l2)  {ss << "    ... L2 solver terminated (ls-failure); its "<< itctx_->cosamp_->nits <<"/"<< itctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
                // if L2 solver hit maxit
                if(conv_maxit)                           {ss << "    ... L2 solver terminated (maxit); its "<< itctx_->cosamp_->nits <<"/"<< itctx_->cosamp_->maxit_newton <<"."; ierr = tuMSG(ss.str()); CHKERRQ(ierr);  ss.str(""); ss.clear();}
                itctx_->cosamp_->cosamp_stage = finalize ?  FINALIZE : POST_RD;
                conv_maxit = itctx_->cosamp_->nits = 0;
                ierr = tuMSG(" << leaving stage FINAL_L2"); CHKERRQ(ierr); ss.str(""); ss.clear();
            }

        // ================
        // this case is executed at once without going back to caller in between
        case POST_RD:
            ierr = tuMSG(" >> entering stage POST_RD"); CHKERRQ(ierr);
            // === (4) reaction/diffusion inversion ===
            if (itctx_->n_misc_->reaction_inversion_) {
                // restrict to new L2 subspace, holding p_i, kappa, and rho
                ierr = restrictSubspace(&itctx_->cosamp_->x_sub, itctx_->cosamp_->x_full, itctx_, true);     CHKERRQ (ierr); // x_sub <-- R(x_full)

                itctx_->cosamp_->cosamp_stage = POST_RD;
                itctx_->optsettings_->newton_maxit = itctx_->cosamp_->maxit_newton;
                // == solve ==
                ierr = solveInverseReacDiff (itctx_->cosamp_->x_sub); /* with current guess as init cond. */  CHKERRQ (ierr);
                ierr = VecCopy (getPrec(), itctx_->cosamp_->x_sub);   /* get solution */                      CHKERRQ (ierr);
                // update full space solution
                ierr = prolongateSubspace(itctx_->cosamp_->x_full, &itctx_->cosamp_->x_sub, itctx_, np_full); CHKERRQ (ierr); // x_full <-- P(x_sub)
            } else {ierr = tuMSGstd("    ... skipping stage, reaction diffusion disabled."); CHKERRQ(ierr);}

            // break; go to finalize
            itctx_->cosamp_->cosamp_stage = FINALIZE;
            ierr = tuMSG(" << leaving stage POST_RD"); CHKERRQ(ierr);
            break;
    }


    // prolongate (in case we are in a subuspace solve, we still need the solution to be prolongated)
    // if (itctx_->cosamp_->cosamp_stage != FINALIZE) {
        // ierr = tuMSG(" >> entering stage FINALIZE"); CHKERRQ(ierr); ss.str(""); ss.clear();
        // ierr = prolongateSubspace(itctx_->cosamp_->x_full, &itctx_->cosamp_->x_sub, itctx_, np_full); CHKERRQ (ierr); // x_full <-- P(x_sub)
    // }
    // pass the reconstructed p vector to the caller (deep copy)
    ierr = VecCopy (itctx_->cosamp_->x_full, xrec_);                                                  CHKERRQ (ierr);
    if (!rs_mode_active && itctx_->cosamp_->cosamp_stage != FINALIZE)
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

    np_full = itctx_->n_misc_->np_; // store np of unrestricted ansatz space
    ierr = VecDuplicate (itctx_->tumor_->p_, &g);                                 CHKERRQ (ierr);
    ierr = VecDuplicate (itctx_->tumor_->p_, &x_L1);                              CHKERRQ (ierr);
    ierr = VecDuplicate (itctx_->tumor_->p_, &x_L1_old);                          CHKERRQ (ierr);
    ierr = VecDuplicate (itctx_->tumor_->p_, &temp);                              CHKERRQ (ierr);
    ierr = VecSet  (g, 0);                                                        CHKERRQ (ierr);
    ierr = VecCopy (itctx_->tumor_->p_, x_L1); /* copy initial guess for p */     CHKERRQ (ierr);
    ierr = VecSet  (x_L1_old, 0);                                                 CHKERRQ (ierr);
    ierr = VecSet  (temp, 0);                                                     CHKERRQ (ierr);


    /* ------------------------------------------------------------------------ */
    // ### (0) (pre-)reaction/diffusion inversion ###
    if (itctx_->n_misc_->pre_reacdiff_solve_ && itctx_->n_misc_->n_[0] > 64) {
    if (itctx_->n_misc_->reaction_inversion_) {
        // restrict to new L2 subspace, holding p_i, kappa, and rho
        ierr = restrictSubspace(&x_L2, x_L1, itctx_, true);                     CHKERRQ (ierr); // x_L2 <-- R(x_L1)
        // solve
        itctx_->cosamp_->cosamp_stage = PRE_RD;
        ierr = solveInverseReacDiff (x_L2);          /* with current guess as init cond. */
        ierr = VecCopy (getPrec(), x_L2);            /* get solution */         CHKERRQ (ierr);
        // update full space solution
        ierr = prolongateSubspace(x_L1, &x_L2, itctx_, np_full);                CHKERRQ (ierr); // x_L1 <-- P(x_L2)
    }
    }

    /* ------------------------------------------------------------------------ */
    // === (1) L1 CoSaMp solver ===
    // set initial guess for k_inv (possibly != zero)
    ierr = VecGetArray(x_L1, &x_L1_ptr);                                          CHKERRQ (ierr);
    if (itctx_->n_misc_->diffusivity_inversion_) x_L1_ptr[np_full] = itctx_->n_misc_->k_;
    else { // set diff ops with this guess -- this will not change during the solve
      ierr = itctx_->tumor_->k_->setValues (itctx_->n_misc_->k_, itctx_->n_misc_->k_gm_wm_ratio_, itctx_->n_misc_->k_glm_wm_ratio_, itctx_->tumor_->mat_prop_, itctx_->n_misc_);  CHKERRQ (ierr);
    }
    ierr = VecRestoreArray(x_L1, &x_L1_ptr);                                      CHKERRQ (ierr);
    ierr = VecCopy        (x_L1, x_L1_old);                                       CHKERRQ (ierr);

    // compute reference value for  objective
    beta_store = itctx_->n_misc_->beta_; itctx_->n_misc_->beta_ = 0.; // set beta to zero for gradient thresholding
    ierr = getObjectiveAndGradient (x_L1, &J_ref, g);                             CHKERRQ (ierr);
    itctx_->n_misc_->beta_ = beta_store;
    ierr = VecNorm (g, NORM_2, &norm_g);                                          CHKERRQ (ierr);
    J = J_ref;

    // print statistics
    printStatistics (its, J_ref, 1, norm_g, 1, x_L1);

    // number of connected components
    itctx_->tumor_->phi_->num_components_ = itctx_->tumor_->phi_->component_weights_.size ();
    // output warmstart (injection) support
    ss << "starting CoSaMP solver with initial support: ["; for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->n_misc_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
    ss << "component label of initial support : [";         for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->tumor_->phi_->gaussian_labels_[itctx_->n_misc_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

    // === L1 solver ===
    while (true) {
        its++;

        /* === hard threshold abs gradient === */
        ierr = VecCopy (g, temp);                                                 CHKERRQ (ierr);
        ierr = VecAbs (temp);                                                     CHKERRQ (ierr);
        // print gradient to file
        if (itctx_->n_misc_->verbosity_ >= 2) {
            ierr = VecGetArray(temp, &grad_ptr);                                    CHKERRQ(ierr);
            for (int i = 0; i < np_full-1; i++) if(procid == 0) itctx_->n_misc_->outfile_glob_grad_ << grad_ptr[i] << ", ";
            if(procid == 0)                                     itctx_->n_misc_->outfile_glob_grad_ << grad_ptr[np_full-1] << ";\n" <<std::endl;
            ierr = VecRestoreArray(temp, &grad_ptr);                                CHKERRQ(ierr);
        }
        // threshold gradient
        idx.clear();
        ierr = hardThreshold (temp, 2 * itctx_->n_misc_->sparsity_level_, np_full, idx, itctx_->tumor_->phi_->gaussian_labels_, itctx_->tumor_->phi_->component_weights_, nnz, itctx_->tumor_->phi_->num_components_);

        /* === update support of prev. solution with new support === */
        itctx_->n_misc_->support_.insert (itctx_->n_misc_->support_.end(), idx.begin(), idx.end());
        std::sort (itctx_->n_misc_->support_.begin(), itctx_->n_misc_->support_.end()); // sort and remove duplicates
        itctx_->n_misc_->support_.erase (std::unique (itctx_->n_misc_->support_.begin(), itctx_->n_misc_->support_.end()), itctx_->n_misc_->support_.end());
        // print out
        ss << "support for corrective L2 solve : [";
        for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->n_misc_->support_[i] << " "; ss << "]";
        ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ss << "component label of support : [";
        for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->tumor_->phi_->gaussian_labels_[itctx_->n_misc_->support_[i]] << " "; ss << "]";
        ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

        /* === corrective L2 solver === */
        ierr = tuMSGstd (""); CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSG("###                                corrective L2 solver in restricted subspace                            ###");CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

        ierr = restrictSubspace(&x_L2, x_L1, itctx_);                             CHKERRQ (ierr); // x_L2 <-- R(x_L1)

        // print vec
        if (procid == 0 && itctx_->n_misc_->verbosity_ >= 4) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}


        // solve interpolation
        // ierr = solveInterpolation (data_);                                        CHKERRQ (ierr);

        // update reference gradient and referenc objective (commented in the solve function)
        itctx_->update_reference_gradient  = true;
        itctx_->update_reference_objective = true;
        ierr = solve ();                                                          CHKERRQ (ierr);
        ierr = VecCopy (getPrec(), x_L2);                                         CHKERRQ (ierr);
        ierr = tuMSG("### ----------------------------------------- L2 solver end --------------------------------------------- ###");CHKERRQ (ierr);
        ierr = tuMSGstd ("");                                                     CHKERRQ (ierr);
        ierr = VecCopy (x_L1, x_L1_old);                                          CHKERRQ (ierr);

        // print support
        ierr = VecDuplicate (itctx_->tumor_->phi_->phi_vec_[0], &all_phis);       CHKERRQ (ierr);
        ierr = VecSet (all_phis, 0.);                                             CHKERRQ (ierr);
        for (int i = 0; i < itctx_->n_misc_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, itctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
        ss << "phiSupport_csitr-" << its << ".nc";
        if (itctx_->n_misc_->writeOutput_) dataOut (all_phis, itctx_->n_misc_, ss.str().c_str()); ss.str(""); ss.clear();
        if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}

        // print vec
        if (procid == 0 && itctx_->n_misc_->verbosity_ >= 4) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}


        ierr = prolongateSubspace(x_L1, &x_L2, itctx_, np_full);                  CHKERRQ (ierr); // x_L1 <-- P(x_L2)

        /* === hard threshold solution to sparsity level === */
        idx.clear();
        if (itctx_->n_misc_->prune_components_) hardThreshold (x_L1, itctx_->n_misc_->sparsity_level_, np_full, idx, itctx_->tumor_->phi_->gaussian_labels_, itctx_->tumor_->phi_->component_weights_, nnz, itctx_->tumor_->phi_->num_components_);
        else                                    hardThreshold (x_L1, itctx_->n_misc_->sparsity_level_, np_full, idx, nnz);
        itctx_->n_misc_->support_.clear ();
        itctx_->n_misc_->support_ = idx;
        // sort and remove duplicates
        std::sort (itctx_->n_misc_->support_.begin(), itctx_->n_misc_->support_.end());
        itctx_->n_misc_->support_.erase (std::unique (itctx_->n_misc_->support_.begin(), itctx_->n_misc_->support_.end()), itctx_->n_misc_->support_.end());
        // print out
        ss << "support after hard thresholding the solution : [";
        for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->n_misc_->support_[i] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();
        ss << "component label of support : [";
        for (int i = 0; i < itctx_->n_misc_->support_.size(); i++) ss << itctx_->tumor_->phi_->gaussian_labels_[itctx_->n_misc_->support_[i]] << " "; ss << "]"; ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

        // set only support values in x_L1 (rest hard thresholded to zero)
        ierr = VecCopy (x_L1, temp);                                              CHKERRQ (ierr);
        ierr = VecSet  (x_L1, 0.0);                                               CHKERRQ (ierr);
        ierr = VecGetArray (temp, &temp_ptr);                                     CHKERRQ (ierr);
        ierr = VecGetArray (x_L1, &x_L1_ptr);                                     CHKERRQ (ierr);
        for (int i = 0; i < itctx_->n_misc_->support_.size(); i++)
          x_L1_ptr[itctx_->n_misc_->support_[i]] = temp_ptr[itctx_->n_misc_->support_[i]];
        if (itctx_->n_misc_->diffusivity_inversion_) {
          x_L1_ptr[np_full] = temp_ptr[np_full];
          if (itctx_->n_misc_->nk_ > 1) x_L1_ptr[np_full+1] = temp_ptr[np_full+1];
          if (itctx_->n_misc_->nk_ > 2) x_L1_ptr[np_full+2] = temp_ptr[np_full+2];
        }
        ierr = VecRestoreArray (x_L1, &x_L1_ptr);                                 CHKERRQ (ierr);
        ierr = VecRestoreArray (temp, &temp_ptr);                                 CHKERRQ (ierr);
        ierr = VecCopy (x_L1, itctx_->tumor_->p_); /* copy initial guess for p */ CHKERRQ (ierr);

        // print initial guess to file
        if (itctx_->n_misc_->writeOutput_) {
        ss << "c0guess_csitr-" << its << ".nc";  dataOut (itctx_->tumor_->c_0_, itctx_->n_misc_, ss.str().c_str()); ss.str(std::string()); ss.clear();
        ss << "c1guess_csitr-" << its << ".nc"; if (itctx_->n_misc_->verbosity_ >= 4) dataOut (itctx_->tumor_->c_t_, itctx_->n_misc_, ss.str().c_str()); ss.str(std::string()); ss.clear();
        }

        /* === convergence check === */
        J_old = J;

        // compute objective (only mismatch term)
        beta_store = itctx_->n_misc_->beta_; itctx_->n_misc_->beta_ = 0.; // set beta to zero for gradient thresholding
        ierr = getObjectiveAndGradient (x_L1, &J, g);                             CHKERRQ (ierr);
        itctx_->n_misc_->beta_ = beta_store;
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
        if (its >= optsettings_->gist_maxit) {ierr = tuMSGwarn (" L1 maxiter reached."); CHKERRQ(ierr); flag_convergence = 1; break;}
        else if (PetscAbsReal (J) < 1E-5)  {ierr = tuMSGwarn (" L1 absolute objective tolerance reached."); CHKERRQ(ierr); flag_convergence = 1; break;}
        else if (PetscAbsReal (J_old - J) < ftol * PetscAbsReal (1 + J_ref)) {ierr = tuMSGwarn (" L1 relative objective tolerance reached."); CHKERRQ(ierr); flag_convergence = 1; break;}
        else { flag_convergence = 0; }  // continue iterating
    } // end while

    /* === (3) if converged: corrective L2 solver === */
    ierr = tuMSGstd ("");                                                       CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);
    ierr = tuMSG("###                                              final L2 solve                                           ###");CHKERRQ (ierr);
    ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###");CHKERRQ (ierr);

    ierr = restrictSubspace(&x_L2, x_L1, itctx_);                               CHKERRQ (ierr); // x_L2 <-- R(x_L1)
    if (procid == 0 && itctx_->n_misc_->verbosity_ >= 4) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}

    // solve interpolation
    // ierr = solveInterpolation (data_);                                        CHKERRQ (ierr);
    // update reference gradient and referenc objective (commented in the solve function)
    itctx_->update_reference_gradient  = true;
    itctx_->update_reference_objective = true;
    ierr = solve ();                                   /* L2 solver    */
    ierr = VecCopy (getPrec(), x_L2);                  /* get solution */       CHKERRQ (ierr);

    // print vec
    if (procid == 0 && itctx_->n_misc_->verbosity_ >= 4) { ierr = VecView (x_L2, PETSC_VIEWER_STDOUT_SELF);               CHKERRQ (ierr);}


    ierr = tuMSG("### -------------------------------------------- L2 solver end ------------------------------------------ ###");CHKERRQ (ierr);
    ierr = tuMSGstd (""); CHKERRQ (ierr);

    // print phi's to file
    if (itctx_->n_misc_->writeOutput_) {
        ierr = VecDuplicate (itctx_->tumor_->phi_->phi_vec_[0], &all_phis);     CHKERRQ (ierr);
        ierr = VecSet       (all_phis, 0.);                                     CHKERRQ (ierr);
        for (int i = 0; i < itctx_->n_misc_->np_; i++) {ierr = VecAXPY (all_phis, 1.0, itctx_->tumor_->phi_->phi_vec_[i]); CHKERRQ (ierr);}
        ss << "phiSupportFinal.nc";  {dataOut (all_phis, itctx_->n_misc_, ss.str().c_str());} ss.str(std::string()); ss.clear();
        ss << "c0FinalGuess.nc";  dataOut (itctx_->tumor_->c_0_, itctx_->n_misc_, ss.str().c_str()); ss.str(std::string()); ss.clear();
        ss << "c1FinalGuess.nc"; if (itctx_->n_misc_->verbosity_ >= 4) { dataOut (itctx_->tumor_->c_t_, itctx_->n_misc_, ss.str().c_str()); } ss.str(std::string()); ss.clear();
        if (all_phis != nullptr) {ierr = VecDestroy (&all_phis); CHKERRQ (ierr); all_phis = nullptr;}
    }

    // write out p vector after IC, k inversion (unscaled)
    if (itctx_->n_misc_->write_p_checkpoint_) {
      writeCheckpoint(x_L2, itctx_->tumor_->phi_, itctx_->n_misc_->writepath_ .str(), std::string("unscaled"));
    }
    // prolongate restricted x_L2 to full x_L1, but do not resize vectors, i.e., call resetOperators
    // if inversion for reaction disabled, also reset operators
    ierr = prolongateSubspace(x_L1, &x_L2, itctx_, np_full, !itctx_->n_misc_->reaction_inversion_);  CHKERRQ (ierr); // x_L1 <-- P(x_L2)

    // === (4) reaction/diffusion inversion ===
    if (itctx_->n_misc_->reaction_inversion_) {
        // restrict to new L2 subspace, holding p_i, kappa, and rho
        ierr = restrictSubspace(&x_L2, x_L1, itctx_, true);                     CHKERRQ (ierr); // x_L2 <-- R(x_L1)
        // solve
        itctx_->cosamp_->cosamp_stage = POST_RD;
        ierr = solveInverseReacDiff (x_L2);          /* with current guess as init cond. */
        ierr = VecCopy (getPrec(), x_L2);            /* get solution */         CHKERRQ (ierr);
        // update full space solution
        ierr = prolongateSubspace(x_L1, &x_L2, itctx_, np_full);                CHKERRQ (ierr); // x_L1 <-- P(x_L2)
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
    if (itctx_->n_misc_->diffusivity_inversion_) {
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx_->n_misc_->np_];
        if (itctx_->n_misc_->nk_ > 1) { s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx_->n_misc_->np_ + 1]; }
    } else {
      s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << itctx_->n_misc_->k_;
    }
    ierr = VecRestoreArray(x_L1, &x_ptr);                                       CHKERRQ(ierr);
    ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
    s.str (""); s.clear ();
    PetscFunctionReturn (ierr);
}

InvSolver::~InvSolver () {
    PetscErrorCode ierr = 0;
    if (tao_     != nullptr) {TaoDestroy (&tao_);      tao_  = nullptr;}
    if (H_       != nullptr) {MatDestroy (&H_);        H_    = nullptr;}
    if (xrec_    != nullptr) {VecDestroy (&xrec_);     xrec_ = nullptr;}
    if (xrec_rd_ != nullptr) {VecDestroy (&xrec_rd_);  xrec_rd_ = nullptr;}
}


// ###                                                                       ###
// ###                                                                       ###
// ###                                                                       ###
// #############################################################################
// ################## non-class methods used for TAO ###########################
// #############################################################################
// ###                                                                       ###
// ###                                                                       ###
// ###                                                                       ###
/* ------------------------------------------------------------------- */
/*
 evaluateObjectiveFunction - evaluates the objective function J(x).
 Input Parameters:
 .  tao - the Tao context
 .  x   - the input vector
 .  ptr - optional user-defined context, as set by TaoSetFunction()
 Output Parameters:
 .  J    - the newly evaluated function
 */
PetscErrorCode evaluateObjectiveFunction (Tao tao, Vec x, PetscReal *J, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tao-eval-obj-tumor");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
    itctx->optfeedback_->nb_objevals++;
    ierr = itctx->derivative_operators_->evaluateObjective (J, x, itctx->data);
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}
/* ------------------------------------------------------------------- */
/*
 evaluateGradient evaluates the gradient g(m0)
 input parameters:
  . tao  - the Tao context
  . x   - input vector p (current estimate for the paramterized initial condition)
  . ptr  - optional user-defined context
 output parameters:
  . dJ    - vector containing the newly evaluated gradient
 */
PetscErrorCode evaluateGradient (Tao tao, Vec x, Vec dJ, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tao-eval-grad-tumor");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
    ierr = VecCopy (x, itctx->derivative_operators_->p_current_);                       CHKERRQ (ierr);

    itctx->optfeedback_->nb_gradevals++;
    ierr = itctx->derivative_operators_->evaluateGradient (dJ, x, itctx->data_gradeval);

    // use petsc default fd gradient
   // itctx->derivative_operators_->disable_verbose_ = true;
   // ierr = TaoDefaultComputeGradient(tao, x, dJ, ptr); CHKERRQ(ierr);
   // itctx->derivative_operators_->disable_verbose_ = false;

    std::stringstream s;
    if (itctx->optsettings_->verbosity > 1) {
        ScalarType gnorm;
        ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
        s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}
/* ------------------------------------------------------------------- */
/*
 evaluateObjectiveFunctionAndGradient - evaluates the function and corresponding gradient
 Input Parameters:
  . tao - the Tao context
  . x   - the input vector
  . ptr - optional user-defined context, as set by TaoSetFunction()
 Output Parameters:
  . J   - the newly evaluated function
  . dJ   - the newly evaluated gradient
 */
PetscErrorCode evaluateObjectiveFunctionAndGradient (Tao tao, Vec x, PetscReal *J, Vec dJ, void *ptr){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj/grad-tumor");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
  ierr = VecCopy (x, itctx->derivative_operators_->p_current_);                       CHKERRQ (ierr);

  itctx->optfeedback_->nb_objevals++;
  itctx->optfeedback_->nb_gradevals++;
  ierr = itctx->derivative_operators_->evaluateObjectiveAndGradient (J, dJ, x, itctx->data_gradeval);

  std::stringstream s;
  if (itctx->optsettings_->verbosity > 1) {
      ScalarType gnorm;
      ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
      s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }

  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

    //ierr = evaluateObjectiveFunction (tao, p, J, ptr);                     CHKERRQ(ierr);
    //ierr = evaluateGradient (tao, p, dJ, ptr);                             CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode evaluateObjectiveReacDiff (Tao tao, Vec x, PetscReal *J, void *ptr){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj-params");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      x->lock = 0;
    }
  #endif

  itctx->optfeedback_->nb_objevals++;
  itctx->optfeedback_->nb_gradevals++;

  // set the last 2-3 entries to the parameters obtained from tao and pass to derivativeoperators
  ScalarType *x_ptr, *x_full_ptr;
  ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  x_full_ptr[itctx->n_misc_->np_] = x_ptr[0];   // k1
  if (itctx->n_misc_->nk_ > 1) x_full_ptr[itctx->n_misc_->np_ + 1] = x_ptr[1];  // k2
  if (itctx->n_misc_->nk_ > 2) x_full_ptr[itctx->n_misc_->np_ + 2] = x_ptr[2];  // k3
  x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_] = x_ptr[itctx->n_misc_->nk_];  // rho
  if (itctx->n_misc_->nr_ > 1) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 1] = x_ptr[itctx->n_misc_->nk_ + 1];  // r2
  if (itctx->n_misc_->nr_ > 2) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 2] = x_ptr[itctx->n_misc_->nk_ + 2];  // r2

  ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      x->lock = lock_state;
    }
  #endif

  ierr = itctx->derivative_operators_->evaluateObjective (J, itctx->x_old, itctx->data);

  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

  PetscFunctionReturn (ierr);
}

PetscErrorCode evaluateGradientReacDiff (Tao tao, Vec x, Vec dJ, void *ptr){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-grad-tumor-params");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      x->lock = 0;
    }
  #endif

  itctx->optfeedback_->nb_objevals++;
  itctx->optfeedback_->nb_gradevals++;

  // set the last 2-3 entries to the parameters obtained from tao and pass to derivativeoperators
  ScalarType *x_ptr, *x_full_ptr;
  ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  x_full_ptr[itctx->n_misc_->np_] = x_ptr[0];   // k1
  if (itctx->n_misc_->nk_ > 1) x_full_ptr[itctx->n_misc_->np_ + 1] = x_ptr[1];  // k2
  if (itctx->n_misc_->nk_ > 2) x_full_ptr[itctx->n_misc_->np_ + 2] = x_ptr[2];  // k3
  x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_] = x_ptr[itctx->n_misc_->nk_];  // rho
  if (itctx->n_misc_->nr_ > 1) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 1] = x_ptr[itctx->n_misc_->nk_ + 1];  // r2
  if (itctx->n_misc_->nr_ > 2) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 2] = x_ptr[itctx->n_misc_->nk_ + 2];  // r2

  ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  Vec dJ_full;
  ierr = VecDuplicate (itctx->x_old, &dJ_full);         CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      x->lock = lock_state;
    }
  #endif

  ierr = itctx->derivative_operators_->evaluateGradient (dJ_full, itctx->x_old, itctx->data_gradeval);

  ScalarType *dj_ptr, *dj_full_ptr;
  ierr = VecGetArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecGetArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  dj_ptr[0] = dj_full_ptr[itctx->n_misc_->np_];
  if (itctx->n_misc_->nk_ > 1) dj_ptr[1] = dj_full_ptr[itctx->n_misc_->np_ + 1];  // k2
  if (itctx->n_misc_->nk_ > 2) dj_ptr[2] = dj_full_ptr[itctx->n_misc_->np_ + 2];  // k3
  dj_ptr[itctx->n_misc_->nk_] = dj_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_];  // rho
  if (itctx->n_misc_->nr_ > 1) dj_ptr[itctx->n_misc_->nk_ + 1] = dj_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 1];  // r2
  if (itctx->n_misc_->nr_ > 2) dj_ptr[itctx->n_misc_->nk_ + 2] = dj_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 2];  // r2

  ierr = VecRestoreArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecRestoreArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  std::stringstream s;
  if (itctx->optsettings_->verbosity > 1) {
      ScalarType gnorm;
      ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
      s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

  if (dJ_full  != nullptr) {VecDestroy (&dJ_full);  CHKERRQ(ierr);  dJ_full  = nullptr;}
  PetscFunctionReturn (ierr);
}

PetscErrorCode evaluateObjectiveAndGradientReacDiff (Tao tao, Vec x, PetscReal *J, Vec dJ, void *ptr){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj/grad-tumor-params");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      x->lock = 0;
    }
  #endif

  itctx->optfeedback_->nb_objevals++;
  itctx->optfeedback_->nb_gradevals++;

  // set the last 2-3 entries to the parameters obtained from tao and pass to derivativeoperators
  ScalarType *x_ptr, *x_full_ptr;
  ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  x_full_ptr[itctx->n_misc_->np_] = x_ptr[0];   // k1
  if (itctx->n_misc_->nk_ > 1) x_full_ptr[itctx->n_misc_->np_ + 1] = x_ptr[1];  // k2
  if (itctx->n_misc_->nk_ > 2) x_full_ptr[itctx->n_misc_->np_ + 2] = x_ptr[2];  // k3
  x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_] = x_ptr[itctx->n_misc_->nk_];  // rho
  if (itctx->n_misc_->nr_ > 1) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 1] = x_ptr[itctx->n_misc_->nk_ + 1];  // r2
  if (itctx->n_misc_->nr_ > 2) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 2] = x_ptr[itctx->n_misc_->nk_ + 2];  // r2

  ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  Vec dJ_full;
  ierr = VecDuplicate (itctx->x_old, &dJ_full);         CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      x->lock = lock_state;
    }
  #endif

  ierr = itctx->derivative_operators_->evaluateObjectiveAndGradient (J, dJ_full, itctx->x_old, itctx->data_gradeval);

  ScalarType *dj_ptr, *dj_full_ptr;
  ierr = VecGetArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecGetArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  dj_ptr[0] = dj_full_ptr[itctx->n_misc_->np_];
  if (itctx->n_misc_->nk_ > 1) dj_ptr[1] = dj_full_ptr[itctx->n_misc_->np_ + 1];  // k2
  if (itctx->n_misc_->nk_ > 2) dj_ptr[2] = dj_full_ptr[itctx->n_misc_->np_ + 2];  // k3
  dj_ptr[itctx->n_misc_->nk_] = dj_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_];  // rho
  if (itctx->n_misc_->nr_ > 1) dj_ptr[itctx->n_misc_->nk_ + 1] = dj_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 1];  // r2
  if (itctx->n_misc_->nr_ > 2) dj_ptr[itctx->n_misc_->nk_ + 2] = dj_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 2];  // r2

  ierr = VecRestoreArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecRestoreArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  std::stringstream s;
  if (itctx->optsettings_->verbosity > 1) {
      ScalarType gnorm;
      ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
      s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

  if (dJ_full  != nullptr) {VecDestroy (&dJ_full);  CHKERRQ(ierr);  dJ_full  = nullptr;}
  PetscFunctionReturn (ierr);
}



/* ------------------------------------------------------------------- */
/*
 hessianMatVec    computes the Hessian matrix-vector product
 input parameters:
  . H       input matrix
  . s       input vector
 output parameters:
  . Hs      solution vector
 */
PetscErrorCode hessianMatVec (Mat A, Vec x, Vec y) {    //y = Ax
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tao-hess-matvec-tumor");
    std::array<double, 7> t = {0}; double self_exec_time = -MPI_Wtime ();
    // get context
    void *ptr;
    ierr = MatShellGetContext (A, &ptr);                                     CHKERRQ (ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>( ptr);
    // eval hessian
    itctx->optfeedback_->nb_matvecs++;
    ierr = itctx->derivative_operators_->evaluateHessian (y, x);
    if (itctx->optsettings_->verbosity > 1) {
        PetscPrintf (MPI_COMM_WORLD, " applying hessian done!\n");
        ScalarType xnorm;
        ierr = VecNorm (x, NORM_2, &xnorm); CHKERRQ(ierr);
        PetscPrintf (MPI_COMM_WORLD, " norm of search direction ||x||_2 = %e\n", xnorm);
    }
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}
/* ------------------------------------------------------------------- */
/*
 constApxHessianMatVec    computes the Hessian matrix-vector product for
                          a constant approximation \beta_p \Phi^T\Phi of the Hessian
 input parameters:
  . H       input matrix
  . s       input vector
 output parameters:
  . Hs      solution vector
 */
PetscErrorCode constApxHessianMatVec (Mat A, Vec x, Vec y) {    //y = Ax
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tao-lmvm-init-hess--matvec");
    std::array<double, 7> t = {0}; double self_exec_time = -MPI_Wtime ();
    // get context
    void *ptr;
    ierr = MatShellGetContext (A, &ptr);                                     CHKERRQ (ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>( ptr);
    // eval hessian
    ierr = itctx->derivative_operators_->evaluateConstantHessianApproximation (y, x);
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}
/* ------------------------------------------------------------------- */
PetscErrorCode matfreeHessian (Tao tao, Vec x, Mat H, Mat precH, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    PetscFunctionReturn (ierr);
}
/* ------------------------------------------------------------------- */
/*
 preconditionerMatVec    computes the matrix-vector product of inverse of
 preconditioner and some input vector
 input parameters:
 pinv    input matrix (shell context)
 x       input vector
 output parameters:
 .       pinvx   inverse of preconditioner applied to output vector
 */
PetscErrorCode preconditionerMatVec (PC pinv, Vec x, Vec pinvx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    void *ptr;
    // get shell context
    ierr = PCShellGetContext (pinv, &ptr);                  CHKERRQ(ierr);
    // apply the hessian
    ierr = applyPreconditioner (ptr, x, pinvx);             CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}
/* ------------------------------------------------------------------- */
/*
 applyPreconditioner  apply preconditioner to a given vector
 input parameters:
 ptr       pointer to user defined context
 x         input vector
 output parameters:
 .       pinvx     inverse of preconditioner applied to input vector
 */
PetscErrorCode applyPreconditioner (void *ptr, Vec x, Vec pinvx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tao-apply-hess-precond");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    ScalarType *ptr_pinvx = NULL, *ptr_x = NULL;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);
    ierr = VecCopy (x, pinvx);
    // === PRECONDITIONER CURRENTLY DISABLED ===
    PetscFunctionReturn (ierr);
    // apply hessian preconditioner
    //ierr = itctx->derivative_operators_->evaluateHessian(pinvx, x);
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}


PetscErrorCode optimizationMonitorMassEffect (Tao tao, void *ptr) {
    // first to monitor then to checkconv in petsc 3.11
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    PetscInt its;
    ScalarType J = 0, gnorm = 0, cnorm = 0 , step = 0, D = 0, J0 = 0, D0 = 0, gnorm0 = 0;
    Vec x = nullptr;
    char msg[256];
    std::string statusmsg;
    std::stringstream s;
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

    Vec tao_grad;

    // get current iteration, objective value, norm of gradient, norm of
    // norm of contraint, step length / trust region readius of iteratore
    // and termination reason
    Vec tao_x;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);  CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &tao_x);                                   CHKERRQ(ierr);
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr =  TaoGetGradientVector(tao, &tao_grad);                               CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    PetscInt num_feval, n2, n3;
    TaoLineSearch ls = nullptr;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = TaoLineSearchGetNumberFunctionEvaluations (ls, &num_feval, &n2, &n3);   CHKERRQ (ierr);

    ScalarType step_tol = std::pow(2, -3);
    // adaptive ls step
    if (step < step_tol) {
        itctx->step_init = step * 2;
    } else {
        itctx->step_init *= 2;
    }
    itctx->step_init = std::min(itctx->step_init, (ScalarType)1);
    //itctx->step_init = 1;

    ierr = TaoLineSearchSetInitialStepLength (ls, itctx->step_init);            CHKERRQ(ierr);


    // update/set reference gradient 
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
        itctx->optfeedback_->gradnorm0 = gnorm;
        itctx->update_reference_gradient = false;
        std::stringstream s; s <<" updated reference gradient for relative convergence criterion: " << itctx->optfeedback_->gradnorm0;
        ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    }
    #endif

    itctx->optfeedback_->nb_newton_it++;

    ScalarType *tao_x_ptr;
    ierr = VecGetArray (tao_x, &tao_x_ptr);                                     CHKERRQ (ierr);

    ScalarType mx, mn;
    ierr = VecMax (itctx->tumor_->c_t_, NULL, &mx); CHKERRQ (ierr);
    ierr = VecMin (itctx->tumor_->c_t_, NULL, &mn); CHKERRQ (ierr);
    // this print helps determine if theres any large aliasing errors which is causing ls failure etc
    s << " ---------- tumor c(1) bounds: max = " << mx << ", min = " << mn << " ----------- ";
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();

    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   ";
        s << std::setw(18) << "gamma";
        s << std::setw(18) << "rho";
        s << std::setw(18) << "kappa";

        if(itctx->optsettings_->newtonsolver == QUASINEWTON) {
          ierr = tuMSGstd (" starting optimization, TAO's Quasi-Newton");            CHKERRQ(ierr);
        } else {
          ierr = tuMSGstd (" starting optimization, TAO's Gauss-Newton");            CHKERRQ(ierr);
        }
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                                  CHKERRQ(ierr);
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }

    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->optfeedback_->gradnorm0
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << tao_x_ptr[0]
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << tao_x_ptr[1]
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << tao_x_ptr[2];

    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");s.clear ();

    ierr = VecRestoreArray (tao_x, &tao_x_ptr);                                    CHKERRQ(ierr);
    if (procid == 0) {
        ierr = VecView (tao_grad, PETSC_VIEWER_STDOUT_SELF);                           CHKERRQ(ierr);
    }
    //Gradient check begin
    // ierr = itctx->derivative_operators_->checkGradient (tao_x, itctx->data);
    // ierr = itctx->derivative_operators_->checkHessian (tao_x, itctx->data);
    //Gradient check end
    PetscFunctionReturn (ierr);
}


/* ------------------------------------------------------------------- */
/*
 optimizationMonitor    mointors the inverse Gauss-Newton solve
 input parameters:
  . tao       TAO object
  . ptr       optional user defined context
 */
PetscErrorCode optimizationMonitor (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    PetscInt its;
    ScalarType J = 0, gnorm = 0, cnorm = 0 , step = 0, D = 0, J0 = 0, D0 = 0, gnorm0 = 0;
    Vec x = nullptr;
    char msg[256];
    std::string statusmsg;
    std::stringstream s;
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

    Vec tao_grad;

    // get current iteration, objective value, norm of gradient, norm of
    // norm of contraint, step length / trust region readius of iteratore
    // and termination reason
    Vec tao_x;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);  CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &tao_x);                                   CHKERRQ(ierr);
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr =  TaoGetGradientVector(tao, &tao_grad);                               CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    if (itctx->n_misc_->verbosity_ >= 2) {
      ScalarType *grad_ptr, *sol_ptr;
      ierr = VecGetArray(tao_x, &sol_ptr);                                        CHKERRQ(ierr);
      ierr = VecGetArray(tao_grad, &grad_ptr);                                    CHKERRQ(ierr);
      for (int i = 0; i < itctx->n_misc_->np_; i++){
        if(procid == 0){
          itctx->n_misc_->outfile_sol_  << sol_ptr[i]  << ", ";
          itctx->n_misc_->outfile_grad_ << grad_ptr[i] << ", ";
        }
      }
      if(procid == 0){
        itctx->n_misc_->outfile_sol_  << sol_ptr[itctx->n_misc_->np_]  << ";" <<std::endl;
        itctx->n_misc_->outfile_grad_ << grad_ptr[itctx->n_misc_->np_] << ";" <<std::endl;
      }
      ierr = VecRestoreArray(tao_x, &sol_ptr);                                    CHKERRQ(ierr);
      ierr = VecRestoreArray(tao_grad, &grad_ptr);                                CHKERRQ(ierr);
    }

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
      Vec dJ, p0;
      ScalarType norm_gref = 0.;
      ierr = VecDuplicate (itctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
      ierr = VecDuplicate (itctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
      ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
      ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);

      if (itctx->n_misc_->flag_reaction_inv_) {
        norm_gref = gnorm;
      } else {
        ierr = evaluateGradient(tao, p0, dJ, (void*) itctx);
        ierr = VecNorm (dJ, NORM_2, &norm_gref);                                CHKERRQ(ierr);
      }
      itctx->optfeedback_->gradnorm0 = norm_gref;
      //ctx->gradnorm0 = gnorm;
      itctx->update_reference_gradient = false;
      s <<" updated reference gradient for relative convergence criterion, Gauss-Newton solver: " << itctx->optfeedback_->gradnorm0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);s.str ("");s.clear ();
      if (dJ  != nullptr) {VecDestroy (&dJ);  CHKERRQ(ierr);  dJ  = nullptr;}
      if (p0  != nullptr) {VecDestroy (&p0);  CHKERRQ(ierr);  p0  = nullptr;}
    }
    #endif

    // ierr = VecAXPY (itctx->x_old, -1.0, tao_x);                                     CHKERRQ (ierr);

    // ScalarType dp_norm, p_norm;
    // ierr = VecNorm (itctx->x_old, NORM_INFINITY, &dp_norm);                         CHKERRQ (ierr);
    // ierr = VecNorm (tao_x, NORM_INFINITY, &p_norm);                                 CHKERRQ (ierr);
    // accumulate number of newton iterations
    itctx->optfeedback_->nb_newton_it++;
    // print out Newton iteration information

    ierr = itctx->tumor_->phi_->apply (itctx->tumor_->c_0_, tao_x);                 CHKERRQ (ierr);
    //Prints a warning if tumor IC is clipped
    ierr = checkClipping (itctx->tumor_->c_0_, itctx->n_misc_);           CHKERRQ (ierr);

    ScalarType mx, mn;
    ierr = VecMax (itctx->tumor_->c_t_, NULL, &mx); CHKERRQ (ierr);
    ierr = VecMin (itctx->tumor_->c_t_, NULL, &mn); CHKERRQ (ierr);
    // this print helps determine if theres any large aliasing errors which is causing ls failure etc
    s << " ---------- tumor c(1) bounds: max = " << mx << ", min = " << mn << " ----------- ";
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();

    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   ";
          if (itctx->n_misc_->diffusivity_inversion_) {
            s << std::setw(18) << "k";
          }

        if(itctx->optsettings_->newtonsolver == QUASINEWTON) {
          ierr = tuMSGstd (" starting optimization, TAO's Quasi-Newton");                   CHKERRQ(ierr);
        } else {
          ierr = tuMSGstd (" starting optimization, TAO's Gauss-Newton");            CHKERRQ(ierr);
        }
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }

    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->optfeedback_->gradnorm0
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      ;
      if (itctx->n_misc_->diffusivity_inversion_) {
        ScalarType *x_ptr;
        ierr = VecGetArray(tao_x, &x_ptr);                                         CHKERRQ(ierr);
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->np_];
        if (itctx->n_misc_->nk_ > 1) {
          s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->np_ + 1]; }
        ierr = VecRestoreArray(tao_x, &x_ptr);                                     CHKERRQ(ierr);
      }
    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");s.clear ();


    //ierr = PetscPrintf (PETSC_COMM_WORLD, "\nKSP number of krylov iterations: %d\n", itctx->optfeedback_->nb_krylov_it);          CHKERRQ(ierr);
    //itctx->optfeedback_->nb_krylov_it = 0;

    //Gradient check begin
    // ierr = itctx->derivative_operators_->checkGradient (tao_x, itctx->data);
    // ierr = itctx->derivative_operators_->checkHessian (tao_x, itctx->data);
    //Gradient check end
    PetscFunctionReturn (ierr);
}

PetscErrorCode optimizationMonitorReacDiff (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    std::stringstream s;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PetscInt its;
    ScalarType J = 0, gnorm = 0, cnorm = 0 , step = 0, D = 0, J0 = 0, D0 = 0, gnorm0 = 0;
    Vec x = nullptr;
    char msg[256];
    std::string statusmsg;
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

    ScalarType norm_gref;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);  CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &x);                                       CHKERRQ(ierr);

    Vec tao_grad;
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr =  TaoGetGradientVector(tao, &tao_grad);                               CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
      norm_gref = gnorm;
      itctx->optfeedback_->gradnorm0 = norm_gref;
      itctx->update_reference_gradient = false;
      s <<" updated reference gradient for relative convergence criterion, Quasi-Newton solver: " << itctx->optfeedback_->gradnorm0;
      ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();
    }
    #endif

    // accumulate number of newton iterations
    itctx->optfeedback_->nb_newton_it++;

    ScalarType *x_ptr, *x_full_ptr;
    ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
    ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

    x_full_ptr[itctx->n_misc_->np_] = x_ptr[0];   // k1
    if (itctx->n_misc_->nk_ > 1) x_full_ptr[itctx->n_misc_->np_ + 1] = x_ptr[1];  // k2
    if (itctx->n_misc_->nk_ > 2) x_full_ptr[itctx->n_misc_->np_ + 2] = x_ptr[2];  // k3
    x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_] = x_ptr[itctx->n_misc_->nk_];  // rho
    if (itctx->n_misc_->nr_ > 1) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 1] = x_ptr[itctx->n_misc_->nk_ + 1];  // r1
    if (itctx->n_misc_->nr_ > 2) x_full_ptr[itctx->n_misc_->np_ + itctx->n_misc_->nk_ + 2] = x_ptr[itctx->n_misc_->nk_ + 2];  // r1

    ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
    ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

    ierr = itctx->tumor_->phi_->apply (itctx->tumor_->c_0_, itctx->x_old);                 CHKERRQ (ierr);
    //Prints a warning if tumor IC is clipped
    ierr = checkClipping (itctx->tumor_->c_0_, itctx->n_misc_);           CHKERRQ (ierr);

    ScalarType mx, mn;
    ierr = VecMax (itctx->tumor_->c_t_, NULL, &mx); CHKERRQ (ierr);
    ierr = VecMin (itctx->tumor_->c_t_, NULL, &mn); CHKERRQ (ierr);
    // this print helps determine if theres any large aliasing errors which is causing ls failure etc
    s << " ---------- tumor c(1) bounds: max = " << mx << ", min = " << mn << " ----------- ";
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();

    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   ";

            s << std::setw(18) << "r1";
            if (itctx->n_misc_->nr_ > 1) s << std::setw(18) << "r2";
            if (itctx->n_misc_->nr_ > 2) s << std::setw(18) << "r3";
            s << std::setw(18) << "k1";
            if (itctx->n_misc_->nk_ > 1) s << std::setw(18) << "k2";
            if (itctx->n_misc_->nk_ > 2) s << std::setw(18) << "k3";


        ierr = tuMSGstd ("starting optimization for only biophysical parameters");                   CHKERRQ(ierr);

        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                             CHKERRQ(ierr);
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }

    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->optfeedback_->gradnorm0
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      ;

      ierr = VecGetArray(x, &x_ptr);                                         CHKERRQ(ierr);
      s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->nk_];
      if (itctx->n_misc_->nr_ > 1) s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->nk_ + 1];
      if (itctx->n_misc_->nr_ > 2) s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->nk_ + 2];
      s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[0];
      if (itctx->n_misc_->nk_ > 1) {
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[1];
      }
      if (itctx->n_misc_->nk_ > 2) {
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[2];
      }


    ierr = VecRestoreArray(x, &x_ptr);                                          CHKERRQ(ierr);

    ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
    s.str ("");
    s.clear ();

    if (itctx->n_misc_->writeOutput_ && itctx->n_misc_->verbosity_ >= 4 && its % 5 == 0) {
        s << "c1guess_paraminvitr-" << its << ".nc";
        dataOut (itctx->tumor_->c_t_, itctx->n_misc_, s.str().c_str());
    }
    s.str(std::string()); s.clear();


    //ierr = PetscPrintf (PETSC_COMM_WORLD, "\nKSP number of krylov iterations: %d\n", itctx->optfeedback_->nb_krylov_it);          CHKERRQ(ierr);
    //itctx->optfeedback_->nb_krylov_it = 0;

    //Gradient check begin
    // ierr = itctx->derivative_operators_->checkGradient (itctx->x_old, itctx->data);
    //Gradient check end
    PetscFunctionReturn (ierr);
}

PetscErrorCode optimizationMonitorL1 (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    PetscInt its;
    ScalarType J = 0, gnorm = 0, cnorm = 0 , step = 0, D = 0, J0 = 0, D0 = 0, gnorm0 = 0;
    Vec x = nullptr;
    char msg[256];
    std::string statusmsg;
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

    // get current iteration, objective value, norm of gradient, norm of
    // norm of contraint, step length / trust region readius of iteratore
    // and termination reason
    Vec tao_x;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);      CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &tao_x);                                   CHKERRQ(ierr);

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
      Vec dJ, p0;
      ScalarType norm_gref = 0.;
      ierr = VecDuplicate (itctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
      ierr = VecDuplicate (itctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
      ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
      ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);

      if (itctx->n_misc_->flag_reaction_inv_) {
        norm_gref = gnorm;
      } else {
        ierr = evaluateGradient(tao, p0, dJ, (void*) itctx);
        ierr = VecNorm (dJ, NORM_2, &norm_gref);                                  CHKERRQ(ierr);
      }
      itctx->optfeedback_->gradnorm0 = norm_gref;
      //ctx->gradnorm0 = gnorm;
      itctx->update_reference_gradient = false;
      std::stringstream s; s <<" updated reference gradient for relative convergence criterion, Gauss-Newton solver: " << itctx->optfeedback_->gradnorm0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
      if (dJ  != nullptr) {VecDestroy (&dJ);  CHKERRQ(ierr);  dJ  = nullptr;}
      if (p0  != nullptr) {VecDestroy (&p0);  CHKERRQ(ierr);  p0  = nullptr;}
    }
    #endif

    TaoLineSearch ls = nullptr;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    LSCtx *lsctx = (LSCtx*) ls->data;
    PetscReal J_old = lsctx->J_old;

    // accumulate number of newton iterations
    itctx->optfeedback_->nb_newton_it++;
    // print out Newton iteration information

    std::stringstream s;
    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "|objective|_rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   " << std::setw(10) << "sparsity";
          if (itctx->n_misc_->diffusivity_inversion_) {
            s << std::setw(18) << "k";
          }

        ierr = tuMSGstd (" User defined L1 optimization");                          CHKERRQ(ierr);
        ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
        ierr = tuMSGstd ("------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }
    ScalarType sparsity;
    ierr = vecSparsity (tao_x, sparsity);                                  CHKERRQ (ierr);
    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << PetscAbsReal (J - J_old)
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << sparsity;
      if (itctx->n_misc_->diffusivity_inversion_) {
        ScalarType *x_ptr;
        ierr = VecGetArray(tao_x, &x_ptr);                                         CHKERRQ(ierr);
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->np_];
        if (itctx->n_misc_->nk_ > 1) {
          s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->n_misc_->np_ + 1]; }
        ierr = VecRestoreArray(tao_x, &x_ptr);                                     CHKERRQ(ierr);
      }


    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");
    s.clear ();

    if (itctx->n_misc_->lambda_continuation_) {
      ScalarType sparsity = 0;
      ScalarType sparsity_old = 0;
      ScalarType threshold = itctx->n_misc_->target_sparsity_;
      if (its > 0) {
        //lambda continuation
        ierr = vecSparsity (tao_x, sparsity);
        ierr = vecSparsity (lsctx->x_work_2, sparsity_old);

        if (sparsity >= threshold) {//Sparse solution
          itctx->flag_sparse = true;
          itctx->lam_right = itctx->n_misc_->lambda_;
          itctx->n_misc_->lambda_ = 0.5 * (itctx->lam_left + itctx->lam_right);
          lsctx->lambda = itctx->n_misc_->lambda_;
        }

        if (sparsity < threshold && sparsity_old >= threshold && its > 1) {
          itctx->lam_left = itctx->n_misc_->lambda_;
          itctx->n_misc_->lambda_ = 0.5 * (itctx->lam_left + itctx->lam_right);
          lsctx->lambda = itctx->n_misc_->lambda_;
        }

        if (sparsity < threshold && sparsity_old < threshold && itctx->flag_sparse) {
          itctx->lam_left = itctx->n_misc_->lambda_;
          itctx->n_misc_->lambda_ = 0.5 * (itctx->lam_left + itctx->lam_right);
          lsctx->lambda = itctx->n_misc_->lambda_;
        }

        s << "Lambda continuation on: Lambda = : " << std::scientific << lsctx->lambda;
        ierr = tuMSGstd(s.str());                                             CHKERRQ(ierr);
        s.str(std::string());
        s.clear();
      }
    }

    //ierr = PetscPrintf (PETSC_COMM_WORLD, "\nKSP number of krylov iterations: %d\n", itctx->optfeedback_->nb_krylov_it);          CHKERRQ(ierr);
    //itctx->optfeedback_->nb_krylov_it = 0;

    //Gradient check begin
    //ierr = itctx->derivative_operators_->checkGradient (tao_x, itctx->data);
    //Gradient check end
    PetscFunctionReturn (ierr);
}

/* ------------------------------------------------------------------- */
/*
 optimizationMonitor    mointors the inner PCG Krylov solve to invert the Hessian
 input parameters:
  . KSP ksp          KSP solver object
    . PetscIntn        iteration number
    . PetscRela rnorm  l2-norm (preconditioned) of residual
  . void *ptr        optional user defined context
 */
PetscErrorCode hessianKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec x; int maxit; PetscScalar divtol, abstol, reltol;
    ierr = KSPBuildSolution (ksp,NULL,&x);
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);             CHKERRQ(ierr);                                                             CHKERRQ(ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
    itctx->optfeedback_->nb_krylov_it++;                // accumulate number of krylov iterations

    std::stringstream s;
    if (its == 0) {
      s << std::setw(3)  << " PCG:" << " computing solution of hessian system (tol="
        << std::scientific << std::setprecision(5) << reltol << ")";
      ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
      s.str (""); s.clear ();
    }
    s << std::setw(3)  << " PCG:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
    << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
    ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    s.str (""); s.clear ();

    // int ksp_itr;
    // ierr = KSPGetIterationNumber (ksp, &ksp_itr);                                 CHKERRQ (ierr);
    // ScalarType e_max, e_min;
    // if (ksp_itr % 10 == 0 || ksp_itr == maxit) {
    //   ierr = KSPComputeExtremeSingularValues (ksp, &e_max, &e_min);       CHKERRQ (ierr);
    //   s << "Condition number of hessian is: " << e_max / e_min << " | largest singular values is: " << e_max << ", smallest singular values is: " << e_min << std::endl;
    //   ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    //   s.str (""); s.clear ();
    // }
	PetscFunctionReturn (ierr);
}

/* ------------------------------------------------------------------- */
/*
 constHessianKSPMonitor    mointors the PCG Krylov solve for the constant apx of initial hessian for lmvm
 input parameters:
  . KSP ksp          KSP solver object
    . PetscIntn        iteration number
    . PetscRela rnorm  l2-norm (preconditioned) of residual
  . void *ptr        optional user defined context
 */
PetscErrorCode constHessianKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec x; int maxit; PetscScalar divtol, abstol, reltol;
    ierr = KSPBuildSolution (ksp,NULL,&x);
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);             CHKERRQ(ierr);                                                             CHKERRQ(ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);     // get user context

    std::stringstream s;
    if (its == 0) {
      s << std::setw(3)  << " PCG:" << " invert constant apx H = (beta Phi^T Phi) as initial guess for L-BFGS  (tol="
        << std::scientific << std::setprecision(5) << reltol << ")";
      ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
      s.str (""); s.clear ();
    }
    s << std::setw(3)  << " PCG:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
    << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
    ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    s.str (""); s.clear ();
    PetscFunctionReturn (ierr);
}



/* ------------------------------------------------------------------- */
/*
 preKrylovSolve    preprocess right hand side and initial condition before entering
                   the krylov subspace method; in the context of numerical optimization
                                     this means we preprocess the gradient and the incremental control variable

 input parameters:
  . KSP ksp       KSP solver object
    . Vec b         right hand side
    . Vec x         solution vector
  . void *ptr     optional user defined context
 */
PetscErrorCode preKrylovSolve (KSP ksp, Vec b, Vec x, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscReal gnorm = 0., g0norm = 1., reltol, abstol = 0., divtol = 0., uppergradbound, lowergradbound;
    PetscInt maxit;
    int nprocs, procid;
    MPI_Comm_rank (PETSC_COMM_WORLD, &procid);
    MPI_Comm_size (PETSC_COMM_WORLD, &nprocs);

    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    ierr = VecNorm (b, NORM_2, &gnorm);                                                 CHKERRQ(ierr);   // compute gradient norm
    if(! itctx->is_ksp_gradnorm_set) {                // set initial gradient norm
        itctx->ksp_gradnorm0 = gnorm;                 // for KSP Hessian solver
        itctx->is_ksp_gradnorm_set = true;
    }
    g0norm = itctx->ksp_gradnorm0;                      // get reference gradient
    gnorm /= g0norm;                                    // normalize gradient
                                                        // get tolerances
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);                   CHKERRQ(ierr);
    uppergradbound = 0.5;                               // assuming quadratic convergence
    lowergradbound = 1E-10;
    // user forcing sequence to estimate adequate tolerance for solution of
    //  KKT system (Eisenstat-Walker)
    if (itctx->optsettings_->fseqtype == QDFS) {
        // assuming quadratic convergence (we do not solver more accurately than 12 digits)
        reltol = PetscMax(lowergradbound, PetscMin(uppergradbound, gnorm));
    }
    else {
        // assuming superlinear convergence (we do not solver  more accurately than 12 digitis)
        reltol = PetscMax (lowergradbound, PetscMin (uppergradbound, std::sqrt(gnorm)));
    }
    // overwrite tolerances with estimate
    ierr = KSPSetTolerances (ksp, reltol, abstol, divtol, maxit);                       CHKERRQ(ierr);

    //if (procid == 0){
    //    std::cout << " ksp rel-tol (Eisenstat/Walker): " << reltol << ", grad0norm: " << g0norm<<", gnorm/grad0norm: " << gnorm << std::endl;
    //}
    PetscFunctionReturn (ierr);
}


PetscErrorCode checkConvergenceGradMassEffect (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscInt its, nl, ng;
    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
    PetscReal jx, jxold, gtolbound, theta, normx, normdx, tolj, tolx, tolg;
    const int nstop = 7;
    bool stop[nstop];
    int verbosity;
    std::stringstream ss;
    Vec x = nullptr, g = nullptr;
    ierr = TaoGetSolutionVector(tao, &x);                                     CHKERRQ(ierr);
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->optsettings_->verbosity;
    minstep = ctx->optsettings_->ls_minstep;
    miniter = ctx->optsettings_->newton_minit;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
    #endif

    // get line-search status
    nl = ctx->n_misc_->n_local_;
    ng = ctx->n_misc_->n_global_;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = VecDuplicate (x, &g);
    ierr = TaoLineSearchGetSolution(ls, x, &J, g, &step, &ls_flag);             CHKERRQ (ierr);
    // display line-search convergence reason
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    ierr = TaoGetMaximumIterations(tao, &maxiter);                              CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);
    jx = J;
    Vec tao_grad;
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr = TaoGetGradientVector(tao, &tao_grad);                                CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    // update/set reference gradient 
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if (ctx->update_reference_gradient) {
        ctx->optfeedback_->gradnorm0 = gnorm;
        ctx->update_reference_gradient = false;
        std::stringstream s; s <<" updated reference gradient for relative convergence criterion: " << ctx->optfeedback_->gradnorm0;
        ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    }
    #endif
    

    g0norm = ctx->optfeedback_->gradnorm0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    // compute tolerances for stopping conditions
    tolj = grtol;
    tolx = std::sqrt(grtol);
    #if __cplusplus > 199711L
    tolg = std::cbrt(grtol);
    #else
    tolg = std::pow(grtol, 1.0/3.0);
    #endif
    // compute theta
    theta = 1.0 + std::abs(ctx->optfeedback_->j0);
    ierr = VecNorm (x, NORM_2, &normx);                                                CHKERRQ(ierr);  // comp norm x
    ierr = VecAXPY (ctx->x_old, -1.0, x);                                              CHKERRQ(ierr);  // comp dx
    ierr = VecNorm (ctx->x_old, NORM_2, &normdx);                                      CHKERRQ(ierr);  // comp norm dx
    ierr = VecCopy (x, ctx->x_old);                                                    CHKERRQ(ierr);  // save old x
    // get old objective function value
    jxold = ctx->jvalold;
    ctx->convergence_message.clear();

    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
          ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
          ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
          ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
          ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn(ierr);
    }
    //if(verbosity >= 1) {
    //    ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    //}
    // only check convergence criteria after a certain number of iterations
    // initialize flags for stopping conditions
    for (int i = 0; i < nstop; i++)
        stop[i] = false;
    ctx->optfeedback_->converged     = false;
    ctx->cosamp_->converged_l2       = false;
    ctx->cosamp_->converged_error_l2 = false;
    if (iter >= miniter && iter > 1) {
        if (step < minstep) {
            ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
            ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
            ss.str(std::string());
            ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
        if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
            ss << "step  = " << std::scientific << step << ". ls failed with status " << ls_flag;
            ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
            ss.str(std::string());
                  ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
        // |j(x_{k-1}) - j(x_k)| < tolj*abs(1+J)
        if (std::abs(jxold-jx) < tolj*theta) {
            stop[0] = true;
        }
        ss << "  " << stop[0] << "    |dJ|  = " << std::setw(18)
        << std::right << std::scientific << std::abs(jxold-jx) << "    <    "
        << std::left << std::setw(18) << tolj*theta << " = " << "tol*|1+J|";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||dx|| < sqrt(tolj)*(1+||x||)
        if (normdx < tolx*(1+normx)) {
           stop[1] = true;
        }
        ss << "  " << stop[1] << "    |dx|  = " << std::setw(18)
        << std::right << std::scientific << normdx << "    <    "
        << std::left << std::setw(18) << tolx*(1+normx) << " = " << "sqrt(tol)*(1+||x||)";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||g_k||_2 < cbrt(tolj)*abs(1+Jc)
        if (gnorm < tolg*theta) {
            stop[2] = true;
        }
        ss  << "  " << stop[2] << "    ||g|| = " << std::setw(18)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(18) << tolg*theta << " = " << "cbrt(tol)*|1+J|";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||g_k||_2 < tol
        if (gnorm < gatol) {
                stop[3] = true;
        }
        ss  << "  " << stop[3] << "    ||g|| = " << std::setw(18)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(18) << gatol << " = " << "tol";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        if (gnorm < gttol*g0norm) {
          stop[4] = true;
      }
        ss << "  " << stop[0] << "    ||g|| = " << std::setw(18)
         << std::right << std::scientific << gnorm << "    <    "
         << std::left << std::setw(18) << gttol*g0norm << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        if (iter > maxiter) {
                stop[5] = true;
        }
        ss  << "  " << stop[5] << "    iter  = " << std::setw(18)
          << std::right << iter  << "    >    "
          << std::left << std::setw(18) << maxiter << " = " << "maxiter";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        //     if (iter > iterbound) {
        //             stop[6] = true;
        //     }
        //     ss  << "  " << stop[6] << "    iter  = " << std::setw(14)
              // << std::right << iter  << "    >    "
              // << std::left << std::setw(14) << iterbound << " = " << "iterbound";
        //     ctx->convergence_message.push_back(ss.str());
        //     ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        //     ss.str(std::string());
        //     ss.clear();

        // store objective function value
        ctx->jvalold = jx;

        if (stop[0] && stop[1] && stop[2]) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER);                  CHKERRQ(ierr);
            ctx->optfeedback_->converged = true;
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        } else if (stop[3]) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                 CHKERRQ(ierr);
            ctx->optfeedback_->converged = true;
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        } else if (stop[4]) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);                 CHKERRQ(ierr);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            ctx->optfeedback_->converged = true;
            PetscFunctionReturn (ierr);
        } else if (stop[5]) {
            ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                 CHKERRQ(ierr);
            ctx->optfeedback_->converged = true;
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
    }
    else {
        // if the gradient is zero, we should terminate immediately
        if (gnorm < gatol) {
            ss << "||g|| = " << std::scientific << " < " << gatol;
            ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);ss.str(std::string()); ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
        // store objective function value
        ctx->jvalold = jx;
    }
    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
}



/* Convergence tests used for L1 regularization: Relative change in objective and solution is
   monitored. Linesearch is user-defined with the tao solver */
PetscErrorCode checkConvergenceFun (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscInt its, nl, ng;
    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep, J_old;
    int verbosity;
    bool stop[2];
    std::stringstream ss, sc;
    Vec x = nullptr, g = nullptr;
    ierr = TaoGetSolutionVector(tao, &x);                                     CHKERRQ(ierr);
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;
    ScalarType norm_g_inf;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->optsettings_->verbosity;
    minstep = ctx->optsettings_->ls_minstep;
    miniter = ctx->optsettings_->newton_minit;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
    ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #else
    ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
    #endif

    // get line-search status
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = VecDuplicate (ctx->tumor_->p_, &g);                                  CHKERRQ(ierr);
    ierr = TaoLineSearchGetSolution (ls, x, &J, g, &step, &ls_flag);            CHKERRQ (ierr);
    // display line-search convergence reason
    ierr = dispLineSearchStatus (tao, ctx, ls_flag);                             CHKERRQ(ierr);
    ierr = TaoGetMaximumIterations (tao, &maxiter);                              CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus (tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);

    //Get ls context
    LSCtx *lsctx = (LSCtx*) ls->data;

    // update/set reference gradient (with p = initial-guess)
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if (ctx->update_reference_objective) {
    ScalarType g_percent = 0.1;
    Vec p0, dJ;
    ierr = VecDuplicate (ctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
    ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
    // ierr = VecSet (p0, ctx->n_misc_->p_scale_);                               CHKERRQ(ierr);
    ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
    //Check for infeasible lambda values
    evaluateGradient(tao, x, dJ, (void*) ctx);
    ierr = VecNorm (dJ, NORM_INFINITY, &norm_g_inf);                           CHKERRQ (ierr);

    // if (ctx->n_misc_->lambda_ >= norm_g_inf) {
      ctx->n_misc_->lambda_ = norm_g_inf - g_percent * norm_g_inf;
      ctx->lam_right = ctx->n_misc_->lambda_;
      ctx->lam_left = 0;
      lsctx->lambda = ctx->n_misc_->lambda_;
    // }
    //Evaluate objective for reference using the correct lambda
    evaluateObjectiveFunction (tao, x, &ctx->optfeedback_->j0, (void*) ctx);
    std::stringstream s; s <<"updated reference objective for relative convergence criterion, ISTA L1 solver: " << ctx->optfeedback_->j0 << " ; regularization : " << ctx->n_misc_->lambda_;
    ctx->update_reference_objective = false;
    ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    if (dJ  != nullptr) {VecDestroy (&dJ);  CHKERRQ(ierr);  dJ  = nullptr;}
    if (p0  != nullptr) {VecDestroy (&p0);  CHKERRQ(ierr);  p0  = nullptr;}
    }
    #endif

    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
    ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
    ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
    ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
    ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn(ierr);
    }
    // only check convergence criteria after a certain number of iterations
    ctx->optfeedback_->converged = false;
    //objective and solution convergence check
    //J_old : prev objective
    //J : current objective
    //J_ref : ref objective

    PetscReal J_ref = ctx->optfeedback_->j0;
    J_old = lsctx->J_old;
    ScalarType ftol = ctx->optsettings_->ftol;
    ScalarType norm_rel, norm;
    stop[0] = false; stop[1] = false;
    if (iter >= miniter) {
    if (verbosity > 1) {
      ss << "step size in linesearch: " << std::scientific << step;
      ierr = tuMSGstd(ss.str());                                             CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
    }
    if (step < minstep) {
      ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);              CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
    if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
      ss << "step  = " << std::scientific << step << ". ls failed. ";
      ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
      ss.str(std::string());
      ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }

    ierr = VecAXPY (lsctx->x_work_2, -1.0, x);                                CHKERRQ (ierr);
    ierr = VecNorm (lsctx->x_work_2, NORM_INFINITY, &norm_rel);               CHKERRQ (ierr);
    ierr = VecNorm (x, NORM_2, &norm);                                        CHKERRQ (ierr);

    ss.str(std::string());
    ss.clear();
    if (PetscAbsReal (J_old - J) < ftol * PetscAbsReal (1 + J_ref) && norm_rel < std::sqrt (ftol) * (1 + norm)) {  //L1 convergence check for objective and solution change
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER);                  CHKERRQ(ierr);
      stop[0] = true;
    }
    ss  << "  " << stop[0] << "    |J_old - J| = " << std::setw(14)
        << std::right << std::scientific << std::abs (J_old - J) << "    <    "
        << std::left << std::setw(14) << ftol * PetscAbsReal (1 + J_ref) << " = " << "tol * |1 + J_ref| = " << ftol << " * |1 + " << J_ref << "|";
    ctx->convergence_message.push_back(ss.str());
    ss  << "  " << stop[0] << "    ||x_old - x||_inf = " << std::setw(14)
        << std::right << std::scientific << norm_rel << "    <    "
        << std::left << std::setw(14) << std::sqrt (ftol) * (1 + norm) << " = " << "sqrt (ftol) * (1 + ||x||_inf^k) = " << std::sqrt (ftol) << " * (1 + " << norm << ")";
    ctx->convergence_message.push_back(ss.str());
    if(verbosity >= 3) {
      ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
    }
    ss.str(std::string());
      ss.clear();
    // iteration number exceeds limit
    if (iter > maxiter) {
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                 CHKERRQ(ierr);
      stop[1] = true;
    }
    ss  << "  " << stop[1] << "     iter = " << std::setw(14)
        << std::right << iter  << "    >    "
        << std::left << std::setw(14) << maxiter << " = " << "maxiter";
    ctx->convergence_message.push_back(ss.str());
    if(verbosity >= 3) {
      ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
    }
    ss.str(std::string());
      ss.clear();
    // store objective function value
    ctx->jvalold = J;
    if (stop[0] || stop[1]) {
      ctx->optfeedback_->converged = true;
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
    }

    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);

    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}

    PetscFunctionReturn (ierr);
}
/* ------------------------------------------------------------------- */
/*
 checkConvergenceGrad    checks convergence of the overall Gauss-Newton tumor inversion

 input parameters:
  . Tao tao       Tao solver object
  . void *ptr     optional user defined context
 */
PetscErrorCode checkConvergenceGrad (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscInt its, nl, ng;
    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
    bool stop[3];
    int verbosity;
    std::stringstream ss;
    Vec x = nullptr, g = nullptr;
    ierr = TaoGetSolutionVector(tao, &x);                                     CHKERRQ(ierr);
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->optsettings_->verbosity;
    minstep = ctx->optsettings_->ls_minstep;
    miniter = ctx->optsettings_->newton_minit;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
    #endif

    // get line-search status
    nl = ctx->n_misc_->n_local_;
    ng = ctx->n_misc_->n_global_;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = VecDuplicate (ctx->tumor_->p_, &g);                                  CHKERRQ(ierr);
    ierr = TaoLineSearchGetSolution(ls, x, &J, g, &step, &ls_flag);             CHKERRQ (ierr);

    // display line-search convergence reason
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    ierr = TaoGetMaximumIterations(tao, &maxiter);                              CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);

    Vec tao_grad;
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr = TaoGetGradientVector(tao, &tao_grad);                                CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    // update/set reference gradient (with p = zeros)
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if (ctx->update_reference_gradient) {
        Vec dJ = nullptr, p0 = nullptr;
        ScalarType norm_gref = 0.;
        ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
        ierr = VecDuplicate (ctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
        ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
        ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);

        if (ctx->n_misc_->flag_reaction_inv_) {
        norm_gref = gnorm;
        } else {
          ierr = evaluateGradient(tao, p0, dJ, (void*) ctx);
          ierr = VecNorm (dJ, NORM_2, &norm_gref);                                  CHKERRQ(ierr);
        }
        ctx->optfeedback_->gradnorm0 = norm_gref;
        //ctx->gradnorm0 = gnorm;
        ctx->update_reference_gradient = false;
        std::stringstream s; s <<" updated reference gradient for relative convergence criterion: " << ctx->optfeedback_->gradnorm0;
        ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
        if (dJ != nullptr) {ierr = VecDestroy(&dJ); CHKERRQ(ierr); dJ = nullptr;}
        if (p0 != nullptr) {ierr = VecDestroy(&p0); CHKERRQ(ierr); p0 = nullptr;}
    }
    #endif
    // get initial gradient
    g0norm = ctx->optfeedback_->gradnorm0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    ctx->convergence_message.clear();

    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
          ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
          ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
          ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
          ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn(ierr);
    }
    //if(verbosity >= 1) {
    //    ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    //}
    // only check convergence criteria after a certain number of iterations
    stop[0] = false; stop[1] = false; stop[2] = false;
    ctx->optfeedback_->converged     = false;
    ctx->cosamp_->converged_l2       = false;
    ctx->cosamp_->converged_error_l2 = false;
    if (iter >= miniter) {
        if (verbosity > 1) {
                ss << "step size in linesearch: " << std::scientific << step;
                ierr = tuMSGstd(ss.str());                                            CHKERRQ(ierr);
                ss.str(std::string());
                ss.clear();
        }
        if (step < minstep) {
                ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
                ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
                ss.str(std::string());
                ss.clear();
                ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
                if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
                ctx->cosamp_->converged_error_l2 = true;
                PetscFunctionReturn(ierr);
        }
        if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
            ss << "step  = " << std::scientific << step << ". ls failed with status " << ls_flag;
            ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
            ss.str(std::string());
                  ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            ctx->cosamp_->converged_error_l2 = true;
            PetscFunctionReturn(ierr);
        }
        // ||g_k||_2 < tol*||g_0||
        if (gnorm < gttol*g0norm) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);               CHKERRQ(ierr);
            stop[0] = true;
        }
        ss << "  " << stop[0] << "    ||g|| = " << std::setw(14)
           << std::right << std::scientific << gnorm << "    <    "
           << std::left << std::setw(14) << gttol*g0norm << " = " << "tol";
        ctx->convergence_message.push_back(ss.str());
        if(verbosity >= 3) { ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); } ss.str(std::string()); ss.clear();

        // ||g_k||_2 < tol
        if (gnorm < gatol) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
            stop[1] = true;
        }
        ss  << "  " << stop[1] << "    ||g|| = " << std::setw(14)
            << std::right << std::scientific << gnorm << "    <    "
            << std::left << std::setw(14) << gatol << " = " << "tol";
        ctx->convergence_message.push_back(ss.str());
        if(verbosity >= 3) { ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); } ss.str(std::string()); ss.clear();

        // iteration number exceeds limit
        if (iter > maxiter) {
            ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                    CHKERRQ(ierr);
            stop[2] = true;
        }
        ss  << "  " << stop[2] << "     iter = " << std::setw(14)
              << std::right << iter  << "    >    "
              << std::left << std::setw(14) << maxiter << " = " << "maxiter";
        ctx->convergence_message.push_back(ss.str());
        if(verbosity >= 3) { ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); } ss.str(std::string()); ss.clear();

        // store objective function value
        ctx->jvalold = J;
        if (stop[0] || stop[1]) {ctx->cosamp_->converged_l2 = true;} // for CoSaMpRS to split up L2 solve
        if (stop[0] || stop[1] || stop[2]) {
            ctx->optfeedback_->converged = true;
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn(ierr);
        }

    // iter < miniter
    } else {
        // if the gradient is zero, we should terminate immediately
        if (gnorm == 0) {
            ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
            ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);
            ss.str(std::string());
            ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            ctx->cosamp_->converged_l2 = true;
            PetscFunctionReturn(ierr);
        }
    }
    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
}

/* ------------------------------------------------------------------- */
/*
 checkConvergenceGradObj    checks convergence of the overall Gauss-Newton tumor inversion

 input parameters:
  . Tao tao       Tao solver object
  . void *ptr     optional user defined context
 */
PetscErrorCode checkConvergenceGradObj (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PetscInt iter, maxiter, miniter, iterbound;
    PetscReal jx, jxold, gnorm, step, gatol, grtol, gttol, g0norm, gtolbound, minstep, theta, normx, normdx, tolj, tolx, tolg;
    const int nstop = 7;
    bool stop[nstop];
    std::stringstream ss;
    Vec x = nullptr, g = nullptr;

    CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
    // get minstep and miniter
    minstep = ctx->optsettings_->ls_minstep;
    miniter = ctx->optsettings_->newton_minit;
    iterbound = ctx->optsettings_->iterbound;
    // get lower bound for gradient
    gtolbound = ctx->optsettings_->gtolbound;

    // get and display line-search convergence reason
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = TaoLineSearchGetSolution(ls, x, &jx, g, &step, &ls_flag);            CHKERRQ (ierr);
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    // create temp vector (make sure it's deleted)
    ierr = VecDuplicate (ctx->tumor_->p_, &g);                                  CHKERRQ(ierr);
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances(tao, &gatol, &grtol, &gttol);                               CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances(tao, NULL, NULL, &gatol, &grtol, &gttol);                   CHKERRQ(ierr);
    #endif
    // get maxiter
    ierr = TaoGetMaximumIterations(tao, &maxiter);                                          CHKERRQ(ierr);
    if (maxiter > iterbound) iterbound = maxiter;
    ierr = TaoGetMaximumIterations(tao, &maxiter);                                          CHKERRQ(ierr);
    // get solution status
    ierr = TaoGetSolutionStatus(tao, &iter, &jx, &gnorm, NULL, &step, NULL);                CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &x);                                                   CHKERRQ(ierr);

    // update/set reference gradient (with p = zeros)
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if(ctx->update_reference_gradient) {
      Vec dJ, p0;
      ScalarType norm_gref = 0.;
      ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
      ierr = VecDuplicate (ctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
      ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
      ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);
      // evaluateGradient(tao, x, dJ, (void*) ctx);
      evaluateObjectiveFunctionAndGradient (tao, p0, &ctx->optfeedback_->j0, dJ, (void*) ctx);
        ierr = VecNorm (dJ, NORM_2, &norm_gref); CHKERRQ(ierr);
        ctx->optfeedback_->gradnorm0 = norm_gref;
      // evaluateObjectiveFunction (tao, x, &ctx->optfeedback_->j0, (void*) ctx);

      std::stringstream s; s <<"updated reference objective for relative convergence criterion: " << ctx->optfeedback_->j0;
        ctx->update_reference_gradient = false;
      ierr = tuMSGstd(s.str());                                                     CHKERRQ(ierr);
      s.str(std::string());
      s.clear();
      s <<" updated reference gradient for relative convergence criterion: " << ctx->optfeedback_->gradnorm0;
        ierr = tuMSGstd(s.str());                                                          CHKERRQ(ierr);
      s.str(std::string());
      s.clear();
      if (dJ != nullptr) {ierr = VecDestroy(&dJ); CHKERRQ(ierr); dJ = nullptr;}
      if (p0 != nullptr) {ierr = VecDestroy(&p0); CHKERRQ(ierr); p0 = nullptr;}
    }
    #endif
    // get initial gradient
    g0norm = ctx->optfeedback_->gradnorm0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    // compute tolerances for stopping conditions
    tolj = grtol;
    tolx = std::sqrt(grtol);
    #if __cplusplus > 199711L
    tolg = std::cbrt(grtol);
    #else
    tolg = std::pow(grtol, 1.0/3.0);
    #endif
    // compute theta
    theta = 1.0 + std::abs(ctx->optfeedback_->j0);
    // compute norm(\Phi x^k - \Phi x^k-1) and norm(\Phi x^k)
    // ierr = ctx->tumor_->phi_->apply (ctx->tmp, x);                                         CHKERRQ(ierr);  // comp \Phi x
    // ierr = VecNorm (ctx->tmp, NORM_2, &normx);                                             CHKERRQ(ierr);  // comp norm \Phi x
    // ierr = VecAXPY(ctx->c0old, -1, ctx->tmp);                                              CHKERRQ(ierr);  // comp dx
    // ierr = VecNorm (ctx->c0old, NORM_2, &normdx);                                          CHKERRQ(ierr);  // comp norm \Phi dx
    // ierr = VecCopy(ctx->tmp, ctx->c0old);                                                  CHKERRQ(ierr);  // save \Phi x
    ierr = VecNorm (x, NORM_2, &normx);                                             CHKERRQ(ierr);  // comp norm x
    ierr = VecAXPY (ctx->x_old, -1.0, x);                                                  CHKERRQ(ierr);  // comp dx
    ierr = VecNorm (ctx->x_old, NORM_2, &normdx);                                   CHKERRQ(ierr);  // comp norm dx
    ierr = VecCopy (x, ctx->x_old);                                                        CHKERRQ(ierr);  // save old x
    // get old objective function value
    jxold = ctx->jvalold;
    ctx->convergence_message.clear();

    // check for NaN value
    if (PetscIsInfOrNanReal(jx)) {
          ierr = tuMSGwarn("objective is NaN");                                             CHKERRQ(ierr);
          ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                              CHKERRQ(ierr);
          PetscFunctionReturn(ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
          ierr = tuMSGwarn("||g|| is NaN");                                                 CHKERRQ(ierr);
          ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                              CHKERRQ(ierr);
          PetscFunctionReturn(ierr);
    }

    // ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    ctx->optfeedback_->converged = false;
    // initialize flags for stopping conditions
    for (int i = 0; i < nstop; i++)
        stop[i] = false;
    // only check convergence criteria after a certain number of iterations
    if (iter >= miniter && iter > 1) {
      if (step < minstep) {
                ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
                ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
                ss.str(std::string());
                ss.clear();
                ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
          if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
                PetscFunctionReturn (ierr);
        }
      if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
        ss << "step  = " << std::scientific << step << ". ls failed with status " << ls_flag;
        ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
        ss.str(std::string());
              ss.clear();
        ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
        PetscFunctionReturn (ierr);
      }
        // |j(x_{k-1}) - j(x_k)| < tolj*abs(1+J)
        if (std::abs(jxold-jx) < tolj*theta) {
            stop[0] = true;
        }
        ss << "  " << stop[0] << "    |dJ|  = " << std::setw(18)
        << std::right << std::scientific << std::abs(jxold-jx) << "    <    "
        << std::left << std::setw(18) << tolj*theta << " = " << "tol*|1+J|";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||dx|| < sqrt(tolj)*(1+||x||)
        if (normdx < tolx*(1+normx)) {
           stop[1] = true;
        }
        ss << "  " << stop[1] << "    |dx|  = " << std::setw(18)
        << std::right << std::scientific << normdx << "    <    "
        << std::left << std::setw(18) << tolx*(1+normx) << " = " << "sqrt(tol)*(1+||x||)";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||g_k||_2 < cbrt(tolj)*abs(1+Jc)
        if (gnorm < tolg*theta) {
            stop[2] = true;
        }
        ss  << "  " << stop[2] << "    ||g|| = " << std::setw(18)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(18) << tolg*theta << " = " << "cbrt(tol)*|1+J|";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||g_k||_2 < tol
        if (gnorm < gatol) {
                stop[3] = true;
        }
        ss  << "  " << stop[3] << "    ||g|| = " << std::setw(18)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(18) << gatol << " = " << "tol";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        if (gnorm < gttol*g0norm) {
          stop[4] = true;
      }
        ss << "  " << stop[0] << "    ||g|| = " << std::setw(18)
         << std::right << std::scientific << gnorm << "    <    "
         << std::left << std::setw(18) << gttol*g0norm << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        if (iter > maxiter) {
                stop[5] = true;
        }
        ss  << "  " << stop[5] << "    iter  = " << std::setw(18)
          << std::right << iter  << "    >    "
          << std::left << std::setw(18) << maxiter << " = " << "maxiter";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

    //     if (iter > iterbound) {
    //             stop[6] = true;
    //     }
    //     ss  << "  " << stop[6] << "    iter  = " << std::setw(14)
          // << std::right << iter  << "    >    "
          // << std::left << std::setw(14) << iterbound << " = " << "iterbound";
    //     ctx->convergence_message.push_back(ss.str());
    //     ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    //     ss.str(std::string());
    //     ss.clear();

        // store objective function value
        ctx->jvalold = jx;

        if (stop[0] && stop[1] && stop[2]) {
              ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER);                  CHKERRQ(ierr);
              ctx->optfeedback_->converged = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
              PetscFunctionReturn (ierr);
        } else if (stop[3]) {
              ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                 CHKERRQ(ierr);
              ctx->optfeedback_->converged = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
              PetscFunctionReturn (ierr);
        } else if (stop[4]) {
              ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);                 CHKERRQ(ierr);
          ctx->optfeedback_->converged = true;
              PetscFunctionReturn (ierr);
        } else if (stop[5]) {
              ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                 CHKERRQ(ierr);
              ctx->optfeedback_->converged = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
    }
    else {
        // if the gradient is zero, we should terminate immediately
        if (gnorm < gatol) {
            ss << "||g|| = " << std::scientific << " < " << gatol;
            ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);ss.str(std::string()); ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
      // store objective function value
      ctx->jvalold = jx;
    }

    // if we're here, we're good to go
    ierr = TaoSetConvergedReason(tao, TAO_CONTINUE_ITERATING);                  CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
}




PetscErrorCode checkConvergenceGradReacDiff (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscInt its, nl, ng;
    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
    bool stop[3];
    int verbosity;
    std::stringstream ss;
    Vec x = nullptr, g = nullptr;
    ierr = TaoGetSolutionVector(tao, &x);                                       CHKERRQ(ierr);
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->optsettings_->verbosity;
    minstep = ctx->optsettings_->ls_minstep;
    miniter = ctx->optsettings_->newton_minit;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
    #endif

    // get line-search status
    nl = ctx->n_misc_->n_local_;
    ng = ctx->n_misc_->n_global_;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = VecDuplicate (x, &g);                                  CHKERRQ(ierr);
    ierr = TaoLineSearchGetSolution(ls, x, &J, g, &step, &ls_flag);             CHKERRQ (ierr);
    // display line-search convergence reason
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    ierr = TaoGetMaximumIterations(tao, &maxiter);                              CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);

    Vec tao_grad;
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr = TaoGetGradientVector(tao, &tao_grad);                                CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    double norm_gref = 0.;
    // update/set reference gradient
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if (ctx->update_reference_gradient) {
      norm_gref = gnorm;
      ctx->optfeedback_->gradnorm0 = norm_gref;
      ctx->update_reference_gradient = false;
      std::stringstream s; s <<" updated reference gradient for relative convergence criterion: " << ctx->optfeedback_->gradnorm0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    }
    #endif
    // get initial gradient
    g0norm = ctx->optfeedback_->gradnorm0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    ctx->convergence_message.clear();
    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
      ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
      ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
      ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
    //if(verbosity >= 1) {
    //  ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    //}
    // only check convergence criteria after a certain number of iterations
    stop[0] = false; stop[1] = false; stop[2] = false;
    ctx->optfeedback_->converged = false;
    if (iter >= miniter) {
      if (verbosity > 1) {
          ss << "step size in linesearch: " << std::scientific << step;
          ierr = tuMSGstd(ss.str());                                            CHKERRQ(ierr);
          ss.str(std::string());
                ss.clear();
      }
      if (step < minstep) {
          ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
          ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
          ss.str(std::string());
                ss.clear();
          ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
          if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn(ierr);
      }
      if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
        ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
        ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
        ss.str(std::string());
              ss.clear();
        ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
        PetscFunctionReturn(ierr);
      }
      // ||g_k||_2 < tol*||g_0||
      if (gnorm < gttol*g0norm) {
          ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);               CHKERRQ(ierr);
          stop[0] = true;
      }
      ss << "  " << stop[0] << "    ||g|| = " << std::setw(14)
         << std::right << std::scientific << gnorm << "    <    "
         << std::left << std::setw(14) << gttol*g0norm << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
        ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
      ss.str(std::string());
        ss.clear();
      // ||g_k||_2 < tol
      if (gnorm < gatol) {
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
      stop[1] = true;
      }
      ss  << "  " << stop[1] << "    ||g|| = " << std::setw(14)
          << std::right << std::scientific << gnorm << "    <    "
          << std::left << std::setw(14) << gatol << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
        ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
      ss.str(std::string());
        ss.clear();
      // iteration number exceeds limit
      if (iter > maxiter) {
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                    CHKERRQ(ierr);
      stop[2] = true;
      }
      ss  << "  " << stop[2] << "     iter = " << std::setw(14)
          << std::right << iter  << "    >    "
          << std::left << std::setw(14) << maxiter << " = " << "maxiter";
      ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
        ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
      ss.str(std::string());
        ss.clear();
      // store objective function value
      ctx->jvalold = J;
      if (stop[0] || stop[1] || stop[2]) {
        ctx->optfeedback_->converged = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
        PetscFunctionReturn(ierr);
      }

    }
    else {
    // if the gradient is zero, we should terminate immediately
    if (gnorm < gatol) {
      ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
      ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);
      ss.str(std::string());
            ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
    }
    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);

    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}

    PetscFunctionReturn (ierr);
}


PetscErrorCode dispTaoConvReason (TaoConvergedReason flag, std::string &msg) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    switch (flag) {
      case TAO_CONVERGED_GATOL : {
          msg = "solver converged: ||g(x)|| <= tol";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_GRTOL : {
          msg = "solver converged: ||g(x)||/J(x) <= tol";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_GTTOL : {
          msg = "solver converged: ||g(x)||/||g(x0)|| <= tol";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_STEPTOL : {
          msg = "step size too small";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_MINF : {
          msg = "objective value to small";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_USER : {
          msg = "solver converged user";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_MAXITS : {
          msg = "maximum number of iterations reached";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_NAN : {
          msg = "numerical problems (NAN detected)";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_MAXFCN : {
          msg = "maximal number of function evaluations reached";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_LS_FAILURE : {
          msg = "line search failed";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_TR_REDUCTION : {
          msg = "trust region failed";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_USER : {
          msg = "user defined divergence criterion met";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONTINUE_ITERATING : {
          // display nothing
          break;
      }
      default : {
          msg = "convergence reason not defined";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
  }
    PetscFunctionReturn (ierr);
}

PetscErrorCode dispLineSearchStatus(Tao tao, void* ptr, TaoLineSearchConvergedReason flag) {
    PetscErrorCode ierr = 0;
    std::string msg;
    PetscFunctionBegin;

    switch(flag) {
        case TAOLINESEARCH_FAILED_INFORNAN:
        {
            msg = "linesearch: function evaluation gave INF or NaN";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_FAILED_BADPARAMETER:
        {
            msg = "linesearch: bad parameter detected";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_FAILED_ASCENT:
        {
            msg = "linesearch: search direction is not a descent direction";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_MAXFCN:
        {
            msg = "linesearch: maximum number of function evaluations reached";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_UPPERBOUND:
        {
            msg = "linesearch: step size reached upper bound";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_LOWERBOUND:
        {
            msg = "linesearch: step size reached lower bound";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_RTOL:
        {
            msg = "linesearch: range of uncertainty is smaller than given tolerance";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_OTHER:
        {
            msg = "linesearch: stopped (other)";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_CONTINUE_ITERATING:
        {
            // do nothing, cause everything's fine
            break;
        }
        case TAOLINESEARCH_SUCCESS:
        {
            msg = "linesearch: successful";
            ierr = tuMSGstd(msg); CHKERRQ(ierr);
            break;
        }
        default:
        {
            msg = "linesearch: status not defined";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
    }
    PetscFunctionReturn (ierr);
}

PetscErrorCode InvSolver::setTaoOptionsMassEffect (Tao tao, CtxInv *ctx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    TaoLineSearch linesearch;        // line-search object
    std::string msg;

    PetscReal minstep;
    minstep = std::pow (2.0, 20.0);
    minstep = 1.0 / minstep;
    itctx_->optsettings_->ls_minstep = minstep;

    if (itctx_->optsettings_->newtonsolver == QUASINEWTON)  {
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
    ub_ptr[0] = itctx_->n_misc_->gamma_ub_;
    ub_ptr[1] = itctx_->n_misc_->rho_ub_;
    ub_ptr[2] = itctx_->n_misc_->k_ub_;
    itctx_->n_misc_->bounds_array_[0] = ub_ptr[0];
    itctx_->n_misc_->bounds_array_[1] = ub_ptr[1];
    itctx_->n_misc_->bounds_array_[2] = ub_ptr[2];
    ierr = VecRestoreArray (upper_bound, &ub_ptr);                        CHKERRQ (ierr);


    ScalarType *lb_ptr;
    ierr = VecGetArray (lower_bound, &lb_ptr);                            CHKERRQ (ierr);
    lb_ptr[2] = itctx_->n_misc_->k_lb_;
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
    // if (itctx_->n_misc_->regularization_norm_ == wL2) ctx->optsettings_->opttolgrad = 1e-6;
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoSetTolerances (tao, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad); CHKERRQ(ierr);
    #else
        ierr = TaoSetTolerances (tao, 1E-12, 1E-12, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad); CHKERRQ(ierr);
    #endif

    ierr = TaoSetMaximumIterations (tao, ctx->optsettings_->newton_maxit); CHKERRQ(ierr);
    ierr = TaoSetConvergenceTest (tao, checkConvergenceGradMassEffect, ctx);                 CHKERRQ(ierr);

    // set linesearch (only for Gauss-Newton, lmvm uses more-thuente type line-search automatically)
    ierr = TaoGetLineSearch (tao, &linesearch);                                   CHKERRQ(ierr);
    linesearch->stepmin = minstep;

    if (ctx->optsettings_->linesearch == ARMIJO) {
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
    s << "   gatol: "<< ctx->optsettings_->gatol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
    s << "   grtol: "<< ctx->optsettings_->grtol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
    s << "   gttol: "<< ctx->optsettings_->opttolgrad;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();

    ierr = TaoSetFromOptions(tao);                                                CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}

PetscErrorCode InvSolver::setTaoOptions (Tao tao, CtxInv *ctx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    TaoLineSearch linesearch;        // line-search object
    std::string msg;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PetscReal minstep;
    minstep = std::pow (2.0, 18.0);
    minstep = 1.0 / minstep;
    itctx_->optsettings_->ls_minstep = minstep;

    if (itctx_->n_misc_->regularization_norm_ == L1) {
      ierr = TaoSetType (tao, "tao_L1");   CHKERRQ (ierr);
      TaoLineSearch ls;
      ierr = TaoGetLineSearch(tao_, &ls);                                          CHKERRQ (ierr);
      LSCtx *lsctx = (LSCtx*) ls->data;
      lsctx->lambda = ctx->n_misc_->lambda_;
      ls->stepmin = minstep;
    } else {
      if (itctx_->optsettings_->newtonsolver == QUASINEWTON)  {
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
    }

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
    if (itctx_->n_misc_->regularization_norm_ == L1) {
      ierr = TaoSetMonitor (tao, optimizationMonitorL1, (void *) ctx, NULL);        CHKERRQ(ierr);
    }
    else {
      ierr = TaoSetMonitor (tao, optimizationMonitor, (void *) ctx, NULL);          CHKERRQ(ierr);
    }
    // Lower and Upper Bounds
    Vec lower_bound;
    ierr = VecDuplicate (ctx->tumor_->p_, &lower_bound);                            CHKERRQ (ierr);
    ierr = VecSet (lower_bound, 0.);                                                CHKERRQ (ierr);
    Vec upper_bound;
    ierr = VecDuplicate (ctx->tumor_->p_, &upper_bound);                            CHKERRQ (ierr);
    ierr = VecSet (upper_bound, PETSC_INFINITY);                                    CHKERRQ (ierr);

    ScalarType *ub_ptr, *lb_ptr;
    ScalarType upper_bound_kappa = itctx_->n_misc_->k_ub_, lower_bound_kappa = itctx_->n_misc_->k_lb_;
    if (itctx_->n_misc_->diffusivity_inversion_) {
      ierr = VecGetArray (upper_bound, &ub_ptr);                                    CHKERRQ (ierr);
      ub_ptr[itctx_->n_misc_->np_] = upper_bound_kappa;
      if (itctx_->n_misc_->nk_ > 1) ub_ptr[itctx_->n_misc_->np_ + 1] = upper_bound_kappa;
      if (itctx_->n_misc_->nk_ > 2) ub_ptr[itctx_->n_misc_->np_ + 2] = upper_bound_kappa;
      ierr = VecRestoreArray (upper_bound, &ub_ptr);                                CHKERRQ (ierr);

      ierr = VecGetArray (lower_bound, &lb_ptr);                                    CHKERRQ (ierr);
      lb_ptr[itctx_->n_misc_->np_] = lower_bound_kappa;
      if (itctx_->n_misc_->nk_ > 1) lb_ptr[itctx_->n_misc_->np_ + 1] = lower_bound_kappa;
      if (itctx_->n_misc_->nk_ > 2) lb_ptr[itctx_->n_misc_->np_ + 2] = lower_bound_kappa;
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
    // if (itctx_->n_misc_->regularization_norm_ == wL2) ctx->optsettings_->opttolgrad = 1e-6;
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoSetTolerances (tao, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad); CHKERRQ(ierr);
    #else
        ierr = TaoSetTolerances (tao, 1E-12, 1E-12, ctx->optsettings_->gatol, ctx->optsettings_->grtol, ctx->optsettings_->opttolgrad); CHKERRQ(ierr);
    #endif
    if (itctx_->n_misc_->regularization_norm_ == L1) {
      ierr = TaoSetMaximumIterations (tao, ctx->optsettings_->gist_maxit); CHKERRQ(ierr);
    }
    else {
      ierr = TaoSetMaximumIterations (tao, ctx->optsettings_->newton_maxit); CHKERRQ(ierr);
    }

    // if (itctx_->n_misc_->flag_reaction_inv_) {
    //   ierr = TaoSetMaximumIterations (tao, 200); CHKERRQ(ierr);
    // }

    if (itctx_->n_misc_->regularization_norm_ == L1) {
      ierr = TaoSetConvergenceTest (tao, checkConvergenceFun, ctx);                  CHKERRQ(ierr);
    }
    else {
      ierr = TaoSetConvergenceTest (tao, checkConvergenceGrad, ctx);                 CHKERRQ(ierr);
      // ierr = TaoSetConvergenceTest(tao, checkConvergenceGradObj, ctx);              CHKERRQ(ierr);
    }

    if (!itctx_->n_misc_->regularization_norm_ == L1) {  //Set other preferences for standard tao solvers
      // set linesearch (only for Gauss-Newton, lmvm uses more-thuente type line-search automatically)
      ierr = TaoGetLineSearch (tao, &linesearch);                                   CHKERRQ(ierr);
      linesearch->stepmin = minstep;

      if (ctx->optsettings_->linesearch == ARMIJO) {
        ierr = TaoLineSearchSetType (linesearch, "armijo");                          CHKERRQ(ierr);
        tuMSGstd(" using line-search type: armijo");
      } else {
        tuMSGstd(" using line-search type: more-thuene");
      }

      //if(itctx_->optsettings_->newtonsolver == GAUSSNEWTON) {
      //  ierr = TaoLineSearchSetType (linesearch, "armijo");                         CHKERRQ(ierr);
      //}
      ierr = TaoLineSearchSetOptionsPrefix (linesearch,"tumor_");                    CHKERRQ(ierr);

      std::stringstream s;
      tuMSGstd(" parameters (optimizer):");
      tuMSGstd(" tolerances (stopping conditions):");
      s << "   gatol: "<< ctx->optsettings_->gatol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
      s << "   grtol: "<< ctx->optsettings_->grtol;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();
      s << "   gttol: "<< ctx->optsettings_->opttolgrad;  /*pout(s.str(), cplctx->_fileOutput);*/ tuMSGstd(s.str()); s.str(""); s.clear();

      ierr = TaoSetFromOptions(tao);                                                CHKERRQ(ierr);
      /* === set the KSP Krylov solver settings === */
      KSP ksp = PETSC_NULL;

      if (itctx_->optsettings_->newtonsolver == QUASINEWTON)  {
        // if (use_intial_hessian_lmvm_) {
        //   // get the ksp of H0 initial matrix
        //   ierr = TaoLMVMGetH0KSP(tao, &ksp);                                        CHKERRQ(ierr);
        //   if (ksp != PETSC_NULL) {
        //       ierr = KSPSetOptionsPrefix(ksp, "init-hessian_");                     CHKERRQ(ierr);
        //       // set default tolerance to 1E-6
        //       ierr = KSPSetTolerances(ksp, 1E-6, PETSC_DEFAULT, PETSC_DEFAULT, ctx->optsettings_->krylov_maxit); CHKERRQ(ierr);
        //       ierr = KSPMonitorSet(ksp, constHessianKSPMonitor,ctx, 0);              CHKERRQ(ierr);
        //   }
        //}
      } else {
        // get the ksp of the optimizer (use gauss-newton-krylov)
        ierr = TaoGetKSP(tao, &ksp);                                                CHKERRQ(ierr);
        if (ksp != PETSC_NULL) {
            ierr = KSPSetOptionsPrefix(ksp, "hessian_");                            CHKERRQ(ierr);
            // set default tolerance to 1E-6
            ierr = KSPSetTolerances(ksp, 1E-6, PETSC_DEFAULT, PETSC_DEFAULT, ctx->optsettings_->krylov_maxit); CHKERRQ(ierr);
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
    }
    PetscFunctionReturn (ierr);
}
