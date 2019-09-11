#ifndef INVSOLVER_H_
#define INVSOLVER_H_

#include <memory>
#include "DerivativeOperators.h"
#include "Utils.h"


struct InterpolationContext {
    InterpolationContext (std::shared_ptr<NMisc> n_misc) : n_misc_ (n_misc) {
        PetscErrorCode ierr = 0;
        ierr = VecCreate (PETSC_COMM_WORLD, &temp_);
        ierr = VecSetSizes (temp_, n_misc->n_local_, n_misc->n_global_);
        ierr = VecSetFromOptions (temp_);
        ierr = VecSet (temp_, 0);
    }
    std::shared_ptr<Tumor> tumor_;
    std::shared_ptr<NMisc> n_misc_;
    Vec temp_;
    ~InterpolationContext () {
        PetscErrorCode ierr = 0;
        ierr = VecDestroy (&temp_);
    }
};

struct CtxCoSaMp {
    int cosamp_stage;               // indicates solver state of CoSaMp function when using warmstart
    int its_l1;                     // cosamp iterations
    int np_full;                    // size of unrestricted subspace
    int maxit_newton;               // global maxit for L2 Newton solver
    int nits;                       // global iterations performed for L2 Newton solver
    int inexact_nits;               // Newton its per inexact solve
    bool compute_reference_values;  // if true, compute and store reference objective and gradient
    bool converged_l1;              // indicates if L1 solver converged
    bool converged_l2;              // indicates if L2 solver converged
    bool converged_error_l2;        // indicates if L2 solver diverged/failed
    bool initialized;               // indicates if vectors are allocated or destroyed
    PetscReal J;                    // objective function value
    PetscReal J_prev;               // previous objective function value
    PetscReal J_ref;                // reference objective function value
    PetscReal g_norm;               // norm of reference gradient
    PetscReal f_tol;                // CoSaMp iteration tolerance
    Vec g;                          // gradient
    Vec x_full;                     // solution vector full space
    Vec x_full_prev;                // solution vector full space
    Vec x_sub;                      // solution vector subspace
    Vec work;

    CtxCoSaMp ()
    :
      cosamp_stage(INIT)
    , its_l1(0)
    , np_full(0)
    , maxit_newton(50)
    , inexact_nits(4)
    , nits(0)
    , compute_reference_values(true)
    , converged_l1(false)
    , converged_l2(false)
    , converged_error_l2(false)
    , initialized(false)
    , J(0)
    , J_prev(0)
    , J_ref(0)
    , g_norm(0)
    , f_tol(1E-5)
    , g(nullptr)
    , x_full(nullptr)
    , x_full_prev(nullptr)
    , x_sub(nullptr)
    {}

    PetscErrorCode initialize(Vec p) {
        PetscFunctionBegin;
        PetscErrorCode ierr = 0;
        ierr = VecDuplicate (p, &g);            CHKERRQ (ierr);
        ierr = VecDuplicate (p, &x_full);       CHKERRQ (ierr);
        ierr = VecDuplicate (p, &x_full_prev);  CHKERRQ (ierr);
        ierr = VecDuplicate (p, &work);         CHKERRQ (ierr);
        ierr = VecSet       (g, 0.0);           CHKERRQ (ierr);
        ierr = VecSet       (x_full_prev, 0.0); CHKERRQ (ierr);
        ierr = VecSet       (work, 0.0);        CHKERRQ (ierr);
        ierr = VecCopy      (p, x_full);        CHKERRQ (ierr);
        initialized = true;
        PetscFunctionReturn(ierr);
    }

    PetscErrorCode cleanup() {
        PetscFunctionBegin;
        PetscErrorCode ierr = 0;
        if(initialized) {
            if (g != nullptr)           { VecDestroy (&g);           g           = nullptr;}
            if (x_full != nullptr)      { VecDestroy (&x_full);      x_full      = nullptr;}
            if (x_full_prev != nullptr) { VecDestroy (&x_full_prev); x_full_prev = nullptr;}
            if (work != nullptr)        { VecDestroy (&work);        work        = nullptr;}
        }
        initialized = false;
        PetscFunctionReturn(ierr);
    }

    ~CtxCoSaMp () {
        if (initialized) {cleanup();}
    }
};

struct CtxInv {
    /// @brief evalJ evalDJ, eval D2J
    std::shared_ptr<DerivativeOperators> derivative_operators_;
    /// @brief required to reset derivative_operators_
    std::shared_ptr<PdeOperators> pde_operators_;
    /// @brief common settings/ parameters
    std::shared_ptr<NMisc> n_misc_;
    /// @brief accumulates all tumor related fields and methods
    std::shared_ptr<Tumor> tumor_;
    /// @brief keeps all the settings, tollerances, maxits for optimization
    std::shared_ptr<OptimizerSettings>  optsettings_;
    /// @brief keeps all the information that is feedbacked to the calling routine
    std::shared_ptr<OptimizerFeedback> optfeedback_;
    /// @brief context for CoSaMp L1 solver
    std::shared_ptr<CtxCoSaMp> cosamp_;

    /* reference values gradient */
    ScalarType ksp_gradnorm0;            // reference gradient for hessian PCG
    /* optimization options/settings */
    ScalarType gttol;                    // used: relative gradient reduction
    ScalarType gatol;                    // absolute tolerance for gradient
    ScalarType grtol;                    // relative tolerance for gradient
    /* steering of reference gradeint reset */
    bool is_ksp_gradnorm_set;        // if false, update reference gradient norm for hessian PCG

    bool flag_sparse;                //flag for tracking sparsity of solution when parameter continuation is used
    ScalarType lam_right;                //Parameters for performing binary search on parameter continuation
    ScalarType lam_left;
    Vec weights;                     //for weighted L2
    bool update_reference_gradient;  // if true, update reference gradient for optimization
    bool update_reference_objective;
    /* optimization state */
    ScalarType jvalold;                 // old value of objective function (previous newton iteration)
    ScalarType last_ls_step;            // remeber line-search step of previous solve
    Vec c0old, tmp;                 // previous initial condition \Phi p^k-1 and tmp vec
    Vec x_old;                      // previous solution
    std::vector<std::string> convergence_message; // convergence message
    int verbosity;                  // controls verbosity of inverse solver
    /* additional data */
    Vec data;                       // data for tumor inversion
    Vec data_gradeval;              // data only for gradient evaluation (may differ)


    CtxInv ()
    :
      derivative_operators_ ()
    , n_misc_ ()
    , tumor_ ()
    , data (nullptr)
    , data_gradeval (nullptr)
    , convergence_message ()
    , cosamp_(nullptr) {
        ksp_gradnorm0 = 1.;
        gttol = 1e-3;
        gatol = 1e-6;
        grtol = 1e-12;
        jvalold = 0;
        last_ls_step = 1.0;
        weights = nullptr;
        c0old = nullptr;
        x_old = nullptr;
        verbosity = 3;
        tmp = nullptr;
        is_ksp_gradnorm_set = false;
        flag_sparse = false;
        update_reference_gradient = true;
        update_reference_objective = true;
        cosamp_ = std::make_shared<CtxCoSaMp>();
    }
    ~CtxInv () {
        if (weights != nullptr) { VecDestroy (&weights); weights = nullptr;}
        if (x_old != nullptr)   { VecDestroy (&x_old);   x_old   = nullptr;}
        if (c0old != nullptr)   { VecDestroy (&c0old);   c0old   = nullptr;}
        if (tmp != nullptr)     { VecDestroy (&tmp);     tmp     = nullptr;}
    }
};


/** Biophysical inversion solver:
 *  (1) inversion for p such that T( \Phi(p) ) = c(1)_target
 *  (2) inversion for diffusivity: we invert for k_1, k_2, k_3 with
 *       k_1 = dk_dm_wm  = k_scale * 1;                     //WM
 *       k_2 = dk_dm_gm  = k_scale * k_gm_wm_ratio_;        //GM
 *       k_3 = dk_dm_glm = k_scale * k_glm_wm_ratio_;       //GLM
 */
class InvSolver {
    public :
        InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators = {}, std::shared_ptr <PdeOperators> pde_operators = {}, std::shared_ptr <NMisc> n_misc = {}, std::shared_ptr <Tumor> tumor = {});
        PetscErrorCode initialize (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc, std::shared_ptr <Tumor> tumor);
        PetscErrorCode allocateTaoObjects (bool initialize_tao = true);
        PetscErrorCode setParams (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, bool npchanged = false);
        PetscErrorCode resetOperators (Vec p);
        PetscErrorCode resetTao(std::shared_ptr<NMisc> n_misc);
        PetscErrorCode solve ();
        PetscErrorCode solveInverseCoSaMp();
        PetscErrorCode solveInverseCoSaMpRS(bool rs_mode_active);
        PetscErrorCode printStatistics (int its, PetscReal J, PetscReal J_rel, PetscReal g_norm, PetscReal p_rel_norm, Vec x_L1);

        PetscErrorCode restrictSubspace (Vec *x_restricted, Vec x_full, std::shared_ptr<CtxInv> itctx, bool create_rho_dofs);
        PetscErrorCode prolongateSubspace (Vec x_full, Vec *x_restricted, std::shared_ptr<CtxInv> itctx, int np_full, bool reset_operators);

        PetscErrorCode setTaoOptions (Tao tao, CtxInv* ctx);
        // setter functions
        void setData (Vec d) {data_ = d;}
        void setDataGradient (Vec d) {data_gradeval_ = d;}
        void updateReferenceGradient (bool b) {if (itctx_ != nullptr) itctx_->update_reference_gradient = b;}
        void setOptFeedback (std::shared_ptr<OptimizerFeedback> optfeed) {optfeedback_ = optfeed; itctx_->optfeedback_ = optfeed;}
        // getter functions
        std::shared_ptr<OptimizerSettings> getOptSettings () {return optsettings_;}
        std::shared_ptr<OptimizerFeedback> getOptFeedback () {return optfeedback_;}
        std::shared_ptr<CtxInv> getInverseSolverContext() {return itctx_;}
        bool isInitialized () {return initialized_;}
        Vec getPrec () {return xrec_;}

        PetscErrorCode getGradient (Vec x, Vec dJ) {
            PetscFunctionBegin; PetscErrorCode ierr = 0;
            ierr = itctx_->derivative_operators_->evaluateGradient (dJ, x, data_);
            PetscFunctionReturn(0);
        }
        PetscErrorCode getObjective (Vec x, PetscReal *J) {
            PetscFunctionBegin; PetscErrorCode ierr = 0;
            ierr = itctx_->derivative_operators_->evaluateObjective (J, x, data_);
            PetscFunctionReturn(0);
        }
        PetscErrorCode getObjectiveAndGradient (Vec x, PetscReal *J, Vec dJ) {
            PetscFunctionBegin; PetscErrorCode ierr = 0;
            ierr = itctx_->derivative_operators_->evaluateObjectiveAndGradient (J, dJ, x, data_);
            PetscFunctionReturn(0);
        }

        std::vector<ScalarType> getInvOutParams () {
            return out_params_;
        }

        PetscErrorCode solveInverseReacDiff (Vec x);
        // solves interpolation with tumor basis phi to fit data
        PetscErrorCode solveInterpolation (Vec data);

        ~InvSolver ();

    private:
        /// @brief true if tumor adapter is correctly initialized. mendatory
        bool initialized_;
        /// @brief true if TAO just recently got reset
        bool tao_is_reset_;
        /// @brief data d1 for tumor inversion (memory managed from outside)
        Vec data_;
        /// @brief data d1_grad for gradient evaluation, may differ from data_ (memory managed from outside)
        Vec data_gradeval_;
        /// @brief holds a copy of the reconstructed p vector
        Vec xrec_;
        /// @brief holds solution vector for reaction/diffusion
        Vec xrec_rd_;
        /// @brief petsc tao object, thet solves the inverse problem
        Tao tao_;
        /// @brief petsc matrix object for hessian matrix
        Mat H_;
        std::shared_ptr<OptimizerSettings> optsettings_;
        std::shared_ptr<OptimizerFeedback> optfeedback_;
        std::shared_ptr<CtxInv> itctx_;

        std::vector<ScalarType> out_params_;
};

// ============================= non-class methods used for TAO ============================
PetscErrorCode evaluateObjectiveFunction (Tao, Vec, PetscReal*, void*);
PetscErrorCode evaluateGradient (Tao, Vec, Vec, void*);
PetscErrorCode evaluateObjectiveFunctionAndGradient (Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode evaluateGradientReacDiff (Tao tao, Vec x, Vec dJ, void *ptr);
PetscErrorCode evaluateObjectiveReacDiff (Tao tao, Vec x, PetscReal *J, void *ptr);
PetscErrorCode evaluateObjectiveAndGradientReacDiff (Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode hessianMatVec (Mat, Vec, Vec);
PetscErrorCode constApxHessianMatVec (Mat, Vec, Vec);
PetscErrorCode matfreeHessian (Tao, Vec, Mat, Mat, void*);
PetscErrorCode preconditionerMatVec (PC, Vec, Vec);
PetscErrorCode applyPreconditioner (void*, Vec, Vec);
PetscErrorCode optimizationMonitor (Tao tao, void *ptr);
PetscErrorCode optimizationMonitorReacDiff (Tao tao, void *ptr);
PetscErrorCode optimizationMonitorL1 (Tao tao, void *ptr);
PetscErrorCode hessianKSPMonitor (KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
PetscErrorCode constHessianKSPMonitor (KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
PetscErrorCode preKrylovSolve (KSP ksp, Vec b, Vec x, void *ptr);
PetscErrorCode checkConvergenceGrad (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradReacDiff (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradObj (Tao tao, void *ptr);
PetscErrorCode dispTaoConvReason (TaoConvergedReason flag, std::string &solverstatus);
PetscErrorCode dispLineSearchStatus(Tao tao, void* ptr, TaoLineSearchConvergedReason flag);

#endif
