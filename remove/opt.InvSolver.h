#ifndef INVSOLVER_H_
#define INVSOLVER_H_

#include <memory>
#include "DerivativeOperators.h"
#include "Parameters.h"


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
public:
    ScalarType jvalold;                 // old value of objective function (previous newton iteration)
    ScalarType last_ls_step;            // remeber line-search step of previous solve
    ScalarType step_init;                // init step length for line search
    ScalarType ksp_gradnorm0;            // reference gradient for hessian PCG
    bool update_reference_gradient_hessian_ksp;
    bool update_reference_gradient;
    bool update_reference_objective;

    Vec tmp;
    Vec c0_old;    // previous initial condition \Phi p^k-1
    Vec x_old;     // previous solution
    Vec data;      // data for tumor inversion

    std::vector<std::string> convergence_message; // convergence message

    /// @brief evalJ evalDJ, eval D2J
    std::shared_ptr<DerivativeOperators> derivative_operators_;
    /// @brief required to reset derivative_operators_
    std::shared_ptr<PdeOperators> pde_operators_;
    /// @brief common settings/ parameters
    std::shared_ptr<Parameters> params_;
    /// @brief accumulates all tumor related fields and methods
    std::shared_ptr<Tumor> tumor_;
    /// @brief context for CoSaMp L1 solver
    std::shared_ptr<CtxCoSaMp> cosamp_;

    CtxInv ()
    :
      jvalold(0)
    , last_ls_step(1.0)
    , step_init(1.0)
    , ksp_gradnorm0(1)
    , update_reference_gradient_hessian_ksp(true)
    , update_reference_gradient(true)
    , update_reference_objective(true)
    , tmp(nullptr)
    , c0_old(nullptr)
    , x_old(nullptr)
    , data(nullptr)
    , derivative_operators_ ()
    , params_ ()
    , tumor_ ()
    , data (nullptr)
    , convergence_message ()
    , cosamp_(nullptr) {
        cosamp_ = std::make_shared<CtxCoSaMp>();
    }
    ~CtxInv () {
        if (x_old != nullptr) { VecDestroy (&x_old); x_old = nullptr;}
        if (c0_old != nullptr) { VecDestroy (&c0_old); c0_old = nullptr;}
        if (tmp != nullptr) { VecDestroy (&tmp); tmp = nullptr;}
    }
};


class InvSolver {
    public :
        InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators = {}, std::shared_ptr <PdeOperators> pde_operators = {}, std::shared_ptr <Parameters> params = {}, std::shared_ptr <Tumor> tumor = {});
        PetscErrorCode initialize (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <Parameters> params, std::shared_ptr <Tumor> tumor);
        PetscErrorCode allocateTaoObjectsMassEffect (bool initialize_tao = true);
        PetscErrorCode allocateTaoObjects (bool initialize_tao = true);
        PetscErrorCode setParams (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr<Parameters> params, std::shared_ptr<Tumor> tumor, bool npchanged = false);
        PetscErrorCode resetOperators (Vec p);
        PetscErrorCode resetTao(std::shared_ptr<Parameters> params);
        PetscErrorCode solve ();
        PetscErrorCode solveForMassEffect ();
        PetscErrorCode solveInverseCoSaMp();
        PetscErrorCode solveInverseCoSaMpRS(bool rs_mode_active);
        PetscErrorCode solveInverseReacDiff (Vec x);

        PetscErrorCode printStatistics (int its, PetscReal J, PetscReal J_rel, PetscReal g_norm, PetscReal p_rel_norm, Vec x_L1);

        PetscErrorCode restrictSubspace (Vec *x_restricted, Vec x_full, std::shared_ptr<CtxInv> itctx, bool create_rho_dofs);
        PetscErrorCode prolongateSubspace (Vec x_full, Vec *x_restricted, std::shared_ptr<CtxInv> itctx, int np_full, bool reset_operators);

        PetscErrorCode setTaoOptions (Tao tao, CtxInv* ctx);
        PetscErrorCode setTaoOptionsMassEffect (Tao tao, CtxInv* ctx);
        // setter functions
        void setData (Vec d) {data_ = d;}
        void setDataGradient (Vec d) {data_gradeval_ = d;}
        void updateReferenceGradient (bool b) {if (ctx_ != nullptr) ctx_->update_reference_gradient = b;}
        void setOptFeedback (std::shared_ptr<OptimizerFeedback> optfeed) {optfeedback_ = optfeed; ctx_->optfeedback_ = optfeed;}
        // getter functions
        std::shared_ptr<CtxInv> getInverseSolverContext() {return ctx_;}
        bool isInitialized () {return initialized_;}
        Vec getPrec () {return xrec_;}

        PetscErrorCode evaluateGradient (Vec x, Vec dJ) {
            PetscFunctionBegin; PetscErrorCode ierr = 0;
            ierr = ctx_->derivative_operators_->evaluateGradient (dJ, x, data_);
            PetscFunctionReturn(0);
        }
        PetscErrorCode evaluateObjective (Vec x, PetscReal *J) {
            PetscFunctionBegin; PetscErrorCode ierr = 0;
            ierr = ctx_->derivative_operators_->evaluateObjective (J, x, data_);
            PetscFunctionReturn(0);
        }
        PetscErrorCode evaluateObjectiveAndGradient (Vec x, PetscReal *J, Vec dJ) {
            PetscFunctionBegin; PetscErrorCode ierr = 0;
            ierr = ctx_->derivative_operators_->evaluateObjectiveAndGradient (J, dJ, x, data_);
            PetscFunctionReturn(0);
        }

        ~InvSolver ();

        std::shared_ptr<CtxInv> ctx_;
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
PetscErrorCode optimizationMonitorMassEffect (Tao tao, void *ptr);
PetscErrorCode optimizationMonitor (Tao tao, void *ptr);
PetscErrorCode optimizationMonitorReacDiff (Tao tao, void *ptr);
PetscErrorCode optimizationMonitorL1 (Tao tao, void *ptr);
PetscErrorCode hessianKSPMonitor (KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
PetscErrorCode constHessianKSPMonitor (KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
PetscErrorCode preKrylovSolve (KSP ksp, Vec b, Vec x, void *ptr);
PetscErrorCode checkConvergenceGrad (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradReacDiff (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradObj (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradObjMassEffect (Tao tao, void *ptr);
PetscErrorCode dispTaoConvReason (TaoConvergedReason flag, std::string &solverstatus);
PetscErrorCode dispLineSearchStatus(Tao tao, void* ptr, TaoLineSearchConvergedReason flag);
PetscErrorCode operatorCreateVecsMassEffect (Mat A, Vec *left, Vec *right);

#endif
