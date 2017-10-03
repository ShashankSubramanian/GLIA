#ifndef INVSOLVER_H_
#define INVSOLVER_H_

#include <memory>
#include "petsctao.h"
#include "accfft.h"
#include "DerivativeOperators.h"
#include "Utils.h"

struct CtxInv {
    /* evalJ evalDJ, eval D2J */
    std::shared_ptr<DerivativeOperators> derivative_operators_;
    /// @brief common settings/ parameters
    std::shared_ptr<NMisc> n_misc_;
    /// @brief accumulates all tumor related fields and methods
    std::shared_ptr<Tumor> tumor_;
    /// @brief keeps all the settings, tollerances, maxits for optimization
    std::shared_ptr<OptimizerSettings>  optsettings_;
    /// @brief keeps all the information that is feedbacked to the calling routine
    std::shared_ptr<OptimizerFeedback> optfeedback_;
    /* reference values gradient */
    double ksp_gradnorm0; // reference gradient for hessian PCG
    /* optimization options/settings */
    double gttol;               // used: relative gradient reduction
    double gatol;               // absolute tolerance for gradient
    double grtol;               // relative tolerance for gradient
    /* steering of reference gradeint reset */
    bool is_ksp_gradnorm_set;   // if false, update reference gradient norm for hessian PCG
    bool update_reference_gradient;  // if true, update reference gradient for optimization
    /* optimization state */
    double jvalold;               // old value of objective function (previous newton iteration)
    Vec c0old, tmp;               // previous initial condition \Phi p^k-1 and tmp vec
    std::vector<std::string> convergence_message; // convergence message
    int verbosity;                // controls verbosity of inverse solver
    /* additional data */
    Vec data;                     // data for tumor inversion
    Vec data_gradeval;            // data only for gradient evaluation (may differ)
    CtxInv () :
    derivative_operators_ ()
    , n_misc_ ()
    , tumor_ ()
    , data (nullptr)
    , data_gradeval (nullptr)
    , convergence_message () {
        ksp_gradnorm0 = 1.;
        gttol = 1e-3;
        gatol = 1e-6;
        grtol = 1e-12;
        jvalold = 0;
        c0old = nullptr;
        tmp = nullptr;
        is_ksp_gradnorm_set = false;
        update_reference_gradient = true;
    }

    ~CtxInv () {
        if (c0old != nullptr) {
            VecDestroy (&c0old);
            c0old = nullptr;
        }
        if (tmp != nullptr) {
            VecDestroy (&tmp);
            tmp = nullptr;
        }
    }
};

class InvSolver {
    public :
        InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators = {}, std::shared_ptr <NMisc> n_misc = {}, std::shared_ptr <Tumor> tumor = {});
        PetscErrorCode initialize (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <NMisc> n_misc, std::shared_ptr <Tumor> tumor);
        PetscErrorCode setParams (std::shared_ptr<DerivativeOperators> derivative_operators, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, bool npchanged = false);
        PetscErrorCode solve ();
        // setter functions
        void setData (Vec d) {data_ = d;}
        void setDataGradient (Vec d) {data_gradeval_ = d;}
        void updateReferenceGradient (bool b) {if (_itctx != nullptr) _itctx->update_reference_gradient = 0;}
        void setOptFeedback (std::shared_ptr<OptimizerFeedback> optfeed) {optfeedback_ = optfeed; itctx_->optfeedback_ = optfeed;}
        // getter functions
        std::shared_ptr<OptimizerSettings> getOptSettings () {return optsettings_;}
        std::shared_ptr<OptimizerFeedback> getOptFeedback () {return optfeedback_;}
        std::shared_ptr<CtxInv> getInverseSolverContext() {return itctx_;}
        bool isInitialized () {return initialized_;}
        Vec getPrec () {return prec_;}
        ~InvSolver ();

    private:
        /// @brief true if tumor adapter is correctly initialized. mendatory
        bool initialized_;
        /// @brief data d1 for tumor inversion (memory managed from outside)
        Vec data_;
        /// @brief data d1_grad for gradient evaluation, may differ from data_ (memory managed from outside)
        Vec data_gradeval_;
        /// @brief holds a copy of the reconstructed p vector
        Vec prec_;
        /// @brief petsc tao object, thet solves the inverse problem
        Tao tao_;
        /// @brief petsc matrix object for hessian matrix
        Mat H_;
        std::shared_ptr<OptimizerSettings> optsettings_;
        std::shared_ptr<OptimizerFeedback> optfeedback_;
        std::shared_ptr<CtxInv> itctx_;
};

// ============================= non-class methods used for TAO ============================
PetscErrorCode evaluateObjectiveFunction (Tao, Vec, PetscReal*, void*);
PetscErrorCode evaluateGradient (Tao, Vec, Vec, void*);
PetscErrorCode evaluateObjectiveFunctionAndGradient (Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode hessianMatVec (Mat, Vec, Vec);
PetscErrorCode hessianeProduct (void*, Vec, Vec);
PetscErrorCode matfreeHessian (Tao, Vec, Mat, Mat, void*);
PetscErrorCode preconditionerMatVec (PC, Vec, Vec);
PetscErrorCode applyPreconditioner (void*, Vec, Vec);
PetscErrorCode optimizationMonitor (Tao tao, void *ptr);
PetscErrorCode hessianKSPMonitor (KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
PetscErrorCode preKrylovSolve (KSP ksp, Vec b, Vec x, void *ptr);
PetscErrorCode checkConvergenceGrad (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradObj (Tao tao, void *ptr);
PetscErrorCode dispTaoConvReason (TaoConvergedReason flag, std::string &solverstatus);
PetscErrorCode setTaoOptions (Tao tao, CtxInv* ctx);
//PetscErrorCode AnalyticFormGradient(Tao, Vec, Vec, void*);
//PetscErrorCode Analytic_HessianMatVec(Mat, Vec, Vec);
//PetscErrorCode AnalyticFormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
//PetscErrorCode AnalyticOptimizationMonitor(Tao tao, void *ptr);

#endif
