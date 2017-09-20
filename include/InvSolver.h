#ifndef INVSOLVER_H_
#define INVSOLVER_H_

#include <memory>
#include "petsctao.h"
#include "accfft.h"
#include "DerivativeOperators.h"

enum {QDFS = 0, SLFS = 1};

struct OptimizerSettings {
	double opttolgrad;           /// @brief l2 gradient tolerance for optimization
	double gtolbound;            /// @brief minimum reduction of gradient (even if maxiter hit earlier)
	double grtol;                /// @brief rtol TAO (relative tolerance for gradient, not used)
	double gatol;                /// @brief atol TAO (absolute tolerance for gradient)
	int    newton_maxit;         /// @brief maximum number of allowed newton iterations
	int    krylov_maxit;         /// @brief maximum number of allowed krylov iterations
	int    newton_minit;         /// @brief minimum number of newton steps
	int    iterbound;            /// @brief if GRADOBJ conv. crit is used, max number newton it
	int    fseqtype;             /// @brief type of forcing sequence (quadratic, superlinear)
	int    verbosity;            /// @brief controls verbosity of solver

	OptimizerSettings()
	:
	opttolgrad(1E-3),
	gtolbound(0.8),
	grtol(1E-12),
	gatol(1E-6),
	newton_maxit(20),
	krylov_maxit(30),
	newton_minit(1),
	iterbound(200),
	fseqtype(SLFS),
	verbosity(1)
	{}
};

struct OptimizerFeedback {
	int nb_newton_it;            /// @brief stores the number of required Newton iterations for the last inverse tumor solve
	int nb_krylov_it;            /// @brief stores the number of required (accumulated) Krylov iterations for the last inverse tumor solve
  std::string solverstatus;    /// @brief gives information about the termination reason of inverse tumor TAO solver
  double gradnorm;             /// @brief final gradient norm
	double gradnorm0;            /// @brief norm of initial gradient (with p = intial guess)
  bool converged;              /// @brief true if solver converged within bounds

	OptimizerFeedback()
	:
	nb_newton_it(-1),
	nb_krylov_it(-1),
	solverstatus(),
	gradnorm(0.),
	gradnorm0(0.),
	converged(false)
	{}
};

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
	CtxInv()
	:
	    derivative_operators_()
		, n_misc_()
		, tumor_()
		, data(nullptr)
		, data_gradeval(nullptr)
		, convergence_message()
	{
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

	~CtxInv() {
		if (c0old != nullptr) {VecDestroy(&c0old); c0old = nullptr; }
		if (tmp != nullptr)   {VecDestroy(&tmp);   tmp = nullptr; }
	}
};

class InvSolver {
	public :
		InvSolver (
			std::shared_ptr <DerivativeOperators> derivative_operators = {},
			std::shared_ptr <NMisc> n_misc = {},
			std::shared_ptr <Tumor> tumor = {});
    ~InvSolver ();

		PetscErrorCode initialize(
			std::shared_ptr <DerivativeOperators> derivative_operators,
			std::shared_ptr <NMisc> n_misc,
		  std::shared_ptr <Tumor> tumor);

		PetscErrorCode solve ();

    // setter functions
    void setOptSettings(std::shared_ptr<OptimizerSettings> optset) {optsettings_ = optset;}

		// getter functions
		std::shared_ptr<OptimizerSettings> getOptSettings() {return optsettings_;}
		std::shared_ptr<OptimizerFeedback> getOptFeedback() {return optfeedback_;}

	private:
		/// @brief true if tumor adapter is correctly initialized. mendatory
		bool initialized_;

    /// @breif regularization parameter for tumor inverse solve
		double betap_;

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
PetscErrorCode evaluateObjectiveFunction(Tao, Vec, PetscReal*, void*);
PetscErrorCode evaluateGradient(Tao, Vec, Vec, void*);
PetscErrorCode evaluateObjectiveFunctionAndGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode hessianMatVec(Mat, Vec, Vec);
PetscErrorCode hessianeProduct(void*, Vec, Vec);
PetscErrorCode matfreeHessian(Tao, Vec, Mat, Mat, void*);
PetscErrorCode preconditionerMatVec(PC, Vec, Vec);
PetscErrorCode applyPreconditioner(void*, Vec, Vec);

PetscErrorCode optimizationMonitor(Tao tao, void* ptr);
PetscErrorCode hessianKSPMonitor(KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
PetscErrorCode preKrylovSolve(KSP ksp, Vec b, Vec x, void* ptr);
PetscErrorCode checkConvergenceGrad(Tao tao, void* ptr);
PetscErrorCode checkConvergenceGradObj(Tao tao, void* ptr);
PetscErrorCode dispTaoConvReason(TaoConvergedReason flag, std::string& solverstatus);
PetscErrorCode setTaoOptions(Tao tao, CtxInv* ctx);

//PetscErrorCode AnalyticFormGradient(Tao, Vec, Vec, void*);
//PetscErrorCode Analytic_HessianMatVec(Mat, Vec, Vec);
//PetscErrorCode AnalyticFormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
//PetscErrorCode AnalyticOptimizationMonitor(Tao tao, void* ptr);

#endif
