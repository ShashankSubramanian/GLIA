#ifndef INVSOLVER_H_
#define INVSOLVER_H_

#include <memory>
#include "petsctao.h"
#include "accfft.h"
#include "DerivativeOperators.h"

enum {QDFS = 0, SLFS = 1};

struct CtxInv {
	/* evalJ evalDJ, eval D2J */
	std::shared_ptr<DerivativeOperators> derivative_operators_;
	std::shared_ptr<NMisc> n_misc_;

	/* reference values gradient */
	double gradnorm0_HessianCG; // reference gradient for hessian PCG
	double gradnorm0;           // norm of initial gradient (with p = intiial guess)

	/* optimization options/settings */
	double gttol;               // used: relative gradient reduction
	double gatol;               // absolute tolerance for gradient
	double grtol;               // relative tolerance for gradient
	double gtolbound;           // minimum reduction of gradient (even if maxiter hit earlier)
	int krylov_maxit;           // maximum number of iterations for hessian PCG
	int newton_maxit;           // maximum number of newton steps
	int newton_minit;           // minimum number of newton steps
	int iterbound;              // upper bound on newton iterations
	int fseqtype;               // type of forcing sequence (quadratic, superlinear)

	/* steering of reference gradeint reset */
	bool isGradNorm0HessianSet;  // if false, update reference gradient norm for hessian PCG
	bool updateGradNorm0;        // if true, update reference gradient for optimization

	/* optimization state */
	int nbKrylovIt;              // count (accumulated) number of PCG iterations
	int nbNewtonIt;              // count number of newton steps
	double jvalold;              // old value of objective function (previous newton iteration)
	Vec c0old, tmp;              // previous initial condition \Phi p^k-1 and tmp vec
	std::vector<std::string> convergenceMessage; // convergence message
	bool converged;              // true if solver converged
	int verbosity;               // controls verbosity of inverse solver

	/* additional data */
	std::shared_ptr<Tumor> tumor;            // to access tumor parameters and eval J, dJ, d2J
	Vec data;   // data for tumor inversion
	Vec data_gradeval; // data only for gradient evaluation (may differ)
	CtxInv()
	:
		tumor(),
		data(nullptr),
		data_gradeval(nullptr),
		convergenceMessage()
	{
		gradnorm0_HessianCG = 1.;
		gradnorm0 = 1.;
		gttol = 1e-3;
		gatol = 1e-6;
		grtol = 1e-12;
		gtolbound = 0.8;
		nbKrylovIt = 0;
		nbNewtonIt = 0;
		iterbound = 200;
		fseqtype = SLFS;
		jvalold = 0;
		c0old = nullptr;
		tmp = nullptr;
		converged = false;
		krylov_maxit = 1000;
		newton_maxit = 1000;
		newton_minit = 1;
		isGradNorm0HessianSet = false;
		updateGradNorm0 = true;
		verbosity = 1;
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
			std::shared_ptr <NMisc> n_misc = {});
    ~InvSolver ();

		PetscErrorCode initialize(
			std::shared_ptr <DerivativeOperators> derivative_operators,
			std::shared_ptr <NMisc> n_misc);

		PetscErrorCode solve ();

    // setter functions
		void setOpttolGrad(double d) {optTolGrad_ = d;}

		// getter functions

	private:
		/// @brief true if tumor adapter is correctly initialized. mendatory
		bool initialized_;

    /// @breif regularization parameter for tumor inverse solve
		double betap_;

		/// @brief l2 gradient tolerance for optimization
    double optTolGrad_;

    /// @brief gives information about the termination reason of inverse tumor TAO solver
		std::string solverstatus_;

    /// @brief stores the number of required Newton iterations for the last inverse tumor solve
		int nbNewtonIt_;

		/// @brief stores the number of required (accumulated) Krylov iterations for the last inverse tumor solve
		int nbKrylovIt_;

		/* @brief flag indicates that update of reference gradient for the relative
	   *        convergence crit. for ITP Newton iteration is neccessary. This should
	   *        only change when beta_reg is changed, as we do warmstarts.
	   */
		bool updateRefGradITPSolver_;
		double refgradITPSolver_;

    /// @brief data d1 for tumor inversion (memory managed from outside)
		Vec data_;

		/// @brief data d1_grad for gradient evaluation, may differ from data_ (memory managed from outside)
		Vec data_gradeval_;

		Vec prec_;

    /// @brief petsc tao object, thet solves the inverse problem
		Tao tao_;

    /// @brief petsc matrix object for hessian matrix
		Mat H_;

		std::shared_ptr<Tumor> tumor_;

		shared_ptr<CtxInv> itctx_;
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
PetscErrorCode setTaoOptions(Tao* tao, CtxInv* ctx);

//PetscErrorCode AnalyticFormGradient(Tao, Vec, Vec, void*);
//PetscErrorCode Analytic_HessianMatVec(Mat, Vec, Vec);
//PetscErrorCode AnalyticFormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
//PetscErrorCode AnalyticOptimizationMonitor(Tao tao, void* ptr);

#endif
