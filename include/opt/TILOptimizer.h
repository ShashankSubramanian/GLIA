#ifndef TIL_OPTIMIZER_H_
#define TIL_OPTIMIZER_H_


#include "Optimizer.h"
#include "Parameters.h"

class TILOptimizer : public Optimizer {
public :
  TILOptimizer()
  : Optimizer()
  {}

  virtual PetscErrorCode initialize(
            std::shared_ptr <DerivativeOperators> derivative_operators,
            std::shared_ptr <PdeOperators> pde_operators,
            std::shared_ptr <Parameters> params,
            std::shared_ptr <Tumor> tumor);

  virtual PetscErrorCode allocateTaoObjects();
  virtual PetscErrorCode setTaoOptions(Tao tao, CtxInv* ctx);
  virtual PetscErrorCode reset(Vec p);
  virtual PetscErrorCode solve();

  virtual ~TILOptimizer(); // TODO(K) implement destructor
};

#endif


// === non-class methods
// derivative ops
PetscErrorCode evaluateObjectiveFunction (Tao, Vec, PetscReal*, void*);
PetscErrorCode evaluateGradient (Tao, Vec, Vec, void*);
PetscErrorCode evaluateObjectiveFunctionAndGradient (Tao, Vec, PetscReal *, Vec, void *);
// hessian
PetscErrorCode hessianMatVec (Mat, Vec, Vec);
PetscErrorCode matfreeHessian (Tao, Vec, Mat, Mat, void*);
PetscErrorCode constApxHessianMatVec (Mat, Vec, Vec);
PetscErrorCode preconditionerMatVec (PC, Vec, Vec);
PetscErrorCode preKrylovSolve (KSP ksp, Vec b, Vec x, void *ptr);
PetscErrorCode applyPreconditioner (void*, Vec, Vec);
PetscErrorCode hessianKSPMonitor (KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
PetscErrorCode constHessianKSPMonitor (KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
// convergence
PetscErrorCode optimizationMonitor (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGrad (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradObj (Tao tao, void *ptr);
PetscErrorCode dispTaoConvReason (TaoConvergedReason flag, std::string &solverstatus);
PetscErrorCode dispLineSearchStatus(Tao tao, void* ptr, TaoLineSearchConvergedReason flag);
