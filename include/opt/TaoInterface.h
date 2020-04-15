#ifndef TAUINTERFACE_H_
#define TAUINTERFACE_H_

#include "Parameters.h"
#include "petsctao.h"

// derivative operators
PetscErrorCode evaluateObjectiveFunction(Tao, Vec, PetscReal*, void*);
PetscErrorCode evaluateGradient(Tao, Vec, Vec, void*);
PetscErrorCode evaluateObjectiveFunctionAndGradient(Tao, Vec, PetscReal *, Vec, void *);
// hessian
PetscErrorCode hessianMatVec (Mat, Vec, Vec);
PetscErrorCode matfreeHessian (Tao, Vec, Mat, Mat, void*);
PetscErrorCode preKrylovSolve (KSP ksp, Vec b, Vec x, void *ptr);
PetscErrorCode preconditionerMatVec (PC, Vec, Vec);
// monitors
PetscErrorCode optimizationMonitor(Tao tao, void *ptr);
PetscErrorCode hessianKSPMonitor (KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
// convergence
PetscErrorCode checkConvergenceGrad(Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradObj(Tao tao, void *ptr);
PetscErrorCode dispTaoConvReason(TaoConvergedReason flag, std::string &solverstatus);
PetscErrorCode dispLineSearchStatus(Tao tao, void* ptr, TaoLineSearchConvergedReason flag);
// misc
PetscErrorCode operatorCreateVecs(Mat A, Vec *left, Vec *right);




// ============== up to here ported ===================
// PetscErrorCode applyPreconditioner (void*, Vec, Vec);
// PetscErrorCode constHessianKSPMonitor (KSP ksp,PetscInt n,PetscReal rnorm, void *dummy);
// reaction + diffusion
PetscErrorCode evaluateGradientReacDiff (Tao tao, Vec x, Vec dJ, void *ptr);
PetscErrorCode evaluateObjectiveReacDiff (Tao tao, Vec x, PetscReal *J, void *ptr);
PetscErrorCode evaluateObjectiveAndGradientReacDiff (Tao, Vec, PetscReal *, Vec, void *);
// convergence
PetscErrorCode optimizationMonitorReacDiff (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradReacDiff (Tao tao, void *ptr);

// mass effect: rection + diffusion + forcing factor
PetscErrorCode optimizationMonitorMassEffect (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradObjMassEffect (Tao tao, void *ptr);

#endif
