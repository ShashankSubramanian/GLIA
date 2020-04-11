#ifndef ME_OPTIMIZER_H_
#define ME_OPTIMIZER_H_


#include "Optimizer.h"
#include "Parameters.h"

class MEOptimizer : public Optimizer {
public :
  MEOptimizer()
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

  virtual ~MEOptimizer(); // TODO(K) implement destructor
};

#endif

// === non-class methods
// convergence
PetscErrorCode optimizationMonitorMassEffect (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradObjMassEffect (Tao tao, void *ptr);
// misc
PetscErrorCode operatorCreateVecsMassEffect (Mat A, Vec *left, Vec *right); // TODO(K) needed?
