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
  virtual PetscErrorCode setTaoOptions();
  virtual PetscErrorCode reset(Vec p);
  virtual PetscErrorCode solve();

  virtual PetscErrorCode setInitialGuess(Vec x_init);
  virtual PetscErrorCode setVariableBounds();
  virtual ~MEOptimizer(); // TODO(K) implement destructor

private:
  ScalarType k_init_;
  ScalarType rho_init_;
  ScalarType gamma_init_;
};

#endif

// === non-class methods
// convergence
PetscErrorCode optimizationMonitorMassEffect (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradObjMassEffect (Tao tao, void *ptr);
