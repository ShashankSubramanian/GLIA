#ifndef RD_OPTIMIZER_H_
#define RD_OPTIMIZER_H_


#include "Optimizer.h"
#include "Parameters.h"

class RDOptimizer : public Optimizer {
public :
  RDOptimizer()
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

  virtual ~RDOptimizer(); // TODO(K) implement destructor
};

#endif


// === non-class methods
// derivative ops
PetscErrorCode evaluateGradientReacDiff (Tao tao, Vec x, Vec dJ, void *ptr);
PetscErrorCode evaluateObjectiveReacDiff (Tao tao, Vec x, PetscReal *J, void *ptr);
PetscErrorCode evaluateObjectiveAndGradientReacDiff (Tao, Vec, PetscReal *, Vec, void *);
// convergence
PetscErrorCode optimizationMonitorReacDiff (Tao tao, void *ptr);
PetscErrorCode checkConvergenceGradReacDiff (Tao tao, void *ptr);
