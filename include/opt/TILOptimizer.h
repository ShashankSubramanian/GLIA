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

  // TODO(K): currently does nothing different than Optimizer, revisit after TaoInterface done
  virtual PetscErrorCode setTaoOptions();
  virtual PetscErrorCode solve();

  virtual PetscErrorCode setInitialGuess(Vec x_init);
  virtual PetscErrorCode setVariableBounds();

  virtual ~TILOptimizer() {};


  // TODO: Not specialized in this class; I think functions that are not specialized do not need to be declared
  // virtual PetscErrorCode allocateTaoObjects();
  // virtual PetscErrorCode resetOperators(Vec p);
private:
  ScalarType k_init_;
};

#endif
