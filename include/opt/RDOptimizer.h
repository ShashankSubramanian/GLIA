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


  // TODO: revisit if necessary after cleanuop of TaoInterface
  virtual PetscErrorCode setTaoOptions();
  virtual PetscErrorCode solve();

  virtual PetscErrorCode setInitialGuess(Vec x_init);
  virtual PetscErrorCode setVariableBounds();

  virtual ~RDOptimizer();

  // TODO: Not specialized in this class; I think functions that are not specialized do not need to be declared
  // virtual PetscErrorCode resetOperators(Vec p);
  // virtual PetscErrorCode allocateTaoObjects();

private:
  ScalarType k_init_;
  ScalarType rho_init_;
};

#endif
