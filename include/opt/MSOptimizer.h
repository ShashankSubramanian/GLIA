#ifndef MS_OPTIMIZER_H_
#define MS_OPTIMIZER_H_

#include "Optimizer.h"
#include "Parameters.h"

class MSOptimizer : public Optimizer {
  public :
    MSOptimizer()
    : Optimizer()
    {}


    virtual PetscErrorCode initialize(
              std::shared_ptr <DerivativeOperators> derivative_operators,
              std::shared_ptr <PdeOperators> pde_operators,
              std::shared_ptr <Parameters> params,
              std::shared_ptr <Tumor> tumor);

    virtual PetscErrorCode allocateTaoObjects();
    virtual PetscErrorCode setTaoOptions();
    virtual PetscErrorCode solve();
    virtual PetscErrorCode setInitialGuess(Vec x_init);


}; 

#endif
