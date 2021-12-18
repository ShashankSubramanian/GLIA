#ifndef MULTISPECIES_OPTIMIZER_H_
#define MULTISPECIES_OPTIMIZER_H_

#include "Optimizer.h"
#include "Parameters.h" 
#include <cmaes.h>
//#include <libcmaes/esoptimizer.h>
//#include <libcmaes/cmastrategy.h>
//#include <libcmaes/llogging.h>

class MultiSpeciesOptimizer : public CMAOptimizer {
 public : 
  MultiSpeciesOptimizer()
  : CMAOptimizer()
  {}
  
  virtual PetscErrorCode initialize (
          std::shared_ptr <DerivativeOperators> derivative_operators,
          std::shared_ptr <PdeOperators> pde_operators, 
          std::shared_ptr <Parameters> params, 
          std::shared_ptr <Tumor> tumor);
  virtual PetscErrorCode allocateObjects();
  virtual PetscErrorCode solve();
  virtual PetscErrorCode setInitialGuess(Vec x_init);
  virtual PetscErrorCode runforward(const double *xtest_, double* J);
  //virtual PetscErrorCode setVariableBounds();

  virtual ~MultiSpeciesOptimizer() {};
 
 private:
  int n_g_;
  ScalarType k_init_;
  ScalarType rho_init_;
  ScalarType gamma_init_;
  ScalarType ox_hypoxia_init_;
  ScalarType death_rate_init_;
  ScalarType alpha_0_init_;
  ScalarType ox_consumption_init_;
  ScalarType ox_source_init_;
  ScalarType beta_0_init_;


  ScalarType k_scale_;
  ScalarType rho_scale_;
  ScalarType gamma_scale_;
  ScalarType ox_hypoxia_scale_; 
  ScalarType death_rate_scale_;
  ScalarType alpha_0_scale_;
  ScalarType ox_consumption_scale_;
  ScalarType ox_source_scale_;
  ScalarType beta_0_scale_;


};

#endif 
          
