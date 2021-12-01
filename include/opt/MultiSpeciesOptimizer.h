#ifndef MULTISPECIES_OPTIMIZER_H_
#define MULTISPECIES_OPTIMIZER_H_

#include <memory>
#include "Parameters.h" 
#include <libcmaes/esoptimizer.h>
#include <libcmaes/cmastrategy.h>

struct CtxInvCMA {

public:
    ScalarType jvalold;                  // old value of the objective function
    ScalarType cma_sigma;                // variance for the cma optimizer
    
    Vec tmp;                             
    Vec c0_old;                          // previous solution
    Vec x_old;                           // previous test solution
    std::shared_ptr<Data> data;           // data for tumor inversion
    std::vector<std::string> convergence_message;  // convergence message
    /// @brief required to reset derivative_operators_
    std::shared_ptr<PdeOperators> pde_operators_;
    /// @brief common settings/ parameters
    std::shared_ptr<Parameters> params_;
    /// @brief accumulates all tumor related fields and methods
    std::shared_ptr<Tumor> tumor_;
    
    CtxInv()
    :
      jvalold(0)
    , cma_sigma(1.0)
    , tmp(nullptr)
    , c0_old(nullptr)
    , x_old(nullptr)
    , data(nullptr)
    , convergence_messsage()
    , params_()
    , tumor_()
    {}

    ~CtxInv () {
        if (x_old != nullptr) {VecDestroy(&x_old; x_old = nullptr;}
        if (c0_old != nullptr) {VecDestroy(&c0_old; c0_old = nullptr;}
        if (tmp != nullptr) {VecDestroy(&tmp; tmp = nullptr;}
    }
};


class MultiSpeciesOptimizer {
 public : 
  Optimizer()
  :
    initialized_(false)
  , data_(nullptr)
  , ctx_(nullptr)
  , xrec_(nullptr)
  , xin_(nullptr)
  , xout_(nullptr)
 {}
  virtual PetscErrorCode initialize (
          std::shared_ptr <PdeOperators> pde_operators, 
          std::shared_ptr <Parameters> params, 
          std::shared_ptr <Tumor> tumor);

  virtual PetscErrorCode solve();
  virtual PetscErrorCode setVariableBounds();

  virtual ~MultiSpeciesOptimizer() {};

 private:
  int n_g_;
  ScalarType k_init_;
  ScalarType rho_init_;
  ScalarType gamma_init_;
  ScalarType ox_hypoxia_;
  ScalarType death_rate_;
  ScalarType alpha_0_;
  ScalarType ox_consumption_;
  ScalarType ox_source_;
  ScalarType beta_0_;


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
          
