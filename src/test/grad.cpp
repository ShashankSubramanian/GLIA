#include "Solver.h"
#include "IO.h"
#include "catch.hpp"
#include "helper.h"

/* objective = 0 at ground truth; objective is always positive */
TEST_CASE( "Evaluating objective function", "[grad]" ) {
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();

  /* override defaults */
  params->tu_->dt_    = 0.01;
  params->tu_->nt_    = 100;
  params->tu_->k_     = 0.01;
  params->tu_->rho_   = 10;
  params->tu_->model_ = 1;

  params->opt_->flag_reaction_inv_     = true; // disables phi apply in obj
  params->opt_->reaction_inversion_    = true;
  params->opt_->diffusivity_inversion_ = true;	
  params->opt_->beta_                  = 0;

#if defined(CUDA) && !defined(MPICUDA)
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(CUFFT);
#else
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(ACCFFT);
#endif

  DISABLE_VERBOSE = true;
  initializeGrid(64, params, spec_ops);

  Vec p;
  createPVec(p, params);

  std::shared_ptr<Tumor> tumor = std::make_shared<Tumor>(params, spec_ops);
  tumor->initialize(p, params, spec_ops, nullptr, nullptr);
  tumor->setSinusoidalCoefficients(params);
  std::shared_ptr<PdeOperators> pde_operators = std::make_shared<PdeOperatorsRD>(tumor, params, spec_ops);
  std::shared_ptr<DerivativeOperators> derivative_operators = std::make_shared<DerivativeOperatorsRD>(pde_operators, params, tumor);

  Vec x;
  // create test functions
  VecDuplicate(tumor->c_t_, &x);
  VecSet(x, 0);
  createTestFunction(x, params);  

  // test update ~ will change all coefficients to zero
  ScalarType normk, normr;
  ScalarType rt, kt;
  rt = params->tu_->k_; kt = params->tu_->k_;
  derivative_operators->updateReactionAndDiffusion(p);
  VecNorm(tumor->k_->kxx_, NORM_2, &normk);
  VecNorm(tumor->rho_->rho_vec_, NORM_2, &normr);
  REQUIRE(normk == 0);
  REQUIRE(normr == 0);

  // set correct values to p
  ScalarType *p_ptr;
  VecGetArray(p, &p_ptr);
  p_ptr[1] = kt;
  p_ptr[params->tu_->nk_ + 1] = rt;
  VecRestoreArray(p, &p_ptr);
  
  // generate test data
  VecCopy(x, tumor->c_0_);
  derivative_operators->updateReactionAndDiffusion(p);
  pde_operators->solveState(0);
  VecCopy(tumor->c_t_, x);
  std::shared_ptr<Data> data = std::make_shared<Data>();
  data->set(x, nullptr);

  // compute objective with ground truth
  ScalarType j;
  derivative_operators->evaluateObjective(&j, p, data);
  REQUIRE(j < 1E-7);
  VecSet(x, 0);
  derivative_operators->evaluateObjective(&j, p, data);
  REQUIRE(j > 0);

  VecDestroy(&x);
  VecDestroy(&p);
}