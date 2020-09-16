#include "Solver.h"
#include "IO.h"
#include "catch.hpp"
#include "helper.h"

/* run a simple diffusion equation \partial c + Dc = 0 with variable coefficients and test the result */
TEST_CASE( "Running diffusion solver", "[pdesolver]" ) {
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
#if defined(CUDA) && !defined(MPICUDA)
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(CUFFT);
#else
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(ACCFFT);
#endif

  DISABLE_VERBOSE = true;

  // initialize a grid
  initializeGrid(64, params, spec_ops);
  // create a test diffusion coefficient
  std::shared_ptr<DiffCoef> k = std::make_shared<DiffCoef>(params, spec_ops);
  // create work vectors
  Vec *work = new Vec[8];
  for (int i = 0; i < 8; i++) {
    VecDuplicate(k->kxx_, &work[i]);
    VecSet(work[i], 0);
  }
  // setting work vecs for diffusion coefficient (first 8)
  k->setWorkVecs(work);
  // set sinusoidal diffusion coefficient
  k->setValuesSinusoidal(params, 1E-2);
  // create a diffusion solver
  std::shared_ptr<DiffusionSolver> diff_solver = std::make_shared<DiffusionSolver>(params, spec_ops, k);
  // create a valid function
  Vec c;
  VecDuplicate(work[0], &c);
  VecSet(c, 0);
  createTestFunction(c, params);
  // solve a diffusion equation for a few time steps
  diff_solver->precFactor();
  for (int i = 0; i < 10; i++)
    diff_solver->solve(c, 0.02);

  // test the norm of c
  ScalarType nrm;
  VecNorm(c, NORM_2, &nrm);
  REQUIRE(nrm == Approx(2.0487));
  // test number of ksp iter for last solve
  REQUIRE(diff_solver->ksp_itr_ == 5);

  // deallocate memory
  for (int i = 0; i < 8; i++) {
    VecDestroy(&work[i]);
  }
  delete[] work;
  VecDestroy(&c);
}

 // run a simple advection equation \partial c + v.\gradc = 0 with non-uniform velocity and advect back with negative velocity 
TEST_CASE( "Running advection solver", "[pdesolver]" ) {
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
#if defined(CUDA) && !defined(MPICUDA)
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(CUFFT);
#else
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators>(ACCFFT);
#endif

  DISABLE_VERBOSE = true;
  ScalarType norm_c, norm;
  Vec c, c_copy;

  // initialize a grid ~ use good resolution to minimize discretization errors
  initializeGrid(256, params, spec_ops);
  // create tumor class
  params->tu_->model_ = 4; // creates velocities, displacements
  std::shared_ptr<Tumor> tumor = std::make_shared<Tumor>(params, spec_ops);
  // create a test velocity
  createTestField(tumor->velocity_, params);
  // create a test function
  VecDuplicate(tumor->c_t_, &c);
  VecDuplicate(tumor->c_t_, &c_copy);
  VecSet(c, 0);
  createTestFunction(c, params);
  VecCopy(c, c_copy);
  // get norm of c
  VecNorm(c, NORM_2, &norm_c);
  // make an advection solver
  std::shared_ptr<AdvectionSolver> adv_solver = std::make_shared<SemiLagrangianSolver>(params, tumor, spec_ops);
  adv_solver->advection_mode_ = 2; // pure advection
  // run adv solver forward
  for (int i = 0; i < 4; i++) {
    adv_solver->solve(c, tumor->velocity_, 0.02);
  }
  // run inverse advection solver
  tumor->velocity_->scale(-1); // reverse the velocity
  adv_solver->trajectoryIsComputed_ = false; // recompute the trajectories
  for (int i = 0; i < 4; i++) {
    adv_solver->solve(c, tumor->velocity_, 0.02);
  }
  // get new c norm
  VecAXPY(c, -1, c_copy);
  VecNorm(c, NORM_2, &norm);

  // less than 1% error
  REQUIRE(norm / norm_c < 1E-2);

  VecDestroy(&c);
  VecDestroy(&c_copy);
}


