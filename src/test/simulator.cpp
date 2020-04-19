#include "Solver.h"
#include "IO.h"
#include "catch.hpp"

// TEST_CASE("Running forward simulator with sinusoidal coefficients", "[sinusoid-simulator]") {
//   std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
//   std::shared_ptr<ApplicationSettings> app_settings = std::make_shared<ApplicationSettings>();
// #if defined(CUDA) && !defined(MPICUDA)
//   std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (CUFFT);
// #else
//   std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
// #endif
//   // disable all I/O for tests
//   DISABLE_VERBOSE = true;

//   /* forward solver test */
//   std::string s = "config/test_forward_sin_config.txt"; //test config
//   RunMode r;
//   PetscErrorCode ierr = 0;
//   ierr = parseConfig(s, params, app_settings, spec_ops, &r);

//   // make a forward solver
//   std::shared_ptr<SolverInterface> solver = std::make_shared<ForwardSolver>();
//   params->tu_->smoothing_factor_atlas_ = 0; // do not smooth atlas ~ sinusoid
//   ierr = solver->initialize(spec_ops, params, app_settings);
//   ierr = solver->run();
//   ierr = solver->finalize();

//   // === create tmp vector for tests according to distributed grid
//   Vec tmp;
//   VecCreate(PETSC_COMM_WORLD, &tmp);
//   VecSetSizes(tmp, params->grid_->nl_, params->grid_->ng_);
//   setupVec(tmp);
//   VecSet(tmp, 0.0);

//   ScalarType norm = 0;

//   // test finalize, displacement norm, c0 norm, c1 norm
//   REQUIRE(ierr == 0);
// #ifdef SINGLE
//   VecNorm(solver->getTumor()->c_0_, NORM_2, &norm);
//   REQUIRE(norm == Approx(22.0161));
//   VecNorm(solver->getTumor()->c_t_, NORM_2, &norm);
//   REQUIRE(norm == Approx(83.9893));
//   solver->getTumor()->displacement_->computeMagnitude(tmp);
//   VecNorm(tmp, NORM_2, &norm);
//   REQUIRE(norm == Approx(43.5494));
// #endif
// }


// TEST_CASE( "Running forward simulator", "[simulator]" ) {
//   std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
//   std::shared_ptr<ApplicationSettings> app_settings = std::make_shared<ApplicationSettings>();
// #if defined(CUDA) && !defined(MPICUDA)
//   std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (CUFFT);
// #else
//   std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
// #endif
//   // disable all I/O for tests
//   DISABLE_VERBOSE = true;

//   /* forward solver test */
//   std::string s = "config/test_forward_config.txt"; //test config
//   RunMode r;
//   PetscErrorCode ierr = 0;
//   ierr = parseConfig(s, params, app_settings, spec_ops, &r);

//   // make a forward solver
//   std::shared_ptr<SolverInterface> solver = std::make_shared<ForwardSolver>();
//   ierr = solver->initialize(spec_ops, params, app_settings);
//   ierr = solver->run();
//   ierr = solver->finalize();

//   // === create tmp vector for tests according to distributed grid
//   Vec tmp;
//   VecCreate(PETSC_COMM_WORLD, &tmp);
//   VecSetSizes(tmp, params->grid_->nl_, params->grid_->ng_);
//   setupVec(tmp);
//   VecSet(tmp, 0.0);

//   ScalarType norm = 0;

//   // test finalize, displacement norm, c0 norm, c1 norm
//   REQUIRE(ierr == 0);
// #ifdef SINGLE
//   VecNorm(solver->getTumor()->c_0_, NORM_2, &norm);
//   REQUIRE(norm == Approx(4.09351));
//   VecNorm(solver->getTumor()->c_t_, NORM_2, &norm);
//   REQUIRE(norm == Approx(32.1486));
//   if (params->tu_->model_ >= 4 && params->tu_->forcing_factor_ > 0) {
//     solver->getTumor()->displacement_->computeMagnitude(tmp);
//     VecNorm(tmp, NORM_2, &norm);
//     REQUIRE(norm == Approx(27.0497));
//   }
// #endif
// }

TEST_CASE( "Running inverse sparse-til simulator with sinusoidal coefficients", "[sinusoid-simulator]" ) {
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
  std::shared_ptr<ApplicationSettings> app_settings = std::make_shared<ApplicationSettings>();
#if defined(CUDA) && !defined(MPICUDA)
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (CUFFT);
#else
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
#endif
  // disable all I/O for tests
  DISABLE_VERBOSE = true;

  std::string s = "config/test_sparsetil_sin_config.txt"; //test config
  RunMode r;
  PetscErrorCode ierr = 0;
  ierr = parseConfig(s, params, app_settings, spec_ops, &r);

  // make an inverse solver
  std::shared_ptr<SolverInterface> solver = std::make_shared<InverseL1Solver>();
  ierr = solver->initialize(spec_ops, params, app_settings);
  ierr = solver->run();
  ierr = solver->finalize();

  ScalarType norm = 0;

  // test finalize, c0_inv norm, c1_inv norm, rho_inv, kappa_inv
  REQUIRE(ierr == 0);
#ifdef SINGLE
  VecNorm(solver->getTumor()->c_0_, NORM_2, &norm);
  REQUIRE(norm == Approx(34.9651));
  VecNorm(solver->getTumor()->c_t_, NORM_2, &norm);
  REQUIRE(norm == Approx(68.3326));
  REQUIRE(params->tu_->rho_ == Approx(3.8765));
  REQUIRE(params->tu_->k_ == Approx(0.000236307));
#endif
}

TEST_CASE( "Running inverse sparse-til simulator", "[simulator]" ) {
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
  std::shared_ptr<ApplicationSettings> app_settings = std::make_shared<ApplicationSettings>();
#if defined(CUDA) && !defined(MPICUDA)
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (CUFFT);
#else
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
#endif
  // disable all I/O for tests
  DISABLE_VERBOSE = true;

  std::string s = "config/test_sparsetil_config.txt"; //test config
  RunMode r;
  PetscErrorCode ierr = 0;
  ierr = parseConfig(s, params, app_settings, spec_ops, &r);

  // make an inverse solver
  std::shared_ptr<SolverInterface> solver = std::make_shared<InverseL1Solver>();
  ierr = solver->initialize(spec_ops, params, app_settings);
  ierr = solver->run();
  ierr = solver->finalize();

  ScalarType norm = 0;

  // test finalize, c0_inv norm, c1_inv norm, rho_inv, kappa_inv
  REQUIRE(ierr == 0);
#ifdef SINGLE
  VecNorm(solver->getTumor()->c_0_, NORM_2, &norm);
  REQUIRE(norm == Approx(28.4884));
  VecNorm(solver->getTumor()->c_t_, NORM_2, &norm);
  REQUIRE(norm == Approx(75.8813));
  REQUIRE(params->tu_->rho_ == Approx(5.22846));
  REQUIRE(params->tu_->k_ == Approx(0.00523482));
#endif
}


