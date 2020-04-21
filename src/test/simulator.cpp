#include "Solver.h"
#include "IO.h"
#include "catch.hpp"

TEST_CASE("Running forward simulator with sinusoidal coefficients", "[sinusoid-simulator]") {
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
  std::shared_ptr<ApplicationSettings> app_settings = std::make_shared<ApplicationSettings>();
#if defined(CUDA) && !defined(MPICUDA)
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (CUFFT);
#else
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
#endif
  // disable all I/O for tests
  DISABLE_VERBOSE = true;

  /* forward solver test */
  std::string s = "config/test_forward_sin_config.txt"; //test config
  RunMode r;
  PetscErrorCode ierr = 0;
  ierr = parseConfig(s, params, app_settings, spec_ops, &r);

  // make a forward solver
  std::shared_ptr<SolverInterface> solver = std::make_shared<ForwardSolver>();
  params->tu_->smoothing_factor_atlas_ = 0; // do not smooth atlas ~ sinusoid
  ierr = solver->initialize(spec_ops, params, app_settings);
  ierr = solver->run();
  ierr = solver->finalize();

  // === create tmp vector for tests according to distributed grid
  Vec tmp;
  VecCreate(PETSC_COMM_WORLD, &tmp);
  VecSetSizes(tmp, params->grid_->nl_, params->grid_->ng_);
  setupVec(tmp);
  VecSet(tmp, 0.0);

  ScalarType norm = 0;

  // test finalize, displacement norm, c0 norm, c1 norm
  REQUIRE(ierr == 0);
#ifdef SINGLE
  VecNorm(solver->getTumor()->c_0_, NORM_2, &norm);
  REQUIRE(norm == Approx(22.0161f));
  VecNorm(solver->getTumor()->c_t_, NORM_2, &norm);
#ifdef CUDA
  REQUIRE(norm == Approx(83.9339f));
#else
  REQUIRE(norm == Approx(83.9893f));
#endif
  solver->getTumor()->displacement_->computeMagnitude(tmp);
  VecNorm(tmp, NORM_2, &norm);
#ifdef CUDA
  REQUIRE(norm == Approx(43.48092f));
#else
  REQUIRE(norm == Approx(43.5494f));
#endif
#endif
}


TEST_CASE( "Running forward simulator", "[simulator]" ) {
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
  std::shared_ptr<ApplicationSettings> app_settings = std::make_shared<ApplicationSettings>();
#if defined(CUDA) && !defined(MPICUDA)
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (CUFFT);
#else
  std::shared_ptr<SpectralOperators> spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
#endif
  // disable all I/O for tests
  DISABLE_VERBOSE = true;

  /* forward solver test */
  std::string s = "config/test_forward_config.txt"; //test config
  RunMode r;
  PetscErrorCode ierr = 0;
  ierr = parseConfig(s, params, app_settings, spec_ops, &r);

  // make a forward solver
  std::shared_ptr<SolverInterface> solver = std::make_shared<ForwardSolver>();
  ierr = solver->initialize(spec_ops, params, app_settings);
  ierr = solver->run();
  ierr = solver->finalize();

  // === create tmp vector for tests according to distributed grid
  Vec tmp;
  VecCreate(PETSC_COMM_WORLD, &tmp);
  VecSetSizes(tmp, params->grid_->nl_, params->grid_->ng_);
  setupVec(tmp);
  VecSet(tmp, 0.0);

  ScalarType norm = 0;

  // test displacement norm, c0 norm, c1 norm
#ifdef SINGLE
  VecNorm(solver->getTumor()->c_0_, NORM_2, &norm);
  REQUIRE(norm == Approx(4.09351f));
  VecNorm(solver->getTumor()->c_t_, NORM_2, &norm);
#ifdef CUDA
  REQUIRE(norm == Approx(31.97134f));
#else
  REQUIRE(norm == Approx(32.1486f));
#endif
  solver->getTumor()->displacement_->computeMagnitude(tmp);
  VecNorm(tmp, NORM_2, &norm);
#ifdef CUDA
  REQUIRE(norm == Approx(26.62458f));  
#else
  REQUIRE(norm == Approx(27.0497f));
#endif
#endif
}

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
  ScalarType rel_error_rho, rel_error_kappa;

  // test c0_inv norm, c1_inv norm, rho_inv, kappa_inv
#ifdef SINGLE
  VecNorm(solver->getTumor()->c_0_, NORM_2, &norm);
  REQUIRE(norm == Approx(34.9651f));
  VecNorm(solver->getTumor()->c_t_, NORM_2, &norm);
  REQUIRE(norm == Approx(68.3326f));
  rel_error_rho = (params->tu_->rho_ - 3.8765f) / 3.8765f;
  REQUIRE(rel_error_rho < 1E-3);
  rel_error_kappa = (params->tu_->k_ - 0.000236307f) / 0.000236307f;
  REQUIRE(rel_error_kappa < 1E-3);
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
  ScalarType rel_error_rho, rel_error_kappa;

  // test c0_inv norm, c1_inv norm, rho_inv, kappa_inv
#ifdef SINGLE
  VecNorm(solver->getTumor()->c_0_, NORM_2, &norm);
  REQUIRE(norm == Approx(28.4884f));
  VecNorm(solver->getTumor()->c_t_, NORM_2, &norm);
  REQUIRE(norm == Approx(75.8813f));
  rel_error_rho = (params->tu_->rho_ - 5.22846f) / 5.22846f;
  REQUIRE(rel_error_rho < 1E-3);
  rel_error_kappa = (params->tu_->k_ - 0.00523482f) / 0.00523482f;
  REQUIRE(rel_error_kappa < 1E-3);
#endif
}


