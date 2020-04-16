#include "Solver.h"
#include "IO.h"
#include "catch.hpp"


TEST_CASE( "Simulator-level tests", "[simulator]" ) {
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
  SECTION( "running forward solver with advection" ) {
    std::string s = "config/test_forward_config.txt"; //test config
    RunMode r;
    PetscErrorCode ierr = 0;
    ierr = parseConfig(s, params, app_settings, spec_ops, &r);

    // make a forward solver
    std::shared_ptr<SolverInterface> solver = std::make_shared<ForwardSolver>();
    ierr = solver->initialize(spec_ops, params, app_settings);
    ierr = solver->run();
    ierr = solver->finalize();

    // ScalarType nrm;
    // might be pointless to test this
    REQUIRE(ierr == 0);
    // what other tests? c0 norm, check for matprop nullptrs, dispnorm > 0 if model has mass effect
    // if (params->tu_->model_ >= 4 && params->tu_->forcing_factor_ > 0) {
    //   solver->tumor_->displacement_->computeMagnitude(solver->tmp_);
    //   VecNorm(solver->tmp_, NORM_2, &nrm);
    //   REQUIRE(nrm > 0);
    // }
  }

  /* inverse sparse til solver test */
  SECTION( "running inverse sparse-til solver" ) {
    std::string s = "config/test_sparsetil_config.txt"; //test config
    RunMode r;
    PetscErrorCode ierr = 0;
    ierr = parseConfig(s, params, app_settings, spec_ops, &r);

    // make an inverse solver
    std::shared_ptr<SolverInterface> solver = std::make_shared<InverseL1Solver>();
    ierr = solver->initialize(spec_ops, params, app_settings);
    ierr = solver->run();
    ierr = solver->finalize();

    REQUIRE(ierr == 0);
  }
}