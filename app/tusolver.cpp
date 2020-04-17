
// system includes
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>

#include "IO.h"
#include "Utils.h"
#include "TestSuite.h"
#include "Parameters.h"
#include "SpectralOperators.h"
#include "SolverInterface.h"
#include "Solver.h"

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
void openFiles(std::shared_ptr<Parameters> params) {
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  if (params->tu_->verbosity_ >= 2) {
      if (procid == 0) {
          ss << params->tu_->writepath_ << "x_it.dat";
          params->tu_->outfile_sol_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
          ss << params->tu_->writepath_ << "g_it.dat";
          params->tu_->outfile_grad_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
          ss << params->tu_->writepath_ << "glob_g_it.dat";
          params->tu_->outfile_glob_grad_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
          params->tu_->outfile_sol_ << std::setprecision(16)<<std::scientific;
          params->tu_->outfile_grad_ << std::setprecision(16)<<std::scientific;
          params->tu_->outfile_glob_grad_ << std::setprecision(16)<<std::scientific;
      }
  }
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
void closeFiles(std::shared_ptr<Parameters> params) {
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  if (procid == 0 && params->tu_->verbosity_ >= 2) {
      params->tu_->outfile_sol_.close();
      params->tu_->outfile_grad_.close();
      params->tu_->outfile_glob_grad_.close();
  }
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv) {

  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc, &argv, reinterpret_cast<char*>(NULL), reinterpret_cast<char*>(NULL)); CHKERRQ(ierr);

  { // begin local scope for all shared pointers (all MPI/PETSC finalize should be out of this scope to allow for
    // safe destruction of all petsc vectors
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  std::stringstream ss;
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
  std::shared_ptr<ApplicationSettings> app_settings = std::make_shared<ApplicationSettings>();

  // verbose
  ierr = tuMSGstd (""); CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###"); CHKERRQ (ierr);
  ierr = tuMSG("###                                         TUMOR INVERSION SOLVER                                        ###"); CHKERRQ (ierr);
  ierr = tuMSG("### ----------------------------------------------------------------------------------------------------- ###"); CHKERRQ (ierr);

  // === create distributed compute grid
  std::shared_ptr<SpectralOperators> spec_ops;
  #if defined(CUDA) && !defined(MPICUDA)
      spec_ops = std::make_shared<SpectralOperators> (CUFFT);
  #else
      spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
  #endif

  // === config file
  std::string config;
  for(int i = 1; i < argc; ++i) {
    if(std::string(argv[i]) == "-config") config = std::string(argv[i+1]);
  }

  // === parse config file, set parameters
  RunMode run_mode = FORWARD;
  ierr = parseConfig(config, params, app_settings, spec_ops, &run_mode);

  // === initialize solvers
  std::shared_ptr<SolverInterface> solver;
  switch(run_mode) {
    case FORWARD:
      solver = std::make_shared<ForwardSolver>();
      break;
    case INVERSE_L2:
      solver = std::make_shared<InverseL2Solver>();
      break;
    case INVERSE_L1:
      solver = std::make_shared<InverseL1Solver>();
      break;
    case INVERSE_RD:
      solver = std::make_shared<InverseReactionDiffusionSolver>();
      break;
    case INVERSE_ME:
      solver = std::make_shared<InverseMassEffectSolver>();
      break;
    case MULTI_SPECIES:
      solver = std::make_shared<MultiSpeciesSolver>();
      break;
    default:
      ierr = tuMSGwarn("Configuration invalid: solver mode not recognized. Exiting."); CHKERRQ(ierr);
      PetscFunctionReturn(ierr);
  }

  // timings
  ierr = solver->initializeEvent(); CHKERRQ(ierr);
  // opens all txt output files on one core only
  openFiles(params);
  // ensures data for specific solver is read and code is set up for running mode
  ierr = solver->initialize(spec_ops, params, app_settings); CHKERRQ(ierr);
  ierr = solver->run(); CHKERRQ(ierr);
  // compute errors, segmentations, other measures of interest, and shutdown solver
  ierr = solver->finalize(); CHKERRQ(ierr);
  // closes all txt output files on one core only
  closeFiles(params);
  // timings
  ierr = solver->finalizeEvent(); CHKERRQ(ierr);
  // memory
  #ifdef CUDA
    cudaPrintDeviceMemory();
  #endif
  } // shared_ptrs scope end

  // shut down
  ierr = PetscFinalize();
  PetscFunctionReturn(ierr);
}
