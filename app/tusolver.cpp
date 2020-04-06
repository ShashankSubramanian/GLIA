/**
 *  SIBIA (Scalable Biophysics-Based Image Analysis)
 *
 *  Copyright (C) 2017-2020, The University of Texas at Austin
 *  This file is part of the SIBIA library.
 *
 *  Authors: Klaudius Scheufele, Shashank Subramanian
 *
 *  SIBIA is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  SIBIA is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see the LICENSE file.
 *
 **/


// system includes

#include "Utils.h"
#include "Parameters.h"
#include "TumorSolverInterface.h"


enum RunMode = {FORWARD, INVERSE_L2, INVERSE_L1, INVERSE_RD, INVERSE_ME, INVERSE_MS, TEST};


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
void openFiles(std::shared_ptr<Parameters> params) {
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  if (params->verbosity_ >= 2) {
      if (procid == 0) {
          ss << params->writepath_.str().c_str() << "x_it.dat";
          params->outfile_sol_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
          ss << params->writepath_.str().c_str() << "g_it.dat";
          params->outfile_grad_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
          ss << params->writepath_.str().c_str() << "glob_g_it.dat";
          params->outfile_glob_grad_.open(ss.str().c_str(), std::ios_base::out); ss.str(std::string()); ss.clear();
          params->outfile_sol_ << std::setprecision(16)<<std::scientific;
          params->outfile_grad_ << std::setprecision(16)<<std::scientific;
          params->outfile_glob_grad_ << std::setprecision(16)<<std::scientific;
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

  if (procid == 0 && params_->verbosity_ >= 2) {
      params_->outfile_sol_.close();
      params_->outfile_grad_.close();
      params_->outfile_glob_grad_.close();
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

  accfft_init();
  MPI_Comm c_comm;
  int c_dims[2] = { 0 };
  accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
  int isize[3], osize[3], istart[3], ostart[3];

  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  std::shared_ptr<SpectralOperators> spec_ops;
  #if defined(CUDA) && !defined(MPICUDA)
      spec_ops = std::make_shared<SpectralOperators> (CUFFT);
  #else
      spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
  #endif
  spec_ops->setup (n, isize, istart, osize, ostart, c_comm);
  int64_t alloc_max = spec_ops->alloc_max_;
  fft_plan *plan = spec_ops->plan_;

  RunMode run_mode = FORWARD;
  std::stringstream ss;
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();

  // TODO(K) parse config file and populate into parameters; also set run_mode
  // TODO(K) params->opt_settings_ have to be populated in arg parse


  EventRegistry::initialize();

  std::unique_ptr<Solver> solver;
  switch(run_mode) {
    case FORWARD:
      solve = std::make_unique<ForwardSolver>(spec_ops);
      break;
    case INVERSE_L2:
      solve = std::make_unique<InverseL2Solver>(spec_ops);
      break;
    case INVERSE_L1:
      solve = std::make_unique<InverseL1Solver>(spec_ops);
      break;
    case INVERSE_RD:
      solve = std::make_unique<InverseReactionDiffusionSolver>(spec_ops);
      break;
    case INVERSE_ME:
      solve = std::unique_ptr<InverseMassEffectSolver>(spec_ops);
      break;
    case INVERSE_MS:
      solve = std::make_unique<InverseMultiSpeciesSolver>(spec_ops);
      break;
    case TEST:
      solve = std::make_unique<TestSuite>(spec_ops);
      break;
    default:
      ierr = cplMSGwarn("Configuration invalid: solver mode not recognized. Exiting."); CHKERRQ(ierr);
      PetscFunctionReturn(ierr);
  }

  openFiles(); // opens all txt output files on one core only
  // ensures data for specific solver is read and code is set up for running mode
  ierr = solver->initialize(); CHKERRQ(ierr);
  ierr = solver->run(); CHKERRQ(ierr);
  // compute errors, segmentations, other measures of interest, and shutdown solver
  ierr = solver->finalize(); CHKERRQ(ierr);

  closeFiles();

  #ifdef CUDA
      cudaPrintDeviceMemory();
  #endif
  EventRegistry::finalize ();
  if (procid == 0) {
      EventRegistry r;
      r.print ();
      r.print ("EventsTimings.log", true);
  }

  PetscFunctionReturn(ierr);
}
