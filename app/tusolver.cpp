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
#include <iostream>
#include <fstream>
#include <algorithm>

#include "Utils.h"
#include "Parameters.h"
#include "TumorSolverInterface.h"


enum RunMode = {FORWARD, INVERSE_L2, INVERSE_L1, INVERSE_RD, INVERSE_ME, INVERSE_MS, TEST};

RunMode run_mode = FORWARD; // global variable

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
void setParameter(std::string name, std::string value, std:shared_ptr<Parameters> params) {
  if(name == "solver") {
    if(value == "sparse_til") {
      run_mode = INVERSE_L1;
      params_->opt_->regularization_norm_ = L1c;
      params_->opt_->diffusivity_inversion_ = true;
      params_->opt_->reaction_inversion = true;
      params_->opt_->pre_reacdiff_solve = true;
    }
    if(value == "nonsparse_til")      {
      run_mode = IINVERSE_L2;
      params_->opt_->regularization_norm_ = L2;
      params_->opt_->diffusivity_inversion_ = true;
      params_->opt_->reaction_inversion = false;
      params_->opt_->pre_reacdiff_solve = false;
    }
    if(value == "reaction_diffusion") {
      run_mode = INVERSE_RD;
      params_->opt_->diffusivity_inversion_ = true;
      params_->opt_->reaction_inversion = true;
    }
    if(value == "mass_effec")         {run_mode = INVERSE_ME;  params_->opt_->invert_mass_effect_ = true;}
    if(value == "multi_species")      {run_mode = INVERSE_MS;}
    if(value == "forward")            {run_mode = FORWARD;     params_->forward_flag_ = true;}
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

  std::stringstream ss;
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();

  // === parse config file, set parameters
  std::string config;
  for(int i = 1; i < argc; ++i) {
    if(std::string(argv[i]) == "-config") config = std::string(argv[i]);
  }
  std::ifstream config_file (config);
  if (config_file.is_open()) {
      std::string line;
      while(getline(config_file, line)){
          line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
          if(line[0] == '#' || line.empty())  // skip empy lines and comments
              continue;
          if (line.find("#") != std::string::npos) { // allow comments after values
            line = line.substr(0, line.find("#"));
          }
          auto delimiter_pos = line.find("=");
          auto name = line.substr(0, delimiter_pos);
          auto value = line.substr(delimiter_pos + 1);
          // std::cout << name << " " << value << '\n';
          setParameter(name, value, params);
      }

  }
  else {
    ierr = tuMSGwarn("No config file given. Terminating Solver."); CHKERRQ(ierr);
    exit(0);
  }


  // TODO(K) parse config file and populate into parameters; also set run_mode
  // TODO(K) params->opt_settings_ have to be populated in arg parse


  // === create distributed compute grid
  std::shared_ptr<SpectralOperators> spec_ops;
  #if defined(CUDA) && !defined(MPICUDA)
      spec_ops = std::make_shared<SpectralOperators> (CUFFT);
  #else
      spec_ops = std::make_shared<SpectralOperators> (ACCFFT);
  #endif
  ierr = spec_ops->setup(n, isize, istart, osize, ostart, c_comm); CHKERRQ(ierr);
  int64_t alloc_max = spec_ops->alloc_max_;
  fft_plan *plan = spec_ops->plan_;
  params->createGrid(n, isize, osize, istart, ostart, plan, c_comm, c_dims);

  EventRegistry::initialize();

  // === initialize solvers
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
