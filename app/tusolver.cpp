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
#include <stdlib.h>

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

PetscErrorCode initializeGrid(n, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  int[3] N = {n, n, n};
  int[3] isize;
  int[3] istart;
  int[3] osize;
  int[3] ostart;
  int c_dims[2] = {0};
  MPI_Comm c_comm;

  accfft_init();
  accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
  ierr = spec_ops->setup(N, isize, istart, osize, ostart, c_comm); CHKERRQ(ierr);
  int64_t alloc_max = spec_ops->alloc_max_;
  fft_plan *plan = spec_ops->plan_;
  params->createGrid(N, isize, osize, istart, ostart, plan, c_comm, c_dims);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
void setParameter(std::string name, std::string value, std:shared_ptr<Parameters> p, std::shared_ptr<ApplicationSettings> a) {
  if(name == "solver") {
    // quick set all neccessary parameters to support minimal config files
    if(value == "sparse_til") {
      run_mode = INVERSE_L1;
      p->opt_->regularization_norm_ = L1c;
      p->opt_->diffusivity_inversion_ = true;
      p->opt_->reaction_inversion = true;
      p->opt_->pre_reacdiff_solve = true;
    }
    if(value == "nonsparse_til") {
      run_mode = IINVERSE_L2;
      p->opt_->regularization_norm_ = L2;
      p->opt_->diffusivity_inversion_ = true;
      p->opt_->reaction_inversion = false;
      p->opt_->pre_reacdiff_solve = false;
    }
    if(value == "reaction_diffusion") {
      run_mode = INVERSE_RD;
      p->opt_->diffusivity_inversion_ = true;
      p->opt_->reaction_inversion = true;
    }
    if(value == "mass_effec") {
      run_mode = INVERSE_ME;
      p->opt_->invert_mass_effect_ = true;
    }
    if(value == "multi_species") {
      run_mode = INVERSE_MS;
    }
    if(value == "forward") {
      run_mode = FORWARD;
      p->time_history_off_ = true;
    }

    // parse all other parameters
    // ### inversion scheme
    if (value == "invert_diff") {p->opt_->diffusivity_inversion_ = std::stoi(value) > 0;}
    if (value == "invert_reac") {p->opt_->reaction_inversion_ = std::stoi(value) > 0;}
    if (value == "multilevel") {p->tu_->multilevel_ = std::stoi(value) > 0;}
    if (value == "inject_solution") {a->inject_solution_ = std::stoi(value) > 0;}
    if (value == "pre_reacdiff_solve") {p->opt_->pre_reacdiff_solve_ = std::stoi(value) > 0;}
    if (value == "verbosity") {p->tu_->verbosity_ = std::stoi(value);}
    // ### optimizer
    if (value == "newton_solver") {p->opt_->newton_solver_ = (value == "GN") ? 0 : 1;}
    if (value == "line_search") {p->opt_->linesearch_ = (value == "armijo") ? 0 : 1;}
    if (value == "ce_loss") {p->opt_->cross_entropy_loss_ = std::stoi(value) > 0;}
    if (value == "regularization") {p->tu_->regularization_norm_ = (value == "L1") ? 0 : 1;}
    if (value == "beta_p") {p->tu_->beta_ = std::stod(value);}
    if (value == "opttol_grad") {p->opt_->opttolgrad_ = std::stod(value);}
    if (value == "newton_maxit") {p->opt_->newton_maxit_ = std::stoi(value);}
    if (value == "krylov_maxit") {p->opt_->krylov_maxit_ = std::stoi(value);}
    if (value == "gist_maxit") {p->opt_->gist_maxit_ = std::stoi(value);}
    if (value == "kappa_lb") {p->opt_->k_lb_ = std::stod(value);}
    if (value == "kappa_ub") {p->opt_->k_ub_ = std::stod(value);}
    if (value == "rho_lb") {p->opt_->rho_lb_ = std::stod(value);}
    if (value == "rho_ub") {p->opt_->rho_ub_ = std::stod(value);}
    if (value == "gamma_ub_") {p->opt_->gamma_ub_ = std::stod(value);}
    if (value == "lbfgs_vectors") {p->opt_->lbfgs_vectors_ = std::stoi(value);}
    if (value == "lbfgs_scale_type") {p->opt_->lbfgs_scale_type = value;}
    if (value == "lbfgs_scale_hist") {p->opt_->lbfgs_scale_hist = std::stoi(value);}
    if (value == "ls_max_func_evals") {p->opt_->ls_max_func_evals = std::stoi(value);}
    // ### forward solver
    if (value == "model") {p->tu_->model_ = std::stoi(value);}
    if (value == "init_rho") {p->tu_->rho_ = std::stod(value);}
    if (value == "init_k") {p->tu_->k_ = std::stod(value);}
    if (value == "init_gamma") {p->tu_->forcing_factor_ = std::stod(value);}
    if (value == "nt_inv") {p->tu_->nt_ = std::stoi(value);}
    if (value == "dt_inv") {p->tu_->dt_ = std::stod(value);}
    if (value == "k_gm_wm") {p->tu_->k_gm_wm_ratio_ = std::stod(value);}
    if (value == "r_gm_wm") {p->tu_->r_gm_wm_ratio_ = std::stod(value);}
    // ### data
    if (value == "smoothing_factor") {p->tu_->smoothing_factor_ = std::stod(value);}
    if (value == "smoothing_factor_data") {p->tu_->smoothing_factor_data_ = std::stod(value);}
    if (value == "obs_threshold_1") {p->tu_->obs_threshold_1_ = std::stod(value);}
    if (value == "obs_threshold_0") {p->tu_->obs_threshold_0_ = std::stod(value);}
    if (value == "obs_threshold_rel") {p->tu_->relative_obs_threshold_ = std::stoi(value) > 0;}
    // ### initial condition
    if (value == "sparsity_level") {p->tu_->sparsity_level_ = std::stoi(value);}
    if (value == "gaussian_selection_mode") {a->gaussian_selection_mode_ = std::stoi(value);}
    if (value == "number_gaussians") {p->tu_->np_ = std::stoi(value);}
    if (value == "sigma_factor") {p->tu_->phi_sigma_ = std::stod(value); p->tu_->phi_sigma_data_driven_ = std::stod(value);}
    if (value == "sigma_spacing") {p->tu_->phi_spacing_factor_ = std::stod(value);}
    if (value == "threshold_data_driven") {p->tu_->data_threshold_ = std::stod(value);}
    if (value == "gaussian_volume_fraction") {p->tu_->gaussian_vol_frac_ = std::stod(value);}
    // ### prediction
    if (value == "prediction") {a->pred_->enabled_ = std::stoi(value) > 0;}
    if (value == "pred_times") {
      std::string v = value.substr(line.find("[")+1);
      value = v.substr(0, line.find("]"));
      auto comma_pos = value.find(",");
      while (comma_pos != std::npos) {
        v = value.substr(0, comma_pos);
        value = value.substr(comma_pos+1);
        a->pred_->t_pred_.push_back(std::stod(v));
        comma_pos = value.find(",");
      }
      a->pred_->t_pred_.push_back(std::stod(value));
    }
    if (value == "dt_pred") {a->pred_->dt_ = std::stod(value);}
    // ### synthetic data
    if (value == "syn_flag") {a->syn_->enabled_ = std::stoi(value) > 0;}
    // TODO(K) user CMs
    if (value == "rho_data") {a->syn_->rho_ = std::stod(value);}
    if (value == "k_data") {a->syn_->k_ = std::stod(value);}
    if (value == "gamma_data") {a->syn_->forcing_factor_ = std::stod(value);}
    if (value == "nt_data") {a->syn_->nt_ = std::stoi(value);}
    if (value == "dt_data") {a->syn_->dt_ = std::stod(value);}
    if (value == "testcase") {a->syn_->testcase_ = std::stod(value);}
    // ### paths
    if (value == "output_dir") {p->writepath_ = value;}
    if (value == "d1_path") {a->path_->data_t1_ = value;}
    if (value == "d0_path") {a->path_->data_t0_ = value;}
    if (value == "a_seg_path") {a->path_->seg_ = value;}
    if (value == "a_wm_path") {a->path_->wm_ = value;}
    if (value == "a_gm_path") {a->path_->gm_ = value;}
    if (value == "a_csf_path") {a->path_->csf_ = value;}
    if (value == "a_glm_path") {a->path_->glm_ = value;}
    if (value == "a_ve_path") {a->path_->ve_ = value;}
    if (value == "p_seg_path") {a->path_->p_seg_ = value;}
    if (value == "p_wm_path") {a->path_->p_wm_ = value;}
    if (value == "p_gm_path") {a->path_->p_gm_ = value;}
    if (value == "p_csf_path") {a->path_->p_csf_ = value;}
    if (value == "p_glm_path") {a->path_->p_glm_ = value;}
    if (value == "p_ve_path") {a->path_->p_ve_ = value;}
    if (value == "mri_path") {a->path_->mri_ = value;}
    if (value == "obs_mask_path") {a->path_->obs_filter_ = value;}
    if (value == "support_data_path") {a->path_->data_support_ = value;} // TODO(K) .nc vs. dat.
    if (value == "gaussian_cm_path") {a->path_->phi_ = value;}
    if (value == "pvec_path") {a->path_->pvec_ = value;}
    if (value == "data_comp_path") {a->path_->data_comps_ = value;}
    if (value == "data_comp_data_path") {a->path_->data_comps_data_ = value;}
    if (value == "velocity_x1") {a->path_->velocity_x1_ = value;}
    if (value == "velocity_x2") {a->path_->velocity_x2_ = value;}
    if (value == "velocity_x3") {a->path_->velocity_x3_ = value;}
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

  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  std::stringstream ss;
  std::shared_ptr<Parameters> params = std::make_shared<Parameters>();
  std::shared_ptr<ApplicationSettings> app_settings = std::make_shared<ApplicationSettings>();


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
    if(std::string(argv[i]) == "-config") config = std::string(argv[i]);
  }

  // === parse config file, set parameters
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
          // initialize grid first
          if(name == "n") {
            ierr = initializeGrid(std::stoi(value), params, spec_ops) CHKERRQ(ierr);
          } else {
            setParameter(name, value, params, app_settings);
          }
      }
  }
  else {
    ierr = tuMSGwarn("No config file given. Terminating Solver."); CHKERRQ(ierr);
    exit(0);
  }


  // TODO(K) parse config file and populate into parameters; also set run_mode
  // TODO(K) params->opt_settings_ have to be populated in arg parse



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
  ierr = solver->initialize(params_, app_settings); CHKERRQ(ierr);
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
