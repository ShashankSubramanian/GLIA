
// system includes
#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>

#include "Utils.h"
#include "Solver.h"
#include "Parameters.h"
#include "SpectralOperators.h"
#include "TumorSolverInterface.h"


enum RunMode {FORWARD, INVERSE_L2, INVERSE_L1, INVERSE_RD, INVERSE_ME, MULTI_SPECIES, TEST};

RunMode run_mode = FORWARD; // global variable

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

PetscErrorCode initializeGrid(int n, std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  int N[3] = {n, n, n};
  int isize[3];
  int istart[3];
  int osize[3];
  int ostart[3];
  int c_dims[2] = {0};
  MPI_Comm c_comm;

  accfft_init();
  accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
  spec_ops->setup(N, isize, istart, osize, ostart, c_comm);
  int64_t alloc_max = spec_ops->alloc_max_;
  fft_plan *plan = spec_ops->plan_;
  params->createGrid(N, isize, osize, istart, ostart, plan, c_comm, c_dims);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
void setParameter(std::string name, std::string value, std::shared_ptr<Parameters> p, std::shared_ptr<ApplicationSettings> a) {
  if(name == "solver") {
    // quick set all neccessary parameters to support minimal config files
    if(value == "sparse_til") {
      run_mode = INVERSE_L1;
      p->opt_->regularization_norm_ = L1;    // TODO(K) set regularization norm to L1
      p->opt_->diffusivity_inversion_ = true;
      p->opt_->reaction_inversion_ = true;
      p->opt_->pre_reacdiff_solve_ = true;
      return;
    }
    if(value == "nonsparse_til") {
      run_mode = INVERSE_L2;
      p->opt_->regularization_norm_ = L2;
      p->opt_->diffusivity_inversion_ = true;
      p->opt_->reaction_inversion_ = false;
      p->opt_->pre_reacdiff_solve_ = false;
      return;
    }
    if(value == "reaction_diffusion") {
      run_mode = INVERSE_RD;
      p->opt_->diffusivity_inversion_ = true;
      p->opt_->reaction_inversion_ = true;
      return;
    }
    if(value == "mass_effect") {
      run_mode = INVERSE_ME;
      p->opt_->invert_mass_effect_ = true;
      return;
    }
    if(value == "multi_species") {
      run_mode = MULTI_SPECIES;
      return;
    }
    if(value == "forward") {
      run_mode = FORWARD;
      p->tu_->time_history_off_ = true;
      return;
    }
  }
  // parse all other parameters
  // ### inversion scheme
  if (name == "invert_diff") {p->opt_->diffusivity_inversion_ = std::stoi(value) > 0; return;}
  if (name == "invert_reac") {p->opt_->reaction_inversion_ = std::stoi(value) > 0; return;}
  if (name == "multilevel") {p->opt_->multilevel_ = std::stoi(value) > 0; p->opt_->rescale_init_cond_ = true; return;}
  if (name == "inject_solution") {a->inject_solution_ = std::stoi(value) > 0; return;}
  if (name == "pre_reacdiff_solve") {p->opt_->pre_reacdiff_solve_ = std::stoi(value) > 0; return;}
  if (name == "verbosity") {p->tu_->verbosity_ = std::stoi(value); return;}
  // ### optimizer
  if (name == "newton_solver") {p->opt_->newton_solver_ = (value == "GN") ? 0 : 1; return;}
  if (name == "line_search") {p->opt_->linesearch_ = (value == "armijo") ? 0 : 1; return;}
  if (name == "ce_loss") {p->opt_->cross_entropy_loss_ = std::stoi(value) > 0; return;}
  if (name == "regularization") {p->opt_->regularization_norm_ = (value == "L1") ? L1 : L2; return;}
  if (name == "beta_p") {p->opt_->beta_ = std::stod(value); return;}
  if (name == "opttol_grad") {p->opt_->opttolgrad_ = std::stod(value); return;}
  if (name == "newton_maxit") {p->opt_->newton_maxit_ = std::stoi(value); return;}
  if (name == "krylov_maxit") {p->opt_->krylov_maxit_ = std::stoi(value); return;}
  if (name == "gist_maxit") {p->opt_->gist_maxit_ = std::stoi(value); return;}
  if (name == "kappa_lb") {p->opt_->k_lb_ = std::stod(value); return;}
  if (name == "kappa_ub") {p->opt_->k_ub_ = std::stod(value); return;}
  if (name == "rho_lb") {p->opt_->rho_lb_ = std::stod(value); return;}
  if (name == "rho_ub") {p->opt_->rho_ub_ = std::stod(value); return;}
  if (name == "gamma_ub_") {p->opt_->gamma_ub_ = std::stod(value); return;}
  if (name == "lbfgs_vectors") {p->opt_->lbfgs_vectors_ = std::stoi(value); return;}
  if (name == "lbfgs_scale_type") {p->opt_->lbfgs_scale_type = value; return;}
  if (name == "lbfgs_scale_hist") {p->opt_->lbfgs_scale_hist = std::stoi(value); return;}
  if (name == "ls_max_func_evals") {p->opt_->ls_max_func_evals = std::stoi(value); return;}
  // ### forward solver
  if (name == "model") {p->tu_->model_ = std::stoi(value); return;}
  if (name == "init_rho") {p->tu_->rho_ = std::stod(value); return;}
  if (name == "init_k") {p->tu_->k_ = std::stod(value); return;}
  if (name == "init_gamma") {p->tu_->forcing_factor_ = std::stod(value); return;}
  if (name == "nt_inv") {p->tu_->nt_ = std::stoi(value); return;}
  if (name == "dt_inv") {p->tu_->dt_ = std::stod(value); return;}
  if (name == "k_gm_wm") {p->tu_->k_gm_wm_ratio_ = std::stod(value); return;}
  if (name == "r_gm_wm") {p->tu_->r_gm_wm_ratio_ = std::stod(value); return;}
  // ### data
  if (name == "smoothing_factor") {p->tu_->smoothing_factor_ = std::stod(value); return;}
  if (name == "smoothing_factor_data") {p->tu_->smoothing_factor_data_ = std::stod(value); return;}
  if (name == "obs_threshold_1") {p->tu_->obs_threshold_1_ = std::stod(value); return;}
  if (name == "obs_threshold_0") {p->tu_->obs_threshold_0_ = std::stod(value); return;}
  if (name == "obs_threshold_rel") {p->tu_->relative_obs_threshold_ = std::stoi(value) > 0; return;}
  // ### initial condition
  if (name == "sparsity_level") {p->tu_->sparsity_level_ = std::stoi(value); return;}
  if (name == "gaussian_selection_mode") {a->gaussian_selection_mode_ = std::stoi(value); return;}
  if (name == "number_gaussians") {p->tu_->np_ = std::stoi(value); return;}
  if (name == "sigma_factor") {
    ScalarType factor = (2.0 * M_PI) / p->grid_->n_[0];
    p->tu_->phi_sigma_ = std::stod(value) * factor; p->tu_->phi_sigma_data_driven_ = std::stod(value) * factor; return;}
  if (name == "sigma_spacing") {p->tu_->phi_spacing_factor_ = std::stod(value); return;}
  if (name == "threshold_data_driven") {p->tu_->data_threshold_ = std::stod(value); return;}
  if (name == "gaussian_volume_fraction") {p->tu_->gaussian_vol_frac_ = std::stod(value); return;}
  // ### prediction
  if (name == "prediction") {a->pred_->enabled_ = std::stoi(value) > 0; return;}
  if (name == "pred_times") {
    std::string v = value.substr(value.find("[")+1);
    value = v.substr(0, value.find("]"));
    size_t pos_loop = 0, pos = 0;
    while ((pos_loop = value.find(",")) != std::string::npos) {
      v = value.substr(0, pos_loop);
      a->pred_->t_pred_.push_back(std::stod(v));
      value.erase(0, pos_loop + 1);
    }
    a->pred_->t_pred_.push_back(std::stod(value));
     return;
  }
  if (name == "dt_pred") {a->pred_->dt_ = std::stod(value); return;}
  // ### synthetic data
  if (name == "syn_flag") {a->syn_->enabled_ = std::stoi(value) > 0; return;}
  if (name == "user_cms") {
    size_t pos_loop = 0, pos = 0;
    std::string cm_str, x_, y_, z_, s_;
    while ((pos_loop = value.find(")")) != std::string::npos) {
      cm_str = value.substr(2, pos_loop);
      pos = cm_str.find(",");
      x_ = cm_str.substr(0, pos);
      cm_str.erase(0, pos + 1);
      pos = cm_str.find(",");
      y_ = cm_str.substr(0, pos);
      cm_str.erase(0, pos + 1);
      pos = cm_str.find(",");
      z_ = cm_str.substr(0, pos);
      s_ = cm_str.substr(pos+1);
      std::array<ScalarType, 4> user_cm = { static_cast<ScalarType>(std::stod(x_)), static_cast<ScalarType>(std::stod(y_)),
                                            static_cast<ScalarType>(std::stod(z_)), static_cast<ScalarType>(std::stod(s_)) };
      a->syn_->user_cms_.push_back(user_cm);
      value.erase(0, pos_loop+1);
    }
    return;
  }
  if (name == "rho_data") {a->syn_->rho_ = std::stod(value); return;}
  if (name == "k_data") {a->syn_->k_ = std::stod(value); return;}
  if (name == "gamma_data") {a->syn_->forcing_factor_ = std::stod(value); return;}
  if (name == "nt_data") {a->syn_->nt_ = std::stoi(value); return;}
  if (name == "dt_data") {a->syn_->dt_ = std::stod(value); return;}
  if (name == "testcase") {a->syn_->testcase_ = std::stod(value); return;}
  // ### paths
  if (name == "output_dir") {p->tu_->writepath_ = value; return;}
  if (name == "input_dir") {p->tu_->readpath_ = value; return;}
  if (name == "d1_path") {a->path_->data_t1_ = value; return;}
  if (name == "d0_path") {a->path_->data_t0_ = value; return;}
  if (name == "a_seg_path") {a->path_->seg_ = value; return;}
  if (name == "a_wm_path") {a->path_->wm_ = value; return;}
  if (name == "a_gm_path") {a->path_->gm_ = value; return;}
  if (name == "a_csf_path") {a->path_->csf_ = value; return;}
  if (name == "a_vt_path") {a->path_->vt_ = value; return;}
  if (name == "p_seg_path") {a->path_->p_seg_ = value; return;}
  if (name == "p_wm_path") {a->path_->p_wm_ = value; return;}
  if (name == "p_gm_path") {a->path_->p_gm_ = value; return;}
  if (name == "p_csf_path") {a->path_->p_csf_ = value; return;}
  if (name == "p_vt_path") {a->path_->p_vt_ = value; return;}
  if (name == "mri_path") {a->path_->mri_ = value; return;}
  if (name == "obs_mask_path") {a->path_->obs_filter_ = value; return;}
  if (name == "support_data_path") {a->path_->data_support_ = value; return;} // TODO(K) .nc vs. dat.
  if (name == "gaussian_cm_path") {a->path_->phi_ = value; return;}
  if (name == "pvec_path") {a->path_->pvec_ = value; return;}
  if (name == "data_comp_path") {a->path_->data_comps_ = value; return;}
  if (name == "data_comp_data_path") {a->path_->data_comps_data_ = value; return;}
  if (name == "velocity_x1") {a->path_->velocity_x1_ = value; return;}
  if (name == "velocity_x2") {a->path_->velocity_x2_ = value; return;}
  if (name == "velocity_x3") {a->path_->velocity_x3_ = value; return;}
  // ### performance
  if (name == "time_history_off") {p->tu_->time_history_off_ = std::stoi(value) > 0; return;}
  if (name == "store_phi") {p->tu_->phi_store_ = std::stoi(value) > 0; return;}
  if (name == "store_adjoint") {p->tu_->adjoint_store_ = std::stoi(value) > 0; return;}
  if (name == "write_output") {p->tu_->write_output_ = std::stoi(value) > 0; return;}
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
            ierr = initializeGrid(std::stoi(value), params, spec_ops); CHKERRQ(ierr);
          } else {
            setParameter(name, value, params, app_settings);
          }
      }
  }
  else {
    ierr = tuMSGwarn("No config file given. Terminating Solver."); CHKERRQ(ierr);
    exit(0);
  }


  EventRegistry::initialize();

  // === initialize solvers
  std::shared_ptr<Solver> solver;
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
    case TEST:
      solver = std::make_shared<TestSuite>();
      break;
    default:
      ierr = tuMSGwarn("Configuration invalid: solver mode not recognized. Exiting."); CHKERRQ(ierr);
      PetscFunctionReturn(ierr);
  }

  openFiles(params); // opens all txt output files on one core only
  // ensures data for specific solver is read and code is set up for running mode
  ierr = solver->initialize(spec_ops, params, app_settings); CHKERRQ(ierr);
  ierr = solver->run(); CHKERRQ(ierr);
  // compute errors, segmentations, other measures of interest, and shutdown solver
  ierr = solver->finalize(); CHKERRQ(ierr);

  closeFiles(params);

  #ifdef CUDA
      cudaPrintDeviceMemory();
  #endif
  EventRegistry::finalize ();
  if (procid == 0) {
      EventRegistry r;
      r.print();
      r.print("EventsTimings.log", true);
  }

  } // shared_ptrs scope end
  ierr = PetscFinalize();
  PetscFunctionReturn(ierr);
}
