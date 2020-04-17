#include "IO.h"
#include "Phi.h"

// parse parameters from config file

PetscErrorCode parseConfig(std::string s, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings, std::shared_ptr<SpectralOperators> spec_ops, RunMode *run_mode) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  std::ifstream config_file(s);
  // issue: not throwing an error is file does not exist in catch2 - check.
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
            setParameter(name, value, params, app_settings, run_mode);
          }
      }
  }
  else {
    ierr = tuMSGwarn("No config file given. Terminating Solver."); CHKERRQ(ierr);
    exit(0);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
void setParameter(std::string name, std::string value, std::shared_ptr<Parameters> p, std::shared_ptr<ApplicationSettings> a, RunMode *run_mode) {
  if(name == "solver") {
    // quick set all neccessary parameters to support minimal config files
    if(value == "sparse_til") {
      (*run_mode) = INVERSE_L1;
      p->opt_->regularization_norm_ = L2;
      p->opt_->diffusivity_inversion_ = true;
      p->opt_->reaction_inversion_ = true;
      p->opt_->pre_reacdiff_solve_ = true;
      return;
    }
    if(value == "nonsparse_til") {
      (*run_mode) = INVERSE_L2;
      p->opt_->regularization_norm_ = L2;
      p->opt_->diffusivity_inversion_ = true;
      p->opt_->reaction_inversion_ = false;
      p->opt_->pre_reacdiff_solve_ = false;
      return;
    }
    if(value == "reaction_diffusion") {
      (*run_mode) = INVERSE_RD;
      p->opt_->diffusivity_inversion_ = true;
      p->opt_->reaction_inversion_ = true;
      return;
    }
    if(value == "mass_effect") {
      (*run_mode) = INVERSE_ME;
      p->opt_->invert_mass_effect_ = true;
      return;
    }
    if(value == "multi_species") {
      (*run_mode) = MULTI_SPECIES;
      return;
    }
    if(value == "forward") {
      (*run_mode) = FORWARD;
      p->tu_->time_history_off_ = true;  // @K: wont this be reset from the params?
      return;
    }
  }
  // parse all other parameters
  // ### inversion scheme
  if (name == "invert_diff") {p->opt_->diffusivity_inversion_ = std::stoi(value) > 0; return;}
  if (name == "invert_reac") {p->opt_->reaction_inversion_ = std::stoi(value) > 0; return;}
  if (name == "multilevel") {p->tu_->multilevel_ = std::stoi(value) > 0; return;}
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
  if (name == "lbfgs_scale_type") {p->opt_->lbfgs_scale_type_ = value; return;}
  if (name == "lbfgs_scale_hist") {p->opt_->lbfgs_scale_hist_ = std::stoi(value); return;}
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
  if (name == "atlas_labels" || name == "patient_labels") { // read in somehting like: [wm=1, gm=2, vt=3, tc=5]
    std::string label, val;
    std::string v = value.substr(value.find("[")+1);
    value = v.substr(0, value.find("]"));
    size_t pos_loop = 0, pos = 0;
    std::vector<int> labels(8);
    for(int i = 0; i < labels.size(); ++i) {
      labels[i] = -1;
    }
    while ((pos_loop = value.find(",")) != std::string::npos) {
      v = value.substr(0, pos_loop);
      pos = v.find("=");
      label = v.substr(0, pos);
      val = v.substr(pos+1);
      if(label == "wm")  labels[0] = std::stoi(val);
      if(label == "gm")  labels[1] = std::stoi(val);
      if(label == "vt")  labels[2] = std::stoi(val);
      if(label == "csf") labels[3] = std::stoi(val);
      if(label == "tc")  labels[4] = std::stoi(val);
      if(label == "nec")  labels[5] = std::stoi(val);
      if(label == "en")  labels[6] = std::stoi(val);
      if(label == "ed")  labels[7] = std::stoi(val);
      value.erase(0, pos_loop + 1);
    }
      if( (pos = value.find("=")) != std::string::npos) {
        label = value.substr(0, pos);
        val = value.substr(pos+1);
        if(label == "wm")  labels[0] = std::stoi(val);
        if(label == "gm")  labels[1] = std::stoi(val);
        if(label == "vt")  labels[2] = std::stoi(val);
        if(label == "csf") labels[3] = std::stoi(val);
        if(label == "tc")  labels[4] = std::stoi(val);
        if(label == "nec")  labels[5] = std::stoi(val);
        if(label == "en")  labels[6] = std::stoi(val);
        if(label == "ed")  labels[7] = std::stoi(val);
      }
    //for(int i = 0; i < labels.size(); ++i) std::cout<<"label["<<i<<"]: "<<labels[i]<<std::endl;
    if(name == "atlas_labels") {
      a->atlas_seg_ = labels;
    } else {
      a->patient_seg_ = labels;
    }
     return;
  }
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
  if (name == "user_cms") { // read in somehting like: [(x1, y1, z1, scale1), (x2, y2, z2, scale2)]
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

// Error handling for IO
PetscErrorCode throwErrorMsg(std::string msg, int line, const char *file) {
  PetscErrorCode ierr = 0;
  std::stringstream ss;
  std::stringstream ss2;

  PetscFunctionBegin;

  ss2 << file << ":" << line;
  ss << std::setw(98 - ss2.str().size()) << std::left << msg << std::right << ss2.str();
  std::string errmsg = "\x1b[31mERROR: " + ss.str() + "\x1b[0m";
  ierr = PetscError(PETSC_COMM_WORLD, __LINE__, PETSC_FUNCTION_NAME, __FILE__, 1, PETSC_ERROR_INITIAL, errmsg.c_str()); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

PetscErrorCode myAssert(bool condition, std::string msg) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;

  if (condition == false) {
    ierr = throwError(msg); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode NCERRQ(int cerr) {
  int rank;
  PetscErrorCode ierr = 0;
  std::stringstream ss;
  PetscFunctionBegin;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  if (cerr != NC_NOERR) {
    ss << ncmpi_strerror(cerr);
    ierr = throwError(ss.str()); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

// Function definitions for netcdf and nifti IO
#ifdef NIFTIIO
template <typename T>
PetscErrorCode readNifti(nifti_image *image, ScalarType *data_global, std::shared_ptr<Parameters> params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int nprocs;
  int ng = params->grid_->ng_;
  MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

  T *data = nullptr;
  std::stringstream ss;
  if (nifti_image_load(image) == -1) {
    ss << "Error in loading nifit image";
    ierr = myAssert(false, ss.str()); CHKERRQ(ierr);
  }
  data = static_cast<T *>(image->data);
  // static cast image data to ScalarType
  int64_t idx = 0;
  int64_t index, x, y, z;
  for (int p = 0; p < nprocs; p++) {
    for (int i = 0; i < params->grid_->isize_gathered_[3 * p + 0]; i++) {
      for (int j = 0; j < params->grid_->isize_gathered_[3 * p + 1]; j++) {
        for (int k = 0; k < params->grid_->isize_gathered_[3 * p + 2]; k++) {
          x = params->grid_->istart_gathered_[3 * p + 0] + i;
          y = params->grid_->istart_gathered_[3 * p + 1] + j;
          z = params->grid_->istart_gathered_[3 * p + 2] + k;
          index = x * params->grid_->n_[1] * params->grid_->n_[2] + y * params->grid_->n_[2] + z;
          data_global[idx++] = static_cast<ScalarType>(data[index]);
        }
      }
    }
  }  // for all the processes

  PetscFunctionReturn(ierr);
}


PetscErrorCode dataIn(ScalarType *p_x, std::shared_ptr<Parameters> params, const char *fname) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  std::string file;
  std::stringstream ss;
  int rank, rval;
  int ng, nl, nglobal, nx[3];
  nifti_image *image = NULL;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  // read header file
  image = nifti_image_read(fname, false);
  ss << "Error in reading file " + std::string(fname);
  ierr = myAssert(image != NULL, ss.str()); CHKERRQ(ierr);
  ss.clear();
  ss.str(std::string());

  nx[2] = static_cast<int>(image->nx);
  nx[1] = static_cast<int>(image->ny);
  nx[0] = static_cast<int>(image->nz);

  ierr = myAssert(nx[0] == params->grid_->n_[0], "Error: dimension mismatch"); CHKERRQ(ierr);
  ierr = myAssert(nx[1] == params->grid_->n_[1], "Error: dimension mismatch"); CHKERRQ(ierr);
  ierr = myAssert(nx[2] == params->grid_->n_[2], "Error: dimension mismatch"); CHKERRQ(ierr);

  // get local size
  nl = params->grid_->nl_;
  ng = params->grid_->ng_;

  ScalarType *data_global = nullptr;
  if (rank == 0) {  // call readNifti only from master rank
    data_global = new ScalarType[ng];
    switch (image->datatype) {
      case NIFTI_TYPE_FLOAT32: {
        ierr = readNifti<float>(image, data_global, params); CHKERRQ(ierr);
        break;
      }
      case NIFTI_TYPE_FLOAT64: {
        ierr = readNifti<double>(image, data_global, params); CHKERRQ(ierr);
        break;
      }
      default: {
        ierr = myAssert(false, "Nifti datatype not supported"); CHKERRQ(ierr);
      }
    }
  }

  MPI_Scatterv(data_global, params->grid_->isize_send_, params->grid_->isize_offset_, MPIType, p_x, nl, MPIType, 0, PETSC_COMM_WORLD);

  // copy the image header, so we can write out files the same way
  params->tu_->nifti_ref_image_ = nifti_copy_nim_info(image);

  if (image != NULL) {
    nifti_image_free(image);
    image = NULL;
  }
  if (data_global != nullptr) delete[] data_global;

  PetscFunctionReturn(0);
}

PetscErrorCode writeNifti(nifti_image **image, ScalarType *p_x, std::shared_ptr<Parameters> params, const char *fname) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int nprocs, rank;
  int ng = params->grid_->ng_;
  int nl = params->grid_->nl_;
  MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  std::stringstream ss;
  ss << params->tu_->writepath_ << fname;
  std::string fnm = ss.str();
  ScalarType *data_global = nullptr;
  if (rank == 0) {
    size_t index = 0;
    index = fnm.find(".nc", index);
    fnm.replace(index, 7, ".nii.gz");
    (*image)->nifti_type = NIFTI_FTYPE_NIFTI1_1;
    char *bnametemp = nifti_makebasename(fnm.c_str());
    const std::string bname(bnametemp);
    free(bnametemp);
    (*image)->fname = nifti_makehdrname(bname.c_str(), (*image)->nifti_type, false, true);
    (*image)->iname = nifti_makeimgname(bname.c_str(), (*image)->nifti_type, false, true);

    data_global = new ScalarType[ng];
  }

  // gather all the data onto rank 0
  MPI_Gatherv(p_x, nl, MPIType, data_global, params->grid_->isize_send_, params->grid_->isize_offset_, MPIType, 0, PETSC_COMM_WORLD);

  ScalarType *data = nullptr;
  if (rank == 0) {
    data = static_cast<ScalarType *>((*image)->data);
    int64_t idx = 0;
    int64_t index, x, y, z;
    for (int p = 0; p < nprocs; p++) {
      for (int i = 0; i < params->grid_->isize_gathered_[3 * p + 0]; i++) {
        for (int j = 0; j < params->grid_->isize_gathered_[3 * p + 1]; j++) {
          for (int k = 0; k < params->grid_->isize_gathered_[3 * p + 2]; k++) {
            x = params->grid_->istart_gathered_[3 * p + 0] + i;
            y = params->grid_->istart_gathered_[3 * p + 1] + j;
            z = params->grid_->istart_gathered_[3 * p + 2] + k;
            index = x * params->grid_->n_[1] * params->grid_->n_[2] + y * params->grid_->n_[2] + z;
            data[index] = static_cast<ScalarType>(data_global[idx++]);
          }
        }
      }
    }  // for all the processes

    nifti_image_write(*image);
  }

  if (data_global != nullptr) delete[] data_global;

  if (*image != NULL) {
    *image = NULL;
    nifti_image_free(*image);
  }

  PetscFunctionReturn(ierr);
}

PetscErrorCode dataOut(ScalarType *p_x, std::shared_ptr<Parameters> params, const char *fname) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  nifti_image *image = NULL;
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  if (params->tu_->nifti_ref_image_ != NULL) {
    image = nifti_copy_nim_info(params->tu_->nifti_ref_image_);
    image->nbyper = sizeof(ScalarType);
    if (rank == 0) image->data = new ScalarType[image->nvox];
  } else {  // create the image
    image = nifti_simple_init_nim();
    image->dim[0] = image->ndim = 5;
    image->dim[1] = params->grid_->n_[2];
    image->dim[2] = params->grid_->n_[1];
    image->dim[3] = params->grid_->n_[0];
    image->pixdim[1] = static_cast<ScalarType>(params->grid_->h_[2]);
    image->pixdim[2] = static_cast<ScalarType>(params->grid_->h_[1]);
    image->pixdim[3] = static_cast<ScalarType>(params->grid_->h_[0]);
    image->dim[4] = image->nt = 1;  // num of time steps
    image->dim[5] = image->nu = 3;  // 3d vector
    image->pixdim[4] = 1;           // temporal size
    image->pixdim[5] = static_cast<ScalarType>(params->grid_->h_[2]);
    image->pixdim[6] = static_cast<ScalarType>(params->grid_->h_[1]);
    image->pixdim[7] = static_cast<ScalarType>(params->grid_->h_[0]);

    image->nvox = 1;
    // count total number of voxels
    for (int i = 1; i <= 4; i++) image->nvox *= image->dim[i];
    if (rank == 0) image->data = new ScalarType[image->nvox];
  }
#ifdef SINGLE
  image->datatype = NIFTI_TYPE_FLOAT32;
#else
  image->datatype = NIFTI_TYPE_FLOAT64
#endif

  ierr = writeNifti(&image, p_x, params, fname); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

#else
PetscErrorCode dataIn(ScalarType *p_x, std::shared_ptr<Parameters> params, const char *fname) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // get local sizes
  MPI_Offset istart[3], isize[3];
  istart[0] = static_cast<MPI_Offset>(params->grid_->istart_[0]);
  istart[1] = static_cast<MPI_Offset>(params->grid_->istart_[1]);
  istart[2] = static_cast<MPI_Offset>(params->grid_->istart_[2]);

  isize[0] = static_cast<MPI_Offset>(params->grid_->isize_[0]);
  isize[1] = static_cast<MPI_Offset>(params->grid_->isize_[1]);
  isize[2] = static_cast<MPI_Offset>(params->grid_->isize_[2]);
  int ncerr, fileid, ndims, nvars, ngatts, unlimited, varid[1];
  // open file
  ncerr = ncmpi_open(PETSC_COMM_WORLD, fname, NC_NOWRITE, MPI_INFO_NULL, &fileid);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);
  // query info about field named "data"
  ncerr = ncmpi_inq(fileid, &ndims, &nvars, &ngatts, &unlimited);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);
  ncerr = ncmpi_inq_varid(fileid, "data", &varid[0]);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);
  ncerr = ncmpi_get_vara_all(fileid, varid[0], istart, isize, p_x, params->grid_->nl_, MPIType);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);
  ncerr = ncmpi_close(fileid);
  PetscFunctionReturn(ierr);
}

PetscErrorCode dataOut(ScalarType *p_x, std::shared_ptr<Parameters> params, const char *fname) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int ncerr, mode, dims[3], varid[1], nx[3], iscdf5, fileid;
  int nl;
  MPI_Offset istart[3], isize[3];
  MPI_Comm c_comm;
  bool usecdf5 = false;  // CDF-5 is mandatory for large files (>= 2x10^9 cells)

  std::stringstream ss;
  ss << params->tu_->writepath_ << fname;
  ierr = myAssert(p_x != NULL, "null pointer"); CHKERRQ(ierr);

  // file creation mode
  mode = NC_CLOBBER;
  if (usecdf5) {
    mode = NC_CLOBBER | NC_64BIT_DATA;
  } else {
    mode = NC_CLOBBER | NC_64BIT_OFFSET;
  }

  c_comm = params->grid_->c_comm_;

  // create netcdf file
  ncerr = ncmpi_create(c_comm, ss.str().c_str(), mode, MPI_INFO_NULL, &fileid);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);

  nx[0] = params->grid_->n_[0];
  nx[1] = params->grid_->n_[1];
  nx[2] = params->grid_->n_[2];
  nl = params->grid_->nl_;

  // set size
  ncerr = ncmpi_def_dim(fileid, "x", nx[0], &dims[0]);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);
  ncerr = ncmpi_def_dim(fileid, "y", nx[1], &dims[1]);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);
  ncerr = ncmpi_def_dim(fileid, "z", nx[2], &dims[2]);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);

  // define name for output field
#ifdef SINGLE
  ncerr = ncmpi_def_var(fileid, "data", NC_FLOAT, 3, dims, &varid[0]);
#else
  ncerr = ncmpi_def_var(fileid, "data", NC_DOUBLE, 3, dims, &varid[0]);
#endif
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);

  iscdf5 = usecdf5 ? 1 : 0;
  ncerr = ncmpi_put_att_int(fileid, NC_GLOBAL, "CDF-5 mode", NC_INT, 1, &iscdf5);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);
  ncerr = ncmpi_enddef(fileid);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);

  // get local sizes
  istart[0] = static_cast<MPI_Offset>(params->grid_->istart_[0]);
  istart[1] = static_cast<MPI_Offset>(params->grid_->istart_[1]);
  istart[2] = static_cast<MPI_Offset>(params->grid_->istart_[2]);

  isize[0] = static_cast<MPI_Offset>(params->grid_->isize_[0]);
  isize[1] = static_cast<MPI_Offset>(params->grid_->isize_[1]);
  isize[2] = static_cast<MPI_Offset>(params->grid_->isize_[2]);

  ierr = myAssert(nl == isize[0] * isize[1] * isize[2], "size error"); CHKERRQ(ierr);

  // write data to file
  ncerr = ncmpi_put_vara_all(fileid, varid[0], istart, isize, p_x, nl, MPIType);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);

  // close file
  ncerr = ncmpi_close(fileid);
  ierr = NCERRQ(ncerr); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}
#endif

PetscErrorCode dataIn(Vec A, std::shared_ptr<Parameters> params, std::string fname) {
  ScalarType *a_ptr;
  PetscErrorCode ierr;
  ierr = VecGetArray(A, &a_ptr); CHKERRQ(ierr);
  dataIn(a_ptr, params, fname.c_str());
  ierr = VecRestoreArray(A, &a_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode dataIn(Vec A, std::shared_ptr<Parameters> params, const char *fname) {
  ScalarType *a_ptr;
  PetscErrorCode ierr;
  ierr = VecGetArray(A, &a_ptr); CHKERRQ(ierr);
  dataIn(a_ptr, params, fname);
  ierr = VecRestoreArray(A, &a_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode dataOut(Vec A, std::shared_ptr<Parameters> params, const char *fname) {
  ScalarType *a_ptr;
  PetscErrorCode ierr;
  ierr = VecGetArray(A, &a_ptr); CHKERRQ(ierr);
  dataOut(a_ptr, params, fname);
  ierr = VecRestoreArray(A, &a_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode dataOut(Vec A, std::shared_ptr<Parameters> params, std::string fname) {
  ScalarType *a_ptr;
  PetscErrorCode ierr;
  ierr = VecGetArray(A, &a_ptr); CHKERRQ(ierr);
  dataOut(a_ptr, params, fname.c_str());
  ierr = VecRestoreArray(A, &a_ptr); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}


PetscErrorCode readVecField(VecField *v, std::string fnx1, std::string fnx2, std::string fnx3, std::shared_ptr<Parameters> params ) {
    PetscErrorCode ierr = 0;
    PetscFunctionBegin;

    ierr = dataIn(v->x_, params, fnx1.c_str()); CHKERRQ(ierr);
    ierr = dataIn(v->y_, params, fnx2.c_str()); CHKERRQ(ierr);
    ierr = dataIn(v->z_, params, fnx3.c_str()); CHKERRQ(ierr);

    PetscFunctionReturn(ierr);
}


// Miscellaneous IO for parameters

// ### _____________________________________________________________________ ___
// ### ///////////////// writeCheckpoint /////////////////////////////////// ###
PetscErrorCode writeCheckpoint(Vec p, std::shared_ptr<Phi> phi, std::string path, std::string suffix) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  // write p vector to bin
  std::string fname_p = path + "p-rec-" + suffix + ".bin";
  std::string fname_p_txt = path + "p-rec-" + suffix + ".txt";
  writeBIN(p, fname_p);

  // write p vector to txt
  ScalarType *p_ptr;
  ierr = VecGetArray(p, &p_ptr);
  std::stringstream pvis;
  pvis << " p = [" << std::endl;
  for (int ptr = 0; ptr < phi->np_; ++ptr) {
    pvis << " " << p_ptr[ptr] << std::endl;
  }
  ierr = VecRestoreArray(p, &p_ptr);
  pvis << "];" << std::endl;

  // write Gaussian centers
  std::string fname_phi = path + "phi-mesh-" + suffix + ".txt";
  std::stringstream phivis;
  phivis << " sigma = " << phi->sigma_ << ", spacing = " << phi->spacing_factor_ * phi->sigma_ << std::endl;
  phivis << " centers = [" << std::endl;
  for (int ptr = 0; ptr < 3 * phi->np_; ptr += 3) {
    phivis << " " << phi->centers_[ptr + 0] << ", " << phi->centers_[ptr + 1] << ", " << phi->centers_[ptr + 2] << std::endl;
  }
  phivis << "];" << std::endl;
  std::fstream phifile;
  std::fstream pfile;
  if (procid == 0) {
    phifile.open(fname_phi, std::ios_base::out);
    phifile << phivis.str() << std::endl;
    phifile.close();
    pfile.open(fname_p_txt, std::ios_base::out);
    pfile << pvis.str() << std::endl;
    pfile.close();
  }
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// readPhiMesh /////////////////////////////////////// ###
PetscErrorCode readPhiMesh(std::vector<ScalarType> &centers, std::shared_ptr<Parameters> params, std::string f, bool read_comp_data, std::vector<int> *comps, bool overwrite_sigma) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  std::ifstream file(f);
  std::stringstream ss;
  int np = 0, k = 0;
  ScalarType sigma = 0, spacing = 0;
  if (file.is_open()) {
    ss << " reading Gaussian centers from file " << f;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    centers.clear();
    if (read_comp_data && comps != nullptr) {
      (*comps).clear();
    }
    std::string line;
    std::getline(file, line);  // sigma, spacing
    std::string token;
    size_t pos1 = line.find("=");
    size_t pos2 = line.find(",");
    if (pos1 != std::string::npos && pos2 != std::string::npos) {
      sigma = atof(line.substr(pos1 + 1, pos2).c_str());
    }
    line.erase(0, pos2 + 1);
    pos1 = line.find("=");
    if (pos1 != std::string::npos) {
      spacing = atof(line.substr(pos1 + 1, line.length()).c_str());
    }
    ss << " reading sigma=" << sigma << ", spacing=" << spacing;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    if (params->tu_->phi_sigma_data_driven_ != sigma) {
      ss << " Warning: specified sigma=" << params->tu_->phi_sigma_data_driven_ << " != sigma=" << sigma << " (read from file).";
      if (overwrite_sigma) {
        ss << " Specified sigma overwritten.";
        params->tu_->phi_sigma_data_driven_ = sigma;
      }
      ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
      ss.str("");
      ss.clear();
    }
    std::getline(file, line);  // throw away;
    std::string t;
    int ii = 0;
    while (std::getline(file, line)) {
      if (line.rfind("]") != std::string::npos) break;  // end of file reached, exit out
      std::stringstream l(line);
      ii = 0;
      while (std::getline(l, t, ',')) {
        if (ii < 3) {
          centers.push_back(atof(t.c_str()));
        } else if (read_comp_data && comps != nullptr) {
          (*comps).push_back(atof(t.c_str()));
        }
        ii++;
      }
      np++;
    }
    file.close();
    params->tu_->np_ = np;
    ss << " np=" << np << " centers read ";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  } else {
    ss << " cannot open file " << f;
    ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    PetscFunctionReturn(1);
  }
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// readPVec ////////////////////////////////////////// ###
PetscErrorCode readPVec(Vec *x, int size, int np, std::string f) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  ScalarType *x_ptr;
  std::string file, msg, path, ext, line;
  std::stringstream ss;

  TU_assert(!f.empty(), "filename not set");
  // get file name without path
  ierr = getFileName(path, file, ext, f); CHKERRQ(ierr);
  msg = "file " + file + " does not exist";
  TU_assert(fileExists(f), msg.c_str());

  if (strcmp(ext.c_str(), ".bin") == 0) {
    ierr = readBIN(&(*x), size, f); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }

  // if not nullptr, clear memory
  if (*x != nullptr) { ierr = VecDestroy(x); CHKERRQ(ierr); *x = nullptr;}

  ierr = VecCreateSeq(PETSC_COMM_SELF, size, &(*x)); CHKERRQ(ierr);
  ierr = setupVec(*x, SEQ); CHKERRQ(ierr);
  ierr = VecSet(*x, 0.); CHKERRQ(ierr);

  // read pvec from file
  std::ifstream pfile(f);
  int pi = 0;
  if (pfile.is_open()) {
    ss << " reading p_i values from file " << f;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    std::getline(pfile, line);  // throw away (p = [);
    ierr = VecGetArray(*x, &x_ptr); CHKERRQ(ierr);
    while (std::getline(pfile, line)) {
      if (line.rfind("]") != std::string::npos) break;  // end of file reached, exit out
      TU_assert(pi < np, "index out of bounds reading p_vec from file.");
      x_ptr[pi] = atof(line.c_str());
      pi++;
    }
    ss << " ... success: " << pi << " values read, size of vector: " << size;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    TU_assert(pi == np, "number of read p_i values does not match with number of read Gaussian centers.");
    ierr = VecRestoreArray(*x, &x_ptr); CHKERRQ(ierr);
    pfile.close();
  } else {
    ss << " cannot open file " << f;
    ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    PetscFunctionReturn(1);
  }
  if (procid == 0) {
    ierr = VecView(*x, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// readBIN /////////////////////////////////////////// ###
PetscErrorCode readBIN(Vec *x, int size, std::string f) {
  PetscFunctionBegin;
  int nprocs, procid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  PetscErrorCode ierr = 0;
  PetscViewer viewer = nullptr;
  std::string file, msg;
  std::stringstream ss;

  ss << "reading p_i values from binary file " << f;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  TU_assert(!f.empty(), "filename not set");
  // get file name without path
  ierr = getFileName(file, f); CHKERRQ(ierr);
  msg = "file " + file + " does not exist";
  TU_assert(fileExists(f), msg.c_str());
  if (*x != nullptr) {
    ierr = VecDestroy(x); CHKERRQ(ierr);
    *x = nullptr;
  }
  ierr = VecCreateSeq(PETSC_COMM_SELF, size, &(*x)); CHKERRQ(ierr);
  ierr = setupVec(*x, SEQ); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF, f.c_str(), FILE_MODE_READ, &viewer); CHKERRQ(ierr);

  TU_assert(viewer != NULL, "could not read binary file");
  ierr = PetscViewerBinarySetFlowControl(viewer, 2); CHKERRQ(ierr);
  ierr = VecLoad(*x, viewer); CHKERRQ(ierr);

  if (procid == 0) {
    ierr = VecView(*x, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  }
  // clean up
  if (viewer != nullptr) {
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    viewer = nullptr;
  }
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// writeBIN ////////////////////////////////////////// ###
PetscErrorCode writeBIN(Vec x, std::string f) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  PetscViewer viewer = nullptr;
  PetscFunctionBegin;
  TU_assert(x != nullptr, "null pointer");
  TU_assert(!f.empty(), "filename not set");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF, f.c_str(), FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);
  TU_assert(viewer != nullptr, "could not write binary file");
  ierr = VecView(x, viewer); CHKERRQ(ierr);
  // clean up
  if (viewer != nullptr) {
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    viewer = nullptr;
  }
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// readConCompDat /////////////////////////////////////// ###
PetscErrorCode readConCompDat(std::vector<ScalarType> &weights, std::vector<ScalarType> &centers, std::string f) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  std::ifstream file(f);
  std::stringstream ss;
  int ncomp = 0;
  if (file.is_open()) {
    ss << " reading concomp.dat file " << f;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    weights.clear();
    centers.clear();

    std::string line;
    std::getline(file, line);  // #components
    std::getline(file, line);  // #components (value)
    ncomp = atoi(line.c_str());
    std::getline(file, line);  // center of mass:
    for (int i = 0; i < ncomp; ++i) {
      std::getline(file, line);  // values
      std::stringstream l(line);
      std::string t;
      while (std::getline(l, t, ',')) {
        centers.push_back(atof(t.c_str()));
      }
    }
    std::getline(file, line);  // relative mass:
    for (int i = 0; i < ncomp; ++i) {
      std::getline(file, line);  // values
      weights.push_back(atof(line.c_str()));
    }

    file.close();
    ss << " ncomp=" << ncomp << " component read ";
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    ss << " weights: ";
    for (int i = 0; i < ncomp; ++i) {
      ss << weights[i];
      if (i < ncomp) {
        ss << ", ";
      }
    }
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    ss << " centers: ";
    for (int i = 0; i < ncomp; ++i) {
      ss << "(";
      for (int j = 0; j < 3; ++j) {
        ss << centers[3 * i + j];
        if (j < 2) {
          ss << ",";
        }
      }
      ss << "); ";
    }
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  } else {
    ss << " cannot open file " << f;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
    PetscFunctionReturn(1);
  }
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// getFileName /////////////////////////////////////// ###
PetscErrorCode getFileName(std::string &filename, std::string file) {
  PetscErrorCode ierr = 0;
  std::string path;
  size_t sep;
  PetscFunctionBegin;

  sep = file.find_last_of("\\/");
  if (sep != std::string::npos) {
    path = file.substr(0, sep);
    filename = file.substr(sep + 1);
  }
  if (filename == "") {
    filename = file;
  }
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// getFileName /////////////////////////////////////// ###
PetscErrorCode getFileName(std::string &path, std::string &filename, std::string &extension, std::string file) {
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  std::string::size_type idx;
  std::stringstream ss;
  PetscFunctionBegin;

  // get path
  idx = file.find_last_of("\\/");
  if (idx != std::string::npos) {
    path = file.substr(0, idx);
    filename = file.substr(idx + 1);
  }
  if (filename == "") {
    filename = file;
  }

  // get extension
  idx = filename.rfind(".");
  if (idx != std::string::npos) {
    extension = filename.substr(idx + 1);

    // handle zipped files
    if (strcmp(extension.c_str(), "gz") == 0) {
      filename = filename.substr(0, idx);
      idx = filename.rfind(".");
      if (idx != std::string::npos) {
        extension = filename.substr(idx + 1);
        extension = extension + ".gz";
      }
    }
    extension = "." + extension;
    filename = filename.substr(0, idx);

  } else {
    ss << "ERROR: no extension found";
    ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  }

  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// fileExists //////////////////////////////////////// ###
bool fileExists(const std::string &filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0);
}
