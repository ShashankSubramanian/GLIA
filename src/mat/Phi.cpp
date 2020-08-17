#include "Phi.h"

Phi::Phi(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops) : params_(params), spec_ops_(spec_ops), gaussian_labels_(), component_weights_(), component_centers_() {
  PetscFunctionBegin;
  PetscErrorCode ierr;

  n_local_ = params->grid_->nl_;
  if (!params_->tu_->phi_store_)
    compute_ = true;
  else
    compute_ = false;

  np_ = params->tu_->np_;
  phi_vec_.resize(np_);
  ierr = VecCreate(PETSC_COMM_WORLD, &phi_vec_[0]);
  ierr = VecSetSizes(phi_vec_[0], params->grid_->nl_, params->grid_->ng_);
  ierr = setupVec(phi_vec_[0]);
  ierr = VecSet(phi_vec_[0], 0);
  for (int i = 1; i < np_; i++) {
    ierr = VecDuplicate(phi_vec_[0], &phi_vec_[i]);
    ierr = VecSet(phi_vec_[i], 0);
  }

  labels_ = nullptr;
  // by default one component
  component_weights_.push_back(1.);

#ifdef CUDA
  initPhiCudaConstants(params->grid_->n_, params->grid_->istart_, params->grid_->isize_);
#endif
}

PetscErrorCode Phi::setGaussians(std::array<ScalarType, 3> &user_cm, ScalarType sigma, ScalarType spacing_factor, int np) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  sigma_ = sigma;
  params_->tu_->phi_sigma_ = sigma_;
  spacing_factor_ = spacing_factor;
  params_->tu_->phi_spacing_factor_ = spacing_factor_;
  np_ = np;
  params_->tu_->np_ = np_;

  ss << " Gaussians set as grid with np: " << np_ << " and sigma: " << sigma_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  centers_.clear();
  // Destroy and clear any previously set phis
  for (int i = 0; i < phi_vec_.size(); i++) {
    ierr = VecDestroy(&phi_vec_[i]); CHKERRQ(ierr);
  }
  phi_vec_.clear();
  phi_vec_.resize(np_);
  ierr = VecCreate(PETSC_COMM_WORLD, &phi_vec_[0]);
  ierr = VecSetSizes(phi_vec_[0], params_->grid_->nl_, params_->grid_->ng_);
  ierr = setupVec(phi_vec_[0]);
  ierr = VecSet(phi_vec_[0], 0);
  for (int i = 1; i < np_; i++) {
    ierr = VecDuplicate(phi_vec_[0], &phi_vec_[i]);
    ierr = VecSet(phi_vec_[i], 0);
  }
  memcpy(cm_, user_cm.data(), 3 * sizeof(ScalarType));
  centers_.resize(3 * np_);
  ierr = phiMesh(&centers_[0]);

  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setValues ///////////////////////////////////////// ###
PetscErrorCode Phi::setValues(std::shared_ptr<MatProp> mat_prop) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  Event e("tumor-phi-setvals");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  mat_prop_ = mat_prop;  // this is needed for phi::apply when computation of phis are on the fly
                         // Use whatever matprop the setvalues uses in default

  // set phis only if compute is disabled: the subspace is small and all the phis
  // are filtered and stored in memory
  if (!compute_) {
    ScalarType *phi_ptr;
    ScalarType sigma_smooth = params_->tu_->smoothing_factor_ * 2.0 * M_PI / params_->grid_->n_[0];
    Vec all_phis;
    ierr = VecDuplicate(phi_vec_[0], &all_phis); CHKERRQ(ierr);
    ierr = VecSet(all_phis, 0); CHKERRQ(ierr);

    ScalarType phi_max = 0, max = 0;

    for (int i = 0; i < np_; i++) {
      // set values of Gaussian function
      ierr = vecGetArray(phi_vec_[i], &phi_ptr); CHKERRQ(ierr);
#ifdef CUDA
      initializeGaussianCuda(phi_ptr, sigma_, centers_[3 * i], centers_[3 * i + 1], centers_[3 * i + 2], params_->grid_->isize_);
#else
      initialize(phi_ptr, params_, &centers_[3 * i]);
#endif
      ierr = vecRestoreArray(phi_vec_[i], &phi_ptr); CHKERRQ(ierr);
      // filter with brain geometry
      ierr = VecPointwiseMult(phi_vec_[i], mat_prop->filter_, phi_vec_[i]); CHKERRQ(ierr);
      // smooth to avoid sharp edges (FFT)
      ierr = spec_ops_->weierstrassSmoother(phi_vec_[i], phi_vec_[i], params_, sigma_smooth); CHKERRQ(ierr);

      // truncate Gaussians after radius of 5*sigma for compact support
      ierr = vecGetArray(phi_vec_[i], &phi_ptr); CHKERRQ(ierr);
#ifdef CUDA
      truncateGaussianCuda(phi_ptr, sigma_, centers_[3 * i], centers_[3 * i + 1], centers_[3 * i + 2], params_->grid_->isize_);
#else
      truncate(phi_ptr, params_, &centers_[3 * i]);
#endif
      ierr = vecRestoreArray(phi_vec_[i], &phi_ptr); CHKERRQ(ierr);
      // find the max
      ierr = vecMax(phi_vec_[i], NULL, &max); CHKERRQ(ierr);
      if (max > phi_max) {
        phi_max = max;
      }
    }
    // Rescale phi so that max is one: this enforces p to be one (needed for reaction inversion)
    for (int i = 0; i < np_; i++) {
      ierr = VecScale(phi_vec_[i], (1.0 / phi_max)); CHKERRQ(ierr);
      ierr = VecAXPY(all_phis, 1.0, phi_vec_[i]); CHKERRQ(ierr);
    }

    if (params_->tu_->write_output_) {
      dataOut(all_phis, params_, "phiGrid.nc");
    }
    ierr = VecDestroy(&all_phis); CHKERRQ(ierr);
  }

  self_exec_time += MPI_Wtime();
  // accumulateTimers (t, t, self_exec_time);
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// phiMesh /////////////////////////////////////////// ###
PetscErrorCode Phi::phiMesh(ScalarType *center) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int nprocs, procid;
  MPI_Comm_rank(PETSC_COMM_WORLD, &procid);
  MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
  int h = round(std::pow(np_, 1.0 / 3.0));
  ScalarType space[3];
  ScalarType mu[3];

#ifdef VISUALIZE_PHI
  std::stringstream phivis;
  phivis << " sigma = " << sigma_ << ", spacing = " << spacing_factor_ * sigma_ << std::endl;
  phivis << " centers = [" << std::endl;
#endif

  space[0] = spacing_factor_ * sigma_;
  space[1] = spacing_factor_ * sigma_;
  space[2] = spacing_factor_ * sigma_;

  if (np_ == 1) {
    center[0] = cm_[0];
    center[1] = cm_[1];
    center[2] = cm_[2];
#ifdef VISUALIZE_PHI
    phivis << "];" << std::endl;
    std::fstream phifile;
    static int ct = 0;
    if (procid == 0) {
      std::stringstream ssct;
      ssct << ct;
      phifile.open(std::string(params_->tu_->writepath_.str() + "phi-mesh-" + ssct.str() + ".dat"), std::ios_base::out);
      phifile << phivis.str() << std::endl;
      phifile.close();
      ct++;
    }
#endif
    PetscFunctionReturn(ierr);
  }
  if (np_ % 2 == 1) {
    int ptr = 0;
    for (int k = -(h - 1) / 2; k <= (h - 1) / 2; k++)
      for (int j = -(h - 1) / 2; j <= (h - 1) / 2; j++)
        for (int i = -(h - 1) / 2; i <= (h - 1) / 2; i++) {
          center[ptr + 0] = i * space[0] + cm_[0];
          center[ptr + 1] = j * space[1] + cm_[1];
          center[ptr + 2] = k * space[2] + cm_[2];
#ifdef VISUALIZE_PHI
          phivis << " " << center[ptr + 0] << ", " << center[ptr + 1] << ", " << center[ptr + 2] << std::endl;
#endif
          ptr += 3;
        }
  } else if (np_ % 2 == 0) {
    int ptr = 0;
    for (int k = -(h) / 2; k <= (h) / 2; k++)
      for (int j = -(h) / 2; j <= (h) / 2; j++)
        for (int i = -(h) / 2; i <= (h) / 2; i++) {
          mu[0] = ((i < 0) ? -1 : ((i > 0) ? 1 : 0)) * 0.5;
          mu[1] = ((j < 0) ? -1 : ((j > 0) ? 1 : 0)) * 0.5;
          mu[2] = ((k < 0) ? -1 : ((k > 0) ? 1 : 0)) * 0.5;
          if ((i != 0) && (j != 0) && (k != 0)) {
            center[ptr + 0] = (i - mu[0]) * space[0] + cm_[0];
            center[ptr + 1] = (j - mu[1]) * space[1] + cm_[1];
            center[ptr + 2] = (k - mu[2]) * space[2] + cm_[2];
#ifdef VISUALIZE_PHI
            phivis << " " << center[ptr + 0] << ", " << center[ptr + 1] << ", " << center[ptr + 2] << std::endl;
#endif
            ptr += 3;
          }
        }
  }
#ifdef VISUALIZE_PHI
  phivis << "];" << std::endl;
  std::fstream phifile;
  static int ct = 0;
  if (procid == 0) {
    std::stringstream ssct;
    ssct << ct;
    phifile.open(std::string(params_->tu_->writepath_.str() + "phi-mesh-" + ssct.str() + ".dat"), std::ios_base::out);
    phifile << phivis.str() << std::endl;
    phifile.close();
    ct++;
  }
#endif

  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// truncate //////////////////////////////////////// ###
PetscErrorCode Phi::truncate(ScalarType *out, std::shared_ptr<Parameters> params, ScalarType *center) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ScalarType twopi = 2.0 * M_PI;
  int64_t X, Y, Z;
  ScalarType r;
  int64_t ptr;
  ScalarType xc = center[0], yc = center[1], zc = center[2];
  ScalarType hx = twopi / params->grid_->n_[0], hy = twopi / params->grid_->n_[1], hz = twopi / params->grid_->n_[2];

  for (int x = 0; x < params->grid_->isize_[0]; x++)
    for (int y = 0; y < params->grid_->isize_[1]; y++)
      for (int z = 0; z < params->grid_->isize_[2]; z++) {
        X = params->grid_->istart_[0] + x;
        Y = params->grid_->istart_[1] + y;
        Z = params->grid_->istart_[2] + z;
        r = sqrt((hx * X - xc) * (hx * X - xc) + (hy * Y - yc) * (hy * Y - yc) + (hz * Z - zc) * (hz * Z - zc));
        ptr = x * params->grid_->isize_[1] * params->grid_->isize_[2] + y * params->grid_->isize_[2] + z;
        // truncate to zero after radius 5*sigma
        out[ptr] = (r / sigma_ <= 5) ? out[ptr] : 0.0;
      }
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// initialize //////////////////////////////////////// ###
PetscErrorCode Phi::initialize(ScalarType *out, std::shared_ptr<Parameters> params, ScalarType *center) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  ScalarType twopi = 2.0 * M_PI;
  ScalarType R, c;
  int64_t X, Y, Z;
  ScalarType r, ratio;
  int64_t ptr;

  ScalarType xc = center[0], yc = center[1], zc = center[2];
  ScalarType hx = twopi / params->grid_->n_[0], hy = twopi / params->grid_->n_[1], hz = twopi / params->grid_->n_[2];
  ScalarType dx = 0, dy = 0, dz = 0, o = 0;

  // PHI = GAUSSIAN
  R = std::sqrt(2.) * sigma_;
  c = 1.;  // 1./(sigma_ * std::sqrt(2*M_PI));

  // PHI = BUMP
  // R = sigma_ * sigma_;
  // c = std::exp(1);

  for (int x = 0; x < params->grid_->isize_[0]; x++)
    for (int y = 0; y < params->grid_->isize_[1]; y++)
      for (int z = 0; z < params->grid_->isize_[2]; z++) {
        X = params->grid_->istart_[0] + x;
        Y = params->grid_->istart_[1] + y;
        Z = params->grid_->istart_[2] + z;
        ptr = x * params->grid_->isize_[1] * params->grid_->isize_[2] + y * params->grid_->isize_[2] + z;

        dx = hx * X - xc;
        dy = hy * Y - yc;
        dz = hz * Z - zc;

        // PHI = GAUSSIAN
        r = sqrt(dx * dx + dy * dy + dz * dz);
        ratio = r / R;
        o = c * std::exp(-ratio * ratio);

        // PHI = BUMP
        // r = sqrt(dx*dx/R + dy*dy/R + dz*dz/R);
        // o = (r < 1.)        ? c * std::exp(-1./(1.-r*r))   : 0.0;

        out[ptr] = o;
      }

  PetscFunctionReturn(ierr);
}

PetscErrorCode Phi::apply(Vec out, Vec p) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-phi-apply");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  const ScalarType *pg_ptr;
  ierr = VecSet(out, 0.); CHKERRQ(ierr);
  ierr = VecGetArrayRead(p, &pg_ptr); CHKERRQ(ierr);

  if (!compute_) {
    for (int i = 0; i < np_; i++) {
      ierr = VecAXPY(out, pg_ptr[i], phi_vec_[i]); CHKERRQ(ierr);
    }
  } else {
    // compute phi and apply on the fly
    // use phi_vec_[0] as proxy for every phi
    ScalarType phi_max = 0, max = 0;
    ScalarType *phi_ptr;
    ScalarType sigma_smooth = params_->tu_->smoothing_factor_ * 2.0 * M_PI / params_->grid_->n_[0];
    bool p_has_nnz = false;
    for (int i = 0; i < np_; i++) {
      if (pg_ptr[i] != 0) {  // if p is zero it adds nothing to \sum_i \phi_i p_i
        p_has_nnz = true;
        ierr = vecGetArray(phi_vec_[0], &phi_ptr); CHKERRQ(ierr);
#ifdef CUDA
        initializeGaussianCuda(phi_ptr, sigma_, centers_[3 * i], centers_[3 * i + 1], centers_[3 * i + 2], params_->grid_->isize_);
#else
        initialize(phi_ptr, params_, &centers_[3 * i]);
#endif
        ierr = vecRestoreArray(phi_vec_[0], &phi_ptr); CHKERRQ(ierr);
        ierr = VecPointwiseMult(phi_vec_[0], mat_prop_->filter_, phi_vec_[0]); CHKERRQ(ierr);
        ierr = spec_ops_->weierstrassSmoother(phi_vec_[0], phi_vec_[0], params_, sigma_smooth); CHKERRQ(ierr);

        // truncate Gaussians after radius of 5*sigma for compact support
        ierr = vecGetArray(phi_vec_[0], &phi_ptr); CHKERRQ(ierr);
#ifdef CUDA
        truncateGaussianCuda(phi_ptr, sigma_, centers_[3 * i], centers_[3 * i + 1], centers_[3 * i + 2], params_->grid_->isize_);
#else
        truncate(phi_ptr, params_, &centers_[3 * i]);
#endif
        ierr = vecRestoreArray(phi_vec_[0], &phi_ptr); CHKERRQ(ierr);
        // find the max
        ierr = vecMax(phi_vec_[0], NULL, &max); CHKERRQ(ierr);
        if (max > phi_max) {
          phi_max = max;
        }

        // accumulate phi*p
        ierr = VecAXPY(out, pg_ptr[i], phi_vec_[0]); CHKERRQ(ierr);
      }
    }
    // scale phi*p with 1/phi_max since all phis are scaled with this.
    if (!p_has_nnz) phi_max = 1;
    ierr = VecScale(out, (1.0 / phi_max)); CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(p, &pg_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  accumulateTimers(t, t, self_exec_time);
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

PetscErrorCode Phi::applyTranspose(Vec pout, Vec in) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e("tumor-phi-applyT");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  ScalarType values[np_];
  ScalarType *pout_ptr;
  ierr = VecGetArray(pout, &pout_ptr); CHKERRQ(ierr);

  if (!compute_) {
    Vec *v = &phi_vec_[0];
    ierr = VecMTDot(in, np_, v, values); CHKERRQ(ierr);
    for (int i = 0; i < np_; i++) {
      pout_ptr[i] = values[i];
    }
  } else {
    // compute the phis on the fly
    // use phi_vec_[0] as proxy for every phi
    ScalarType *phi_ptr;
    ScalarType phi_max = 0, max = 0;
    ScalarType sigma_smooth = params_->tu_->smoothing_factor_ * 2.0 * M_PI / params_->grid_->n_[0];
    for (int i = 0; i < np_; i++) {
      ierr = vecGetArray(phi_vec_[0], &phi_ptr); CHKERRQ(ierr);
#ifdef CUDA
      initializeGaussianCuda(phi_ptr, sigma_, centers_[3 * i], centers_[3 * i + 1], centers_[3 * i + 2], params_->grid_->isize_);
#else
      initialize(phi_ptr, params_, &centers_[3 * i]);
#endif
      ierr = vecRestoreArray(phi_vec_[0], &phi_ptr); CHKERRQ(ierr);
      ierr = VecPointwiseMult(phi_vec_[0], mat_prop_->filter_, phi_vec_[0]); CHKERRQ(ierr);

      ierr = spec_ops_->weierstrassSmoother(phi_vec_[0], phi_vec_[0], params_, sigma_smooth); CHKERRQ(ierr);
      // truncate Gaussians after radius of 5*sigma for compact support
      ierr = vecGetArray(phi_vec_[0], &phi_ptr); CHKERRQ(ierr);
#ifdef CUDA
      truncateGaussianCuda(phi_ptr, sigma_, centers_[3 * i], centers_[3 * i + 1], centers_[3 * i + 2], params_->grid_->isize_);
#else
      truncate(phi_ptr, params_, &centers_[3 * i]);
#endif
      ierr = vecRestoreArray(phi_vec_[0], &phi_ptr); CHKERRQ(ierr);
      // find the max
      ierr = vecMax(phi_vec_[0], NULL, &max); CHKERRQ(ierr);
      if (max > phi_max) {
        phi_max = max;
      }

      // compute phi^T*p
      ierr = VecDot(phi_vec_[0], in, &pout_ptr[i]); CHKERRQ(ierr);
    }
    ierr = VecScale(pout, (1.0 / phi_max)); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(pout, &pout_ptr); CHKERRQ(ierr);

  self_exec_time += MPI_Wtime();
  accumulateTimers(t, t, self_exec_time);
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

int isInLocalProc(int64_t X, int64_t Y, int64_t Z, std::shared_ptr<Parameters> params) {  // Check if global index (X, Y, Z) is inside the local proc
  int check = 0;
  int end_x, end_y, end_z;
  end_x = params->grid_->istart_[0] + params->grid_->isize_[0] - 1;
  end_y = params->grid_->istart_[1] + params->grid_->isize_[1] - 1;
  end_z = params->grid_->istart_[2] + params->grid_->isize_[2] - 1;
  if (X < params->grid_->istart_[0] || Y < params->grid_->istart_[1] || Z < params->grid_->istart_[2])
    check = 0;
  else if (X > end_x || Y > end_y || Z > end_z)
    check = 0;
  else
    check = 1;
  return check;
}

// x, y, z are local coordinates
void checkTumorExistence(int64_t x, int64_t y, int64_t z, ScalarType radius, ScalarType *data, std::shared_ptr<Parameters> params, std::vector<int> &local_tumor_marker) {
  int flag, num_tumor;
  num_tumor = 0;
  ScalarType distance;
  ScalarType threshold = params->tu_->data_threshold_;

  int64_t ptr;
  for (int i = x - radius; i <= x + radius; i++)
    for (int j = y - radius; j <= y + radius; j++)
      for (int k = z - radius; k <= z + radius; k++) {
        if (k < 0) continue;
        if (k >= params->grid_->isize_[2]) continue;  // Dont bother in the z direction as there is no partition here
        assert(isInLocalProc(x + params->grid_->istart_[0], y + params->grid_->istart_[1], z + params->grid_->istart_[2], params));

        distance = sqrt((i - x) * (i - x) + (j - y) * (j - y) + (k - z) * (k - z));
        if (distance <= radius) {
          ptr = i * params->grid_->isize_[1] * params->grid_->isize_[2] + j * params->grid_->isize_[2] + k;
          if (data[ptr] > threshold) {
            num_tumor++;
          }
        }
      }
  local_tumor_marker[x * params->grid_->isize_[1] * params->grid_->isize_[2] + y * params->grid_->isize_[2] + z] = num_tumor;
}

// Check tumor presence for boundary points and their neighbours in other procs: x,y,z are global indices

void checkTumorExistenceOutOfProc(int64_t x, int64_t y, int64_t z, ScalarType radius, ScalarType *data, std::shared_ptr<Parameters> params, std::vector<int64_t> &center_comm,
                                  std::vector<int> &local_tumor_marker, int local_check) {
  int flag, num_tumor;
  num_tumor = 0;
  ScalarType distance;
  ScalarType threshold = params->tu_->data_threshold_;

  int check_local_pos = 0;

  int64_t ptr;
  for (int i = x - radius; i <= x + radius; i++)
    for (int j = y - radius; j <= y + radius; j++)
      for (int k = z - radius; k <= z + radius; k++) {
        check_local_pos = isInLocalProc(i, j, k, params);
        if (k < 0) check_local_pos = 0;
        if (k >= params->grid_->isize_[2]) check_local_pos = 0;  // Dont bother in the z direction as there is no partition here
        if (check_local_pos) {
          distance = sqrt((i - x) * (i - x) + (j - y) * (j - y) + (k - z) * (k - z));
          if (distance <= radius) {
            ptr = (i - params->grid_->istart_[0]) * params->grid_->isize_[1] * params->grid_->isize_[2] + (j - params->grid_->istart_[1]) * params->grid_->isize_[2] +
                  (k - params->grid_->istart_[2]);  // Local index for data
            if (data[ptr] > threshold) {
              num_tumor++;
            }
          }
        }
      }

  if (local_check) {  // The center is in the local process
    // Get local index of center
    ptr = (x - params->grid_->istart_[0]) * params->grid_->isize_[1] * params->grid_->isize_[2] + (y - params->grid_->istart_[1]) * params->grid_->isize_[2] + (z - params->grid_->istart_[2]);
    local_tumor_marker[ptr] = num_tumor;
  } else {
    int64_t index_g = x * params->grid_->n_[1] * params->grid_->n_[2] + y * params->grid_->n_[2] + z;
    center_comm.push_back(index_g);
    center_comm.push_back(num_tumor);
  }
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setGaussians ////////////////////////////////////// ###
PetscErrorCode Phi::setGaussians(std::string file, bool read_comp_data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  ss << " Gaussian basis functions overwritten (from file) -- bounding box not set.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  ss << " file: " << file;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();

  // read centers from file (clears centers_)
  if (!read_comp_data) {
    ierr = readPhiMesh(centers_, params_, file); CHKERRQ(ierr);
  } else {
    ierr = readPhiMesh(centers_, params_, file, true, &gaussian_labels_); CHKERRQ(ierr);
  }

  ScalarType twopi = 2.0 * M_PI;
  ScalarType hx = twopi / params_->grid_->n_[0], hy = twopi / params_->grid_->n_[1], hz = twopi / params_->grid_->n_[2];
  sigma_ = params_->tu_->phi_sigma_data_driven_;  // This spacing corresponds to 1mm sigma -- tumor width of say 4*sigma
  spacing_factor_ = 2.0;
  params_->tu_->phi_spacing_factor_ = spacing_factor_;
  ScalarType space = spacing_factor_ * sigma_ / hx;

  // Get gaussian volume
  ScalarType dist = 0.0;
  int gaussian_interior = 0;
  for (int i = -sigma_ / hx; i <= sigma_ / hx; i++)
    for (int j = -sigma_ / hx; j <= sigma_ / hx; j++)
      for (int k = -sigma_ / hx; k <= sigma_ / hx; k++) {
        dist = sqrt(i * i + j * j + k * k);
        if (dist <= sigma_ / hx) gaussian_interior++;
      }
  ss << " .. parameters: sigma:" << sigma_ << " | radius: " << sigma_ / hx << " | center spacing: " << space << " | gaussian interior: " << gaussian_interior
     << " | gvf: " << params_->tu_->gaussian_vol_frac_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  np_ = params_->tu_->np_;
  ss << " np: " << np_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
// write centers to file
#ifdef VISUALIZE_PHI
  std::stringstream phivis;
  phivis << " sigma = " << sigma_ << ", spacing = " << spacing_factor_ * sigma_ << std::endl;
  phivis << " centers = [" << std::endl;
  for (int ptr = 0; ptr < 3 * np_; ptr += 3) {
    phivis << " " << centers_[ptr + 0] << ", " << centers_[ptr + 1] << ", " << centers_[ptr + 2] << std::endl;
  }
  phivis << "];" << std::endl;
  std::fstream phifile;
  static int ct = 0;
  if (procid == 0) {
    std::stringstream ssct;
    ssct << ct;
    phifile.open(std::string(params_->tu_->writepath_.str() + "phi-mesh-" + ssct.str() + ".dat"), std::ios_base::out);
    phifile << phivis.str() << std::endl;
    phifile.close();
    ct++;
  }
#endif

  // Destroy and clear any previously set phis
  for (int i = 0; i < phi_vec_.size(); i++) {
    ierr = VecDestroy(&phi_vec_[i]); CHKERRQ(ierr);
  }
  phi_vec_.clear();
  phi_vec_.resize(np_);
  ierr = VecCreate(PETSC_COMM_WORLD, &phi_vec_[0]);
  ierr = VecSetSizes(phi_vec_[0], params_->grid_->nl_, params_->grid_->ng_);
  ierr = setupVec(phi_vec_[0]);
  ierr = VecSet(phi_vec_[0], 0);
  for (int i = 1; i < np_; i++) {
    ierr = VecDuplicate(phi_vec_[0], &phi_vec_[i]);
    ierr = VecSet(phi_vec_[i], 0);
  }
  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// setGaussians ////////////////////////////////////// ###
PetscErrorCode Phi::setGaussians(Vec data) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;

  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  ss << " .. basis functions set based on tumor input data.";
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();

  Vec num_tumor_output;
  ierr = VecDuplicate(data, &num_tumor_output); CHKERRQ(ierr);
  ierr = VecSet(num_tumor_output, 0); CHKERRQ(ierr);

  ScalarType twopi = 2.0 * M_PI;
  int64_t X, Y, Z;
  int ptr;
  int gaussian_interior = 0;
  ScalarType hx = twopi / params_->grid_->n_[0], hy = twopi / params_->grid_->n_[1], hz = twopi / params_->grid_->n_[2];
  ScalarType h_64 = twopi / 64;
  ScalarType h_256 = twopi / 256;
  sigma_ = params_->tu_->phi_sigma_data_driven_;  // This spacing corresponds to 1mm sigma -- tumor width of say 4*sigma

  ScalarType sigma_smooth = 2.0 * M_PI / params_->grid_->n_[0];
  spacing_factor_ = 2.0;
  params_->tu_->phi_spacing_factor_ = spacing_factor_;
  ScalarType space = spacing_factor_ * sigma_ / hx;

  // Get gaussian volume
  ScalarType dist = 0.0;
  for (int i = -sigma_ / hx; i <= sigma_ / hx; i++)
    for (int j = -sigma_ / hx; j <= sigma_ / hx; j++)
      for (int k = -sigma_ / hx; k <= sigma_ / hx; k++) {
        dist = sqrt(i * i + j * j + k * k);
        if (dist <= sigma_ / hx) gaussian_interior++;
      }

  ss << " .. parameters: sigma:" << sigma_ << " | radius: " << sigma_ / hx << " | center spacing: " << space << " | gaussian interior: " << gaussian_interior
     << " | gvf: " << params_->tu_->gaussian_vol_frac_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  int flag = 0;
  np_ = 0;
  std::vector<ScalarType> center;
  std::vector<int64_t> center_comm;  // Vector of global indices of centers to be communicated across procs
                                     // because they are either boundary centers or out of proc neighbour centers
  std::vector<int> local_tumor_marker(params_->grid_->nl_, 0);  // Local marker for boundary centers. This is updated everytime the local
                                                                // proc receives computation results from the neighbors.

  ScalarType *data_ptr, *num_top_ptr;
  ierr = VecGetArray(data, &data_ptr); CHKERRQ(ierr);
  ierr = VecGetArray(num_tumor_output, &num_top_ptr); CHKERRQ(ierr);

  int start_x, start_y, start_z, end_x, end_y, end_z;
  bool break_check = false;

  // Find the global index of the first and last center in the current proc
  for (int x = 0; x < params_->grid_->isize_[0]; x++) {
    for (int y = 0; y < params_->grid_->isize_[1]; y++) {
      for (int z = 0; z < params_->grid_->isize_[2]; z++) {
        X = params_->grid_->istart_[0] + x;
        Y = params_->grid_->istart_[1] + y;
        Z = params_->grid_->istart_[2] + z;
        if (fmod(X, space) == 0 && fmod(Y, space) == 0 && fmod(Z, space) == 0) {
          start_x = x;
          start_y = y;
          start_z = z;
          break_check = true;
          break;
        }
      }
      if (break_check) break;
    }
    if (break_check) break;
  }
  break_check = false;
  for (int x = params_->grid_->isize_[0] - 1; x >= 0; x--) {
    for (int y = params_->grid_->isize_[1] - 1; y >= 0; y--) {
      for (int z = params_->grid_->isize_[2] - 1; z >= 0; z--) {
        X = params_->grid_->istart_[0] + x;
        Y = params_->grid_->istart_[1] + y;
        Z = params_->grid_->istart_[2] + z;
        if (fmod(X, space) == 0 && fmod(Y, space) == 0 && fmod(Z, space) == 0) {
          end_x = x;
          end_y = y;
          end_z = z;
          break_check = true;
          break;
        }
      }
      if (break_check) break;
    }
    if (break_check) break;
  }

  // Loop over centers in the local process
  // Domain is parallelized only in x and y direction in accordance with accfft
  // The boundary centers in z direction are hence irrelevant
  for (int x = start_x; x <= end_x; x += space)
    for (int y = start_y; y <= end_y; y += space)
      for (int z = start_z; z <= end_z; z += space) {
        X = params_->grid_->istart_[0] + x;
        Y = params_->grid_->istart_[1] + y;
        Z = params_->grid_->istart_[2] + z;

        if (x == start_x || y == start_y || x == end_x || y == end_y) {                                               // Get boundary points
          checkTumorExistenceOutOfProc(X, Y, Z, sigma_ / hx, data_ptr, params_, center_comm, local_tumor_marker, 1);  // In proc
          // Out of Proc checks
          if (x == start_x) {  // Check for left neighbor
            checkTumorExistenceOutOfProc((X - space > 0) ? (X - space) : (params_->grid_->n_[0] - space), Y, Z, sigma_ / hx, data_ptr, params_, center_comm, local_tumor_marker, 0);
          }
          if (y == start_y) {  // Check for bottom neighbor
            checkTumorExistenceOutOfProc(X, (Y - space > 0) ? (Y - space) : (params_->grid_->n_[1] - space), Z, sigma_ / hx, data_ptr, params_, center_comm, local_tumor_marker, 0);
          }
          if (x == end_x) {  // Check for right neighbor
            checkTumorExistenceOutOfProc((X + space < params_->grid_->n_[0]) ? (X + space) : space, Y, Z, sigma_ / hx, data_ptr, params_, center_comm, local_tumor_marker, 0);
          }
          if (y == end_y) {  // Check for top neighbor
            checkTumorExistenceOutOfProc(X, (Y + space < params_->grid_->n_[1]) ? (Y + space) : space, Z, sigma_ / hx, data_ptr, params_, center_comm, local_tumor_marker, 0);
          }
          // Now check for corner neighbors
          if (x == start_x && y == start_y) {
            checkTumorExistenceOutOfProc((X - space > 0) ? (X - space) : (params_->grid_->n_[0] - space), (Y - space > 0) ? (Y - space) : (params_->grid_->n_[1] - space), Z, sigma_ / hx, data_ptr,
                                         params_, center_comm, local_tumor_marker, 0);
          }
          if (x == start_x && y == end_y) {
            checkTumorExistenceOutOfProc((X - space > 0) ? (X - space) : (params_->grid_->n_[0] - space), (Y + space < params_->grid_->n_[1]) ? (Y + space) : space, Z, sigma_ / hx, data_ptr, params_,
                                         center_comm, local_tumor_marker, 0);
          }
          if (x == end_x && y == start_y) {
            checkTumorExistenceOutOfProc((X + space < params_->grid_->n_[0]) ? (X + space) : space, (Y - space > 0) ? (Y - space) : (params_->grid_->n_[1] - space), Z, sigma_ / hx, data_ptr, params_,
                                         center_comm, local_tumor_marker, 0);
          }
          if (x == end_x && y == end_y) {
            checkTumorExistenceOutOfProc((X + space < params_->grid_->n_[0]) ? (X + space) : space, (Y + space < params_->grid_->n_[1]) ? (Y + space) : space, Z, sigma_ / hx, data_ptr, params_,
                                         center_comm, local_tumor_marker, 0);
          }
        } else {  // Not a boundary center: Computation can be completed locally
          checkTumorExistence(x, y, z, sigma_ / hx, data_ptr, params_, local_tumor_marker);
        }
      }

  ierr = VecRestoreArray(data, &data_ptr); CHKERRQ(ierr);

  std::vector<int> center_comm_sizes;
  center_comm_sizes.resize(nprocs);
  int size = center_comm.size();
  MPI_Allgather(&size, 1, MPI_INT, &center_comm_sizes[0], 1, MPI_INT, PETSC_COMM_WORLD);

  int max_center_size;  // If odd number of procs are used, the points will be distributed unevenly
  max_center_size = *std::max_element(center_comm_sizes.begin(), center_comm_sizes.end());

  MPI_Request request[16];
  for (int i = 0; i < 16; i++) request[i] = MPI_REQUEST_NULL;
  // Communicate partial computation of boundary centers to neighbouring processes
  std::vector<int64_t> receive_buffer(8 * max_center_size, 0);
  std::vector<int64_t> zero_vector(center_comm.size(), 0);
  int proc_i, proc_j, procid_neigh, count, neigh_i, neigh_j, periodic_check;
  periodic_check = 0;
  count = 0;
  proc_i = procid / params_->grid_->c_dims_[1];
  proc_j = procid - proc_i * params_->grid_->c_dims_[1];
  for (int ni = proc_i - 1; ni <= proc_i + 1; ni++)
    for (int nj = proc_j - 1; nj <= proc_j + 1; nj++) {  // Get all neighboring processes
      // Check for boundary procs and implement periodic send and recv
      neigh_i = ni;
      neigh_j = nj;
      request[count] = MPI_REQUEST_NULL;

      if (neigh_i < 0) {
        neigh_i = params_->grid_->c_dims_[0] - 1;
        periodic_check = 1;
      }
      if (neigh_i >= params_->grid_->c_dims_[0]) {
        neigh_i = 0;
        periodic_check = 1;
      }
      if (neigh_j < 0) {
        neigh_j = params_->grid_->c_dims_[1] - 1;
        periodic_check = 1;
      }
      if (neigh_j >= params_->grid_->c_dims_[1]) {
        neigh_j = 0;
        periodic_check = 1;
      }
      if (neigh_i == proc_i && neigh_j == proc_j) {
        periodic_check = 0;
        continue;
      }

      procid_neigh = neigh_i * params_->grid_->c_dims_[1] + neigh_j;

      if (periodic_check) {
        MPI_Isend(&zero_vector[0], center_comm.size(), MPI_LONG_LONG, procid_neigh, 0, MPI_COMM_WORLD, &request[count]);
      } else {
        MPI_Isend(&center_comm[0], center_comm.size(), MPI_LONG_LONG, procid_neigh, 0, MPI_COMM_WORLD, &request[count]);
      }

      count++;
      periodic_check = 0;
    }

  count = 0;
  proc_i = procid / params_->grid_->c_dims_[1];
  proc_j = procid - proc_i * params_->grid_->c_dims_[1];
  for (int ni = proc_i - 1; ni <= proc_i + 1; ni++)
    for (int nj = proc_j - 1; nj <= proc_j + 1; nj++) {  // Get all neighboring processes
      // Check for boundary procs and implement periodic send and recv
      neigh_i = ni;
      neigh_j = nj;
      request[count + 8] = MPI_REQUEST_NULL;

      if (neigh_i < 0) neigh_i = params_->grid_->c_dims_[0] - 1;
      if (neigh_i >= params_->grid_->c_dims_[0]) neigh_i = 0;
      if (neigh_j < 0) neigh_j = params_->grid_->c_dims_[1] - 1;
      if (neigh_j >= params_->grid_->c_dims_[1]) neigh_j = 0;
      if (neigh_i == proc_i && neigh_j == proc_j) continue;

      procid_neigh = neigh_i * params_->grid_->c_dims_[1] + neigh_j;

      MPI_Irecv(&receive_buffer[count * max_center_size], center_comm_sizes[procid_neigh], MPI_LONG_LONG, procid_neigh, 0, MPI_COMM_WORLD, &request[count + 8]);
      count++;
    }

  MPI_Status status;

  for (int i = 0; i < 16; i++) {
    if (request[i] != MPI_REQUEST_NULL) MPI_Wait(&request[i], MPI_STATUS_IGNORE);
  }

  // Check receive buffer for the local centers and add their value to local_tumor_marker
  for (int i = 0; i < receive_buffer.size(); i += 2) {  // Every alternate value is a center global index
    // Get global coordinates from global index
    X = receive_buffer[i] / (params_->grid_->n_[1] * params_->grid_->n_[2]);
    Y = fmod(receive_buffer[i], params_->grid_->n_[1] * params_->grid_->n_[2]) / params_->grid_->n_[2];
    Z = receive_buffer[i] - Y * params_->grid_->n_[2] - X * params_->grid_->n_[1] * params_->grid_->n_[2];
    flag = isInLocalProc(X, Y, Z, params_);

    if (flag) {  // The point is inside the local processor
      ptr = (X - params_->grid_->istart_[0]) * params_->grid_->isize_[1] * params_->grid_->isize_[2] + (Y - params_->grid_->istart_[1]) * params_->grid_->isize_[2] + (Z - params_->grid_->istart_[2]);
      local_tumor_marker[ptr] += receive_buffer[i + 1];  // Add the contirbution from the neighbour to the local boundary center
    }
  }

  ScalarType *label_ptr;
  if (labels_ != nullptr) {
    // connected components has updated the labels
    ierr = VecGetArray(labels_, &label_ptr); CHKERRQ(ierr);
  }

  // Add the local boundary centers to the selected centers vector
  for (int i = 0; i < local_tumor_marker.size(); i++) {
    num_top_ptr[i] = (ScalarType)local_tumor_marker[i] / gaussian_interior;  // For visualization

    if (local_tumor_marker[i] > params_->tu_->gaussian_vol_frac_ * gaussian_interior) {  // Boundary center with tumors in its vicinity
      X = i / (params_->grid_->isize_[1] * params_->grid_->isize_[2]);
      Y = fmod(i, params_->grid_->isize_[1] * params_->grid_->isize_[2]) / params_->grid_->isize_[2];
      Z = i - Y * params_->grid_->isize_[2] - X * params_->grid_->isize_[1] * params_->grid_->isize_[2];
      X += params_->grid_->istart_[0];
      Y += params_->grid_->istart_[1];
      Z += params_->grid_->istart_[2];
      np_++;
      center.push_back(X * hx);
      center.push_back(Y * hy);
      center.push_back(Z * hz);
      if (labels_ != nullptr)
        gaussian_labels_.push_back(label_ptr[i]);  // each gaussian has the component label
      else
        gaussian_labels_.push_back(1);  // if no labels, assume one component: all gaussians are component 1
    }
  }

  if (labels_ != nullptr) {
    ierr = VecRestoreArray(labels_, &label_ptr); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(num_tumor_output, &num_top_ptr); CHKERRQ(ierr);

  int np_global;
  MPI_Allreduce(&np_, &np_global, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
  std::vector<int> center_size, displs, rcount;
  std::vector<ScalarType> center_global;
  center_size.resize(nprocs);
  displs.resize(nprocs);
  rcount.resize(nprocs);
  size = center.size();
  center_global.resize(3 * np_global);
  MPI_Allgather(&size, 1, MPI_INT, &center_size[0], 1, MPI_INT, PETSC_COMM_WORLD);

  displs[0] = 0;
  rcount[0] = center_size[0];
  for (int i = 1; i < nprocs; i++) {
    displs[i] = displs[i - 1] + center_size[i - 1];
    rcount[i] = center_size[i];
  }
  MPI_Allgatherv(&center[0], center.size(), MPIType, &center_global[0], &rcount[0], &displs[0], MPIType, PETSC_COMM_WORLD);

  // gather gaussian labels
  std::vector<int> g_labels;
  g_labels.resize(np_global);
  size = gaussian_labels_.size();
  MPI_Allgather(&size, 1, MPI_INT, &center_size[0], 1, MPI_INT, PETSC_COMM_WORLD);
  displs[0] = 0;
  rcount[0] = center_size[0];
  for (int i = 1; i < nprocs; i++) {
    displs[i] = displs[i - 1] + center_size[i - 1];
    rcount[i] = center_size[i];
  }
  MPI_Allgatherv(&gaussian_labels_[0], gaussian_labels_.size(), MPI_INT, &g_labels[0], &rcount[0], &displs[0], MPI_INT, PETSC_COMM_WORLD);
  gaussian_labels_.clear();
  gaussian_labels_ = g_labels;

  np_ = np_global;
  params_->tu_->np_ = np_;
  ss << " .. np: " << np_;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
  ss.str("");
  ss.clear();
  centers_.clear();
  centers_.resize(3 * np_);
  centers_ = center_global;

#ifdef VISUALIZE_PHI
  std::stringstream phivis;
  phivis << " sigma = " << sigma_ << ", spacing = " << spacing_factor_ * sigma_ << std::endl;
  phivis << " centers = [" << std::endl;
  for (int ptr = 0; ptr < 3 * np_; ptr += 3) {
    phivis << " " << centers_[ptr + 0] << ", " << centers_[ptr + 1] << ", " << centers_[ptr + 2] << std::endl;
  }
  phivis << "];" << std::endl;
  std::fstream phifile;
  static int ct = 0;
  if (procid == 0) {
    std::stringstream ssct;
    ssct << ct;
    phifile.open(std::string(params_->tu_->writepath_.str() + "phi-mesh-" + ssct.str() + ".dat"), std::ios_base::out);
    phifile << phivis.str() << std::endl;
    phifile.close();
    ct++;
  }
#endif

  // create 3 * sparsity_level phis to be re-used every cosamp iteration: if not cosamp this is unused and phi is computed
  // max size of subspace can be 3 * sparsity_level
  // on the fly
  int num_phi_store = (params_->tu_->phi_store_) ? np_ : 3 * params_->tu_->sparsity_level_;
#ifdef CUDA
  size_t free_mem;
  size_t total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  ScalarType free = static_cast<ScalarType>(free_mem) / 1048576.0;
  ScalarType size_of_vector_256 = 68.;
  // insufficient memory for even restricted subspace; +1 because there is a temp vector created in TILOptimzer (might have to check if there are more like this)
  if (std::floor(free / size_of_vector_256) <= num_phi_store + 1) {
    params_->tu_->force_phi_compute_ = true;
    ss << "Insufficient memory for gaussians on restricted subspace = " << std::floor(free / size_of_vector_256) << " <= " << num_phi_store + 1 << " ; all computations performed on the fly";
    ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
   }
#endif
  if (params_->tu_->force_phi_compute_) num_phi_store = 1;

  // Destroy and clear any previously set phis
  for (int i = 0; i < phi_vec_.size(); i++) {
    ierr = VecDestroy(&phi_vec_[i]); CHKERRQ(ierr);
  }
  phi_vec_.clear();

  phi_vec_.resize(num_phi_store);
  ierr = VecCreate(PETSC_COMM_WORLD, &phi_vec_[0]);
  ierr = VecSetSizes(phi_vec_[0], params_->grid_->nl_, params_->grid_->ng_);
  ierr = setupVec(phi_vec_[0]);
  ierr = VecSet(phi_vec_[0], 0);

  for (int i = 1; i < num_phi_store; i++) {
    ierr = VecDuplicate(phi_vec_[0], &phi_vec_[i]);
    ierr = VecSet(phi_vec_[i], 0);
  }

  if (params_->tu_->write_output_) {
    dataOut(num_tumor_output, params_, "phiNumTumor.nc");
  }
  ierr = VecDestroy(&num_tumor_output); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode Phi::setAdditionalMemory(int n) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int sz = phi_vec_.size();
  phi_vec_.resize(sz + n);

  for (int i = sz; i < sz + n; i++) {
    ierr = VecDuplicate(phi_vec_[0], &phi_vec_[i]); CHKERRQ(ierr);
    ierr = VecSet(phi_vec_[i], 0); CHKERRQ(ierr);
  }

  PetscFunctionReturn(ierr);
}

void Phi::modifyCenters(std::vector<int> support_idx) {
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  std::stringstream ss;

  centers_temp_ = centers_;
  centers_.clear();

  int counter = 0;
  int idx;
  for (int i = 0; i < support_idx.size(); i++) {
    idx = support_idx[i];  // Get the required center idx
    counter = 3 * idx;
    centers_.push_back(centers_temp_[counter]);
    centers_.push_back(centers_temp_[counter + 1]);
    centers_.push_back(centers_temp_[counter + 2]);
  }

  // resize np
  np_ = support_idx.size();
  if (!params_->tu_->force_phi_compute_) {
    // restricted subspace; can store phis
    if (!params_->tu_->phi_store_) compute_ = false;
  } else {
    // phi compute is forced to be on the fly
    compute_ = true;
  }
  ss << " size of restricted subspace: " << np_;
  tuMSGstd(ss.str());
  ss.str("");
  ss.clear();
}

void Phi::resetCenters() {
  centers_.clear();
  centers_ = centers_temp_;
  np_ = params_->tu_->np_;
  if (!params_->tu_->phi_store_) compute_ = true;
}

Phi::~Phi() {
  PetscErrorCode ierr = 0;
  for (int i = 0; i < phi_vec_.size(); i++) {
    ierr = VecDestroy(&phi_vec_[i]);
  }
  phi_vec_.clear();
}
