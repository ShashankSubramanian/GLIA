#include "TumorSolverInterface.h"

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
TumorSolverInterface::TumorSolverInterface(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Phi> phi, std::shared_ptr<MatProp> mat_prop)
    : initialized_(false),
      initializedFFT_(false),
      optimizer_settings_changed_(false),
      regularization_norm_changed_(false),
      newton_solver_type_changed_(false),
      params_(params),
      spec_ops_(spec_ops),
      tumor_(),
      pde_operators_(),
      derivative_operators_(),
      inv_solver_() {
  if (params != nullptr and spec_ops != nullptr) initialize(params, spec_ops, phi, mat_prop);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::initializeEvent() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  EventRegistry::initialize();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::finalizeEvent() {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  EventRegistry::finalize();
  if (procid == 0) {
    EventRegistry r;
    r.print("TumorSolverTimings.log", true);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::finalize(DataDistributionParameters &ivars) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  accfft_destroy_plan(ivars.plan);
  accfft_cleanup();
  MPI_Comm_free(&ivars.comm);
  initializedFFT_ = false;
  initialized_ = false;
  // ierr = PetscFinalize ();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::initializeFFT(DataDistributionParameters &ivars) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  if (initializedFFT_) PetscFunctionReturn(0);
  // initialize accfft, data distribution and comm plan
  accfft_init();
  accfft_create_comm(MPI_COMM_WORLD, ivars.cdims, &ivars.comm);
#if defined(CUDA) && !defined(MPICUDA)
  spec_ops_ = std::make_shared<SpectralOperators>(CUFFT);
#else
  spec_ops_ = std::make_shared<SpectralOperators>(ACCFFT);
#endif
  spec_ops_->setup(ivars.n, ivars.isize, ivars.istart, ivars.osize, ivars.ostart, ivars.comm);
  ivars.plan = spec_ops_->plan_;
  ivars.alloc_max = spec_ops_->alloc_max_;
  ivars.nlocal = ivars.isize[0] * ivars.isize[1] * ivars.isize[2];
  ivars.nglobal = ivars.n[0] * ivars.n[1] * ivars.n[2];
  initializedFFT_ = true;

  PetscFunctionReturn(ierr);
}

// ### _____________________________________________________________________ ___
// ### ///////////////// TODO ////////////////////////////////////////////// ###
// # TODO: rewrite initialization process
// #  - need to do all the setup stuff from app/inverse.cpp
// #  - function that reads in atlas (i.e., initial mat probs)
// #  - function that reads in data
// #  - possibly InitVars als otake np and Gaussian selection mode, and create PHi from that (so does't need to be recreated)
// ### _____________________________________________________________________ ___
// ### ///////////////// TODO ////////////////////////////////////////////// ###

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::initialize(DataDistributionParameters &ivars, std::shared_ptr<TumorParameters> tumor_params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // don't do it twice
  if (initialized_) PetscFunctionReturn(0);
  // FFT needs to be initialized
  if (!initializedFFT_) {
    ierr = tuMSGwarn("Error: FFT needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  // create params
  params_ = std::make_shared<Parameters>();
  params_->createGrid(ivars.n, ivars.isize, ivars.osize, ivars.istart, ivars.ostart, ivars.plan, ivars.comm, ivars.cdims);
  // set tumor params from outside
  if (tumor_params != nullptr) {
    ierr = setParams(tumor_params); CHKERRQ(ierr);
  }
  // initialize tumor, initialize dummy phi, initialize mat probs
  initialize(params_, spec_ops_, nullptr, nullptr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::initialize(std::shared_ptr<Parameters> params, std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Phi> phi, std::shared_ptr<MatProp> mat_prop) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  if (initialized_) PetscFunctionReturn(0);

  tumor_ = std::make_shared<Tumor>(params, spec_ops);
  params_ = params;
  // set up vector p (should also add option to pass a p vec, that is used to initialize tumor)
  Vec p;
  int np = params_->tu_->np_;
  int nk = params_->get_nk();

  ierr = VecCreateSeq(PETSC_COMM_SELF, np + nk, &p); CHKERRQ(ierr);
  ierr = setupVec(p, SEQ); CHKERRQ(ierr);
  ierr = VecSet(p, 1); CHKERRQ(ierr);
  ierr = tumor_->initialize(p, params_, spec_ops, phi, mat_prop); CHKERRQ(ierr);

  // create pde and derivative operators
  if (params_->tu_->model_ == 1) {
    pde_operators_ = std::make_shared<PdeOperatorsRD>(tumor_, params_, spec_ops);
    if (params_->opt_->cross_entropy_loss_)
      derivative_operators_ = std::make_shared<DerivativeOperatorsKL>(pde_operators_, params_, tumor_);
    else
      derivative_operators_ = std::make_shared<DerivativeOperatorsRD>(pde_operators_, params_, tumor_);
  }
  if (params_->tu_->model_ == 2) {
    pde_operators_ = std::make_shared<PdeOperatorsRD>(tumor_, params_, spec_ops);
    derivative_operators_ = std::make_shared<DerivativeOperatorsPos>(pde_operators_, params_, tumor_);
  }
  if (params_->tu_->model_ == 3) {
    pde_operators_ = std::make_shared<PdeOperatorsRD>(tumor_, params_, spec_ops);
    derivative_operators_ = std::make_shared<DerivativeOperatorsRDObj>(pde_operators_, params_, tumor_);
  }
  if (params_->tu_->model_ == 4) {
    pde_operators_ = std::make_shared<PdeOperatorsMassEffect>(tumor_, params_, spec_ops);
    derivative_operators_ = std::make_shared<DerivativeOperatorsMassEffect>(pde_operators_, params_, tumor_);
  }
  if (params_->tu_->model_ == 5) {
    pde_operators_ = std::make_shared<PdeOperatorsMultiSpecies>(tumor_, params_, spec_ops);
    derivative_operators_ = std::make_shared<DerivativeOperatorsRD>(pde_operators_, params_, tumor_);
  }
  // create tumor inverse solver
  inv_solver_ = std::make_shared<InvSolver>(derivative_operators_, pde_operators_, params_, tumor_); CHKERRQ(ierr);
  ierr = inv_solver_->initialize(derivative_operators_, pde_operators_, params_, tumor_); CHKERRQ(ierr);
  initialized_ = true;
  ierr = VecDestroy(&p); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setParams(std::shared_ptr<TumorParameters> tumor_params) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  TU_assert(initialized_, "TumorSolverInterface::setParams(): TumorSolverInterface needs to be initialized.")
      TU_assert(tumor_params != nullptr, "TumorSolverInterface::setParams(): Tumor Parameters must not be NULL.") std::stringstream ss;
  bool np_changed = false, model_changed = false, nt_changed = false, nx_changed = false;

  if (tumor_params != nullptr) {
    ierr = tuMSGstd(" Overwriting Tumor Parameters."); CHKERRQ(ierr);
    np_changed = (params_->tu_->np_ + params_->tu_->nk_) != (tumor_params->np_ + tumor_params->nk_);
    nt_changed = params_->tu_->nt_ != tumor_params->nt_;
    model_changed = params_->tu_->model_ != tumor_params->model_;
    // overwriting params
    params_->tu_ = tumor_params;
    ss << "np_changed: " << np_changed << ", nt_changed: " << nt_changed << ", model_changed: " << model_changed;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  } else {
    {
      ierr = tuMSGwarn("Error: (setParams) Tumor Parameter must not be NULL. Exiting."); CHKERRQ(ierr);
      PetscFunctionReturn(1);
    }
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setParams(Vec p, std::shared_ptr<TumorParameters> tumor_params = {}) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  TU_assert(initialized_, "TumorSolverInterface::setParams(): TumorSolverInterface needs to be initialized.") bool np_changed = true, model_changed = false,
                                                                                                                   nt_changed = true;  // nt_changed = true for synthetic tests using app/inverse.cpp

  // === re-set parameters
  if (tumor_params != nullptr) {
    // if one of these parameters has changed, we need to re-allocate objects and memory
    np_changed = (params_->tu_->np_ + params_->tu_->nk_) != (tumor_params->np_ + tumor_params->nk_);
    nt_changed = params_->tu_->nt_ != tumor_params->nt_;
    model_changed = params_->tu_->model_ != tumor_params->model_;
    ierr = setParams(tumor_params); CHKERRQ(ierr);
  }
  if (np_changed) {
    ierr = tuMSGstd(" number of basis functions changed, resetting Phi and DerivativeOperators."); CHKERRQ(ierr);
  }
  if (nt_changed) {
    ierr = tuMSGstd(" number of time steps changed, resetting PdeOperators (time history)."); CHKERRQ(ierr);
  }
  if (model_changed) {
    ierr = tuMSGstd(" tumor model changed, resetting PdeOperators and DerivativeOperators."); CHKERRQ(ierr);
  }
  // === re-initialize Tumor
  ierr = tumor_->setParams(p, params_, np_changed); CHKERRQ(ierr);
  // === re-initialize pdeoperators and derivativeoperators; invcludes re-allocating time history for adjoint.
  if (model_changed) {
    switch (params_->tu_->model_) {
      case 1:
        pde_operators_ = std::make_shared<PdeOperatorsRD>(tumor_, params_, spec_ops_);
        if (params_->opt_->cross_entropy_loss_) {
          derivative_operators_ = std::make_shared<DerivativeOperatorsKL>(pde_operators_, params_, tumor_);
        } else {
          derivative_operators_ = std::make_shared<DerivativeOperatorsRD>(pde_operators_, params_, tumor_);
        }
        break;
      case 2:
        pde_operators_ = std::make_shared<PdeOperatorsRD>(tumor_, params_, spec_ops_);
        derivative_operators_ = std::make_shared<DerivativeOperatorsPos>(pde_operators_, params_, tumor_);
        break;
      case 3:
        pde_operators_ = std::make_shared<PdeOperatorsRD>(tumor_, params_, spec_ops_);
        derivative_operators_ = std::make_shared<DerivativeOperatorsRDObj>(pde_operators_, params_, tumor_);
        break;
      case 4:
        pde_operators_ = std::make_shared<PdeOperatorsMassEffect>(tumor_, params_, spec_ops_);
        derivative_operators_ = std::make_shared<DerivativeOperatorsRD>(pde_operators_, params_, tumor_);
        break;
      default:
        break;
    }
  } else {  // re-allocate vectors, same model
    if (np_changed) {
      ierr = derivative_operators_->reset(p, pde_operators_, params_, tumor_); CHKERRQ(ierr);
    }
    if (nt_changed) {
      ierr = pde_operators_->reset(params_, tumor_); CHKERRQ(ierr);
    }
  }
  // ++ re-initialize InvSolver ++, i.e. H matrix, p_rec vectores etc..
  inv_solver_->setParams(derivative_operators_, pde_operators_, params_, tumor_, np_changed); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveForward(Vec c1, Vec c0) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // timing
  Event e("solveForward()");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  if (!initialized_) {
    ierr = tuMSGwarn("Error: (solveForward) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  ierr = VecCopy(c0, tumor_->c_0_); CHKERRQ(ierr);
  ierr = pde_operators_->solveState(0); CHKERRQ(ierr);
  ierr = VecCopy(tumor_->c_t_, c1); CHKERRQ(ierr);
  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveForward(Vec c1, Vec c0, std::map<std::string, Vec> *species) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  // timing
  Event e("solveForward()");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  if (!initialized_) {
    ierr = tuMSGwarn("Error: (solveForward) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  ierr = VecCopy(c0, tumor_->c_0_); CHKERRQ(ierr);
  ierr = pde_operators_->solveState(0); CHKERRQ(ierr);
  ierr = VecCopy(tumor_->c_t_, c1); CHKERRQ(ierr);
  species = &(tumor_->species_);
  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveInverseCoSaMp(Vec prec, Vec data, Vec data_gradeval) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  std::stringstream ss;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  // timing
  Event e("solveInverseCoSaMp()");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  if (!initialized_) {
    ierr = tuMSGwarn("Error: (solveInverseCoSaMp) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (data == nullptr) {
    ierr = tuMSGwarn("Error: (solveInverseCoSaMp) Variable data cannot be nullptr. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (prec == nullptr) {
    ierr = tuMSGwarn("Error: (solveInverseCoSaMp) Variable prec cannot be nullptr. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (!optimizer_settings_changed_) {
    ierr = tuMSGwarn(" Tumor inverse solver running with default settings."); CHKERRQ(ierr);
  }

  // set target data for inversion (just sets the vector, no deep copy)
  inv_solver_->setData(data);
  if (data_gradeval == nullptr) data_gradeval = data;
  inv_solver_->setDataGradient(data_gradeval);

  // count the number of observed voxels
  if (params_->tu_->verbosity_ > 2) {
    int sum = 0, global_sum = 0;
    ScalarType *pixel_ptr;
    ierr = VecGetArray(data, &pixel_ptr); CHKERRQ(ierr);
    for (int i = 0; i < params_->grid_->nl_; i++)
      if (pixel_ptr[i] > params_->tu_->obs_threshold_1_) sum++;
    ierr = VecRestoreArray(data, &pixel_ptr); CHKERRQ(ierr);
    MPI_Reduce(&sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
    ss << " number of observed voxels: " << global_sum;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  }
  // inexact Newton
  inv_solver_->getInverseSolverContext()->cosamp_->maxit_newton = params_->opt_->newton_maxit_;
  // inv_solver_->getInverseSolverContext()->cosamp_->inexact_nits = params_->opt_->newton_maxit_;
  // solve
  ierr = inv_solver_->solveInverseCoSaMp(); CHKERRQ(ierr);
  // inv_solver_->solveInverseCoSaMpRS(false);
  ierr = VecCopy(inv_solver_->getPrec(), prec); CHKERRQ(ierr);
  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveInverseMassEffect(ScalarType *xrec, Vec data, Vec data_gradeval) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  std::stringstream ss;

  // timing
  Event e("solveInverseMassEffect()");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  if (!initialized_) {
    ierr = tuMSGwarn("Error: (solveInverseMassEffect) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (data == nullptr) {
    ierr = tuMSGwarn("Error: (solveInverseMassEffect) Variable data cannot be nullptr. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (!optimizer_settings_changed_) {
    ierr = tuMSGwarn(" Tumor inverse solver running with default settings."); CHKERRQ(ierr);
  }

  // set target data for inversion (just sets the vector, no deep copy)
  inv_solver_->setData(data);
  if (data_gradeval == nullptr) data_gradeval = data;
  inv_solver_->setDataGradient(data_gradeval);

  // count the number of observed voxels
  if (params_->tu_->verbosity_ > 2) {
    int sum = 0, global_sum = 0;
    ScalarType *pixel_ptr;
    ierr = VecGetArray(data, &pixel_ptr); CHKERRQ(ierr);
    for (int i = 0; i < params_->grid_->nl_; i++)
      if (pixel_ptr[i] > params_->tu_->obs_threshold_1_) sum++;
    ierr = VecRestoreArray(data, &pixel_ptr); CHKERRQ(ierr);
    MPI_Reduce(&sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
    ss << " number of observed voxels: " << global_sum;
    ierr = tuMSGstd(ss.str()); CHKERRQ(ierr);
    ss.str("");
    ss.clear();
  }

  ierr = inv_solver_->solveForMassEffect(); CHKERRQ(ierr);

  ScalarType *x_ptr;
  ierr = VecGetArray(inv_solver_->getPrec(), &x_ptr); CHKERRQ(ierr);
  *xrec = x_ptr[0];
  ierr = VecRestoreArray(inv_solver_->getPrec(), &x_ptr); CHKERRQ(ierr);

  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveInverse(Vec prec, Vec data, Vec data_gradeval) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  // timing
  Event e("solveInverse()");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();

  if (!initialized_) {
    ierr = tuMSGwarn("Error: (solveInverse) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (data == nullptr) {
    ierr = tuMSGwarn("Error: (solveInverse) Variable data cannot be nullptr. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (prec == nullptr) {
    ierr = tuMSGwarn("Error: (solveInverse) Variable prec cannot be nullptr. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (!optimizer_settings_changed_) {
    ierr = tuMSGwarn(" Tumor inverse solver running with default settings."); CHKERRQ(ierr);
  }

  // reset TAO object if either the solver type changes or the regularization norm
  if (newton_solver_type_changed_ || regularization_norm_changed_) {
    ierr = resetTaoSolver(); CHKERRQ(ierr);
    newton_solver_type_changed_ = false;
    regularization_norm_changed_ = false;
  }

  ierr = tumor_->obs_->apply(data, data, 1); CHKERRQ(ierr);

  // set target data for inversion (just sets the vector, no deep copy) and solve
  inv_solver_->setData(data);
  if (data_gradeval == nullptr) data_gradeval = data;
  inv_solver_->setDataGradient(data_gradeval);
  ierr = inv_solver_->solve(); CHKERRQ(ierr);
  // pass the reconstructed p vector to the caller (deep copy)
  ierr = VecCopy(inv_solver_->getPrec(), prec); CHKERRQ(ierr);
  ierr = inv_solver_->updateReferenceGradient(false); CHKERRQ(ierr);

  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::solveInverseReacDiff(Vec prec, Vec data, Vec data_gradeval) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  int procid, nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  // timing
  Event e("solveInverseReacDiff()");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ScalarType *ptr_pr_rec;
  int np = params_->tu_->np_;  // TODO(K): change to get_nk() function ..
  int nk = (params_->opt_->reaction_inversion_ || params_->opt_->diffusivity_inversion_) ? params_->tu_->nk_ : 0;
  int nr = (params_->opt_->reaction_inversion_) ? params_->tu_->nr_ : 0;

  // ------------- L2 solve to solve for p_i, given sparsity -------------------
  ierr = VecGetArray(prec, &ptr_pr_rec); CHKERRQ(ierr);
  // set initial guess for k_inv (possibly != zero)
  if (params_->opt_->diffusivity_inversion_) {
    ptr_pr_rec[np] = params_->tu_->k_;
    if (nk > 1) ptr_pr_rec[np + 1] = params_->tu_->k_ * params_->tu_->k_gm_wm_ratio_;
    if (nk > 2) ptr_pr_rec[np + 2] = params_->tu_->k_ * params_->tu_->k_glm_wm_ratio_;
  } else {
    // set diff ops with this guess -- this will not change during the solve
    ierr = getTumor()->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, getTumor()->mat_prop_, params_); CHKERRQ(ierr);
  }
  // set initial guess for rho (possibly != zero)
  if (params_->opt_->reaction_inversion_) {
    ptr_pr_rec[np + nk] = params_->tu_->rho_;
    if (nr > 1) ptr_pr_rec[np + nk + 1] = params_->tu_->rho_ * params_->tu_->r_gm_wm_ratio_;
    if (nr > 2) ptr_pr_rec[np + nk + 2] = params_->tu_->rho_ * params_->tu_->r_glm_wm_ratio_;
  }
  ierr = VecRestoreArray(prec, &ptr_pr_rec); CHKERRQ(ierr);

  // set target data for inversion (just sets the vector, no deep copy) and solve
  inv_solver_->setData(data);
  if (data_gradeval == nullptr) data_gradeval = data;
  inv_solver_->setDataGradient(data_gradeval);

  ierr = resetTaoSolver(); CHKERRQ(ierr);
  ierr = setParams(prec, nullptr); CHKERRQ(ierr);
  ierr = tuMSGstd("### ------------------------------------------------- ###"); CHKERRQ(ierr);
  ierr = tuMSG("###                  Initial L2 Solve                 ###"); CHKERRQ(ierr);
  ierr = tuMSGstd("### ------------------------------------------------- ###"); CHKERRQ(ierr);
  ierr = inv_solver_->solve(); CHKERRQ(ierr);
  ierr = VecCopy(inv_solver_->getPrec(), prec); CHKERRQ(ierr);

  if (params_->opt_->reaction_inversion_) {
    inv_solver_->itctx_->cosamp_->cosamp_stage = POST_RD;
    ierr = tuMSGstd("### ------------------------------------------------- ###"); CHKERRQ(ierr);
    ierr = tuMSG("### rho/kappa inversion with scaled L2 solution guess ###"); CHKERRQ(ierr);
    ierr = tuMSGstd("### ------------------------------------------------- ###"); CHKERRQ(ierr);
    inv_solver_->setData(data);
    if (data_gradeval == nullptr) data_gradeval = data;
    inv_solver_->setDataGradient(data_gradeval);
    ierr = inv_solver_->solveInverseReacDiff(prec); CHKERRQ(ierr);
    ierr = VecCopy(inv_solver_->getPrec(), prec); CHKERRQ(ierr);
  } else {
    ierr = tuMSGwarn(" WARNING: Attempting to solve for reaction and diffusion, but reaction inversion is nor enabled."); CHKERRQ(ierr);
  }

  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::computeGradient(Vec dJ, Vec p, Vec data_gradeval) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: (solveForward) computeGradient needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  // TODO[SETTING]: set tumor regularization weight
  // compute gradient for given data 'data_gradeval' and control variable 'p'
  ierr = derivative_operators_->evaluateGradient(dJ, p, data_gradeval); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setOptimizerFeedback(std::shared_ptr<OptimizerFeedback> optfeed) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  params_->optf_ = opt_feedback;
  inv_solver_->setOptFeedback(optfeed);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setOptimizerSettings(std::shared_ptr<OptimizerSettings> optset) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  TU_assert(inv_solver_->isInitialized(), "TumorSolverInterface::setOptimizerSettings(): InvSolver needs to be initialized.")
      TU_assert(optset != nullptr, "TumorSolverInterface::setOptimizerSettings(): requires non-null input.");
  PetscFunctionBegin;
  params_->opt_ = optset;
  inv_solver_->setOptSettings(optset);
  optimizer_settings_changed_ = true;
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setMassEffectData(Vec gm, Vec wm, Vec csf, Vec glm) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: (setMassEffectData) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (wm == nullptr) {
    ierr = tuMSGwarn("Warning: (setMassEffectData) Vector wm is nullptr."); CHKERRQ(ierr);
  }
  if (gm == nullptr) {
    ierr = tuMSGwarn("Warning: (setMassEffectData) Vector gm is nullptr."); CHKERRQ(ierr);
  }
  if (csf == nullptr) {
    ierr = tuMSGwarn("Warning: (setMassEffectData) Vector csf is nullptr."); CHKERRQ(ierr);
  }
  if (glm == nullptr) {
    ierr = tuMSGwarn("Warning: (setMassEffectData) Vector glm is nullptr."); CHKERRQ(ierr);
  }

  return derivative_operators_->setMaterialProperties(gm, wm, csf, glm);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setDistMeassureSimulationGeoImages(Vec wm, Vec gm, Vec csf, Vec bg) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: (setDistMeassureSimulationGeoImages) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (wm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureSimulationGeoImages) Vector wm is nullptr."); CHKERRQ(ierr);
  }
  if (gm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureSimulationGeoImages) Vector gm is nullptr."); CHKERRQ(ierr);
  }
  if (csf == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureSimulationGeoImages) Vector csf is nullptr."); CHKERRQ(ierr);
  }
  /** Sets the image vectors for the simulation geometry material properties
   *  - MOVING PATIENT: mA(0) (= initial helathy atlas)
   *  - MOVING ATLAS:   mA(1) (= initial helathy patient)
   */
  return derivative_operators_->setDistMeassureSimulationGeoImages(wm, gm, csf, nullptr, nullptr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setDistMeassureTargetDataImages(Vec wm, Vec gm, Vec csf, Vec bg) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: (setDistMeassureTargetDataImages) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (wm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureTargetDataImages) Vector wm is nullptr."); CHKERRQ(ierr);
  }
  if (gm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureTargetDataImages) Vector gm is nullptr."); CHKERRQ(ierr);
  }
  if (csf == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureTargetDataImages) Vector csf is nullptr."); CHKERRQ(ierr);
  }
  /** Sets the image vectors for the simulation geometry material properties
  /** Sets the image vectors for the target (patient) geometry material properties
   *  - MOVING PATIENT: mP(1) (= advected patient)
   *  - MOVING ATLAS:   mR    (= patient data)
   */
  return derivative_operators_->setDistMeassureTargetDataImages(wm, gm, csf, nullptr, nullptr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setDistMeassureDiffImages(Vec wm, Vec gm, Vec csf, Vec bg) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (!initialized_) {
    ierr = tuMSGwarn("Error: (setDistMeassureDiffImages) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (wm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureDiffImages) Vector wm is nullptr."); CHKERRQ(ierr);
  }
  if (gm == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureDiffImages) Vector gm is nullptr."); CHKERRQ(ierr);
  }
  if (csf == nullptr) {
    ierr = tuMSGwarn("Warning: (setDistMeassureDiffImages) Vector csf is nullptr."); CHKERRQ(ierr);
  }
  /** Sets the image vectors for the simulation geometry material properties
  /** Sets the image vectors for the distance measure difference
   *  - MOVING PATIENT: || mA(0)(1-c(1)) - mP(1) ||^2
   *  - MOVING ATLAS:   || mA(1)(1-c(1)) - mR    ||^2
   */
  return derivative_operators_->setDistMeassureDiffImages(wm, gm, csf, nullptr, nullptr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::computeTumorContributionRegistration(Vec q1, Vec q2, Vec q4) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  if (pde_operators_ != nullptr) {
    ierr = pde_operators_->computeTumorContributionRegistration(q1, q2, nullptr, q4); CHKERRQ(ierr);
  } else {
    ierr = tuMSGwarn("Error: (in computeTumorContributionRegistration()) PdeOperators not initialized. Exiting .."); CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setGaussians(Vec data) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // timing
  Event e("setGaussians()");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  ierr = tumor_->phi_->setGaussians(data); CHKERRQ(ierr);
  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setGaussians(ScalarType *cm, ScalarType sigma, ScalarType spacing, int np) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // timing
  Event e("setGaussians()");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  std::array<ScalarType, 3> cm_ = {{cm[0], cm[1], cm[2]}};
  params_->tu_->user_cm_ = cm_;
  params_->tu_->phi_spacing_factor_ = spacing;
  params_->tu_->phi_sigma_ = sigma;
  ierr = tumor_->phi_->setGaussians(cm_, sigma, spacing, np); CHKERRQ(ierr);
  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setGaussians(std::array<ScalarType, 3> cm, ScalarType sigma, ScalarType spacing, int np) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  // timing
  Event e("setGaussians()");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  params_->tu_->user_cm_ = cm;
  params_->tu_->phi_spacing_factor_ = spacing;
  params_->tu_->phi_sigma_ = sigma;
  ierr = tumor_->phi_->setGaussians(cm, sigma, spacing, np); CHKERRQ(ierr);
  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::applyPhi(Vec phi_p, Vec p) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ierr = tumor_->phi_->apply(phi_p, p); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::resetTaoSolver() {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  ierr = inv_solver_->resetTao(params_); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setInitialGuess(Vec p) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  TU_assert(p != nullptr, "TumorSolverInterface::setInitialGuess(): requires non-null input.");
  ierr = VecCopy(p, tumor_->p_); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::setInitialGuess(ScalarType d) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  ierr = VecSet(tumor_->p_, d); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::smooth(Vec x, ScalarType num_voxels) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ScalarType sigma_smooth = num_voxels * 2.0 * M_PI / params_->grid_->->n_[0];
  ierr = spec_ops_->weierstrassSmoother(x, x, params_, sigma_smooth); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::readNetCDF(Vec A, std::string filename) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ierr = dataIn(A, params_, filename.c_str()); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::writeNetCDF(Vec A, std::string filename) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  ierr = dataOut(A, params_, filename.c_str()); CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode TumorSolverInterface::updateTumorCoefficients(Vec wm, Vec gm, Vec glm, Vec csf, Vec bg) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  TU_assert(initialized_, "TumorSolverInterface::updateTumorCoefficients(): TumorSolverInterface needs to be initialized.") if (!initialized_) {
    ierr = tuMSGwarn("Error: (updateTumorCoefficients) TumorSolverInterface needs to be initialized before calling this function. Exiting .."); CHKERRQ(ierr);
    PetscFunctionReturn(ierr);
  }
  if (wm == nullptr) {
    ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector wm is nullptr."); CHKERRQ(ierr);
  }
  if (gm == nullptr) {
    ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector gm is nullptr."); CHKERRQ(ierr);
  }
  if (csf == nullptr) {
    ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector csf is nullptr."); CHKERRQ(ierr);
  }
  if (glm == nullptr) {
    ierr = tuMSGwarn("Warning: (updateTumorCoefficients) Vector glm is nullptr."); CHKERRQ(ierr);
  }
  // timing
  Event e("update-tumor-coefficients");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime();
  std::stringstream s;

  ierr = tumor_->mat_prop_->setValuesCustom(gm, wm, glm, csf, bg, params_); CHKERRQ(ierr);
  ierr = tumor_->k_->setValues(params_->tu_->k_, params_->tu_->k_gm_wm_ratio_, params_->tu_->k_glm_wm_ratio_, tumor_->mat_prop_, params_); CHKERRQ(ierr);
  ierr = tumor_->rho_->setValues(params_->tu_->rho_, params_->tu_->r_gm_wm_ratio_, params_->tu_->r_glm_wm_ratio_, tumor_->mat_prop_, params_); CHKERRQ(ierr);
  ierr = tumor_->phi_->setValues(tumor_->mat_prop_); CHKERRQ(ierr);
  ierr = pde_operators_->diff_solver_->precFactor(); CHKERRQ(ierr);

  // timing
  self_exec_time += MPI_Wtime();
  t[5] = self_exec_time;
  e.addTimings(t);
  e.stop();
  PetscFunctionReturn(ierr);
}
