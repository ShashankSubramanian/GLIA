#include "TumorSolverInterface.h"
#include "EventTimings.hpp"

TumorSolverInterface::TumorSolverInterface (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Phi> phi, std::shared_ptr<MatProp> mat_prop) :
initialized_ (false),
optimizer_settings_changed_ (false),
n_misc_ (n_misc),
tumor_ (),
pde_operators_ (),
derivative_operators_ (),
inv_solver_ () {
    PetscErrorCode ierr = 0;
    if (n_misc != nullptr)
        initialize (n_misc, phi, mat_prop);
}

PetscErrorCode TumorSolverInterface::initialize (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Phi> phi, std::shared_ptr<MatProp> mat_prop) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    if (initialized_) PetscFunctionReturn (0);

    tumor_ = std::make_shared<Tumor> (n_misc);
    n_misc_ = n_misc;
    // set up vector p (should also add option to pass a p vec, that is used to initialize tumor)
    Vec p;
    int np = n_misc_->np_;
    int nk = (n_misc_->diffusivity_inversion_) ? n_misc_->nk_ : 0;

    #ifdef SERIAL
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p);                     CHKERRQ (ierr);
    #else
        ierr = VecCreate (PETSC_COMM_WORLD, &p);                                CHKERRQ (ierr);
        ierr = VecSetSizes (p, PETSC_DECIDE, n_misc->np_);                      CHKERRQ (ierr);
        ierr = VecSetFromOptions (p);                                           CHKERRQ (ierr);
    #endif

    ierr = VecSet (p, n_misc->p_scale_);                                        CHKERRQ (ierr);
    ierr = tumor_->initialize (p, n_misc, phi, mat_prop);

    // create pde and derivative operators
    if (n_misc->model_ == 1) {
        pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc);
        derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc, tumor_);
    }
    if (n_misc->model_ == 2) {
        pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc);
        derivative_operators_ = std::make_shared<DerivativeOperatorsPos> (pde_operators_, n_misc, tumor_);
    }
    if (n_misc->model_ == 3) {
        pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc);
        derivative_operators_ = std::make_shared<DerivativeOperatorsRDObj> (pde_operators_, n_misc, tumor_);
    }
    if (n_misc->model_ == 4) {
        pde_operators_ = std::make_shared<PdeOperatorsMassEffect> (tumor_, n_misc);
        derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc, tumor_);
    }
    // create tumor inverse solver
    inv_solver_ = std::make_shared<InvSolver> (derivative_operators_, pde_operators_, n_misc, tumor_);
    ierr = inv_solver_->initialize (derivative_operators_, pde_operators_, n_misc, tumor_);
    initialized_ = true;
    // cleanup
    ierr = VecDestroy (&p);                                                     CHKERRQ (ierr);
    PetscFunctionReturn (0);
}

PetscErrorCode TumorSolverInterface::setParams (Vec p, std::shared_ptr<TumorSettings> tumor_params = {}) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert (initialized_, "TumorSolverInterface::setParams(): TumorSolverInterface needs to be initialized.")

    bool npchanged = true, modelchanged = false, ntchanged = false;
    // ++ re-initialize nmisc ==
    if(tumor_params != nullptr) {
        // if one of these parameters has changed, we need to re-allocate maemory
        npchanged                       = (n_misc_->np_+n_misc_->nk_) != (tumor_params->np+tumor_params->nk);
        modelchanged                    = n_misc_->model_ != tumor_params->tumor_model;
        ntchanged                       = n_misc_->nt_ != tumor_params->time_steps;
        n_misc_->model_                 = tumor_params->tumor_model;
        n_misc_->dt_                    = tumor_params->time_step_size;
        n_misc_->nt_                    = tumor_params->time_steps;
        n_misc_->time_horizon_          = tumor_params->time_horizon;
        n_misc_->np_                    = tumor_params->np;
        n_misc_->beta_                  = tumor_params->betap;
        n_misc_->writeOutput_           = tumor_params->writeOutput;
        n_misc_->verbosity_             = tumor_params->verbosity;
        n_misc_->obs_threshold_         = tumor_params->obs_threshold;
        n_misc_->k_                     = tumor_params->diff_coeff_scale;
        n_misc_->kf_                    = tumor_params->diff_coeff_scale_anisotropic;
        n_misc_->rho_                   = tumor_params->reaction_coeff_scale;
        n_misc_->k_gm_wm_ratio_         = tumor_params->diffusion_ratio_gm_wm;
        n_misc_->k_glm_wm_ratio_        = tumor_params->diffusion_ratio_glm_wm;
        n_misc_->r_gm_wm_ratio_         = tumor_params->reaction_ratio_gm_wm;
        n_misc_->r_glm_wm_ratio_        = tumor_params->reaction_ratio_glm_wm;
        n_misc_->user_cm_               = tumor_params->phi_center_of_mass;
        n_misc_->phi_spacing_factor_    = tumor_params->phi_spacing_factor;
        n_misc_->phi_sigma_             = tumor_params->phi_sigma;
        n_misc_->phi_sigma_data_driven_ = tumor_params->phi_sigma_data_driven;
        n_misc_->gaussian_vol_frac_     = tumor_params->gaussian_volume_fraction;
        n_misc_->target_sparsity_       = tumor_params->target_sparsity;
        n_misc_->bounding_box_          = tumor_params->phi_selection_mode_bbox;
        n_misc_->diffusivity_inversion_ = tumor_params->diffusivity_inversion;
        n_misc_->reaction_inversion_    = tumor_params->reaction_inversion;
        n_misc_->nk_                    = tumor_params->nk;
        n_misc_->phi_store_             = tumor_params->phi_store;
        n_misc_->adjoint_store_         = tumor_params->adjoint_store;
        n_misc_->multilevel_            = tumor_params->multilevel;
        n_misc_->prune_components_      = tumor_params->prune_components;
        n_misc_->sparsity_level_        = tumor_params->sparsity_level;
    }
    // ++ re-initialize Tumor ++
    ierr = tumor_->setParams (p, n_misc_, npchanged);                           CHKERRQ (ierr);
    // ++ re-initialize pdeoperators and derivativeoperators ++
    // invcludes re-allocating time history for adjoint,
    if(modelchanged) {
        switch (n_misc_->model_) {
            case 1: pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc_);
                    derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc_, tumor_);
                    break;
            case 2: pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc_);
                    derivative_operators_ = std::make_shared<DerivativeOperatorsPos> (pde_operators_, n_misc_, tumor_);
                    break;
            case 3: pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc_);
                    derivative_operators_ = std::make_shared<DerivativeOperatorsRDObj> (pde_operators_, n_misc_, tumor_);
                    break;
            case 4: pde_operators_ = std::make_shared<PdeOperatorsMassEffect> (tumor_, n_misc_);
                    derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc_, tumor_);
                    break;
            default: break;
        }
    } else {
        // re-allocate vectors, same model
        if (npchanged) derivative_operators_->reset(p, pde_operators_, n_misc_, tumor_);
        if (ntchanged) pde_operators_->reset(n_misc_, tumor_);
    }
    // ++ re-initialize InvSolver ++, i.e. H matrix, p_rec vectores etc..
    inv_solver_->setParams(derivative_operators_, pde_operators_, n_misc_, tumor_, npchanged);  CHKERRQ (ierr);

    PetscFunctionReturn(0);
}

// TODO: add switch, if we want to copy or take the pointer from incoming and outgoing data
PetscErrorCode TumorSolverInterface::solveForward (Vec cT, Vec c0) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    // set the initial condition
    ierr = VecCopy (c0, tumor_->c_0_);                                          CHKERRQ (ierr);
    // solve forward
    ierr = pde_operators_->solveState (0);                                      CHKERRQ (ierr);
    // get solution
    ierr = VecCopy (tumor_->c_t_, cT);                                          CHKERRQ (ierr);
    PetscFunctionReturn(0);
}
// TODO: add switch, if we want to copy or take the pointer from incoming and outgoing data
PetscErrorCode TumorSolverInterface::solveInverse (Vec prec, Vec d1, Vec d1g) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert (inv_solver_->isInitialized (), "TumorSolverInterface::setOptimizerSettings(): InvSolver needs to be initialized.")
    if (!optimizer_settings_changed_) {
        ierr = tuMSGwarn (" Tumor inverse solver running with default settings."); CHKERRQ (ierr);
    }

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    // set the observation operator filter : default filter
    ierr = tumor_->obs_->setDefaultFilter (d1);
    ierr = tumor_->obs_->apply (d1, d1);

    // set target data for inversion (just sets the vector, no deep copy)
    inv_solver_->setData (d1); if (d1g == nullptr) d1g = d1;
    inv_solver_->setDataGradient (d1g);

    // solve
    ierr = inv_solver_->solve ();
    // pass the reconstructed p vector to the caller (deep copy)
    ierr= VecCopy (inv_solver_->getPrec(), prec);                               CHKERRQ (ierr);

    Vec x_L2;
    int np, nk, nr;
    double *x_L2_ptr, *prec_ptr;
    np = n_misc_->np_;
    nk = n_misc_->nk_;
    nr = n_misc_->nr_;
    ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk + nr, &x_L2);                 CHKERRQ (ierr);              // Create the L2 solution vector
    ierr = VecSet (x_L2, 0);                                                    CHKERRQ (ierr);
    ierr = VecGetArray (x_L2, &x_L2_ptr);                                       CHKERRQ (ierr);
    ierr = VecGetArray (prec, &prec_ptr);                                       CHKERRQ (ierr);

    for (int i = 0; i < np + nk; i++)
        x_L2_ptr[i] = prec_ptr[i]; // solution + diffusivity copied from L2 solve
    x_L2_ptr[np + nk] = n_misc_->rho_; // Last entry is the current reaction coefficient guess
    if (nr > 1) x_L2_ptr[np + nk + 1] = n_misc_->rho_ * n_misc_->r_gm_wm_ratio_;
    if (nr > 2) x_L2_ptr[np + nk + 2] = n_misc_->rho_ * n_misc_->r_glm_wm_ratio_;


    ierr = VecRestoreArray (x_L2, &x_L2_ptr);                                   CHKERRQ (ierr);
    ierr = VecRestoreArray (prec, &prec_ptr);                                   CHKERRQ (ierr);

    if (n_misc_->reaction_inversion_) {
        ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
        ierr = tuMSG    ("### rho/kappa inversion with scaled L2 solution guess ###"); CHKERRQ (ierr);
        ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);


        ierr = resetTaoSolver();
        ierr = setParams (x_L2, nullptr);

        // Now, we invert for only reaction and diffusion after scaling the IC appropriately
        n_misc_->flag_reaction_inv_ = true; // invert only for reaction and diffusion now

        inv_solver_->setData (d1); if (d1g == nullptr) d1g = d1;
        inv_solver_->setDataGradient (d1g);

        // scale p to one according to our modeling assumptions
        double ic_max, g_norm_ref;
        ic_max = 0;
        // get c0
        ierr = getTumor()->phi_->apply (getTumor()->c_0_, x_L2);
        ierr = VecMax (getTumor()->c_0_, NULL, &ic_max);                        CHKERRQ (ierr);  // max of IC

        ierr = VecGetArray (x_L2, &x_L2_ptr);                                   CHKERRQ (ierr);
        for (int i = 0; i < np; i++){
            if(n_misc_->multilevel_) {
              // scales INT_Omega phi(x) dx = const across levels, factor in between levels: 2
              // scales nx=256 to max {Phi p} = 1, nx=128 to max {Phi p} = 0.5, nx=64 to max {Phi p} = 0.25
              x_L2_ptr[i] *= (1.0/4.0 * n_misc_->n_[0]/64.  / ic_max);
            } else {
              x_L2_ptr[i] *= (1.0 / ic_max);
            }
          }
        ierr = VecRestoreArray (x_L2, &x_L2_ptr);                               CHKERRQ (ierr);

        // write out p vector after IC, k inversion (unscaled)
        if (n_misc_->write_p_checkpoint_) {
          writeCheckpoint(x_L2, getTumor()->phi_, n_misc_->writepath_ .str(), std::string("scaled"));
        }

        // reaction -inversion solve
        ierr = inv_solver_->solveInverseReacDiff (x_L2);          // With IC as current guess
        ierr = VecCopy (inv_solver_->getPrec(), x_L2);                          CHKERRQ (ierr);

        // update rho
        ierr = VecGetArray (x_L2, &x_L2_ptr);                                   CHKERRQ (ierr);
        n_misc_->rho_ = x_L2_ptr[np + nk];

        double r1, r2, r3;
        r1 = x_L2_ptr[np + nk];
        r2 = (n_misc_->nr_ > 1) ? x_L2_ptr[np + nk + 1] : 0;
        r3 = (n_misc_->nr_ > 2) ? x_L2_ptr[np + nk + 2] : 0;

        PCOUT << "\nEstimated reaction coefficients " <<  std::endl;
        PCOUT << "r1: " << r1 << std::endl;
        PCOUT << "r2: " << r2 << std::endl;
        PCOUT << "r3: " << r3 << std::endl;

        ierr = VecGetArray (prec, &prec_ptr);                                   CHKERRQ (ierr);

        for (int i = 0; i < np + nk; i++)
            prec_ptr[i] = x_L2_ptr[i]; // solution + diffusivity copied from L2 solve

        ierr = VecRestoreArray (x_L2, &x_L2_ptr);                               CHKERRQ (ierr);
        ierr = VecRestoreArray (prec, &prec_ptr);                               CHKERRQ (ierr);

    }

    ierr = VecDestroy (&x_L2);                                                  CHKERRQ (ierr);

    PetscFunctionReturn (0);
}


PetscErrorCode TumorSolverInterface::solveInverseReacDiff(Vec prec, Vec d1, Vec d1g) {
  PetscErrorCode ierr = 0;
  PetscFunctionBegin;
  int procid, nprocs;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  double *ptr_pr_rec;
  int np = n_misc_->np_;
  int nk = (n_misc_->reaction_inversion_ || n_misc_->diffusivity_inversion_) ? n_misc_->nk_ : 0;
  int nr = (n_misc_->reaction_inversion_) ? n_misc_->nr_ : 0;

  // ------------- L2 solve to solve for p_i, given sparsity -------------------
  ierr = VecGetArray(prec, &ptr_pr_rec);                                        CHKERRQ (ierr);
  // set initial guess for k_inv (possibly != zero)
  if (n_misc_->diffusivity_inversion_) {
      ptr_pr_rec[np] = n_misc_->k_;
      if (nk > 1) ptr_pr_rec[np+1] = n_misc_->k_ * n_misc_->k_gm_wm_ratio_;
      if (nk > 2) ptr_pr_rec[np+2] = n_misc_->k_ * n_misc_->k_glm_wm_ratio_;
  } else {
      // set diff ops with this guess -- this will not change during the solve
      ierr = getTumor()->k_->setValues (n_misc_->k_, n_misc_->k_gm_wm_ratio_, n_misc_->k_glm_wm_ratio_, getTumor()->mat_prop_, n_misc_);  CHKERRQ (ierr);
  }
  // set initial guess for rho (possibly != zero)
  if (n_misc_->reaction_inversion_) {
      ptr_pr_rec[np + nk] = n_misc_->rho_;
      if (nr > 1) ptr_pr_rec[np + nk + 1] = n_misc_->rho_ * n_misc_->r_gm_wm_ratio_;
      if (nr > 2) ptr_pr_rec[np + nk + 2] = n_misc_->rho_ * n_misc_->r_glm_wm_ratio_;
  }
  ierr = VecRestoreArray(prec, &ptr_pr_rec);                                    CHKERRQ (ierr);

  // set data
  inv_solver_->setData (d1);
  if (d1g == nullptr) {d1g = d1;}
  inv_solver_->setDataGradient (d1g);

  ierr = resetTaoSolver();                                                      CHKERRQ (ierr);
  ierr = setParams (prec, nullptr);                                             CHKERRQ (ierr);

  PCOUT << "\n\n\n-------------------------------------------------------------------- Initial L2 solve -------------------------------------------------------------------- " << std::endl;
  PCOUT << "-------------------------------------------------------------------- -------------------- -------------------------------------------------------------------- " << std::endl;

  //	    bool temp_flag = n_misc_->diffusivity_inversion_;
  //	    n_misc_->diffusivity_inversion_ = (temp_flag == true) ? temp_flag : true;
  ierr = inv_solver_->solve ();       // L2 solver
  ierr = VecCopy (inv_solver_->getPrec(), prec);                                               CHKERRQ (ierr);
  //	    n_misc_->diffusivity_inversion_ = temp_flag;

  // ---------------------------------------------------------------------------

  if (n_misc_->reaction_inversion_) {
    ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
    ierr = tuMSG    ("### rho/kappa inversion with scaled L2 solution guess ###"); CHKERRQ (ierr);
    ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);

    n_misc_->flag_reaction_inv_ = true; // invert only for reaction and diffusion now

    // scale p to one according to our modeling assumptions
    double ic_max, g_norm_ref;
    ic_max = 0;
    // get c0
    // ierr = getTumor()->phi_->apply (getTumor()->c_0_, x_L2);
    ierr = VecMax (getTumor()->c_0_, NULL, &ic_max);                          CHKERRQ (ierr);  // max of IC
    ierr = VecGetArray (prec, &ptr_pr_rec);                                   CHKERRQ (ierr);
    for (int i = 0; i < np; i++){
        if(n_misc_->multilevel_) {
          // scales INT_Omega phi(x) dx = const across levels, factor in between levels: 2
          // scales nx=256 to max {Phi p} = 1, nx=128 to max {Phi p} = 0.5, nx=64 to max {Phi p} = 0.25
          ptr_pr_rec[i] *= (1.0/4.0 * n_misc_->n_[0]/64.  / ic_max);
        } else {
          ptr_pr_rec[i] *= (1.0 / ic_max);
        }
      }
    ierr = VecRestoreArray (prec, &ptr_pr_rec);                       CHKERRQ (ierr);

    ierr = tuMSG    ("### l2 scaled-solution guess with incorrect reaction coefficient ###"); CHKERRQ (ierr);
    if (procid == 0) {
        ierr = VecView (prec, PETSC_VIEWER_STDOUT_SELF);            CHKERRQ (ierr);
    }
    ierr = tuMSGstd ("### ------------------------------------------------- ###"); CHKERRQ (ierr);
    // write out p vector after IC, k inversion (scaled)
    if (n_misc_->write_p_checkpoint_) {
      writeCheckpoint(prec, getTumor()->phi_, n_misc_->writepath_ .str(), std::string("scaled"));
    }

    // set data
    inv_solver_->setData (d1);
    if (d1g == nullptr) {d1g = d1;}
    inv_solver_->setDataGradient (d1g);

    // reaction-inversion solve
    ierr = inv_solver_->solveInverseReacDiff (prec);          // With IC as current guess
    ierr = VecCopy (inv_solver_->getPrec(), prec);                            CHKERRQ (ierr);
  } else {
    ierr = tuMSGwarn ( " WARNING: Attempting to solve for reaction and diffusion, but reaction inversion is nor enabled."); CHKERRQ (ierr);
  }

  ierr = VecGetArray (prec, &ptr_pr_rec);                                   CHKERRQ (ierr);
  double r1, r2, r3, k1, k2, k3;
  r1 = ptr_pr_rec[np + nk];
  r2 = (n_misc_->nr_ > 1) ? ptr_pr_rec[np + nk + 1] : 0;
  r3 = (n_misc_->nr_ > 2) ? ptr_pr_rec[np + nk + 2] : 0;

  k1 = ptr_pr_rec[np];
  k2 = (nk > 1) ? ptr_pr_rec[np + 1] : 0;
  k3 = (nk > 2) ? ptr_pr_rec[np + 2] : 0;

  PCOUT << "\nEstimated reaction coefficients " <<  std::endl;
  PCOUT << "r1: " << r1 << std::endl;
  PCOUT << "r2: " << r2 << std::endl;
  PCOUT << "r3: " << r3 << std::endl;

  PCOUT << "\nEstimated diffusion coefficients " <<  std::endl;
  PCOUT << "k1: " << k1 << std::endl;
  PCOUT << "k2: " << k2 << std::endl;
  PCOUT << "k3: " << k3 << std::endl;
  ierr = VecRestoreArray (prec, &ptr_pr_rec);                       CHKERRQ (ierr);

  n_misc_->rho_ = r1;  //update n_misc rho

  PetscFunctionReturn(ierr);
}


PetscErrorCode TumorSolverInterface::computeGradient (Vec dJ, Vec p, Vec data_gradeval) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	TU_assert (initialized_, "TumorSolverInterface::computeGradient(): TumorSolverInterface needs to be initialized.")
    //_tumor->t_history_->Reset();
    // compute gradient for given data 'data_gradeval' and control variable 'p'
    ierr = derivative_operators_->evaluateGradient (dJ, p, data_gradeval);      CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


void TumorSolverInterface::setOptimizerSettings (std::shared_ptr<OptimizerSettings> optset) {
    PetscErrorCode ierr = 0;
    TU_assert (inv_solver_->isInitialized(), "TumorSolverInterface::setOptimizerSettings(): InvSolver needs to be initialized.")
    TU_assert (optset != nullptr,            "TumorSolverInterface::setOptimizerSettings(): requires non-null input.");
    inv_solver_->getOptSettings ()->beta                  = optset->beta;
    inv_solver_->getOptSettings ()->opttolgrad            = optset->opttolgrad;
    inv_solver_->getOptSettings ()->gtolbound             = optset->gtolbound;
    inv_solver_->getOptSettings ()->grtol                 = optset->grtol;
    inv_solver_->getOptSettings ()->gatol                 = optset->gatol;
    inv_solver_->getOptSettings ()->newton_maxit          = optset->newton_maxit;
    inv_solver_->getOptSettings ()->krylov_maxit          = optset->krylov_maxit;
    inv_solver_->getOptSettings ()->gist_maxit            = optset->gist_maxit;
    inv_solver_->getOptSettings ()->iterbound             = optset->iterbound;
    inv_solver_->getOptSettings ()->fseqtype              = optset->fseqtype;
    inv_solver_->getOptSettings ()->newtonsolver          = optset->newtonsolver;
    inv_solver_->getOptSettings ()->lmvm_set_hessian      = optset->lmvm_set_hessian;
    inv_solver_->getOptSettings ()->verbosity             = optset->verbosity;
    inv_solver_->getOptSettings ()->regularization_norm   = optset->regularization_norm;
    inv_solver_->getOptSettings ()->diffusivity_inversion = optset->diffusivity_inversion;
    inv_solver_->getOptSettings ()->reaction_inversion    = optset->reaction_inversion;
    inv_solver_->getOptSettings ()->k_lb                  = optset->k_lb;
    inv_solver_->getOptSettings ()->k_ub                  = optset->k_ub;
    // overwrite n_misc params
    n_misc_->regularization_norm_                         = optset->regularization_norm;
    n_misc_->diffusivity_inversion_                       = optset->diffusivity_inversion;
    n_misc_->reaction_inversion_                          = optset->reaction_inversion;
    n_misc_->k_lb_                                        = optset->k_lb;
    n_misc_->k_ub_                                        = optset->k_ub;
    optimizer_settings_changed_ = true;
}


PetscErrorCode TumorSolverInterface::resetTaoSolver() {
  PetscErrorCode ierr;
  ierr = inv_solver_->resetTao(n_misc_);                                        CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TumorSolverInterface::setInitialGuess(Vec p) {
  PetscErrorCode ierr;
  TU_assert (p != nullptr,                  "TumorSolverInterface::setInitialGuess(): requires non-null input.");
  ierr = VecCopy (p, tumor_->p_);                                               CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TumorSolverInterface::setInitialGuess (double d) {
  PetscErrorCode ierr;
  ierr = VecSet (tumor_->p_, d);                                                CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TumorSolverInterface::updateTumorCoefficients (Vec wm, Vec gm, Vec glm, Vec csf, Vec filter, std::shared_ptr<TumorSettings> tumor_params, bool use_nmisc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert(initialized_,      "TumorSolverInterface::updateTumorCoefficients(): TumorSolverInterface needs to be initialized.")
    // timing
    Event e("update-tumor-coefficients");
    std::array<double, 7> t = {0}; double self_exec_time = -MPI_Wtime ();

    if (!use_nmisc) {
        // update matprob, deep copy of probability maps
    		if(wm != nullptr)      { ierr = VecCopy (wm, tumor_->mat_prop_->wm_);   CHKERRQ(ierr); }
    		else                   { ierr = VecSet (tumor_->mat_prop_->wm_, 0.0);   CHKERRQ(ierr); }
    		if(gm != nullptr)      { ierr = VecCopy (gm, tumor_->mat_prop_->gm_);   CHKERRQ(ierr); }
    		else                   { ierr = VecSet (tumor_->mat_prop_->gm_, 0.0);   CHKERRQ(ierr); }
    		if(csf != nullptr)     { ierr = VecCopy (csf, tumor_->mat_prop_->csf_); CHKERRQ(ierr); }
    		else                   { ierr = VecSet (tumor_->mat_prop_->csf_, 0.0);  CHKERRQ(ierr); }
    		if(glm != nullptr)     { ierr = VecCopy (gm, tumor_->mat_prop_->glm_);  CHKERRQ(ierr); }
    		else                   { ierr = VecSet (tumor_->mat_prop_->glm_, 0.0);  CHKERRQ(ierr); }
    		if(filter != nullptr)  { ierr = VecCopy (filter, tumor_->mat_prop_->filter_); CHKERRQ(ierr); }
    		else                   { ierr = VecSet (tumor_->mat_prop_->filter_, 0.0);     CHKERRQ(ierr); }

        // don't apply k_i values coming from outside if we invert for diffusivity
        if(n_misc_->diffusivity_inversion_) {
          tumor_params->diffusion_ratio_gm_wm  = tumor_->k_->k_gm_wm_ratio_;
          tumor_params->diffusion_ratio_glm_wm = tumor_->k_->k_glm_wm_ratio_;
          tumor_params->diff_coeff_scale       = tumor_->k_->k_scale_;
        }

        // update diffusion coefficient
        tumor_->k_->setValues (tumor_params->diff_coeff_scale, tumor_params->diffusion_ratio_gm_wm, tumor_params->diffusion_ratio_glm_wm,
                                tumor_->mat_prop_, n_misc_);                    CHKERRQ (ierr);
        // update reaction coefficient
        tumor_->rho_->setValues (tumor_params->reaction_coeff_scale, tumor_params->reaction_ratio_gm_wm, tumor_params->reaction_ratio_glm_wm,
                                tumor_->mat_prop_, n_misc_);                    CHKERRQ (ierr);

        // update the phi values, i.e., update the filter
        tumor_->phi_->setValues (tumor_->mat_prop_);
        // need to update prefactors for diffusion KSP preconditioner, as k changed
        pde_operators_->diff_solver_->precFactor();

    } else { //Use n_misc to update tumor coefficients. Needed if matprop is changed from tumor solver application
        // update diffusion coefficient
        ierr = tumor_->k_->setValues (n_misc_->k_, n_misc_->k_gm_wm_ratio_, n_misc_->k_glm_wm_ratio_, tumor_->mat_prop_, n_misc_);
        ierr = tumor_->rho_->setValues (n_misc_->rho_, n_misc_->r_gm_wm_ratio_, n_misc_->r_glm_wm_ratio_, tumor_->mat_prop_, n_misc_);

        // update the phi values, i.e., update the filter
        tumor_->phi_->setValues (tumor_->mat_prop_);
        // need to update prefactors for diffusion KSP preconditioner, as k changed
        pde_operators_->diff_solver_->precFactor();
    }

    // timing
    self_exec_time += MPI_Wtime ();
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (0);
}


PetscErrorCode TumorSolverInterface::solveInverseCoSaMp (Vec prec, Vec d1, Vec d1g) {
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  int procid, nprocs;
  std::stringstream ss;
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procid);

  // set target data for inversion (just sets the vector, no deep copy)
  inv_solver_->setData (d1); if (d1g == nullptr) d1g = d1;
  inv_solver_->setDataGradient (d1g);

  // count the number of observed voxels
  int sum = 0, global_sum = 0;
  double *pixel_ptr;
  ierr = VecGetArray (d1, &pixel_ptr);                                          CHKERRQ (ierr);
  for (int i = 0; i < n_misc_->n_local_; i++)
      if (pixel_ptr[i] > n_misc_->obs_threshold_) sum++;
  ierr = VecRestoreArray (d1, &pixel_ptr);                                      CHKERRQ (ierr);
  MPI_Reduce (&sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
  ss << " number of observed voxels: " << global_sum;
  ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); ss.str(""); ss.clear();

  // solve
  inv_solver_->solveInverseCoSaMp();
  // inv_solver_->solveInverseCoSaMpRS();
  ierr = VecCopy (inv_solver_->getPrec(), prec);                                CHKERRQ (ierr);

  PetscFunctionReturn(ierr);
}
