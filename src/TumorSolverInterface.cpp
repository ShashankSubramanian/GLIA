#include "TumorSolverInterface.h"
#include "EventTimings.hpp"

TumorSolverInterface::TumorSolverInterface (std::shared_ptr<NMisc> n_misc) :
initialized_ (false),
optimizer_settings_changed_ (false),
n_misc_ (n_misc),
tumor_ (),
pde_operators_ (),
derivative_operators_ (),
inv_solver_ () {
    PetscErrorCode ierr = 0;
    if (n_misc != nullptr)
        initialize (n_misc);
}

PetscErrorCode TumorSolverInterface::initialize (std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    tumor_ = std::make_shared<Tumor> (n_misc);
    // set up vector p (should also add option to pass a p vec, that is used to initialize tumor)
    Vec p;
    ierr = VecCreate (PETSC_COMM_WORLD, &p);                    CHKERRQ (ierr);
    ierr = VecSetSizes (p, PETSC_DECIDE, n_misc->np_);          CHKERRQ (ierr);
    ierr = VecSetFromOptions (p);                               CHKERRQ (ierr);
    ierr = VecSet (p, n_misc->p_scale_);                        CHKERRQ (ierr);
    ierr = tumor_->initialize (p, n_misc);                      CHKERRQ (ierr);
    // create pde and derivative operators
    if (n_misc->rd_) {
        pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc);
        derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc, tumor_);
    }
    // create tumor inverse solver
    inv_solver_ = std::make_shared<InvSolver> (derivative_operators_, n_misc, tumor_);
    ierr = inv_solver_->initialize (derivative_operators_, n_misc, tumor_);      
    initialized_ = true;
    // cleanup
    ierr = VecDestroy (&p);                                     CHKERRQ (ierr);
    PetscFunctionReturn (0);
}

// TODO: add switch, if we want to copy or take the pointer from incoming and outgoing data
PetscErrorCode TumorSolverInterface::solveForward (Vec cT, Vec c0) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    // set the initial condition
    ierr = VecCopy (c0, tumor_->c_0_);                           CHKERRQ (ierr);
    // solve forward
    ierr = pde_operators_->solveState (0);                       CHKERRQ (ierr);
    // get solution
    ierr = VecCopy (tumor_->c_t_, cT);                           CHKERRQ (ierr);
    PetscFunctionReturn(0);
}
// TODO: add switch, if we want to copy or take the pointer from incoming and outgoing data
PetscErrorCode TumorSolverInterface::solveInverse (Vec prec, Vec d1, Vec d1g) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert (inv_solver_->isInitialized (), "TumorSolverInterface::setOptimizerSettings(): InvSolver needs to be initialized.")
    if (!optimizer_settings_changed_) {
        ierr = tuMSGwarn (" Tumor inverse solver running with default settings.");              CHKERRQ (ierr);
    }
    // set target data for inversion (just sets the vector, no deep copy)
    inv_solver_->setData(d1);
    if (d1g == nullptr)
        d1g = d1;
    inv_solver_->setDataGradient(d1g);
    // solve
    ierr = inv_solver_->solve ();
    // pass the reconstructed p vector to the caller (deep copy)
    ierr= VecCopy (inv_solver_->getPrec(), prec);                                               CHKERRQ (ierr);
    PetscFunctionReturn (0);
}

PetscErrorCode TumorSolverInterface::computeGradient (Vec dJ, Vec p, Vec data_gradeval) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	TU_assert (initialized_, "TumorSolverInterface::computeGradient(): TumorSolverInterface needs to be initialized.")
    //_tumor->t_history_->Reset();
    // compute gradient for given data 'data_gradeval' and control variable 'p'
    ierr = derivative_operators_->evaluateGradient (dJ, p, data_gradeval); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

void TumorSolverInterface::setOptimizerSettings (std::shared_ptr<OptimizerSettings> optset) {
    PetscErrorCode ierr = 0;
    TU_assert (inv_solver_->isInitialized(), "TumorSolverInterface::setOptimizerSettings(): InvSolver needs to be initialized.")
    TU_assert (optset != nullptr,            "TumorSolverInterface::setOptimizerSettings(): requires non-null input.");
    inv_solver_->getOptSettings ()->beta          = optset->beta;
    inv_solver_->getOptSettings ()->opttolgrad    = optset->opttolgrad;
    inv_solver_->getOptSettings ()->gtolbound     = optset->gtolbound;
    inv_solver_->getOptSettings ()->grtol         = optset->grtol;
    inv_solver_->getOptSettings ()->gatol         = optset->gatol;
    inv_solver_->getOptSettings ()->newton_maxit  = optset->newton_maxit;
    inv_solver_->getOptSettings ()->krylov_maxit  = optset->krylov_maxit;
    inv_solver_->getOptSettings ()->iterbound     = optset->iterbound;
    inv_solver_->getOptSettings ()->fseqtype      = optset->fseqtype;
    inv_solver_->getOptSettings ()->verbosity     = optset->verbosity;
    optimizer_settings_changed_ = true;
}

PetscErrorCode TumorSolverInterface::setInitialGuess(Vec p) {
  PetscErrorCode ierr;
  TU_assert (p != nullptr,                  "TumorSolverInterface::setInitialGuess(): requires non-null input.");
  ierr = VecCopy (p, tumor_->p_); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TumorSolverInterface::updateTumorCoefficients (Vec wm, Vec gm, Vec glm, Vec csf, Vec filter, std::shared_ptr<TumorParameters> tumor_params) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    TU_assert(initialized_,      "TumorSolverInterface::updateTumorCoefficients(): TumorSolverInterface needs to be initialized.")
    TU_assert(wm != nullptr,     "TumorSolverInterface::updateTumorCoefficients(): WM needs to be non-null.");
    TU_assert(gm != nullptr,     "TumorSolverInterface::updateTumorCoefficients(): GM needs to be non-null.");
    TU_assert(csf != nullptr,    "TumorSolverInterface::updateTumorCoefficients(): CSF needs to be non-null.");
    TU_assert(glm != nullptr,    "TumorSolverInterface::updateTumorCoefficients(): GLM needs to be non-null.");
    TU_assert(filter != nullptr, "TumorSolverInterface::updateTumorCoefficients(): Filter needs to be non-null.");
    // timing
    Event e("update-tumor-coefficients");
    std::array<double, 7> t = {0}; double self_exec_time = -MPI_Wtime ();
    /*
    tumor_->k_->k_scale = tumor_params->diff_coeff_scale;
    //tumor_->k_->kf_scale_ = tumor_params->diff_coefficient_anisotropic;
    tumor_->rho_ = tumor_params->reaction_coeff_scale;
    tumor_->diff_ratio_ = tumor_params->diffusion_ratio;
    tumor_->reac_ratio_ = tumor_params->reaction_ratio;
    */
    // update matprob, deep copy of probability maps
		if(wm != nullptr)      { ierr = VecCopy (wm, tumor_->mat_prop_->wm_);         CHKERRQ(ierr); }
		else                   { ierr = VecSet (tumor_->mat_prop_->wm_, 0.0);         CHKERRQ(ierr); }
		if(gm != nullptr)      { ierr = VecCopy (gm, tumor_->mat_prop_->gm_);         CHKERRQ(ierr); }
		else                   { ierr = VecSet (tumor_->mat_prop_->gm_, 0.0);         CHKERRQ(ierr); }
		if(csf != nullptr)     { ierr = VecCopy (csf, tumor_->mat_prop_->csf_);       CHKERRQ(ierr); }
		else                   { ierr = VecSet (tumor_->mat_prop_->csf_, 0.0);        CHKERRQ(ierr); }
		if(glm != nullptr)     { ierr = VecCopy (gm, tumor_->mat_prop_->glm_);        CHKERRQ(ierr); }
		else                   { ierr = VecSet (tumor_->mat_prop_->glm_, 0.0);        CHKERRQ(ierr); }
		if(filter != nullptr)  { ierr = VecCopy (filter, tumor_->mat_prop_->filter_); CHKERRQ(ierr); }
		else                   { ierr = VecSet (tumor_->mat_prop_->filter_, 0.0);     CHKERRQ(ierr); }
    // update diffusion coefficient
    tumor_->k_->setValues (tumor_params->diff_coeff_scale, tumor_params->diffusion_ratio_gm_wm, tumor_params->diffusion_ratio_glm_wm,
                            tumor_->mat_prop_, n_misc_);                                                    CHKERRQ (ierr);
    // update reaction coefficient
    tumor_->rho_->setValues (tumor_params->reaction_coeff_scale, tumor_params->reaction_ratio_gm_wm, tumor_params->reaction_ratio_glm_wm,
                            tumor_->mat_prop_, n_misc_);                                                    CHKERRQ (ierr);
    // update mesh of Gaussians, new phi spacing, center, sigma
    tumor_->phi_->setValues (tumor_params->phi_center_of_mass, tumor_params->phi_sigma, tumor_params->phi_spacing_factor,
                            tumor_->mat_prop_, n_misc_);                                                    CHKERRQ (ierr);

    // timing
    self_exec_time += MPI_Wtime ();
    t[5] = self_exec_time;
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (0);
}
