#include "TumorSolverInterface.h"
#include "EventTimings.hpp"

struct InterpolationContext {
    InterpolationContext (std::shared_ptr<NMisc> n_misc) : n_misc_ (n_misc) {
        PetscErrorCode ierr = 0;
        ierr = VecCreate (PETSC_COMM_WORLD, &temp_);
        ierr = VecSetSizes (temp_, n_misc->n_local_, n_misc->n_global_);
        ierr = VecSetFromOptions (temp_);
        ierr = VecSet (temp_, 0);
    }
    std::shared_ptr<Tumor> tumor_;
    std::shared_ptr<NMisc> n_misc_;
    Vec temp_;
    ~InterpolationContext () {
        PetscErrorCode ierr = 0;
        ierr = VecDestroy (&temp_);
    }
};

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
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &p);     CHKERRQ (ierr);
    #else
        ierr = VecCreate (PETSC_COMM_WORLD, &p);                    CHKERRQ (ierr);
        ierr = VecSetSizes (p, PETSC_DECIDE, n_misc->np_);          CHKERRQ (ierr);
        ierr = VecSetFromOptions (p);                               CHKERRQ (ierr);
    #endif

    ierr = VecSet (p, n_misc->p_scale_);                        CHKERRQ (ierr);
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
    // create tumor inverse solver
    inv_solver_ = std::make_shared<InvSolver> (derivative_operators_, n_misc, tumor_);
    ierr = inv_solver_->initialize (derivative_operators_, n_misc, tumor_);
    initialized_ = true;
    // cleanup
    ierr = VecDestroy (&p);                                     CHKERRQ (ierr);
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
        n_misc_->nk_                    = tumor_params->nk;
    }
    // ++ re-initialize Tumor ++
    ierr = tumor_->setParams (p, n_misc_, npchanged);                           CHKERRQ (ierr);
    // ++ re-initialize pdeoperators and derivativeoperators ++ if either tumor model or np or nt changed
    // invcludes re-allocating time history for adjoint,
    if (n_misc_->model_ == 1 && (modelchanged || npchanged || ntchanged)) {
      pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc_);
      derivative_operators_ = std::make_shared<DerivativeOperatorsRD> (pde_operators_, n_misc_, tumor_);
    }
    if (n_misc_->model_ == 2 && (modelchanged || npchanged || ntchanged)) {
      pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc_);
      derivative_operators_ = std::make_shared<DerivativeOperatorsPos> (pde_operators_, n_misc_, tumor_);
    }
    if (n_misc_->model_ == 3 && (modelchanged || npchanged || ntchanged)) {
      pde_operators_ = std::make_shared<PdeOperatorsRD> (tumor_, n_misc_);
      derivative_operators_ = std::make_shared<DerivativeOperatorsRDObj> (pde_operators_, n_misc_, tumor_);
    }
    // ++ re-initialize InvSolver ++, i.e. H matrix, p_rec vectores etc..
    inv_solver_->setParams(derivative_operators_, n_misc_, tumor_, npchanged);   CHKERRQ (ierr);

    PetscFunctionReturn(0);
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

    // set the observation operator filter : default filter
    ierr = tumor_->obs_->setDefaultFilter (d1);

    // set target data for inversion (just sets the vector, no deep copy)
    inv_solver_->setData (d1);
    if (d1g == nullptr)
        d1g = d1;
    inv_solver_->setDataGradient (d1g);

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


PetscErrorCode phiMult (Mat A, Vec x, Vec y) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    //A = phiT * OT * O * phi
    InterpolationContext *ctx;
    ierr = MatShellGetContext (A, &ctx);                                CHKERRQ (ierr);
    ierr = ctx->tumor_->phi_->apply (ctx->temp_, x);                    CHKERRQ (ierr);
    ierr = ctx->tumor_->obs_->apply (ctx->temp_, ctx->temp_);           CHKERRQ (ierr);
    ierr = ctx->tumor_->obs_->apply (ctx->temp_, ctx->temp_);           CHKERRQ (ierr);
    ierr = ctx->tumor_->phi_->applyTranspose (y, ctx->temp_);           CHKERRQ (ierr);

    // Regularization
    double beta = 1e-3;
    ierr = VecAXPY (y, beta, x);                                        CHKERRQ (ierr);

    PetscFunctionReturn (0);
}

PetscErrorCode interpolationKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec x; int maxit; PetscScalar divtol, abstol, reltol;
    ierr = KSPBuildSolution (ksp,NULL,&x);
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);           CHKERRQ(ierr);                                                             CHKERRQ(ierr);
    InterpolationContext *itctx = reinterpret_cast<InterpolationContext*>(ptr);     // get user context

    std::stringstream s;
    if (its == 0) {
      s << std::setw(3)  << " PCG:" << " computing solution of interpolation system (tol="
        << std::scientific << std::setprecision(5) << reltol << ")";
      ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
      s.str (""); s.clear ();
    }
    s << std::setw(3)  << " PCG:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
    << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
    ierr = tuMSGstd (s.str());                                                  CHKERRQ(ierr);
    s.str (""); s.clear ();

    PetscFunctionReturn (0);
}

PetscErrorCode TumorSolverInterface::solveInterpolation (Vec data, Vec p_out, std::shared_ptr<Phi> phi, std::shared_ptr<NMisc> n_misc) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PCOUT << " -------  Interpolation begin ---------" << std::endl;

    KSP ksp;
    Mat A;
    Vec p, rhs;
    Vec phi_p;

    int np = n_misc_->np_;
    int nk = (n_misc_->diffusivity_inversion_) ? n_misc_->nk_ : 0;


    ierr = VecDuplicate (data, &phi_p);                                 CHKERRQ (ierr);
    ierr = VecDuplicate (p_out, &p);                                    CHKERRQ (ierr);
    ierr = VecDuplicate (p_out, &rhs);                                  CHKERRQ (ierr);
    ierr = VecSet (rhs, 0);                                             CHKERRQ (ierr);
    ierr = VecSet (p_out, 0);                                           CHKERRQ (ierr);
    ierr = VecSet (p, 0);                                               CHKERRQ (ierr);

    std::shared_ptr<Tumor> tumor = tumor_;
    std::shared_ptr<InterpolationContext> ctx = std::make_shared<InterpolationContext> (n_misc_);
    ctx->tumor_ = tumor;

    ierr = KSPCreate (PETSC_COMM_SELF, &ksp);                          CHKERRQ (ierr);
    ierr = MatCreateShell (PETSC_COMM_SELF, np + nk, np + nk, np + nk, np + nk, ctx.get(), &A);      CHKERRQ (ierr);
    ierr = MatShellSetOperation (A, MATOP_MULT, (void (*) (void))phiMult);                                              CHKERRQ (ierr);
    ierr = KSPSetOperators (ksp, A, A);                                 CHKERRQ (ierr);
    ierr = KSPSetTolerances (ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, 5000);                                            CHKERRQ (ierr);   //Max iter is 100
    ierr = KSPSetType (ksp, KSPCG);                                     CHKERRQ (ierr);
    ierr = KSPSetOptionsPrefix (ksp, "phieq_");                         CHKERRQ (ierr);
    ierr = KSPSetComputeSingularValues(ksp, PETSC_TRUE);                CHKERRQ (ierr);  // To compute the condition number
    ierr = KSPSetFromOptions (ksp);                                     CHKERRQ (ierr);
    ierr = KSPSetUp (ksp);                                              CHKERRQ (ierr);
    ierr = KSPMonitorSet (ksp, interpolationKSPMonitor, ctx.get(), 0);  CHKERRQ (ierr);

    //RHS -- phiT * OT * O * d
    ierr = tumor->obs_->apply (ctx->temp_, data);                       CHKERRQ (ierr);
    ierr = tumor->obs_->apply (ctx->temp_, ctx->temp_);                 CHKERRQ (ierr);
    ierr = tumor->phi_->applyTranspose (rhs, ctx->temp_);               CHKERRQ (ierr);

    ierr = KSPSolve (ksp, rhs, p);                                      CHKERRQ (ierr);
    // Compute extreme singular values for condition number
    double e_max, e_min;
    ierr = KSPComputeExtremeSingularValues (ksp, &e_max, &e_min);       CHKERRQ (ierr);
    PCOUT << "Condition number of PhiTPhi is: " << e_max / e_min << " | largest singular values is: " << e_max << ", smallest singular values is: " << e_min << std::endl;
    ierr = VecCopy (p, p_out);                                          CHKERRQ (ierr);

    //Compute reconstruction error
    double error_norm, p_norm, d_norm;
    ierr = tumor->phi_->apply (phi_p, p);                               CHKERRQ (ierr);
    ierr = tumor->obs_->apply (phi_p, phi_p);                           CHKERRQ (ierr);
    dataOut (phi_p, ctx->n_misc_, "CInterp.nc");

    ierr = tumor->obs_->apply (ctx->temp_, data);                       CHKERRQ (ierr);
    ierr = VecNorm (ctx->temp_, NORM_2, &d_norm);                       CHKERRQ (ierr);
    ierr = VecAXPY (phi_p, -1.0, ctx->temp_);                           CHKERRQ (ierr);
    ierr = VecNorm (phi_p, NORM_2, &error_norm);                        CHKERRQ (ierr);
    ierr = VecNorm (p_out, NORM_2, &p_norm);                            CHKERRQ (ierr);

    double err_p0;
    Vec p_zero;
    ierr = VecDuplicate (p_out, &p_zero);                               CHKERRQ (ierr);
    ierr = VecSet (p_zero, 0);                                          CHKERRQ (ierr);

    ierr = tumor->phi_->apply (phi_p, p_zero);                          CHKERRQ (ierr);
    ierr = tumor->obs_->apply (phi_p, phi_p);                           CHKERRQ (ierr);
    ierr = VecAXPY (phi_p, -1.0, ctx->temp_);                           CHKERRQ (ierr);
    ierr = VecNorm (phi_p, NORM_2, &err_p0);                            CHKERRQ (ierr);

    PCOUT << "Data mismatch using interpolated basis is: " << error_norm << std::endl;
    PCOUT << "Data mismatch using zero vector is: " << err_p0 << std::endl;
    PCOUT << "Rel error in interpolation reconstruction: " << error_norm / d_norm << std::endl;
    PCOUT << "Norm of reconstructed p: " << p_norm << std::endl;
    PCOUT << " -------  Interpolation end ---------" << std::endl;

    // std::ofstream outfile;
    // outfile.open ("./interperror.dat", std::ios_base::app);
    // if (procid == 0)
    //     outfile << n_misc->phi_sigma_ / (2 * M_PI / 64) << " " << error_norm / d_norm << " " <<  e_max / e_min << std::endl;
    // outfile.close ();

    ierr = VecDestroy (&p_zero);                                        CHKERRQ (ierr);
    ierr = VecDestroy (&p);                                             CHKERRQ (ierr);
    ierr = VecDestroy (&rhs);                                           CHKERRQ (ierr);
    ierr = VecDestroy (&phi_p);                                         CHKERRQ (ierr);
    ierr = MatDestroy (&A);                                             CHKERRQ (ierr);
    ierr = KSPDestroy (&ksp);                                           CHKERRQ (ierr);


    PetscFunctionReturn (0);
}


void TumorSolverInterface::setOptimizerSettings (std::shared_ptr<OptimizerSettings> optset) {
    PetscErrorCode ierr = 0;
    TU_assert (inv_solver_->isInitialized(), "TumorSolverInterface::setOptimizerSettings(): InvSolver needs to be initialized.")
    TU_assert (optset != nullptr,            "TumorSolverInterface::setOptimizerSettings(): requires non-null input.");
    inv_solver_->getOptSettings ()->beta                = optset->beta;
    inv_solver_->getOptSettings ()->opttolgrad          = optset->opttolgrad;
    inv_solver_->getOptSettings ()->gtolbound           = optset->gtolbound;
    inv_solver_->getOptSettings ()->grtol               = optset->grtol;
    inv_solver_->getOptSettings ()->gatol               = optset->gatol;
    inv_solver_->getOptSettings ()->newton_maxit        = optset->newton_maxit;
    inv_solver_->getOptSettings ()->krylov_maxit        = optset->krylov_maxit;
    inv_solver_->getOptSettings ()->gist_maxit          = optset->gist_maxit;
    inv_solver_->getOptSettings ()->iterbound           = optset->iterbound;
    inv_solver_->getOptSettings ()->fseqtype            = optset->fseqtype;
    inv_solver_->getOptSettings ()->newtonsolver        = optset->newtonsolver;
    inv_solver_->getOptSettings ()->lmvm_set_hessian    = optset->lmvm_set_hessian;
    inv_solver_->getOptSettings ()->verbosity           = optset->verbosity;
    inv_solver_->getOptSettings ()->regularization_norm = optset->regularization_norm;
    // overwrite n_misc params
    n_misc_->regularization_norm_                       = optset->regularization_norm;
    n_misc_->diffusivity_inversion_                     = optset->diffusivity_inversion;
    optimizer_settings_changed_ = true;
}


PetscErrorCode TumorSolverInterface::resetTaoSolver() {
  PetscErrorCode ierr;
  ierr = inv_solver_->resetTao(n_misc_); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TumorSolverInterface::setInitialGuess(Vec p) {
  PetscErrorCode ierr;
  TU_assert (p != nullptr,                  "TumorSolverInterface::setInitialGuess(): requires non-null input.");
  ierr = VecCopy (p, tumor_->p_); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TumorSolverInterface::setInitialGuess (double d) {
  PetscErrorCode ierr;
  ierr = VecSet (tumor_->p_, d); CHKERRQ(ierr);
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

        // don't apply k_i values coming from outside if we invert for diffusivity
        if(n_misc_->diffusivity_inversion_) {
          tumor_params->diffusion_ratio_gm_wm  = tumor_->k_->k_gm_wm_ratio_;
          tumor_params->diffusion_ratio_glm_wm = tumor_->k_->k_glm_wm_ratio_;
          tumor_params->diff_coeff_scale       = tumor_->k_->k_scale_;
        }

        // update diffusion coefficient
        tumor_->k_->setValues (tumor_params->diff_coeff_scale, tumor_params->diffusion_ratio_gm_wm, tumor_params->diffusion_ratio_glm_wm,
                                tumor_->mat_prop_, n_misc_);                          CHKERRQ (ierr);
        // update reaction coefficient
        tumor_->rho_->setValues (tumor_params->reaction_coeff_scale, tumor_params->reaction_ratio_gm_wm, tumor_params->reaction_ratio_glm_wm,
                                tumor_->mat_prop_, n_misc_);                          CHKERRQ (ierr);

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

PetscErrorCode TumorSolverInterface::printStatistics (int its, PetscReal J, PetscReal J_rel, PetscReal g_norm, PetscReal p_rel_norm, Vec x_L1) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    /* -------------------------------------------------------------------- PRINT -------------------------------------------------------------------- */


    std::stringstream s;

    s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
    << std::setw(18) << "||objective||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
    << "   "  << std::setw(18) << "||dp||_rel"
    << std::setw(18) << "k";
        
    ierr = tuMSGstd ("-------------------------------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
    ierr = tuMSGstd ("-------------------------------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
    s.str ("");
    s.clear ();

    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
    << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
    << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J_rel
    << "   " << std::scientific << std::setprecision(12) << std::setw(18) << g_norm
    << "   " << std::scientific << std::setprecision(12) << std::setw(18) << p_rel_norm;

    double *x_ptr;

    if (n_misc_->diffusivity_inversion_) {
        ierr = VecGetArray(x_L1, &x_ptr);                                         CHKERRQ(ierr);
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[n_misc_->np_];
        if (n_misc_->nk_ > 1) {
            s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[n_misc_->np_ + 1]; 
        }
        ierr = VecRestoreArray(x_L1, &x_ptr);                                     CHKERRQ(ierr);
    } else {
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << "0";
    }
      
    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");
    s.clear ();

    PetscFunctionReturn (0);
    /* -------------------------------------------------------------------- PRINT -------------------------------------------------------------------- */
}

// L1 solve is done outside InvSolver because the corrective L2 solve needs to be done with tao
PetscErrorCode TumorSolverInterface::solveInverseCoSaMp (Vec prec, Vec d1, Vec d1g) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    // No regularization for L1 constrainied optimization
    n_misc_->beta_ = 0;

    // set the observation operator filter : default filter
    ierr = tumor_->obs_->setDefaultFilter (d1);

    // set target data for inversion (just sets the vector, no deep copy)
    inv_solver_->setData (d1);
    if (d1g == nullptr)
        d1g = d1;
    inv_solver_->setDataGradient (d1g);

    // solve
    Vec g, g_ref;              // Holds the gradient and reference gradient
    Vec temp;                  // Temp vector 
    PetscReal J, J_ref, J_old;        // Objective
    Vec x_L1, x_L1_old;                  // Holds the L1 solution and the previous guess
    // Tolerance for L1 solver 
    double ftol = inv_solver_->getOptSettings()->ftol;                                                                                                                            
    double *x_L2_ptr, *x_L1_ptr, *temp_ptr;
    double norm_rel, norm, norm_g;
    std::vector<int> idx;  // Holds the idx list after
    int np, np_original, nk;
    Vec x_L2;  
    np_original = n_misc_->np_;                         // Keeps track of the original number of gaussians

    ierr = VecDuplicate (tumor_->p_, &g);           CHKERRQ (ierr);
    ierr = VecDuplicate (tumor_->p_, &g_ref);       CHKERRQ (ierr);
    ierr = VecDuplicate (tumor_->p_, &x_L1);        CHKERRQ (ierr);
    ierr = VecDuplicate (tumor_->p_, &x_L1_old);    CHKERRQ (ierr);
    ierr = VecDuplicate (tumor_->p_, &temp);        CHKERRQ (ierr);
    ierr = VecSet (g, 0);                           CHKERRQ (ierr);
    ierr = VecSet (g_ref, 0);                       CHKERRQ (ierr);
    ierr = VecSet (x_L1, 0);                        CHKERRQ (ierr);  // Initial guess for L1 solver is zero
    ierr = VecSet (x_L1_old, 0);                    CHKERRQ (ierr);
    ierr = VecSet (temp, 0);                        CHKERRQ (ierr);

    // Compute reference quantities
    ierr = inv_solver_->getObjective (x_L1, &J_ref);   
    ierr = inv_solver_->getGradient (x_L1, g_ref);


    J = J_ref;
    ierr = VecCopy (g_ref, g);                      CHKERRQ (ierr);
    ierr = VecNorm (g, NORM_2, &norm_g);              CHKERRQ (ierr);

    int its = 0;

    // print statistics
    printStatistics (its, J_ref, 1, norm_g, 1, x_L1);

    // Solver begin
    while (true) {
        its++;
        if (its > inv_solver_->getOptSettings()->gist_maxit) { // max iter reached
            PCOUT << "Max L1 iter reached" << std::endl;
            break;
        }

        /* -------------------------------------------------------------------- 1) Hard threshold neg gradient   --------------------------------------------------------------------  */
        ierr = VecCopy (g, temp);                               CHKERRQ (ierr);
        ierr = VecScale (temp, -1.0);                           CHKERRQ (ierr);
        ierr = hardThreshold (temp, 2 * n_misc_->sparsity_level_, np_original, idx);

        /* -------------------------------------------------------------------- 2) Update the prev soln's support with the 2K sparse guess' support -------------------------------------------------------------------- */
        n_misc_->support_.insert (n_misc_->support_.end(), idx.begin(), idx.end());

        // Sort and remove duplicates
        std::sort (n_misc_->support_.begin(), n_misc_->support_.end());
        n_misc_->support_.erase (std::unique (n_misc_->support_.begin(), n_misc_->support_.end()), n_misc_->support_.end());

        PCOUT << "Support for corrective L2 solve : ";
        for (int i = 0; i < n_misc_->support_.size(); i++) {
            PCOUT << n_misc_->support_[i] << " ";
        }
        PCOUT << std::endl;

        /* -------------------------------------------------------------------- 3) Take a Gauss-Newton/Quasi-Newton step -------------------------------------------------------------------- */

        PCOUT << "\n\n\n-------------------------------------------------------------------- Corrective L2 solver -------------------------------------------------------------------- " << std::endl;
        PCOUT << "-------------------------------------------------------------------- -------------------- -------------------------------------------------------------------- " << std::endl;


        np = n_misc_->support_.size();    // Length of vector in the restricted subspace: Note that this is not always 2s as
                                          // the support is merged with the previous support and hence could be larger
        nk = (n_misc_->diffusivity_inversion_) ? n_misc_->nk_ : 0;
        n_misc_->np_ = np;                    // Change np to solve the smaller L2 subsystem
        ierr = VecCreateSeq (PETSC_COMM_SELF, np + nk, &x_L2);                 CHKERRQ (ierr);              // Create the L2 solution vector
        ierr = VecSet (x_L2, 0);                                               CHKERRQ (ierr);
        ierr = VecGetArray (x_L2, &x_L2_ptr);                                  CHKERRQ (ierr);
        ierr = VecGetArray (x_L1, &x_L1_ptr);                                  CHKERRQ (ierr);
        for (int i = 0; i < np; i++) {
            x_L2_ptr[i] = x_L1_ptr[n_misc_->support_[i]];   // Starting guess for L2 solver is prev guess support values
        }

        if (n_misc_->diffusivity_inversion_) {
            // Use the same diffusivity too
            x_L2_ptr[np] = x_L1_ptr[np_original];
            if (nk > 1) x_L2_ptr[np+1] = x_L1_ptr[np_original+1];
        }

        ierr = VecRestoreArray (x_L2, &x_L2_ptr);                              CHKERRQ (ierr);
        ierr = VecRestoreArray (x_L1, &x_L1_ptr);                              CHKERRQ (ierr);
        

        tumor_->phi_->modifyCenters (n_misc_->support_);                // Modifies the centers
        // Reset tao solver explicitly: TODO: Ask Klaudius why inv_solver_->setParams() does not reset tao solver
        ierr = resetTaoSolver();
        ierr = setParams (x_L2, nullptr);                               // Resets the phis and other operators, x_L2 is copied into tumor->p_ and is used as 
                                                                        // IC for the L2 solver. This needs to be done every iteration as the while the number
                                                                        // of basis fns remain the same, their location needs to be constantly updated.

        ierr = inv_solver_->solve ();       // L2 solver
        ierr = VecCopy (inv_solver_->getPrec(), x_L2);                                               CHKERRQ (ierr);

        PCOUT << "--------------------------------------------------------------------     L2 solver end     -------------------------------------------------------------------- " << std::endl;
        PCOUT << "-------------------------------------------------------------------- -------------------- -------------------------------------------------------------------- \n\n\n" << std::endl;

        /* -------------------------------------------------------------------- 4) Updates --------------------------------------------------------------------  */

        ierr = VecCopy (x_L1, x_L1_old);                CHKERRQ (ierr);     // Keep track of the previous L1 guess for convergence checks

        ierr = VecGetArray (x_L2, &x_L2_ptr);                                  CHKERRQ (ierr);
        ierr = VecGetArray (x_L1, &x_L1_ptr);                                  CHKERRQ (ierr);
        for (int i = 0; i < np; i++) {
            x_L1_ptr[n_misc_->support_[i]] = x_L2_ptr[i];   // Correct L1 guess
        }
        // Correct the diffusivity
        if (n_misc_->diffusivity_inversion_) {
            x_L1_ptr[np_original] = x_L2_ptr[np];
            if (nk > 1) x_L1_ptr[np_original+1] = x_L2_ptr[np+1];
        }

        // Hard threshold L1 guess to sparsity level
        idx.clear();
        ierr = hardThreshold (x_L1, n_misc_->sparsity_level_, np_original, idx);

        //clear the support
        n_misc_->support_.clear ();
        // Add the support of the solution. This will be merged with the support of the proxy before the L2 solve
        n_misc_->support_.insert (n_misc_->support_.end(), idx.begin(), idx.end());

        PCOUT << "Support of current solution : ";
        for (int i = 0; i < n_misc_->support_.size(); i++) {
            PCOUT << n_misc_->support_[i] << " ";
        }
        PCOUT << std::endl;

        // Set only idx values in x_L1. Rest are hard thresholded to zero
        ierr = VecSet (temp, 0);                        CHKERRQ (ierr);
        ierr = VecGetArray (temp, &temp_ptr);           CHKERRQ (ierr);
        for (int i = 0; i < idx.size(); i++) {
            temp_ptr[idx[i]] = x_L1_ptr[idx[i]];
        }
        if (n_misc_->diffusivity_inversion_) {
            temp_ptr[np_original] = x_L1_ptr[np_original];
            if (nk > 1) temp_ptr[np_original+1] = x_L1_ptr[np_original+1];
        }

        ierr = VecRestoreArray (x_L2, &x_L2_ptr);                              CHKERRQ (ierr);
        ierr = VecRestoreArray (x_L1, &x_L1_ptr);                              CHKERRQ (ierr);
        ierr = VecRestoreArray (temp, &temp_ptr);                              CHKERRQ (ierr);

        // Copy thresholded vector to current L1 solution
        ierr = VecCopy (temp, x_L1);                    CHKERRQ (ierr);

        // Reset all values to initial space
        np = np_original;
        n_misc_->np_ = np;

        tumor_->phi_->resetCenters ();              // Reset all the basis centers

        ierr = setParams (x_L1, nullptr);           // Reset phis and other operators

        // Reset all data as this is turned to nullptr after every tao solve. TODO: Ask Klaudius why?
        ierr = tumor_->obs_->setDefaultFilter (d1);
        inv_solver_->setData (d1);
        if (d1g == nullptr)
            d1g = d1;
        inv_solver_->setDataGradient (d1g);


        // Destroy the L2 solution vector as its size could potentially change in subsequent iterations
        ierr = VecDestroy (&x_L2);                      CHKERRQ (ierr);


        /* -------------------------------------------------------------------- 5) Convergence check -------------------------------------------------------------------- */

        J_old = J;
        // Compute new objective -- again, this is now only the mismatch term
        ierr = inv_solver_->getObjective (x_L1, &J); 
        ierr = inv_solver_->getGradient (x_L1, g); 
        ierr = VecNorm (x_L1, NORM_INFINITY, &norm);        CHKERRQ (ierr);
        ierr = VecAXPY (temp, -1.0, x_L1_old);              CHKERRQ (ierr);     // temp holds x_L1
        ierr = VecNorm (temp, NORM_INFINITY, &norm_rel);    CHKERRQ (ierr);     // Norm change in the solution
        ierr = VecNorm (g, NORM_2, &norm_g);              CHKERRQ (ierr);
        // print statistics

        PCOUT << "\n\n\n-------------------------------------------------------------------- L1 solver statistics -------------------------------------------------------------------- " << std::endl;
        printStatistics (its, J, PetscAbsReal (J_old - J) / PetscAbsReal (1 + J_ref), norm_g, norm_rel / (1 + norm), x_L1);
        PCOUT << "-------------------------------------------------------------------- -------------------- -------------------------------------------------------------------- \n\n\n" << std::endl;

        if (PetscAbsReal (J_old - J) < ftol * PetscAbsReal (1 + J_ref) && norm_rel < std::sqrt (ftol) * (1 + norm)) {
            PCOUT << "L1 tolerance reached." << std::endl;
            break;
        }  

    }

    // pass the reconstructed p vector to the caller (deep copy)
    ierr= VecCopy (x_L1, prec);                                               CHKERRQ (ierr);

    ierr = VecDestroy (&g);                            CHKERRQ (ierr);
    ierr = VecDestroy (&x_L1);                         CHKERRQ (ierr);
    ierr = VecDestroy (&g_ref);                        CHKERRQ (ierr);
    ierr = VecDestroy (&x_L1_old);                     CHKERRQ (ierr);
    ierr = VecDestroy (&temp);                         CHKERRQ (ierr);

    PetscFunctionReturn (0);
}
