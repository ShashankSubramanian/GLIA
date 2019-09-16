#include "AdvectionSolver.h"

AdvectionSolver::AdvectionSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops) : ctx_ () {
	PetscErrorCode ierr = 0;

    spec_ops_ = spec_ops;

    ctx_ = std::make_shared<CtxAdv> ();
    ctx_->n_misc_ = n_misc;
    ctx_->spec_ops_ = spec_ops_;
    ctx_->dt_ = n_misc->dt_;
    ctx_->temp_.resize (3);
    for (int i = 0; i < 3; i++)
    	ctx_->temp_[i] = tumor->work_[11 - i]; 	// Choose some tumor work vector

    ctx_->velocity_ = tumor->velocity_;
    advection_mode_ = 1;    // 1 -- mass conservation
                            // 2 -- pure advection (csf uses this to allow for leakage etc)
    ctx_->advection_mode_ = advection_mode_;
    trajectoryIsComputed_ = false;
}


TrapezoidalSolver::TrapezoidalSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops) : AdvectionSolver (n_misc, tumor, spec_ops) {
    PetscErrorCode ierr = 0;
    ierr = MatCreateShell (PETSC_COMM_WORLD, n_misc->n_local_, n_misc->n_local_, n_misc->n_global_, n_misc->n_global_, ctx_.get(), &A_);
    ierr = MatShellSetOperation (A_, MATOP_MULT, (void(*)(void)) operatorAdv);

    ierr = KSPCreate (PETSC_COMM_WORLD, &ksp_);
    ierr = KSPSetOperators (ksp_, A_, A_);
    ierr = KSPSetTolerances (ksp_, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    ierr = KSPSetType (ksp_, KSPGMRES);
    ierr = KSPSetFromOptions (ksp_);
    ierr = KSPSetUp (ksp_);

    ierr = VecCreate (PETSC_COMM_WORLD, &rhs_);
    ierr = VecSetSizes (rhs_, n_misc->n_local_, n_misc->n_global_);
    ierr = setupVec (rhs_);
    ierr = VecSet (rhs_, 0);
}


// LHS for transport equation using Crank-Nicolson 
// y = Ax = (x + dt/2 div (xv))
PetscErrorCode operatorAdv (Mat A, Vec x, Vec y) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	Event e ("tumor-adv-ksp-matvec");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    CtxAdv *ctx;
    ierr = MatShellGetContext (A, &ctx);                        CHKERRQ (ierr);

    ScalarType alph = 1.0 / 2.0 * ctx->dt_;

    ierr = VecPointwiseMult (ctx->temp_[0], ctx->velocity_->x_, x);			CHKERRQ (ierr);
    ierr = VecPointwiseMult (ctx->temp_[1], ctx->velocity_->y_, x);			CHKERRQ (ierr);
    ierr = VecPointwiseMult (ctx->temp_[2], ctx->velocity_->z_, x);			CHKERRQ (ierr);

    ctx->spec_ops_->computeDivergence (y, ctx->temp_[0], ctx->temp_[1], ctx->temp_[2], t.data());

    ierr = VecScale (y, alph);									CHKERRQ (ierr);
    ierr = VecAXPY (y, 1.0, x);									CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
	PetscFunctionReturn (ierr);
}

PetscErrorCode TrapezoidalSolver::solve (Vec scalar, std::shared_ptr<VecField> velocity, ScalarType dt) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	Event e ("tumor-adv-solve");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    CtxAdv *ctx;
    ierr = MatShellGetContext (A_, &ctx);                       CHKERRQ (ierr);
    ctx->dt_ = dt;
    ctx->velocity_->x_ = velocity->x_;
    ctx->velocity_->y_ = velocity->y_;
    ctx->velocity_->z_ = velocity->z_;

    ScalarType alph = -1.0 / 2.0 * ctx->dt_;
    //rhs for advection solve: b = scalar - dt/2 div(scalar v)
    ierr = VecPointwiseMult (ctx->temp_[0], velocity->x_, scalar);			CHKERRQ (ierr);
    ierr = VecPointwiseMult (ctx->temp_[1], velocity->y_, scalar);			CHKERRQ (ierr);
    ierr = VecPointwiseMult (ctx->temp_[2], velocity->z_, scalar);			CHKERRQ (ierr);

    spec_ops_->computeDivergence (rhs_, ctx->temp_[0], ctx->temp_[1], ctx->temp_[2], t.data());

    ierr = VecScale (rhs_, alph);									CHKERRQ (ierr);
    ierr = VecAXPY (rhs_, 1.0, scalar);							    CHKERRQ (ierr);
    //KSP solve
    ierr = KSPSolve (ksp_, rhs_, scalar);                            CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
	PetscFunctionReturn (ierr);
}

TrapezoidalSolver::~TrapezoidalSolver () {
    PetscErrorCode ierr = 0;
    ierr = MatDestroy (&A_);
    ierr = KSPDestroy (&ksp_);
    ierr = VecDestroy (&rhs_);
}


ImplicitEulerSolver::ImplicitEulerSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops) : AdvectionSolver (n_misc, tumor, spec_ops) {
    PetscErrorCode ierr = 0;
    ierr = MatCreateShell (PETSC_COMM_WORLD, n_misc->n_local_, n_misc->n_local_, n_misc->n_global_, n_misc->n_global_, ctx_.get(), &A_);
    ierr = MatShellSetOperation (A_, MATOP_MULT, (void(*)(void)) operatorAdv);

    ierr = KSPCreate (PETSC_COMM_WORLD, &ksp_);
    ierr = KSPSetOperators (ksp_, A_, A_);
    ierr = KSPSetTolerances (ksp_, 1e-3, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    ierr = KSPSetType (ksp_, KSPGMRES);
    // ierr = KSPMonitorSet(ksp_, advSolverKSPMonitor, ctx_.get(), 0);    
    ierr = KSPSetFromOptions (ksp_);
    ierr = KSPSetUp (ksp_);

    ierr = VecCreate (PETSC_COMM_WORLD, &rhs_);
    ierr = VecSetSizes (rhs_, n_misc->n_local_, n_misc->n_global_);
    ierr = setupVec (rhs_);
    ierr = VecSet (rhs_, 0);
}

PetscErrorCode advSolverKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec x; int maxit; ScalarType divtol, abstol, reltol;
    ierr = KSPBuildSolution (ksp,NULL,&x);
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);             CHKERRQ(ierr);                                                             CHKERRQ(ierr);
    CtxAdv *ctx = reinterpret_cast<CtxAdv*>(ptr);     // get user context

    std::stringstream s;
    if (its == 0) {
      s << std::setw(3)  << " KSP: computing solution of advection system (tol="
        << std::scientific << std::setprecision(5) << reltol << ")";
      ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
      s.str (""); s.clear ();
    }
    s << std::setw(3)  << " KSP:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
    << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
    ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    s.str (""); s.clear ();

    // int ksp_itr;
    // ierr = KSPGetIterationNumber (ksp, &ksp_itr);                                 CHKERRQ (ierr);
    // ScalarType e_max, e_min;
    // if (ksp_itr % 10 == 0 || ksp_itr == maxit) {
    //   ierr = KSPComputeExtremeSingularValues (ksp, &e_max, &e_min);       CHKERRQ (ierr);
    //   s << "Condition number of matrix is: " << e_max / e_min << " | largest singular values is: " << e_max << ", smallest singular values is: " << e_min << std::endl;
    //   ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    //   s.str (""); s.clear ();
    // }
    PetscFunctionReturn (ierr);
}


// y = Ax = (x + dt div (xv)) or (x + dt delx . v)
PetscErrorCode operatorAdvEuler (Mat A, Vec x, Vec y) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tumor-adv-ksp-matvec");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    CtxAdv *ctx;
    ierr = MatShellGetContext (A, &ctx);                        CHKERRQ (ierr);

    ScalarType alph = ctx->dt_;
    std::bitset<3> XYZ;
    XYZ[0] = 1;
    XYZ[1] = 1;
    XYZ[2] = 1;
    if (ctx->advection_mode_ == 1) {
        // compute (x + dt div(xv))
        ierr = VecPointwiseMult (ctx->temp_[0], ctx->velocity_->x_, x);         CHKERRQ (ierr);
        ierr = VecPointwiseMult (ctx->temp_[1], ctx->velocity_->y_, x);         CHKERRQ (ierr);
        ierr = VecPointwiseMult (ctx->temp_[2], ctx->velocity_->z_, x);         CHKERRQ (ierr);
        ierr = ctx->spec_ops_->computeDivergence (y, ctx->temp_[0], ctx->temp_[1], ctx->temp_[2], t.data());    CHKERRQ (ierr);
        ierr = VecScale (y, alph);                                  CHKERRQ (ierr);
        ierr = VecAXPY (y, 1.0, x);                                 CHKERRQ (ierr);
    } else { //pure advection
        // compute (x + dt * gradx .v)
        ierr = ctx->spec_ops_->computeGradient (ctx->temp_[0], ctx->temp_[1], ctx->temp_[2], x, &XYZ, t.data()); CHKERRQ (ierr);
        ierr = VecPointwiseMult (y, ctx->temp_[0], ctx->velocity_->x_);                             CHKERRQ (ierr);
        ierr = VecPointwiseMult (ctx->temp_[0], ctx->temp_[1], ctx->velocity_->y_);                 CHKERRQ (ierr);
        ierr = VecAXPY (y, 1.0, ctx->temp_[0]);                                                     CHKERRQ (ierr);
        ierr = VecPointwiseMult (ctx->temp_[0], ctx->temp_[2], ctx->velocity_->z_);                 CHKERRQ (ierr);
        ierr = VecAXPY (y, 1.0, ctx->temp_[0]);                                                     CHKERRQ (ierr);
        ierr = VecScale (y, alph);                                  CHKERRQ (ierr);
        ierr = VecAXPY (y, 1.0, x);                                 CHKERRQ (ierr);                
    }   

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode ImplicitEulerSolver::solve (Vec scalar, std::shared_ptr<VecField> velocity, ScalarType dt) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tumor-adv-solve");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    CtxAdv *ctx;
    ierr = MatShellGetContext (A_, &ctx);                       CHKERRQ (ierr);
    ctx->dt_ = dt;
    ctx->velocity_->x_ = velocity->x_;
    ctx->velocity_->y_ = velocity->y_;
    ctx->velocity_->z_ = velocity->z_;

    ierr = VecCopy (scalar, rhs_);                              CHKERRQ (ierr);
    // RHS is simply c^n
    //KSP solve
    ierr = KSPSolve (ksp_, rhs_, scalar);                       CHKERRQ (ierr);

    int itr;
    ierr = KSPGetIterationNumber (ksp_, &itr);                  CHKERRQ (ierr);
    ScalarType res_norm;
    ierr = KSPGetResidualNorm (ksp_, &res_norm);                CHKERRQ (ierr);

    std::stringstream s;
    s << "[Advection solver] GMRES convergence - iterations: " << itr << "    residual: " << res_norm;
    ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
    s.str (""); s.clear ();

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

ImplicitEulerSolver::~ImplicitEulerSolver () {
    PetscErrorCode ierr = 0;
    ierr = MatDestroy (&A_);
    ierr = KSPDestroy (&ksp_);
    ierr = VecDestroy (&rhs_);
}

AdvectionSolver::~AdvectionSolver () {
}


SemiLagrangianSolver::SemiLagrangianSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops) : AdvectionSolver (n_misc, tumor, spec_ops) {
    PetscErrorCode ierr = 0;
    m_dofs_[0] = 1;   // one set of query points for scalar field
    m_dofs_[1] = 3;   // one set of query points (Euler points) for the velocity field: the semilagrangian is second order

    n_ghost_ = 3;
    scalar_field_ghost_ = NULL;
    vector_field_ghost_ = NULL;
    temp_interpol1_ = NULL;
    temp_interpol2_ = NULL;
    interp_plan_scalar_ = nullptr;
    interp_plan_vector_ = nullptr;
    interp_plan_ = nullptr;

    // three versions of interpolation - multi-CPU, single-GPU, and multi-GPU (<none>, CUDA, MPICUDA)
    #if defined(CUDA) && !defined(MPICUDA)
        m_texture_ = gpuInitEmptyTexture (n_misc->n_);
        cudaMalloc ((void**) &temp_interpol1_, sizeof(float) * n_misc->n_[0] * n_misc->n_[1] * n_misc->n_[2]);
        cudaMalloc ((void**) &temp_interpol2_, sizeof(float) * n_misc->n_[0] * n_misc->n_[1] * n_misc->n_[2]);
        coords_ = std::make_shared<VecField> (n_misc->n_local_, n_misc->n_global_);
        // Different interpolation implementations require the global coordinates in different formats
        // Single GPU version requires it as three separate vectors for each coordinate
        ierr = setCoords (coords_);  // sets the global coordinates
    #else
        n_alloc_ = accfft_ghost_xyz_local_size_dft_r2c (n_misc->plan_, n_ghost_, isize_g_, istart_g_); // memory allocate
        scalar_field_ghost_ = reinterpret_cast<ScalarType*> (accfft_alloc (n_alloc_));    // scalar field with ghost points
        vector_field_ghost_ = reinterpret_cast<ScalarType*> (accfft_alloc (3 * n_alloc_));    // vector field with ghost points
        #ifdef MPICUDA
            interp_plan_scalar_ = std::make_shared<InterpPlan> (n_alloc_);
            interp_plan_scalar_->allocate (n_misc->n_local_, 1);
            interp_plan_vector_ = std::make_shared<InterpPlan> (n_alloc_);
            interp_plan_vector_->allocate (n_misc->n_local_, 3);
        #else
            interp_plan_ = std::make_shared<InterpPlan> ();
            interp_plan_->allocate (n_misc->n_local_, m_dofs_, 2);  // allocate memory for two sets of plans - scalars and vectors
        #endif
    #endif
    int factor = 3;
    ierr = VecCreate (PETSC_COMM_WORLD, &query_points_);
    ierr = VecSetSizes (query_points_, factor * n_misc->n_local_, factor * n_misc->n_global_);
    ierr = setupVec (query_points_);
    ierr = VecSet (query_points_, 0);

    work_field_ = tumor->work_field_;

    temp_ = new Vec[4];
    for (int i = 0; i < 4; i++) {
        // point to tumor work vectors -- no memory allocation
        temp_[i] = tumor->work_[i];
    }
}

PetscErrorCode SemiLagrangianSolver::setCoords (std::shared_ptr<VecField> coords) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    std::shared_ptr<NMisc> n_misc = ctx_->n_misc_;

    #ifdef CUDA
        ScalarType *x_ptr, *y_ptr, *z_ptr;
        ierr = VecCUDAGetArrayReadWrite (coords->x_, &x_ptr);                 CHKERRQ (ierr);
        ierr = VecCUDAGetArrayReadWrite (coords->y_, &y_ptr);                 CHKERRQ (ierr);
        ierr = VecCUDAGetArrayReadWrite (coords->z_, &z_ptr);                 CHKERRQ (ierr);
        setCoordsCuda (x_ptr, y_ptr, z_ptr, n_misc->isize_);
        ierr = VecCUDARestoreArrayReadWrite (coords->x_, &x_ptr);                 CHKERRQ (ierr);
        ierr = VecCUDARestoreArrayReadWrite (coords->y_, &y_ptr);                 CHKERRQ (ierr);
        ierr = VecCUDARestoreArrayReadWrite (coords->z_, &z_ptr);                 CHKERRQ (ierr);
    #else
        TU_assert (false, "Not implemented for CPUs.")
    #endif

    PetscFunctionReturn (ierr);
}

// Interpolate scalar fields
PetscErrorCode SemiLagrangianSolver::interpolate (Vec output, Vec input) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tumor-interp-scafield");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    std::shared_ptr<NMisc> n_misc = ctx_->n_misc_;

    ScalarType *in_ptr, *out_ptr, *query_ptr;

    // Interpolation
    #ifdef MPICUDA
        ierr = VecGetArray (input, &in_ptr);                        CHKERRQ (ierr);
        ierr = VecGetArray (output, &out_ptr);                      CHKERRQ (ierr);
        accfft_get_ghost_xyz (n_misc->plan_, n_ghost_, isize_g_, in_ptr, scalar_field_ghost_);  // populate scalar ghost field with input
        interp_plan_scalar_->interpolate (scalar_field_ghost_, 1, n_misc->n_, n_misc->isize_, n_misc->istart_,
                                     n_misc->n_local_, n_ghost_, out_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data());
        ierr = VecRestoreArray (input, &in_ptr);                    CHKERRQ (ierr);
        ierr = VecRestoreArray (output, &out_ptr);                  CHKERRQ (ierr);
    #elif CUDA
        ierr = VecCUDAGetArrayReadWrite (query_points_, &query_ptr);                 CHKERRQ (ierr);
        ierr = VecCUDAGetArrayReadWrite (input, &in_ptr);                            CHKERRQ (ierr);
        ierr = VecCUDAGetArrayReadWrite (output, &out_ptr);                          CHKERRQ (ierr);
        #ifdef SINGLE
            gpuInterp3D (in_ptr, &query_ptr[0], &query_ptr[n_misc->n_local_], &query_ptr[2*n_misc->n_local_], 
                     out_ptr, temp_interpol1_, temp_interpol2_, n_misc->n_, m_texture_, n_ghost_, (float*)t.data());
        #endif
        ierr = VecCUDARestoreArrayReadWrite (query_points_, &query_ptr);                 CHKERRQ (ierr);
        ierr = VecCUDARestoreArrayReadWrite (input, &in_ptr);                            CHKERRQ (ierr);
        ierr = VecCUDARestoreArrayReadWrite (output, &out_ptr);                          CHKERRQ (ierr);
    #else
        ierr = VecGetArray (input, &in_ptr);                        CHKERRQ (ierr);
        ierr = VecGetArray (output, &out_ptr);                      CHKERRQ (ierr);
        accfft_get_ghost_xyz (n_misc->plan_, n_ghost_, isize_g_, in_ptr, scalar_field_ghost_);  // populate scalar ghost field with input
        interp_plan_->interpolate (scalar_field_ghost_, n_misc->n_, n_misc->isize_, n_misc->istart_,
                                     n_misc->n_local_, n_ghost_, out_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data(), 0);
        ierr = VecRestoreArray (input, &in_ptr);                    CHKERRQ (ierr);
        ierr = VecRestoreArray (output, &out_ptr);                  CHKERRQ (ierr);
    #endif


    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx_->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

// Interpolate vector fields
PetscErrorCode SemiLagrangianSolver::interpolate (std::shared_ptr<VecField> output, std::shared_ptr<VecField> input) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tumor-interp-vecfield");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    std::shared_ptr<NMisc> n_misc = ctx_->n_misc_;

    #if defined(CUDA) && !defined(MPICUDA)
        ScalarType *ix_ptr, *iy_ptr, *iz_ptr, *ox_ptr, *oy_ptr, *oz_ptr, *query_ptr;
        ierr = VecCUDAGetArrayReadWrite (query_points_, &query_ptr);                 CHKERRQ (ierr);
        ierr = input->getComponentArrays (ix_ptr, iy_ptr, iz_ptr);              CHKERRQ (ierr);
        ierr = output->getComponentArrays (ox_ptr, oy_ptr, oz_ptr);             CHKERRQ (ierr);
        #ifdef SINGLE
            gpuInterpVec3D (ix_ptr, iy_ptr, iz_ptr, &query_ptr[0], &query_ptr[n_misc->n_local_], &query_ptr[2*n_misc->n_local_], 
                     ox_ptr, oy_ptr, oz_ptr, temp_interpol1_, temp_interpol2_, n_misc->n_, m_texture_, n_ghost_, (float*)t.data());
        #endif
        ierr = VecCUDARestoreArrayReadWrite (query_points_, &query_ptr);                 CHKERRQ (ierr);
        ierr = input->restoreComponentArrays (ix_ptr, iy_ptr, iz_ptr);              CHKERRQ (ierr);
        ierr = output->restoreComponentArrays (ox_ptr, oy_ptr, oz_ptr);             CHKERRQ (ierr);
    #else
        // flatten input
        ierr = input->getIndividualComponents (query_points_);      // query points is just a temp now
        int nl_ghost = 1;
        for (int i = 0; i < 3; i++) {
            nl_ghost *= static_cast<int> (isize_g_[i]);
        }
        ScalarType *query_ptr;
        ierr = VecGetArray (query_points_, &query_ptr);              CHKERRQ (ierr);
        for (int i = 0; i < 3; i++) {
            accfft_get_ghost_xyz (n_misc->plan_, n_ghost_, isize_g_, &query_ptr[i * n_misc->n_local_], &vector_field_ghost_[i * nl_ghost]);  // populate vector ghost field with input
        }
        // Interpolation
        #ifdef MPICUDA
            interp_plan_vector_->interpolate (vector_field_ghost_, 3, n_misc->n_, n_misc->isize_, n_misc->istart_,
                                         n_misc->n_local_, n_ghost_, query_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data());
        #else
            interp_plan_->interpolate (vector_field_ghost_, n_misc->n_, n_misc->isize_, n_misc->istart_,
                                         n_misc->n_local_, n_ghost_, query_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data(), 1);
        #endif
        ierr = VecRestoreArray (query_points_, &query_ptr);          CHKERRQ (ierr);
        // set interpolated values back into array
        ierr = output->setIndividualComponents (query_points_);      CHKERRQ (ierr);
    #endif
    
    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx_->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}


PetscErrorCode SemiLagrangianSolver::computeTrajectories () {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tumor-adv-eulercomp");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();

    ScalarType dt = ctx_->dt_;
    std::shared_ptr<NMisc> n_misc = ctx_->n_misc_;
    std::shared_ptr<VecField> velocity = ctx_->velocity_;

    ScalarType *vx_ptr, *vy_ptr, *vz_ptr, *query_ptr, *wx_ptr, *wy_ptr, *wz_ptr;
    ScalarType x1, x2, x3;
    
    // single-GPU
    #if defined(CUDA) && !defined(MPICUDA)
        ierr = VecWAXPY (work_field_->x_, -dt / n_misc->h_[0], velocity->x_, coords_->x_);           CHKERRQ (ierr);
        ierr = VecWAXPY (work_field_->y_, -dt / n_misc->h_[1], velocity->y_, coords_->y_);           CHKERRQ (ierr);
        ierr = VecWAXPY (work_field_->z_, -dt / n_misc->h_[2], velocity->z_, coords_->z_);           CHKERRQ (ierr);
        ierr = work_field_->getIndividualComponents (query_points_);                 CHKERRQ (ierr);
    // multi-GPU
    #elif defined(MPICUDA)
        ierr = VecCUDAGetArrayReadWrite (query_points_, &query_ptr);                 CHKERRQ (ierr);
        ierr = velocity->getComponentArrays (vx_ptr, vy_ptr, vz_ptr);
        computeEulerPointsCuda (query_ptr, vx_ptr, vy_ptr, vz_ptr, dt, n_misc->isize_);
        ierr = VecCUDARestoreArrayReadWrite (query_points_, &query_ptr);             CHKERRQ (ierr);
        ierr = velocity->restoreComponentArrays (vx_ptr, vy_ptr, vz_ptr);
    // only CPU (MPI)
    #else
        ierr = velocity->getComponentArrays (vx_ptr, vy_ptr, vz_ptr);
        ierr = VecGetArray (query_points_, &query_ptr);             CHKERRQ (ierr);
        int64_t ptr;
        for (int i1 = 0; i1 < n_misc->isize_[0]; i1++) {
            for (int i2 = 0; i2 < n_misc->isize_[1]; i2++) {
                for (int i3 = 0; i3 < n_misc->isize_[2]; i3++) {
                    x1 = n_misc->h_[0] * static_cast<ScalarType> (i1 + n_misc->istart_[0]);
                    x2 = n_misc->h_[1] * static_cast<ScalarType> (i2 + n_misc->istart_[1]);
                    x3 = n_misc->h_[2] * static_cast<ScalarType> (i3 + n_misc->istart_[2]);

                    ptr = i1 * n_misc->isize_[1] * n_misc->isize_[2] + i2 * n_misc->isize_[2] + i3;

                    // compute the Euler points: xstar = x - dt * vel.
                    // coords are normalized - requirement from interpolation
                    query_ptr[ptr * 3 + 0] = (x1 - dt * vx_ptr[ptr]) / (2.0 * M_PI);   
                    query_ptr[ptr * 3 + 1] = (x2 - dt * vy_ptr[ptr]) / (2.0 * M_PI);   
                    query_ptr[ptr * 3 + 2] = (x3 - dt * vz_ptr[ptr]) / (2.0 * M_PI);   
                }
            }
        }
        ierr = VecRestoreArray (query_points_, &query_ptr);         CHKERRQ (ierr);
        ierr = velocity->restoreComponentArrays (vx_ptr, vy_ptr, vz_ptr);
    #endif
   
    // communicate coordinates to all processes: this function keeps track of query points
    // coordinates must always be scattered before any interpolation, otherwise the plan
    // will use whatever query points that was set before (if at all)
    // only used if MPICUDA OR MPI
    #ifdef MPICUDA
        ierr = VecGetArray (query_points_, &query_ptr);             CHKERRQ (ierr);
        interp_plan_scalar_->scatter (1, n_misc->n_, n_misc->isize_, n_misc->istart_, n_misc->n_local_, 
                                n_ghost_, query_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data());
        interp_plan_vector_->scatter (3, n_misc->n_, n_misc->isize_, n_misc->istart_, n_misc->n_local_, 
                                n_ghost_, query_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data());
        ierr = VecRestoreArray (query_points_, &query_ptr);         CHKERRQ (ierr);
    #elif !defined(CUDA)
        ierr = VecGetArray (query_points_, &query_ptr);             CHKERRQ (ierr);
        interp_plan_->scatter (n_misc->n_, n_misc->isize_, n_misc->istart_, n_misc->n_local_, 
                                n_ghost_, query_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data());
        ierr = VecRestoreArray (query_points_, &query_ptr);         CHKERRQ (ierr);
    #endif
    
    // Interpolate velocity fields at Euler points
    ierr = work_field_->set (0.);                                   CHKERRQ (ierr);
    ierr = interpolate (work_field_, velocity);

    // Compute RK2 queries
    // single-GPU
    #if defined(CUDA) && !defined(MPICUDA)
        ierr = VecAYPX (work_field_->x_, -0.5*dt / n_misc->h_[0], coords_->x_);                     CHKERRQ (ierr);
        ierr = VecAYPX (work_field_->y_, -0.5*dt / n_misc->h_[1], coords_->y_);                     CHKERRQ (ierr);
        ierr = VecAYPX (work_field_->z_, -0.5*dt / n_misc->h_[2], coords_->z_);                     CHKERRQ (ierr);
        ierr = VecAXPY (work_field_->x_, -0.5*dt / n_misc->h_[0], velocity->x_);                    CHKERRQ (ierr);
        ierr = VecAXPY (work_field_->y_, -0.5*dt / n_misc->h_[1], velocity->y_);                    CHKERRQ (ierr);
        ierr = VecAXPY (work_field_->z_, -0.5*dt / n_misc->h_[2], velocity->z_);                    CHKERRQ (ierr);
        ierr = work_field_->getIndividualComponents (query_points_);                CHKERRQ (ierr);
    // multi-GPU
    #elif defined(MPICUDA)
        ierr = velocity->getComponentArrays (vx_ptr, vy_ptr, vz_ptr);
        ierr = work_field_->getComponentArrays (wx_ptr, wy_ptr, wz_ptr);
        ierr = VecCUDAGetArrayReadWrite (query_points_, &query_ptr);                 CHKERRQ (ierr);
        computeSecondOrderEulerPointsCuda (query_ptr, vx_ptr, vy_ptr, vz_ptr, wx_ptr, wy_ptr, wz_ptr, dt, n_misc->isize_);
        ierr = VecCUDARestoreArrayReadWrite (query_points_, &query_ptr);             CHKERRQ (ierr);
        ierr = velocity->restoreComponentArrays (vx_ptr, vy_ptr, vz_ptr);
        ierr = work_field_->restoreComponentArrays (wx_ptr, wy_ptr, wz_ptr);
    #else
        ierr = velocity->getComponentArrays (vx_ptr, vy_ptr, vz_ptr);
        ierr = work_field_->getComponentArrays (wx_ptr, wy_ptr, wz_ptr);
        ierr = VecGetArray (query_points_, &query_ptr);             CHKERRQ (ierr);
        for (int i1 = 0; i1 < n_misc->isize_[0]; i1++) {
            for (int i2 = 0; i2 < n_misc->isize_[1]; i2++) {
                for (int i3 = 0; i3 < n_misc->isize_[2]; i3++) {
                    x1 = n_misc->h_[0] * static_cast<ScalarType> (i1 + n_misc->istart_[0]);
                    x2 = n_misc->h_[1] * static_cast<ScalarType> (i2 + n_misc->istart_[1]);
                    x3 = n_misc->h_[2] * static_cast<ScalarType> (i3 + n_misc->istart_[2]);

                    ptr = i1 * n_misc->isize_[1] * n_misc->isize_[2] + i2 * n_misc->isize_[2] + i3;

                    // compute query points
                    query_ptr[ptr * 3 + 0] = (x1 - 0.5 * dt * (vx_ptr[ptr] + wx_ptr[ptr])) / (2.0 * M_PI);   
                    query_ptr[ptr * 3 + 1] = (x2 - 0.5 * dt * (vy_ptr[ptr] + wy_ptr[ptr])) / (2.0 * M_PI);   
                    query_ptr[ptr * 3 + 2] = (x3 - 0.5 * dt * (vz_ptr[ptr] + wz_ptr[ptr])) / (2.0 * M_PI);   
                }
            }
        }
        ierr = VecRestoreArray (query_points_, &query_ptr);         CHKERRQ (ierr);
        ierr = velocity->restoreComponentArrays (vx_ptr, vy_ptr, vz_ptr);
        ierr = work_field_->restoreComponentArrays (wx_ptr, wy_ptr, wz_ptr);
    #endif
    
    // scatter final query points
    // only used if MPICUDA OR MPI
    #ifdef MPICUDA
        ierr = VecGetArray (query_points_, &query_ptr);             CHKERRQ (ierr);
        interp_plan_scalar_->scatter (1, n_misc->n_, n_misc->isize_, n_misc->istart_, n_misc->n_local_, 
                                n_ghost_, query_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data());
        interp_plan_vector_->scatter (3, n_misc->n_, n_misc->isize_, n_misc->istart_, n_misc->n_local_, 
                                n_ghost_, query_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data());
        ierr = VecRestoreArray (query_points_, &query_ptr);         CHKERRQ (ierr);
    #elif !defined(CUDA)
        ierr = VecGetArray (query_points_, &query_ptr);             CHKERRQ (ierr);
        interp_plan_->scatter (n_misc->n_, n_misc->isize_, n_misc->istart_, n_misc->n_local_, 
                                n_ghost_, query_ptr, n_misc->c_dims_, n_misc->c_comm_, t.data());
        ierr = VecRestoreArray (query_points_, &query_ptr);         CHKERRQ (ierr);
    #endif

    ierr = work_field_->set (0.);                                   CHKERRQ (ierr);

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx_->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode SemiLagrangianSolver::solve (Vec scalar, std::shared_ptr<VecField> velocity, ScalarType dt) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tumor-adv-semilag-solve");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    
    ctx_->dt_ = dt;
    ctx_->velocity_ = velocity;
    std::shared_ptr<NMisc> n_misc = ctx_->n_misc_;

    if (!trajectoryIsComputed_) {
        ierr = computeTrajectories ();                        CHKERRQ (ierr);
        trajectoryIsComputed_ = true;
    }

    for (int i = 0; i < 4; i++) {
        ierr = VecSet (temp_[i], 0.);                         CHKERRQ (ierr);
    }

    if (advection_mode_ == 1) {
        // Mass conservation equation in transport form is:
        // d_t \nu + grad \nu . v = -\nu (div v)

        // Interpolate scalar at query points
        ierr = interpolate (temp_[0], scalar);

        // Compute source term: -\nu (div v)
        ierr = spec_ops_->computeDivergence (temp_[1], velocity->x_, velocity->y_, velocity->z_, t.data());

        // Interpolate div using the same query points
        ierr = interpolate (temp_[2], temp_[1]);

        // Compute interpolated source: -\nu_interp * interp(div v)
        ierr = VecPointwiseMult (temp_[2], temp_[0], temp_[2]);   CHKERRQ (ierr);
        ierr = VecScale (temp_[2], -1.0);                         CHKERRQ (ierr);

        // Compute \nu_star = interp\nu + dt * interp\source
        ierr = VecWAXPY (temp_[3], dt, temp_[2], temp_[0]);

        // Compute \source_star : -\nu_star * (div v)
        ierr = VecPointwiseMult (temp_[1], temp_[3], temp_[1]);   CHKERRQ (ierr);
        ierr = VecScale (temp_[1], -1.0);                         CHKERRQ (ierr);

        // Compute output: \nu_interp + dt / 2 * (source_interp + source_star)
        ierr = VecAXPY (temp_[1], 1.0, temp_[2]);                 CHKERRQ (ierr);
        ierr = VecWAXPY (scalar, 0.5 * dt, temp_[1], temp_[0]);   CHKERRQ (ierr);

    } else if (advection_mode_ == 2) {
        // Pure advection
        // d_t \nu + grad \nu . v = 0

        // Interpolate scalar at query points
        ierr = interpolate (temp_[0], scalar);

        // Output is simply the interpolated scalar
        ierr = VecCopy (temp_[0], scalar);                      CHKERRQ (ierr);

    } else {
        TU_assert (false, "advection mode not implemented.");   CHKERRQ(ierr);
    }
    

    self_exec_time += MPI_Wtime();
    accumulateTimers (ctx_->n_misc_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}


SemiLagrangianSolver::~SemiLagrangianSolver () {
    PetscErrorCode ierr = 0;

    if (scalar_field_ghost_ != NULL) fft_free (scalar_field_ghost_);
    if (vector_field_ghost_ != NULL) fft_free (vector_field_ghost_);
    if (temp_interpol1_ != NULL) fft_free (temp_interpol1_);
    if (temp_interpol2_ != NULL) fft_free (temp_interpol2_);

    delete [] temp_;

    #ifdef CUDA
        cudaDestroyTextureObject (m_texture_);
    #endif

    ierr = VecDestroy (&query_points_);  
}