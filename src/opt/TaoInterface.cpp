

/* #### ------------------------------------------------------------------- #### */
/* #### ========                TIL + DIffusion                    ======== #### */
/* #### ------------------------------------------------------------------- #### */

/* ------------------------------------------------------------------- */
/*
 evaluateObjectiveFunction - evaluates the objective function J(x).
 Input Parameters:
 .  tao - the Tao context
 .  x   - the input vector
 .  ptr - optional user-defined context, as set by TaoSetFunction()
 Output Parameters:
 .  J    - the newly evaluated function
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode evaluateObjectiveFunction (Tao tao, Vec x, PetscReal *J, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tao-eval-obj-tumor");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
    itctx->params_->optf_->nb_objevals_++;
    ierr = itctx->derivative_operators_->evaluateObjective (J, x, itctx->data);
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

/* ------------------------------------------------------------------- */
/*
 evaluateGradient evaluates the gradient g(m0)
 input parameters:
  . tao  - the Tao context
  . x   - input vector p (current estimate for the paramterized initial condition)
  . ptr  - optional user-defined context
 output parameters:
  . dJ    - vector containing the newly evaluated gradient
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode evaluateGradient (Tao tao, Vec x, Vec dJ, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tao-eval-grad-tumor");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
    ierr = VecCopy (x, itctx->derivative_operators_->p_current_);                       CHKERRQ (ierr);

    itctx->params_->optf_->nb_gradevals_++;
    ierr = itctx->derivative_operators_->evaluateGradient (dJ, x, itctx->data);

    // use petsc default fd gradient
   // itctx->derivative_operators_->disable_verbose_ = true;
   // ierr = TaoDefaultComputeGradient(tao, x, dJ, ptr); CHKERRQ(ierr);
   // itctx->derivative_operators_->disable_verbose_ = false;

    std::stringstream s;
    if (itctx->params_->tu_->verbosity_ > 1) {
        ScalarType gnorm;
        ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
        s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
    }
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
    e.addTimings (t);
    e.stop ();
    PetscFunctionReturn (ierr);
}

/* ------------------------------------------------------------------- */
/*
 evaluateObjectiveFunctionAndGradient - evaluates the function and corresponding gradient
 Input Parameters:
  . tao - the Tao context
  . x   - the input vector
  . ptr - optional user-defined context, as set by TaoSetFunction()
 Output Parameters:
  . J   - the newly evaluated function
  . dJ   - the newly evaluated gradient
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode evaluateObjectiveFunctionAndGradient (Tao tao, Vec x, PetscReal *J, Vec dJ, void *ptr){
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj/grad-tumor");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);
  ierr = VecCopy (x, itctx->derivative_operators_->p_current_);                       CHKERRQ (ierr);

  itctx->params_->optf_->nb_objevals_++;
  itctx->params_->optf_->nb_gradevals_++;
  ierr = itctx->derivative_operators_->evaluateObjectiveAndGradient (J, dJ, x, itctx->data);

  std::stringstream s;
  if (itctx->params_->tu_->verbosity_ > 1) {
      ScalarType gnorm;
      ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
      s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }

  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

    //ierr = evaluateObjectiveFunction (tao, p, J, ptr);                     CHKERRQ(ierr);
    //ierr = evaluateGradient (tao, p, dJ, ptr);                             CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}

/* #### ------------------------------------------------------------------- #### */
// #### convergence


/* ------------------------------------------------------------------- */
/*
 optimizationMonitor    mointors the inverse Gauss-Newton solve
 input parameters:
  . tao       TAO object
  . ptr       optional user defined context
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode optimizationMonitor (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    PetscInt its;
    ScalarType J = 0, gnorm = 0, cnorm = 0 , step = 0, D = 0, J0 = 0, D0 = 0, gnorm0 = 0;
    Vec x = nullptr;
    char msg[256];
    std::string statusmsg;
    std::stringstream s;
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

    Vec tao_grad;

    // get current iteration, objective value, norm of gradient, norm of
    // norm of contraint, step length / trust region readius of iteratore
    // and termination reason
    Vec tao_x;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);  CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &tao_x);                                   CHKERRQ(ierr);
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr =  TaoGetGradientVector(tao, &tao_grad);                               CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    if (itctx->params_->tu_->verbosity_ >= 2) {
      ScalarType *grad_ptr, *sol_ptr;
      ierr = VecGetArray(tao_x, &sol_ptr);                                        CHKERRQ(ierr);
      ierr = VecGetArray(tao_grad, &grad_ptr);                                    CHKERRQ(ierr);
      for (int i = 0; i < itctx->params_->tu_->np_; i++){
        if(procid == 0){
          itctx->params_->tu_->outfile_sol_  << sol_ptr[i]  << ", ";
          itctx->params_->tu_->outfile_grad_ << grad_ptr[i] << ", ";
        }
      }
      if(procid == 0){
        itctx->params_->tu_->outfile_sol_  << sol_ptr[itctx->params_->tu_->np_]  << ";" <<std::endl;
        itctx->params_->tu_->outfile_grad_ << grad_ptr[itctx->params_->tu_->np_] << ";" <<std::endl;
      }
      ierr = VecRestoreArray(tao_x, &sol_ptr);                                    CHKERRQ(ierr);
      ierr = VecRestoreArray(tao_grad, &grad_ptr);                                CHKERRQ(ierr);
    }

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
      Vec dJ, p0;
      ScalarType norm_gref = 0.;
      ierr = VecDuplicate (itctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
      ierr = VecDuplicate (itctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
      ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
      ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);

      if (itctx->params_->opt_->flag_reaction_inv_) {
        norm_gref = gnorm;
      } else {
        ierr = evaluateGradient(tao, p0, dJ, (void*) itctx);
        ierr = VecNorm (dJ, NORM_2, &norm_gref);                                CHKERRQ(ierr);
      }
      itctx->params_->optf_->gradnorm_0 = norm_gref;
      //ctx->gradnorm0 = gnorm;
      itctx->update_reference_gradient = false;
      s <<" updated reference gradient for relative convergence criterion, Gauss-Newton solver: " << itctx->params_->optf_->gradnorm_0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);s.str ("");s.clear ();
      if (dJ  != nullptr) {VecDestroy (&dJ);  CHKERRQ(ierr);  dJ  = nullptr;}
      if (p0  != nullptr) {VecDestroy (&p0);  CHKERRQ(ierr);  p0  = nullptr;}
    }
    #endif

    // ierr = VecAXPY (itctx->x_old, -1.0, tao_x);                                     CHKERRQ (ierr);

    // ScalarType dp_norm, p_norm;
    // ierr = VecNorm (itctx->x_old, NORM_INFINITY, &dp_norm);                         CHKERRQ (ierr);
    // ierr = VecNorm (tao_x, NORM_INFINITY, &p_norm);                                 CHKERRQ (ierr);
    // accumulate number of newton iterations
    itctx->params_->optf_->nb_newton_it_++;
    // print out Newton iteration information

    ierr = itctx->tumor_->phi_->apply (itctx->tumor_->c_0_, tao_x);                 CHKERRQ (ierr);
    //Prints a warning if tumor IC is clipped
    ierr = checkClipping (itctx->tumor_->c_0_, itctx->params_);           CHKERRQ (ierr);

    ScalarType mx, mn;
    ierr = VecMax (itctx->tumor_->c_t_, NULL, &mx); CHKERRQ (ierr);
    ierr = VecMin (itctx->tumor_->c_t_, NULL, &mn); CHKERRQ (ierr);
    // this print helps determine if theres any large aliasing errors which is causing ls failure etc
    s << " ---------- tumor c(1) bounds: max = " << mx << ", min = " << mn << " ----------- ";
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();

    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   ";
          if (itctx->params_->opt_->diffusivity_inversion_) {
            s << std::setw(18) << "k";
          }

        if(itctx->params_->opt_->newton_solver_ == QUASINEWTON) {
          ierr = tuMSGstd (" starting optimization, TAO's Quasi-Newton");                   CHKERRQ(ierr);
        } else {
          ierr = tuMSGstd (" starting optimization, TAO's Gauss-Newton");            CHKERRQ(ierr);
        }
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }

    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->params_->optf_->gradnorm_0
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      ;
      if (itctx->params_->opt_->diffusivity_inversion_) {
        ScalarType *x_ptr;
        ierr = VecGetArray(tao_x, &x_ptr);                                         CHKERRQ(ierr);
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->params_->tu_->np_];
        if (itctx->params_->tu_->nk_ > 1) {
          s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->params_->tu_->np_ + 1]; }
        ierr = VecRestoreArray(tao_x, &x_ptr);                                     CHKERRQ(ierr);
      }
    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");s.clear ();


    //ierr = PetscPrintf (PETSC_COMM_WORLD, "\nKSP number of krylov iterations: %d\n", itctx->params_->optf_->nb_krylov_it_);          CHKERRQ(ierr);
    //itctx->params_->optf_->nb_krylov_it_ = 0;

    //Gradient check begin
    // ierr = itctx->derivative_operators_->checkGradient (tao_x, itctx->data);
    // ierr = itctx->derivative_operators_->checkHessian (tao_x, itctx->data);
    //Gradient check end
    PetscFunctionReturn (ierr);
}



/* ------------------------------------------------------------------- */
/*
 checkConvergenceGrad    checks convergence of the overall Gauss-Newton tumor inversion

 input parameters:
  . Tao tao       Tao solver object
  . void *ptr     optional user defined context
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode checkConvergenceGrad (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscInt its, nl, ng;
    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
    bool stop[3];
    int verbosity;
    std::stringstream ss;
    Vec x = nullptr, g = nullptr;
    ierr = TaoGetSolutionVector(tao, &x);                                     CHKERRQ(ierr);
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->params_->tu_->verbosity_;
    minstep = ctx->params_->opt_->ls_minstep_;
    miniter = ctx->params_->opt_->newton_minit_;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
    #endif

    // get line-search status
    nl = ctx->params_->grid_->nl_;
    ng = ctx->params_->grid_->ng_;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = VecDuplicate (ctx->tumor_->p_, &g);                                  CHKERRQ(ierr);
    ierr = TaoLineSearchGetSolution(ls, x, &J, g, &step, &ls_flag);             CHKERRQ (ierr);

    // display line-search convergence reason
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    ierr = TaoGetMaximumIterations(tao, &maxiter);                              CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);

    Vec tao_grad;
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr = TaoGetGradientVector(tao, &tao_grad);                                CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    // update/set reference gradient (with p = zeros)
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if (ctx->update_reference_gradient) {
        Vec dJ = nullptr, p0 = nullptr;
        ScalarType norm_gref = 0.;
        ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
        ierr = VecDuplicate (ctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
        ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
        ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);

        if (ctx->params_->opt_->flag_reaction_inv_) {
        norm_gref = gnorm;
        } else {
          ierr = evaluateGradient(tao, p0, dJ, (void*) ctx);
          ierr = VecNorm (dJ, NORM_2, &norm_gref);                                  CHKERRQ(ierr);
        }
        ctx->params_->optf_->gradnorm_0 = norm_gref;
        //ctx->gradnorm0 = gnorm;
        ctx->update_reference_gradient = false;
        std::stringstream s; s <<" updated reference gradient for relative convergence criterion: " << ctx->params_->optf_->gradnorm_0;
        ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
        if (dJ != nullptr) {ierr = VecDestroy(&dJ); CHKERRQ(ierr); dJ = nullptr;}
        if (p0 != nullptr) {ierr = VecDestroy(&p0); CHKERRQ(ierr); p0 = nullptr;}
    }
    #endif
    // get initial gradient
    g0norm = ctx->params_->optf_->gradnorm_0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    ctx->convergence_message.clear();

    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
          ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
          ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
          ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
          ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn(ierr);
    }
    //if(verbosity >= 1) {
    //    ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    //}
    // only check convergence criteria after a certain number of iterations
    stop[0] = false; stop[1] = false; stop[2] = false;
    ctx->params_->optf_->converged_     = false;
    ctx->cosamp_->converged_l2       = false;
    ctx->cosamp_->converged_error_l2 = false;
    if (iter >= miniter) {
        if (verbosity > 1) {
                ss << "step size in linesearch: " << std::scientific << step;
                ierr = tuMSGstd(ss.str());                                            CHKERRQ(ierr);
                ss.str(std::string());
                ss.clear();
        }
        if (step < minstep) {
                ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
                ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
                ss.str(std::string());
                ss.clear();
                ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
                if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
                ctx->cosamp_->converged_error_l2 = true;
                PetscFunctionReturn(ierr);
        }
        if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
            ss << "step  = " << std::scientific << step << ". ls failed with status " << ls_flag;
            ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
            ss.str(std::string());
                  ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            ctx->cosamp_->converged_error_l2 = true;
            PetscFunctionReturn(ierr);
        }
        // ||g_k||_2 < tol*||g_0||
        if (gnorm < gttol*g0norm) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);               CHKERRQ(ierr);
            stop[0] = true;
        }
        ss << "  " << stop[0] << "    ||g|| = " << std::setw(14)
           << std::right << std::scientific << gnorm << "    <    "
           << std::left << std::setw(14) << gttol*g0norm << " = " << "tol";
        ctx->convergence_message.push_back(ss.str());
        if(verbosity >= 3) { ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); } ss.str(std::string()); ss.clear();

        // ||g_k||_2 < tol
        if (gnorm < gatol) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
            stop[1] = true;
        }
        ss  << "  " << stop[1] << "    ||g|| = " << std::setw(14)
            << std::right << std::scientific << gnorm << "    <    "
            << std::left << std::setw(14) << gatol << " = " << "tol";
        ctx->convergence_message.push_back(ss.str());
        if(verbosity >= 3) { ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); } ss.str(std::string()); ss.clear();

        // iteration number exceeds limit
        if (iter > maxiter) {
            ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                    CHKERRQ(ierr);
            stop[2] = true;
        }
        ss  << "  " << stop[2] << "     iter = " << std::setw(14)
              << std::right << iter  << "    >    "
              << std::left << std::setw(14) << maxiter << " = " << "maxiter";
        ctx->convergence_message.push_back(ss.str());
        if(verbosity >= 3) { ierr = tuMSGstd(ss.str()); CHKERRQ(ierr); } ss.str(std::string()); ss.clear();

        // store objective function value
        ctx->jvalold = J;
        if (stop[0] || stop[1]) {ctx->cosamp_->converged_l2 = true;} // for CoSaMpRS to split up L2 solve
        if (stop[0] || stop[1] || stop[2]) {
            ctx->params_->optf_->converged_ = true;
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn(ierr);
        }

    // iter < miniter
    } else {
        // if the gradient is zero, we should terminate immediately
        if (gnorm == 0) {
            ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
            ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);
            ss.str(std::string());
            ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            ctx->cosamp_->converged_l2 = true;
            PetscFunctionReturn(ierr);
        }
    }
    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
}



/* ------------------------------------------------------------------- */
/*
 checkConvergenceGradObj    checks convergence of the overall Gauss-Newton tumor inversion

 input parameters:
  . Tao tao       Tao solver object
  . void *ptr     optional user defined context
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode checkConvergenceGradObj (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PetscInt iter, maxiter, miniter, iterbound;
    PetscReal jx, jxold, gnorm, step, gatol, grtol, gttol, g0norm, gtolbound, minstep, theta, normx, normdx, tolj, tolx, tolg;
    const int nstop = 7;
    bool stop[nstop];
    std::stringstream ss;
    Vec x = nullptr, g = nullptr;

    CtxInv *ctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
    // get minstep and miniter
    minstep = ctx->params_->opt_->ls_minstep_;
    miniter = ctx->params_->opt_->newton_minit_;
    iterbound = ctx->params_->opt_->iterbound_;
    // get lower bound for gradient
    gtolbound = ctx->params_->opt_->gtolbound_;

    // get and display line-search convergence reason
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = TaoLineSearchGetSolution(ls, x, &jx, g, &step, &ls_flag);            CHKERRQ (ierr);
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    // create temp vector (make sure it's deleted)
    ierr = VecDuplicate (ctx->tumor_->p_, &g);                                  CHKERRQ(ierr);
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances(tao, &gatol, &grtol, &gttol);                               CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances(tao, NULL, NULL, &gatol, &grtol, &gttol);                   CHKERRQ(ierr);
    #endif
    // get maxiter
    ierr = TaoGetMaximumIterations(tao, &maxiter);                                          CHKERRQ(ierr);
    if (maxiter > iterbound) iterbound = maxiter;
    ierr = TaoGetMaximumIterations(tao, &maxiter);                                          CHKERRQ(ierr);
    // get solution status
    ierr = TaoGetSolutionStatus(tao, &iter, &jx, &gnorm, NULL, &step, NULL);                CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &x);                                                   CHKERRQ(ierr);

    // update/set reference gradient (with p = zeros)
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if(ctx->update_reference_gradient) {
      Vec dJ, p0;
      ScalarType norm_gref = 0.;
      ierr = VecDuplicate (ctx->tumor_->p_, &dJ);                               CHKERRQ(ierr);
      ierr = VecDuplicate (ctx->tumor_->p_, &p0);                               CHKERRQ(ierr);
      ierr = VecSet (dJ, 0.);                                                   CHKERRQ(ierr);
      ierr = VecSet (p0, 0.);                                                   CHKERRQ(ierr);
      // evaluateGradient(tao, x, dJ, (void*) ctx);
      evaluateObjectiveFunctionAndGradient (tao, p0, &ctx->params_->optf_->j0_, dJ, (void*) ctx);
        ierr = VecNorm (dJ, NORM_2, &norm_gref); CHKERRQ(ierr);
        ctx->params_->optf_->gradnorm_0 = norm_gref;
      // evaluateObjectiveFunction (tao, x, &ctx->params_->optf_->j0_, (void*) ctx);

      std::stringstream s; s <<"updated reference objective for relative convergence criterion: " << ctx->params_->optf_->j0_;
        ctx->update_reference_gradient = false;
      ierr = tuMSGstd(s.str());                                                     CHKERRQ(ierr);
      s.str(std::string());
      s.clear();
      s <<" updated reference gradient for relative convergence criterion: " << ctx->params_->optf_->gradnorm_0;
        ierr = tuMSGstd(s.str());                                                          CHKERRQ(ierr);
      s.str(std::string());
      s.clear();
      if (dJ != nullptr) {ierr = VecDestroy(&dJ); CHKERRQ(ierr); dJ = nullptr;}
      if (p0 != nullptr) {ierr = VecDestroy(&p0); CHKERRQ(ierr); p0 = nullptr;}
    }
    #endif
    // get initial gradient
    g0norm = ctx->params_->optf_->gradnorm_0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    // compute tolerances for stopping conditions
    tolj = grtol;
    tolx = std::sqrt(grtol);
    #if __cplusplus > 199711L
    tolg = std::cbrt(grtol);
    #else
    tolg = std::pow(grtol, 1.0/3.0);
    #endif
    // compute theta
    theta = 1.0 + std::abs(ctx->params_->optf_->j0_);
    // compute norm(\Phi x^k - \Phi x^k-1) and norm(\Phi x^k)
    // ierr = ctx->tumor_->phi_->apply (ctx->tmp, x);                                         CHKERRQ(ierr);  // comp \Phi x
    // ierr = VecNorm (ctx->tmp, NORM_2, &normx);                                             CHKERRQ(ierr);  // comp norm \Phi x
    // ierr = VecAXPY(ctx->c0_old, -1, ctx->tmp);                                              CHKERRQ(ierr);  // comp dx
    // ierr = VecNorm (ctx->c0_old, NORM_2, &normdx);                                          CHKERRQ(ierr);  // comp norm \Phi dx
    // ierr = VecCopy(ctx->tmp, ctx->c0_old);                                                  CHKERRQ(ierr);  // save \Phi x
    ierr = VecNorm (x, NORM_2, &normx);                                             CHKERRQ(ierr);  // comp norm x
    ierr = VecAXPY (ctx->x_old, -1.0, x);                                                  CHKERRQ(ierr);  // comp dx
    ierr = VecNorm (ctx->x_old, NORM_2, &normdx);                                   CHKERRQ(ierr);  // comp norm dx
    ierr = VecCopy (x, ctx->x_old);                                                        CHKERRQ(ierr);  // save old x
    // get old objective function value
    jxold = ctx->jvalold;
    ctx->convergence_message.clear();

    // check for NaN value
    if (PetscIsInfOrNanReal(jx)) {
          ierr = tuMSGwarn("objective is NaN");                                             CHKERRQ(ierr);
          ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                              CHKERRQ(ierr);
          PetscFunctionReturn(ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
          ierr = tuMSGwarn("||g|| is NaN");                                                 CHKERRQ(ierr);
          ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                              CHKERRQ(ierr);
          PetscFunctionReturn(ierr);
    }

    // ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    ctx->params_->optf_->converged_ = false;
    // initialize flags for stopping conditions
    for (int i = 0; i < nstop; i++)
        stop[i] = false;
    // only check convergence criteria after a certain number of iterations
    if (iter >= miniter && iter > 1) {
      if (step < minstep) {
                ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
                ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
                ss.str(std::string());
                ss.clear();
                ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
          if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
                PetscFunctionReturn (ierr);
        }
      if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
        ss << "step  = " << std::scientific << step << ". ls failed with status " << ls_flag;
        ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
        ss.str(std::string());
              ss.clear();
        ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
        PetscFunctionReturn (ierr);
      }
        // |j(x_{k-1}) - j(x_k)| < tolj*abs(1+J)
        if (std::abs(jxold-jx) < tolj*theta) {
            stop[0] = true;
        }
        ss << "  " << stop[0] << "    |dJ|  = " << std::setw(18)
        << std::right << std::scientific << std::abs(jxold-jx) << "    <    "
        << std::left << std::setw(18) << tolj*theta << " = " << "tol*|1+J|";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||dx|| < sqrt(tolj)*(1+||x||)
        if (normdx < tolx*(1+normx)) {
           stop[1] = true;
        }
        ss << "  " << stop[1] << "    |dx|  = " << std::setw(18)
        << std::right << std::scientific << normdx << "    <    "
        << std::left << std::setw(18) << tolx*(1+normx) << " = " << "sqrt(tol)*(1+||x||)";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||g_k||_2 < cbrt(tolj)*abs(1+Jc)
        if (gnorm < tolg*theta) {
            stop[2] = true;
        }
        ss  << "  " << stop[2] << "    ||g|| = " << std::setw(18)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(18) << tolg*theta << " = " << "cbrt(tol)*|1+J|";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||g_k||_2 < tol
        if (gnorm < gatol) {
                stop[3] = true;
        }
        ss  << "  " << stop[3] << "    ||g|| = " << std::setw(18)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(18) << gatol << " = " << "tol";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        if (gnorm < gttol*g0norm) {
          stop[4] = true;
      }
        ss << "  " << stop[0] << "    ||g|| = " << std::setw(18)
         << std::right << std::scientific << gnorm << "    <    "
         << std::left << std::setw(18) << gttol*g0norm << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        if (iter > maxiter) {
                stop[5] = true;
        }
        ss  << "  " << stop[5] << "    iter  = " << std::setw(18)
          << std::right << iter  << "    >    "
          << std::left << std::setw(18) << maxiter << " = " << "maxiter";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

    //     if (iter > iterbound) {
    //             stop[6] = true;
    //     }
    //     ss  << "  " << stop[6] << "    iter  = " << std::setw(14)
          // << std::right << iter  << "    >    "
          // << std::left << std::setw(14) << iterbound << " = " << "iterbound";
    //     ctx->convergence_message.push_back(ss.str());
    //     ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
    //     ss.str(std::string());
    //     ss.clear();

        // store objective function value
        ctx->jvalold = jx;

        if (stop[0] && stop[1] && stop[2]) {
              ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER);                  CHKERRQ(ierr);
              ctx->params_->optf_->converged_ = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
              PetscFunctionReturn (ierr);
        } else if (stop[3]) {
              ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                 CHKERRQ(ierr);
              ctx->params_->optf_->converged_ = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
              PetscFunctionReturn (ierr);
        } else if (stop[4]) {
              ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);                 CHKERRQ(ierr);
          ctx->params_->optf_->converged_ = true;
              PetscFunctionReturn (ierr);
        } else if (stop[5]) {
              ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                 CHKERRQ(ierr);
              ctx->params_->optf_->converged_ = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
    }
    else {
        // if the gradient is zero, we should terminate immediately
        if (gnorm < gatol) {
            ss << "||g|| = " << std::scientific << " < " << gatol;
            ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);ss.str(std::string()); ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
      // store objective function value
      ctx->jvalold = jx;
    }

    // if we're here, we're good to go
    ierr = TaoSetConvergedReason(tao, TAO_CONTINUE_ITERATING);                  CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
}


/* #### ------------------------------------------------------------------- #### */
// #### hessian


/* ------------------------------------------------------------------- */
/*
 hessianMatVec    computes the Hessian matrix-vector product
 input parameters:
  . H       input matrix
  . s       input vector
 output parameters:
  . Hs      solution vector
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode hessianMatVec (Mat A, Vec x, Vec y) {    //y = Ax
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tao-hess-matvec-tumor");
    std::array<double, 7> t = {0}; double self_exec_time = -MPI_Wtime ();
    // get context
    void *ptr;
    ierr = MatShellGetContext (A, &ptr);                                     CHKERRQ (ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>( ptr);
    // eval hessian
    itctx->params_->optf_->nb_matvecs_++;
    ierr = itctx->derivative_operators_->evaluateHessian (y, x);
    if (itctx->params_->tu_->verbosity_ > 1) {
        PetscPrintf (MPI_COMM_WORLD, " applying hessian done!\n");
        ScalarType xnorm;
        ierr = VecNorm (x, NORM_2, &xnorm); CHKERRQ(ierr);
        PetscPrintf (MPI_COMM_WORLD, " norm of search direction ||x||_2 = %e\n", xnorm);
    }
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}

PetscErrorCode matfreeHessian (Tao, Vec, Mat, Mat, void*);

/* ------------------------------------------------------------------- */
/*
 constApxHessianMatVec    computes the Hessian matrix-vector product for
                          a constant approximation \beta_p \Phi^T\Phi of the Hessian
 input parameters:
  . H       input matrix
  . s       input vector
 output parameters:
  . Hs      solution vector
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode constApxHessianMatVec (Mat A, Vec x, Vec y) {    //y = Ax
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Event e ("tao-lmvm-init-hess--matvec");
    std::array<double, 7> t = {0}; double self_exec_time = -MPI_Wtime ();
    // get context
    void *ptr;
    ierr = MatShellGetContext (A, &ptr);                                     CHKERRQ (ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>( ptr);
    // eval hessian
    ierr = itctx->derivative_operators_->evaluateConstantHessianApproximation (y, x);
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode matfreeHessian (Tao tao, Vec x, Mat H, Mat precH, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    PetscFunctionReturn (ierr);
}

/* ------------------------------------------------------------------- */
/*
 preconditionerMatVec    computes the matrix-vector product of inverse of
 preconditioner and some input vector
 input parameters:
 pinv    input matrix (shell context)
 x       input vector
 output parameters:
 .       pinvx   inverse of preconditioner applied to output vector
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode preconditionerMatVec (PC pinv, Vec x, Vec pinvx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    void *ptr;
    // get shell context
    ierr = PCShellGetContext (pinv, &ptr);                  CHKERRQ(ierr);
    // apply the hessian
    ierr = applyPreconditioner (ptr, x, pinvx);             CHKERRQ(ierr);
    PetscFunctionReturn (ierr);
}



/* ------------------------------------------------------------------- */
/*
 preKrylovSolve    preprocess right hand side and initial condition before entering
                   the krylov subspace method; in the context of numerical optimization
                                     this means we preprocess the gradient and the incremental control variable

 input parameters:
  . KSP ksp       KSP solver object
    . Vec b         right hand side
    . Vec x         solution vector
  . void *ptr     optional user defined context
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode preKrylovSolve (KSP ksp, Vec b, Vec x, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscReal gnorm = 0., g0norm = 1., reltol, abstol = 0., divtol = 0., uppergradbound, lowergradbound;
    PetscInt maxit;
    int nprocs, procid;
    MPI_Comm_rank (PETSC_COMM_WORLD, &procid);
    MPI_Comm_size (PETSC_COMM_WORLD, &nprocs);

    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    ierr = VecNorm (b, NORM_2, &gnorm); CHKERRQ(ierr);   // compute gradient norm
    if(itctx->update_reference_gradient_hessian_ksp) {   // set initial gradient norm
        itctx->ksp_gradnorm0 = gnorm;                    // for KSP Hessian solver
        itctx->update_reference_gradient_hessian_ksp = false;
    }
    g0norm = itctx->ksp_gradnorm0;                      // get reference gradient
    gnorm /= g0norm;                                    // normalize gradient
                                                        // get tolerances
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);                   CHKERRQ(ierr);
    uppergradbound = 0.5;                               // assuming quadratic convergence
    lowergradbound = 1E-10;
    // user forcing sequence to estimate adequate tolerance for solution of
    //  KKT system (Eisenstat-Walker)
    if (itctx->params_->opt_->fseqtype_ == QDFS) {
        // assuming quadratic convergence (we do not solver more accurately than 12 digits)
        reltol = PetscMax(lowergradbound, PetscMin(uppergradbound, gnorm));
    }
    else {
        // assuming superlinear convergence (we do not solver  more accurately than 12 digitis)
        reltol = PetscMax (lowergradbound, PetscMin (uppergradbound, std::sqrt(gnorm)));
    }
    // overwrite tolerances with estimate
    ierr = KSPSetTolerances (ksp, reltol, abstol, divtol, maxit);                       CHKERRQ(ierr);

    //if (procid == 0){
    //    std::cout << " ksp rel-tol (Eisenstat/Walker): " << reltol << ", grad0norm: " << g0norm<<", gnorm/grad0norm: " << gnorm << std::endl;
    //}
    PetscFunctionReturn (ierr);
}

/* ------------------------------------------------------------------- */
/*
 applyPreconditioner  apply preconditioner to a given vector
 input parameters:
 ptr       pointer to user defined context
 x         input vector
 output parameters:
 .       pinvx     inverse of preconditioner applied to input vector
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode applyPreconditioner (void *ptr, Vec x, Vec pinvx) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    Event e ("tao-apply-hess-precond");
    std::array<double, 7> t = {0};
    double self_exec_time = -MPI_Wtime ();
    ScalarType *ptr_pinvx = NULL, *ptr_x = NULL;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);
    ierr = VecCopy (x, pinvx);
    // === PRECONDITIONER CURRENTLY DISABLED ===
    PetscFunctionReturn (ierr);
    // apply hessian preconditioner
    //ierr = itctx->derivative_operators_->evaluateHessian(pinvx, x);
    self_exec_time += MPI_Wtime ();
    accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
    e.addTimings (t); e.stop ();
    PetscFunctionReturn (ierr);
}


/* ------------------------------------------------------------------- */
/*
 optimizationMonitor    mointors the inner PCG Krylov solve to invert the Hessian
 input parameters:
  . KSP ksp          KSP solver object
    . PetscIntn        iteration number
    . PetscRela rnorm  l2-norm (preconditioned) of residual
  . void *ptr        optional user defined context
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode hessianKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec x; int maxit; PetscScalar divtol, abstol, reltol;
    ierr = KSPBuildSolution (ksp,NULL,&x);
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);             CHKERRQ(ierr);                                                             CHKERRQ(ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);     // get user context
    itctx->params_->optf_->nb_krylov_it_++;                // accumulate number of krylov iterations

    std::stringstream s;
    if (its == 0) {
      s << std::setw(3)  << " PCG:" << " computing solution of hessian system (tol="
        << std::scientific << std::setprecision(5) << reltol << ")";
      ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
      s.str (""); s.clear ();
    }
    s << std::setw(3)  << " PCG:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
    << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
    ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    s.str (""); s.clear ();

    // int ksp_itr;
    // ierr = KSPGetIterationNumber (ksp, &ksp_itr);                                 CHKERRQ (ierr);
    // ScalarType e_max, e_min;
    // if (ksp_itr % 10 == 0 || ksp_itr == maxit) {
    //   ierr = KSPComputeExtremeSingularValues (ksp, &e_max, &e_min);       CHKERRQ (ierr);
    //   s << "Condition number of hessian is: " << e_max / e_min << " | largest singular values is: " << e_max << ", smallest singular values is: " << e_min << std::endl;
    //   ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    //   s.str (""); s.clear ();
    // }
	PetscFunctionReturn (ierr);
}


/* ------------------------------------------------------------------- */
/*
 constHessianKSPMonitor    mointors the PCG Krylov solve for the constant apx of initial hessian for lmvm
 input parameters:
  . KSP ksp          KSP solver object
    . PetscIntn        iteration number
    . PetscRela rnorm  l2-norm (preconditioned) of residual
  . void *ptr        optional user defined context
 */
 // ### ______________________________________________________________________ ___
 // ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode constHessianKSPMonitor (KSP ksp, PetscInt its, PetscReal rnorm, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    Vec x; int maxit; PetscScalar divtol, abstol, reltol;
    ierr = KSPBuildSolution (ksp,NULL,&x);
    ierr = KSPGetTolerances (ksp, &reltol, &abstol, &divtol, &maxit);             CHKERRQ(ierr);                                                             CHKERRQ(ierr);
    CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);     // get user context

    std::stringstream s;
    if (its == 0) {
      s << std::setw(3)  << " PCG:" << " invert constant apx H = (beta Phi^T Phi) as initial guess for L-BFGS  (tol="
        << std::scientific << std::setprecision(5) << reltol << ")";
      ierr = tuMSGstd (s.str());                                                CHKERRQ(ierr);
      s.str (""); s.clear ();
    }
    s << std::setw(3)  << " PCG:" << std::setw(15) << " " << std::setfill('0') << std::setw(3)<< its
    << "   ||r||_2 = " << std::scientific << std::setprecision(5) << rnorm;
    ierr = tuMSGstd (s.str());                                                    CHKERRQ(ierr);
    s.str (""); s.clear ();
    PetscFunctionReturn (ierr);
}



/* #### ------------------------------------------------------------------- #### */
/* #### ========           Reaction + DIffusion                    ======== #### */
/* #### ------------------------------------------------------------------- #### */



// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode evaluateObjectiveReacDiff (Tao tao, Vec x, PetscReal *J, void *ptr){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj-params");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      x->lock = 0;
    }
  #endif

  itctx->params_->optf_->nb_objevals_++;
  itctx->params_->optf_->nb_gradevals_++;

  // set the last 2-3 entries to the parameters obtained from tao and pass to derivativeoperators
  ScalarType *x_ptr, *x_full_ptr;
  ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  x_full_ptr[itctx->params_->tu_->np_] = x_ptr[0];   // k1
  if (itctx->params_->tu_->nk_ > 1) x_full_ptr[itctx->params_->tu_->np_ + 1] = x_ptr[1];  // k2
  if (itctx->params_->tu_->nk_ > 2) x_full_ptr[itctx->params_->tu_->np_ + 2] = x_ptr[2];  // k3
  x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_] = x_ptr[itctx->params_->tu_->nk_];  // rho
  if (itctx->params_->tu_->nr_ > 1) x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 1] = x_ptr[itctx->params_->tu_->nk_ + 1];  // r2
  if (itctx->params_->tu_->nr_ > 2) x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 2] = x_ptr[itctx->params_->tu_->nk_ + 2];  // r2

  ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      x->lock = lock_state;
    }
  #endif

  ierr = itctx->derivative_operators_->evaluateObjective (J, itctx->x_old, itctx->data);

  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

  PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode evaluateGradientReacDiff (Tao tao, Vec x, Vec dJ, void *ptr){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-grad-tumor-params");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      x->lock = 0;
    }
  #endif

  itctx->params_->optf_->nb_objevals_++;
  itctx->params_->optf_->nb_gradevals_++;

  // set the last 2-3 entries to the parameters obtained from tao and pass to derivativeoperators
  ScalarType *x_ptr, *x_full_ptr;
  ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  x_full_ptr[itctx->params_->tu_->np_] = x_ptr[0];   // k1
  if (itctx->params_->tu_->nk_ > 1) x_full_ptr[itctx->params_->tu_->np_ + 1] = x_ptr[1];  // k2
  if (itctx->params_->tu_->nk_ > 2) x_full_ptr[itctx->params_->tu_->np_ + 2] = x_ptr[2];  // k3
  x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_] = x_ptr[itctx->params_->tu_->nk_];  // rho
  if (itctx->params_->tu_->nr_ > 1) x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 1] = x_ptr[itctx->params_->tu_->nk_ + 1];  // r2
  if (itctx->params_->tu_->nr_ > 2) x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 2] = x_ptr[itctx->params_->tu_->nk_ + 2];  // r2

  ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  Vec dJ_full;
  ierr = VecDuplicate (itctx->x_old, &dJ_full);         CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      x->lock = lock_state;
    }
  #endif

  ierr = itctx->derivative_operators_->evaluateGradient (dJ_full, itctx->x_old, itctx->data);

  ScalarType *dj_ptr, *dj_full_ptr;
  ierr = VecGetArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecGetArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  dj_ptr[0] = dj_full_ptr[itctx->params_->tu_->np_];
  if (itctx->params_->tu_->nk_ > 1) dj_ptr[1] = dj_full_ptr[itctx->params_->tu_->np_ + 1];  // k2
  if (itctx->params_->tu_->nk_ > 2) dj_ptr[2] = dj_full_ptr[itctx->params_->tu_->np_ + 2];  // k3
  dj_ptr[itctx->params_->tu_->nk_] = dj_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_];  // rho
  if (itctx->params_->tu_->nr_ > 1) dj_ptr[itctx->params_->tu_->nk_ + 1] = dj_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 1];  // r2
  if (itctx->params_->tu_->nr_ > 2) dj_ptr[itctx->params_->tu_->nk_ + 2] = dj_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 2];  // r2

  ierr = VecRestoreArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecRestoreArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  std::stringstream s;
  if (itctx->params_->tu_->verbosity_ > 1) {
      ScalarType gnorm;
      ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
      s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

  if (dJ_full  != nullptr) {VecDestroy (&dJ_full);  CHKERRQ(ierr);  dJ_full  = nullptr;}
  PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode evaluateObjectiveAndGradientReacDiff (Tao tao, Vec x, PetscReal *J, Vec dJ, void *ptr){
  PetscFunctionBegin;
  PetscErrorCode ierr = 0;
  Event e ("tao-eval-obj/grad-tumor-params");
  std::array<double, 7> t = {0};
  double self_exec_time = -MPI_Wtime ();
  CtxInv *itctx = reinterpret_cast<CtxInv*>(ptr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    int lock_state;
    ierr = VecLockGet (x, &lock_state);     CHKERRQ (ierr);
    if (lock_state != 0) {
      x->lock = 0;
    }
  #endif

  itctx->params_->optf_->nb_objevals_++;
  itctx->params_->optf_->nb_gradevals_++;

  // set the last 2-3 entries to the parameters obtained from tao and pass to derivativeoperators
  ScalarType *x_ptr, *x_full_ptr;
  ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  x_full_ptr[itctx->params_->tu_->np_] = x_ptr[0];   // k1
  if (itctx->params_->tu_->nk_ > 1) x_full_ptr[itctx->params_->tu_->np_ + 1] = x_ptr[1];  // k2
  if (itctx->params_->tu_->nk_ > 2) x_full_ptr[itctx->params_->tu_->np_ + 2] = x_ptr[2];  // k3
  x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_] = x_ptr[itctx->params_->tu_->nk_];  // rho
  if (itctx->params_->tu_->nr_ > 1) x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 1] = x_ptr[itctx->params_->tu_->nk_ + 1];  // r2
  if (itctx->params_->tu_->nr_ > 2) x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 2] = x_ptr[itctx->params_->tu_->nk_ + 2];  // r2

  ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
  ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

  Vec dJ_full;
  ierr = VecDuplicate (itctx->x_old, &dJ_full);         CHKERRQ (ierr);

  #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (lock_state != 0) {
      x->lock = lock_state;
    }
  #endif

  ierr = itctx->derivative_operators_->evaluateObjectiveAndGradient (J, dJ_full, itctx->x_old, itctx->data);

  ScalarType *dj_ptr, *dj_full_ptr;
  ierr = VecGetArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecGetArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  dj_ptr[0] = dj_full_ptr[itctx->params_->tu_->np_];
  if (itctx->params_->tu_->nk_ > 1) dj_ptr[1] = dj_full_ptr[itctx->params_->tu_->np_ + 1];  // k2
  if (itctx->params_->tu_->nk_ > 2) dj_ptr[2] = dj_full_ptr[itctx->params_->tu_->np_ + 2];  // k3
  dj_ptr[itctx->params_->tu_->nk_] = dj_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_];  // rho
  if (itctx->params_->tu_->nr_ > 1) dj_ptr[itctx->params_->tu_->nk_ + 1] = dj_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 1];  // r2
  if (itctx->params_->tu_->nr_ > 2) dj_ptr[itctx->params_->tu_->nk_ + 2] = dj_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 2];  // r2

  ierr = VecRestoreArray (dJ, &dj_ptr);             CHKERRQ (ierr);
  ierr = VecRestoreArray (dJ_full, &dj_full_ptr);   CHKERRQ (ierr);

  std::stringstream s;
  if (itctx->params_->tu_->verbosity_ > 1) {
      ScalarType gnorm;
      ierr = VecNorm (dJ, NORM_2, &gnorm);                                            CHKERRQ(ierr);
      s << " norm of gradient ||g||_2 = " << std::scientific << gnorm; ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(""); s.clear();
  }
  self_exec_time += MPI_Wtime ();
  accumulateTimers (itctx->params_->tu_->timers_, t, self_exec_time);
  e.addTimings (t);
  e.stop ();

  if (dJ_full  != nullptr) {VecDestroy (&dJ_full);  CHKERRQ(ierr);  dJ_full  = nullptr;}
  PetscFunctionReturn (ierr);
}


/* #### ------------------------------------------------------------------- #### */
// #### convergence


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode optimizationMonitorReacDiff (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    std::stringstream s;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PetscInt its;
    ScalarType J = 0, gnorm = 0, cnorm = 0 , step = 0, D = 0, J0 = 0, D0 = 0, gnorm0 = 0;
    Vec x = nullptr;
    char msg[256];
    std::string statusmsg;
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

    ScalarType norm_gref;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);  CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &x);                                       CHKERRQ(ierr);

    Vec tao_grad;
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr =  TaoGetGradientVector(tao, &tao_grad);                               CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
      norm_gref = gnorm;
      itctx->params_->optf_->gradnorm_0 = norm_gref;
      itctx->update_reference_gradient = false;
      s <<" updated reference gradient for relative convergence criterion, Quasi-Newton solver: " << itctx->params_->optf_->gradnorm_0;
      ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();
    }
    #endif

    // accumulate number of newton iterations
    itctx->params_->optf_->nb_newton_it_++;

    ScalarType *x_ptr, *x_full_ptr;
    ierr = VecGetArray (x, &x_ptr);       CHKERRQ (ierr);
    ierr = VecGetArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

    x_full_ptr[itctx->params_->tu_->np_] = x_ptr[0];   // k1
    if (itctx->params_->tu_->nk_ > 1) x_full_ptr[itctx->params_->tu_->np_ + 1] = x_ptr[1];  // k2
    if (itctx->params_->tu_->nk_ > 2) x_full_ptr[itctx->params_->tu_->np_ + 2] = x_ptr[2];  // k3
    x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_] = x_ptr[itctx->params_->tu_->nk_];  // rho
    if (itctx->params_->tu_->nr_ > 1) x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 1] = x_ptr[itctx->params_->tu_->nk_ + 1];  // r1
    if (itctx->params_->tu_->nr_ > 2) x_full_ptr[itctx->params_->tu_->np_ + itctx->params_->tu_->nk_ + 2] = x_ptr[itctx->params_->tu_->nk_ + 2];  // r1

    ierr = VecRestoreArray (x, &x_ptr);       CHKERRQ (ierr);
    ierr = VecRestoreArray (itctx->x_old, &x_full_ptr);   CHKERRQ (ierr);

    ierr = itctx->tumor_->phi_->apply (itctx->tumor_->c_0_, itctx->x_old);                 CHKERRQ (ierr);
    //Prints a warning if tumor IC is clipped
    ierr = checkClipping (itctx->tumor_->c_0_, itctx->params_);           CHKERRQ (ierr);

    ScalarType mx, mn;
    ierr = VecMax (itctx->tumor_->c_t_, NULL, &mx); CHKERRQ (ierr);
    ierr = VecMin (itctx->tumor_->c_t_, NULL, &mn); CHKERRQ (ierr);
    // this print helps determine if theres any large aliasing errors which is causing ls failure etc
    s << " ---------- tumor c(1) bounds: max = " << mx << ", min = " << mn << " ----------- ";
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();

    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   ";

            s << std::setw(18) << "r1";
            if (itctx->params_->tu_->nr_ > 1) s << std::setw(18) << "r2";
            if (itctx->params_->tu_->nr_ > 2) s << std::setw(18) << "r3";
            s << std::setw(18) << "k1";
            if (itctx->params_->tu_->nk_ > 1) s << std::setw(18) << "k2";
            if (itctx->params_->tu_->nk_ > 2) s << std::setw(18) << "k3";


        ierr = tuMSGstd ("starting optimization for only biophysical parameters");                   CHKERRQ(ierr);

        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                             CHKERRQ(ierr);
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }

    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->params_->optf_->gradnorm_0
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      ;

      ierr = VecGetArray(x, &x_ptr);                                         CHKERRQ(ierr);
      s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->params_->tu_->nk_];
      if (itctx->params_->tu_->nr_ > 1) s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->params_->tu_->nk_ + 1];
      if (itctx->params_->tu_->nr_ > 2) s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[itctx->params_->tu_->nk_ + 2];
      s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[0];
      if (itctx->params_->tu_->nk_ > 1) {
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[1];
      }
      if (itctx->params_->tu_->nk_ > 2) {
        s << "   " << std::scientific << std::setprecision(12) << std::setw(18) << x_ptr[2];
      }


    ierr = VecRestoreArray(x, &x_ptr);                                          CHKERRQ(ierr);

    ierr = tuMSGwarn (s.str());                                                 CHKERRQ(ierr);
    s.str ("");
    s.clear ();

    if (itctx->params_->write_output_ && itctx->params_->tu_->verbosity_ >= 4 && its % 5 == 0) {
        s << "c1guess_paraminvitr-" << its << ".nc";
        dataOut (itctx->tumor_->c_t_, itctx->params_, s.str().c_str());
    }
    s.str(std::string()); s.clear();


    //ierr = PetscPrintf (PETSC_COMM_WORLD, "\nKSP number of krylov iterations: %d\n", itctx->params_->optf_->nb_krylov_it_);          CHKERRQ(ierr);
    //itctx->params_->optf_->nb_krylov_it_ = 0;

    //Gradient check begin
    // ierr = itctx->derivative_operators_->checkGradient (itctx->x_old, itctx->data);
    //Gradient check end
    PetscFunctionReturn (ierr);
}




// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode checkConvergenceGradReacDiff (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscInt its, nl, ng;
    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
    bool stop[3];
    int verbosity;
    std::stringstream ss;
    Vec x = nullptr, g = nullptr;
    ierr = TaoGetSolutionVector(tao, &x);                                       CHKERRQ(ierr);
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->params_->tu_->verbosity_;
    minstep = ctx->params_->opt_->ls_minstep_;
    miniter = ctx->params_->opt_->newton_minit_;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
    #endif

    // get line-search status
    nl = ctx->params_->grid_->nl_;
    ng = ctx->params_->grid_->ng_;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = VecDuplicate (x, &g);                                  CHKERRQ(ierr);
    ierr = TaoLineSearchGetSolution(ls, x, &J, g, &step, &ls_flag);             CHKERRQ (ierr);
    // display line-search convergence reason
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    ierr = TaoGetMaximumIterations(tao, &maxiter);                              CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);

    Vec tao_grad;
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr = TaoGetGradientVector(tao, &tao_grad);                                CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    double norm_gref = 0.;
    // update/set reference gradient
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if (ctx->update_reference_gradient) {
      norm_gref = gnorm;
      ctx->params_->optf_->gradnorm_0 = norm_gref;
      ctx->update_reference_gradient = false;
      std::stringstream s; s <<" updated reference gradient for relative convergence criterion: " << ctx->params_->optf_->gradnorm_0;
      ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    }
    #endif
    // get initial gradient
    g0norm = ctx->params_->optf_->gradnorm_0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    ctx->convergence_message.clear();
    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
      ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
      ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
      ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
    //if(verbosity >= 1) {
    //  ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    //}
    // only check convergence criteria after a certain number of iterations
    stop[0] = false; stop[1] = false; stop[2] = false;
    ctx->params_->optf_->converged_ = false;
    if (iter >= miniter) {
      if (verbosity > 1) {
          ss << "step size in linesearch: " << std::scientific << step;
          ierr = tuMSGstd(ss.str());                                            CHKERRQ(ierr);
          ss.str(std::string());
                ss.clear();
      }
      if (step < minstep) {
          ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
          ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
          ss.str(std::string());
                ss.clear();
          ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
          if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn(ierr);
      }
      if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
        ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
        ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
        ss.str(std::string());
              ss.clear();
        ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
        PetscFunctionReturn(ierr);
      }
      // ||g_k||_2 < tol*||g_0||
      if (gnorm < gttol*g0norm) {
          ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);               CHKERRQ(ierr);
          stop[0] = true;
      }
      ss << "  " << stop[0] << "    ||g|| = " << std::setw(14)
         << std::right << std::scientific << gnorm << "    <    "
         << std::left << std::setw(14) << gttol*g0norm << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
        ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
      ss.str(std::string());
        ss.clear();
      // ||g_k||_2 < tol
      if (gnorm < gatol) {
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
      stop[1] = true;
      }
      ss  << "  " << stop[1] << "    ||g|| = " << std::setw(14)
          << std::right << std::scientific << gnorm << "    <    "
          << std::left << std::setw(14) << gatol << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
        ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
      ss.str(std::string());
        ss.clear();
      // iteration number exceeds limit
      if (iter > maxiter) {
      ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                    CHKERRQ(ierr);
      stop[2] = true;
      }
      ss  << "  " << stop[2] << "     iter = " << std::setw(14)
          << std::right << iter  << "    >    "
          << std::left << std::setw(14) << maxiter << " = " << "maxiter";
      ctx->convergence_message.push_back(ss.str());
      if(verbosity >= 3) {
        ierr = tuMSGstd(ss.str());                                              CHKERRQ(ierr);
      }
      ss.str(std::string());
        ss.clear();
      // store objective function value
      ctx->jvalold = J;
      if (stop[0] || stop[1] || stop[2]) {
        ctx->params_->optf_->converged_ = true;
        if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
        PetscFunctionReturn(ierr);
      }

    }
    else {
    // if the gradient is zero, we should terminate immediately
    if (gnorm < gatol) {
      ss << "||g|| = " << std::scientific << 0.0 << " < " << gatol  << " = " << "bound";
      ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);
      ss.str(std::string());
            ss.clear();
      ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
      PetscFunctionReturn(ierr);
    }
    }
    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);

    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}

    PetscFunctionReturn (ierr);
}



/* #### ------------------------------------------------------------------- #### */
/* #### ======== Mass-Effect: Reaction + Diffusion + Forcing Factor ======= #### */
/* #### ------------------------------------------------------------------- #### */


/* #### ------------------------------------------------------------------- #### */
// #### convergence


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode optimizationMonitorMassEffect (Tao tao, void *ptr) {
    // first to monitor then to checkconv in petsc 3.11
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
    PetscInt its;
    ScalarType J = 0, gnorm = 0, cnorm = 0 , step = 0, D = 0, J0 = 0, D0 = 0, gnorm0 = 0;
    Vec x = nullptr;
    char msg[256];
    std::string statusmsg;
    std::stringstream s;
    TaoConvergedReason flag;
    CtxInv *itctx = reinterpret_cast<CtxInv*> (ptr);

    Vec tao_grad;

    // get current iteration, objective value, norm of gradient, norm of
    // norm of contraint, step length / trust region readius of iteratore
    // and termination reason
    Vec tao_x;
    ierr = TaoGetSolutionStatus (tao, &its, &J, &gnorm, &cnorm, &step, &flag);  CHKERRQ(ierr);
    ierr = TaoGetSolutionVector(tao, &tao_x);                                   CHKERRQ(ierr);
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr =  TaoGetGradientVector(tao, &tao_grad);                               CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    PetscInt num_feval, n2, n3;
    TaoLineSearch ls = nullptr;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = TaoLineSearchGetNumberFunctionEvaluations (ls, &num_feval, &n2, &n3);   CHKERRQ (ierr);

    ScalarType step_tol = std::pow(2, -3);
    // adaptive ls step
    if (step < step_tol) {
        itctx->step_init = step * 2;
    } else {
        itctx->step_init *= 2;
    }
    itctx->step_init = std::min(itctx->step_init, (ScalarType)1);
    //itctx->step_init = 1;

    ierr = TaoLineSearchSetInitialStepLength (ls, itctx->step_init);            CHKERRQ(ierr);


    // update/set reference gradient
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
    if (itctx->update_reference_gradient) {
        itctx->params_->optf_->gradnorm_0 = gnorm;
        itctx->params_->optf_->j0_ = J;
        itctx->update_reference_gradient = false;
        std::stringstream s; s <<" updated reference gradient for relative convergence criterion: " << itctx->params_->optf_->gradnorm_0;
        ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    }
    #endif

    itctx->params_->optf_->nb_newton_it_++;

    ScalarType *tao_x_ptr;
    ierr = VecGetArray (tao_x, &tao_x_ptr);                                     CHKERRQ (ierr);

    ScalarType mx, mn;
    ierr = VecMax (itctx->tumor_->c_t_, NULL, &mx); CHKERRQ (ierr);
    ierr = VecMin (itctx->tumor_->c_t_, NULL, &mn); CHKERRQ (ierr);
    // this print helps determine if theres any large aliasing errors which is causing ls failure etc
    s << " ---------- tumor c(1) bounds: max = " << mx << ", min = " << mn << " ----------- ";
    ierr = tuMSGstd(s.str()); CHKERRQ(ierr);s.str ("");s.clear ();

    if (its == 0) {
        s << std::setw(4)  << " iter"              << "   " << std::setw(18) << "objective (abs)" << "   "
          << std::setw(18) << "||gradient||_2,rel" << "   " << std::setw(18) << "||gradient||_2"  << "   "
          << std::setw(18) << "step" << "   ";
        s << std::setw(18) << "gamma";
        s << std::setw(18) << "rho";
        s << std::setw(18) << "kappa";

        if(itctx->params_->opt_->newton_solver_ == QUASINEWTON) {
          ierr = tuMSGstd (" starting optimization, TAO's Quasi-Newton");            CHKERRQ(ierr);
        } else {
          ierr = tuMSGstd (" starting optimization, TAO's Gauss-Newton");            CHKERRQ(ierr);
        }
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        ierr = tuMSGwarn (s.str());                                                  CHKERRQ(ierr);
        ierr = tuMSGstd ("---------------------------------------------------------------------------------------------------------------"); CHKERRQ(ierr);
        s.str ("");
        s.clear ();
    }

    s << " "   << std::scientific << std::setprecision(5) << std::setfill('0') << std::setw(4) << its << std::setfill(' ')
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << J
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm/itctx->params_->optf_->gradnorm_0
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << gnorm
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << step
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << tao_x_ptr[0]
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << tao_x_ptr[1]
      << "   " << std::scientific << std::setprecision(12) << std::setw(18) << tao_x_ptr[2];

    ierr = tuMSGwarn (s.str());                                                    CHKERRQ(ierr);
    s.str ("");s.clear ();

    ierr = VecRestoreArray (tao_x, &tao_x_ptr);                                    CHKERRQ(ierr);
    if (procid == 0) {
        ierr = VecView (tao_grad, PETSC_VIEWER_STDOUT_SELF);                           CHKERRQ(ierr);
    }
    //Gradient check begin
    // ierr = itctx->derivative_operators_->checkGradient (tao_x, itctx->data);
    // ierr = itctx->derivative_operators_->checkHessian (tao_x, itctx->data);
    //Gradient check end
    PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode checkConvergenceGradMassEffect (Tao tao, void *ptr) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    PetscInt its, nl, ng;
    PetscInt iter, maxiter, miniter;
    PetscReal J, gnorm, step, gatol, grtol, gttol, g0norm, minstep;
    PetscReal jx, jxold, gtolbound, theta, normx, normdx, tolj, tolx, tolg;
    const int nstop = 7;
    bool stop[nstop];
    int verbosity;
    std::stringstream ss;
    Vec x = nullptr, g = nullptr;
    ierr = TaoGetSolutionVector(tao, &x);                                     CHKERRQ(ierr);
    TaoLineSearch ls = nullptr;
    TaoLineSearchConvergedReason ls_flag;

    CtxInv *ctx = reinterpret_cast<CtxInv*> (ptr);     // get user context
    verbosity = ctx->params_->tu_->verbosity_;
    minstep = ctx->params_->opt_->ls_minstep_;
    miniter = ctx->params_->opt_->newton_minit_;
    // get tolerances
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 7)
        ierr = TaoGetTolerances (tao, &gatol, &grtol, &gttol);                  CHKERRQ(ierr);
    #else
        ierr = TaoGetTolerances( tao, NULL, NULL, &gatol, &grtol, &gttol);      CHKERRQ(ierr);
    #endif

    // get line-search status
    nl = ctx->params_->grid_->nl_;
    ng = ctx->params_->grid_->ng_;
    ierr = TaoGetLineSearch(tao, &ls);                                          CHKERRQ (ierr);
    ierr = VecDuplicate (x, &g);
    ierr = TaoLineSearchGetSolution(ls, x, &J, g, &step, &ls_flag);             CHKERRQ (ierr);
    // display line-search convergence reason
    ierr = dispLineSearchStatus(tao, ctx, ls_flag);                             CHKERRQ(ierr);
    ierr = TaoGetMaximumIterations(tao, &maxiter);                              CHKERRQ(ierr);
    ierr = TaoGetSolutionStatus(tao, &iter, &J, &gnorm, NULL, &step, NULL);     CHKERRQ(ierr);
    jx = J;
    Vec tao_grad;
    // get gradient vector norm for bqnls since gnorm is a different residual in this algorithm
    ierr = TaoGetGradientVector(tao, &tao_grad);                                CHKERRQ(ierr);
    ierr = VecNorm (tao_grad, NORM_2, &gnorm);                                  CHKERRQ (ierr);

    // update/set reference gradient
    #if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR < 9)
    if (ctx->update_reference_gradient) {
        ctx->params_->optf_->gradnorm_0 = gnorm;
        ctx->params_->optf_->j0_ = jx;
        ctx->update_reference_gradient = false;
        std::stringstream s; s <<" updated reference gradient for relative convergence criterion: " << ctx->params_->optf_->gradnorm_0;
        ierr = tuMSGstd(s.str());                                                 CHKERRQ(ierr);
    }
    #endif


    g0norm = ctx->params_->optf_->gradnorm_0;
    g0norm = (g0norm > 0.0) ? g0norm : 1.0;
    // compute tolerances for stopping conditions
    tolj = grtol;
    tolx = std::sqrt(grtol);
    #if __cplusplus > 199711L
    tolg = std::cbrt(grtol);
    #else
    tolg = std::pow(grtol, 1.0/3.0);
    #endif
    // compute theta
    theta = 1.0 + std::abs(ctx->params_->optf_->j0_);
    ierr = VecNorm (x, NORM_2, &normx);                                                CHKERRQ(ierr);  // comp norm x
    ierr = VecAXPY (ctx->x_old, -1.0, x);                                              CHKERRQ(ierr);  // comp dx
    ierr = VecNorm (ctx->x_old, NORM_2, &normdx);                                      CHKERRQ(ierr);  // comp norm dx
    ierr = VecCopy (x, ctx->x_old);                                                    CHKERRQ(ierr);  // save old x
    // get old objective function value
    jxold = ctx->jvalold;
    ctx->convergence_message.clear();

    // check for NaN value
    if (PetscIsInfOrNanReal(J)) {
          ierr = tuMSGwarn ("objective is NaN");                                      CHKERRQ(ierr);
          ierr = TaoSetConvergedReason (tao, TAO_DIVERGED_NAN);                       CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn (ierr);
    }
    // check for NaN value
    if (PetscIsInfOrNanReal(gnorm)) {
          ierr = tuMSGwarn("||g|| is NaN");                                           CHKERRQ(ierr);
          ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_NAN);                        CHKERRQ(ierr);
      if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
          PetscFunctionReturn(ierr);
    }
    //if(verbosity >= 1) {
    //    ierr = PetscPrintf (MPI_COMM_WORLD, "||g(x)|| / ||g(x0)|| = %6E, ||g(x0)|| = %6E \n", gnorm/g0norm, g0norm);
    //}
    // only check convergence criteria after a certain number of iterations
    // initialize flags for stopping conditions
    for (int i = 0; i < nstop; i++)
        stop[i] = false;
    ctx->params_->optf_->converged_     = false;
    ctx->cosamp_->converged_l2       = false;
    ctx->cosamp_->converged_error_l2 = false;
    if (iter >= miniter && iter > 1) {
        if (step < minstep) {
            ss << "step  = " << std::scientific << step << " < " << minstep << " = " << "bound";
            ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
            ss.str(std::string());
            ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_STEPTOL);             CHKERRQ(ierr);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
        if (ls_flag != 1 && ls_flag != 0 && ls_flag != 2) {
            ss << "step  = " << std::scientific << step << ". ls failed with status " << ls_flag;
            ierr = tuMSGwarn(ss.str()); CHKERRQ(ierr);
            ss.str(std::string());
                  ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_LS_FAILURE);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
        // |j(x_{k-1}) - j(x_k)| < tolj*abs(1+J)
        if (std::abs(jxold-jx) < tolj*theta) {
            stop[0] = true;
        }
        ss << "  " << stop[0] << "    |dJ|  = " << std::setw(18)
        << std::right << std::scientific << std::abs(jxold-jx) << "    <    "
        << std::left << std::setw(18) << tolj*theta << " = " << "tol*|1+J|";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||dx|| < sqrt(tolj)*(1+||x||)
        if (normdx < tolx*(1+normx)) {
           stop[1] = true;
        }
        ss << "  " << stop[1] << "    |dx|  = " << std::setw(18)
        << std::right << std::scientific << normdx << "    <    "
        << std::left << std::setw(18) << tolx*(1+normx) << " = " << "sqrt(tol)*(1+||x||)";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||g_k||_2 < cbrt(tolj)*abs(1+Jc)
        if (gnorm < tolg*theta) {
            stop[2] = true;
        }
        ss  << "  " << stop[2] << "    ||g|| = " << std::setw(18)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(18) << tolg*theta << " = " << "cbrt(tol)*|1+J|";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();
        // ||g_k||_2 < tol
        if (gnorm < gatol || std::abs(jxold-jx) <= PETSC_MACHINE_EPSILON)  {
                stop[3] = true;
        }
        ss  << "  " << stop[3] << "    ||g|| = " << std::setw(18)
        << std::right << std::scientific << gnorm << "    <    "
        << std::left << std::setw(18) << gatol << " = " << "tol";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        if (gnorm < gttol*g0norm) {
          stop[4] = true;
      }
        ss << "  " << stop[4] << "    ||g|| = " << std::setw(18)
         << std::right << std::scientific << gnorm << "    <    "
         << std::left << std::setw(18) << gttol*g0norm << " = " << "tol";
      ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        if (iter > maxiter) {
                stop[5] = true;
        }
        ss  << "  " << stop[5] << "    iter  = " << std::setw(18)
          << std::right << iter  << "    >    "
          << std::left << std::setw(18) << maxiter << " = " << "maxiter";
        ctx->convergence_message.push_back(ss.str());
        ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        ss.str(std::string());
        ss.clear();

        //     if (iter > iterbound) {
        //             stop[6] = true;
        //     }
        //     ss  << "  " << stop[6] << "    iter  = " << std::setw(14)
              // << std::right << iter  << "    >    "
              // << std::left << std::setw(14) << iterbound << " = " << "iterbound";
        //     ctx->convergence_message.push_back(ss.str());
        //     ierr = tuMSGstd(ss.str());                                                     CHKERRQ(ierr);
        //     ss.str(std::string());
        //     ss.clear();

        // store objective function value
        ctx->jvalold = jx;

        if (stop[0] && stop[1] && stop[2]) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_USER);                  CHKERRQ(ierr);
            ctx->params_->optf_->converged_ = true;
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        } else if (stop[3]) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                 CHKERRQ(ierr);
            ctx->params_->optf_->converged_ = true;
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        } else if (stop[4]) {
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GTTOL);                 CHKERRQ(ierr);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            ctx->params_->optf_->converged_ = true;
            PetscFunctionReturn (ierr);
        } else if (stop[5]) {
            ierr = TaoSetConvergedReason(tao, TAO_DIVERGED_MAXITS);                 CHKERRQ(ierr);
            ctx->params_->optf_->converged_ = true;
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
    }
    else {
        // if the gradient is zero, we should terminate immediately
        if (gnorm < gatol) {
            ss << "||g|| = " << std::scientific << " < " << gatol;
            ierr = tuMSGwarn(ss.str());                                               CHKERRQ(ierr);ss.str(std::string()); ss.clear();
            ierr = TaoSetConvergedReason(tao, TAO_CONVERGED_GATOL);                   CHKERRQ(ierr);
            if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
            PetscFunctionReturn (ierr);
        }
        // store objective function value
        ctx->jvalold = jx;
    }
    // if we're here, we're good to go
    ierr = TaoSetConvergedReason (tao, TAO_CONTINUE_ITERATING);                 CHKERRQ(ierr);
    if (g != NULL) {ierr = VecDestroy(&g); CHKERRQ(ierr); g = NULL;}
    PetscFunctionReturn (ierr);
}



/* #### ------------------------------------------------------------------- #### */
// #### misc

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode operatorCreateVecsMassEffect (Mat A, Vec *left, Vec *right) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;

    CtxInv *ctx;
    ierr = MatShellGetContext (A, &ctx);                        CHKERRQ (ierr);

    if (right) {
        ierr = VecDuplicate (ctx->x_old, right);             CHKERRQ(ierr);
    }
    if (left) {
        ierr = VecDuplicate (ctx->x_old, left);              CHKERRQ(ierr);
    }
    PetscFunctionReturn (ierr);
}

// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode dispTaoConvReason (TaoConvergedReason flag, std::string &msg) {
    PetscFunctionBegin;
    PetscErrorCode ierr = 0;
    switch (flag) {
      case TAO_CONVERGED_GATOL : {
          msg = "solver converged: ||g(x)|| <= tol";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_GRTOL : {
          msg = "solver converged: ||g(x)||/J(x) <= tol";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_GTTOL : {
          msg = "solver converged: ||g(x)||/||g(x0)|| <= tol";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_STEPTOL : {
          msg = "step size too small";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_MINF : {
          msg = "objective value to small";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONVERGED_USER : {
          msg = "solver converged user";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_MAXITS : {
          msg = "maximum number of iterations reached";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_NAN : {
          msg = "numerical problems (NAN detected)";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_MAXFCN : {
          msg = "maximal number of function evaluations reached";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_LS_FAILURE : {
          msg = "line search failed";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_TR_REDUCTION : {
          msg = "trust region failed";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_DIVERGED_USER : {
          msg = "user defined divergence criterion met";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
      case TAO_CONTINUE_ITERATING : {
          // display nothing
          break;
      }
      default : {
          msg = "convergence reason not defined";
          ierr = tuMSGwarn(msg);                                    CHKERRQ(ierr);
          break;
      }
  }
    PetscFunctionReturn (ierr);
}


// ### ______________________________________________________________________ ___
// ### ////////////////////////////////////////////////////////////////////// ###
PetscErrorCode dispLineSearchStatus(Tao tao, void* ptr, TaoLineSearchConvergedReason flag) {
    PetscErrorCode ierr = 0;
    std::string msg;
    PetscFunctionBegin;

    switch(flag) {
        case TAOLINESEARCH_FAILED_INFORNAN:
        {
            msg = "linesearch: function evaluation gave INF or NaN";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_FAILED_BADPARAMETER:
        {
            msg = "linesearch: bad parameter detected";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_FAILED_ASCENT:
        {
            msg = "linesearch: search direction is not a descent direction";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_MAXFCN:
        {
            msg = "linesearch: maximum number of function evaluations reached";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_UPPERBOUND:
        {
            msg = "linesearch: step size reached upper bound";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_LOWERBOUND:
        {
            msg = "linesearch: step size reached lower bound";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_RTOL:
        {
            msg = "linesearch: range of uncertainty is smaller than given tolerance";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_HALTED_OTHER:
        {
            msg = "linesearch: stopped (other)";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
        case TAOLINESEARCH_CONTINUE_ITERATING:
        {
            // do nothing, cause everything's fine
            break;
        }
        case TAOLINESEARCH_SUCCESS:
        {
            msg = "linesearch: successful";
            ierr = tuMSGstd(msg); CHKERRQ(ierr);
            break;
        }
        default:
        {
            msg = "linesearch: status not defined";
            ierr = tuMSGwarn(msg); CHKERRQ(ierr);
            break;
        }
    }
    PetscFunctionReturn (ierr);
}
