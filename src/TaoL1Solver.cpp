#include "TaoL1Solver.h"

PetscErrorCode TaoCreate_ISTA (Tao tao) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	TaoCtx *ctx = (TaoCtx*) tao->data;
	ierr = PetscNewLog (tao, &ctx);				CHKERRQ (ierr);
	tao->data = (void*) ctx;

	//Setup tao routines
	tao->ops->setup = TaoSetup_ISTA;
	tao->ops->solve = TaoSolve_ISTA;
	tao->ops->view = TaoView_ISTA;
	// tao->ops->setfromoptions = TaoSetFromOptions_ISTA;
	tao->ops->destroy = TaoDestroy_ISTA;

	//Setup tao linesearch
	ierr = TaoLineSearchRegister ("ista_ls", TaoLineSearchCreate_ISTA);						CHKERRQ (ierr);
	ierr = TaoLineSearchCreate (((PetscObject)tao)->comm, &tao->linesearch);				CHKERRQ (ierr);
	ierr = TaoLineSearchSetType (tao->linesearch, "ista_ls"); 								CHKERRQ (ierr);
	ierr = TaoLineSearchUseTaoRoutines (tao->linesearch, tao); 								CHKERRQ (ierr);

	PetscFunctionReturn (0);
}

PetscErrorCode TaoLineSearchCreate_ISTA (TaoLineSearch ls) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	LSCtx *ctx;
	ierr = PetscNewLog (ls, &ctx);				CHKERRQ (ierr);

	ctx->sigma = 1e-5;
	ctx->J_old = 0;
	ctx->x_work_1 = nullptr;
	ctx->x_work_2 = nullptr;
	ctx->x_sol = nullptr;

	ls->data = (void*) ctx;
	ls->initstep = 1.0;
	ls->ops->setup = 0;
	ls->ops->apply = TaoLineSearchApply_ISTA;
	ls->ops->destroy = TaoLineSearchDestroy_ISTA;

	PetscFunctionReturn (0);
}

PetscErrorCode TaoLineSearchDestroy_ISTA (TaoLineSearch ls) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	LSCtx *ctx = (LSCtx*) ls->data;
	ierr = VecDestroy (&ctx->x_work_1);		    CHKERRQ (ierr);
	ierr = VecDestroy (&ctx->x_sol);			CHKERRQ (ierr);
	ierr = VecDestroy (&ctx->x_work_2);		    CHKERRQ (ierr);

	ierr = PetscFree (ls->data);			    CHKERRQ (ierr);
	ls->data = NULL;

	PetscFunctionReturn (0);
}

PetscErrorCode proximalOperator (Vec y, Vec x, double lambda, PetscReal step) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	ierr = vecSign (x); //x = sign (x)
	double *y_ptr;
	int size;
	ierr = VecGetArray (y, &y_ptr);				CHKERRQ (ierr);
	ierr = VecGetSize (y, &size);				CHKERRQ (ierr);

	for (int i = 0; i < size; i++) {
		y_ptr[i] = PetscAbsReal (y_ptr[i]) - step * lambda;
		y_ptr[i] = PetscMax (0, y_ptr[i]);
	}

	ierr = VecRestoreArray (y, &y_ptr);			CHKERRQ (ierr);
	ierr = VecPointwiseMult (y, y, x);			CHKERRQ (ierr);

	PetscFunctionReturn (0);
}

PetscErrorCode TaoLineSearchApply_ISTA (TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);

    PCOUT << "(user-defined linesearch begin)\n";

	LSCtx *ctx = (LSCtx*) ls->data;

	if (PetscIsInfOrNanReal (*f)) {
		ierr = PetscInfo (ls, "ISTA linesearch error: function is inf or nan\n");		CHKERRQ (ierr);
		ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
		PetscFunctionReturn (0);
	}

	PetscReal f_old = *f;
	ctx->J_old = f_old;
	double norm = 0;

	ierr = VecNorm (g, NORM_2, &norm);									CHKERRQ (ierr);
	if (PetscIsInfOrNanReal (norm)) {
		ierr = PetscInfo (ls, "ISTA linesearch error: gradient is inf or nan\n");		CHKERRQ (ierr);
		ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
		PetscFunctionReturn (0);
	}

	ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;

	if (ctx->x_work_1 == nullptr) {
		ierr = VecDuplicate (x, &ctx->x_work_1);								CHKERRQ (ierr);
	}
	if (ctx->x_work_2 == nullptr) {
		ierr = VecDuplicate (x, &ctx->x_work_2);								CHKERRQ (ierr);
	}
	if (ctx->x_sol == nullptr) {
		ierr = VecDuplicate (x, &ctx->x_sol);					    			CHKERRQ (ierr);
	}
	
	ls->step = ls->initstep;
	while (ls->step >= ls->stepmin) {	//Default step min is 1e-20
		ierr = VecCopy (x, ctx->x_work_1);									CHKERRQ (ierr);
		ierr = VecAXPY (ctx->x_work_1, -1.0 * ls->step, g);					CHKERRQ (ierr); //gradient descent guess

		ierr = VecCopy (ctx->x_work_1, ctx->x_work_2);						CHKERRQ (ierr);
		ierr = proximalOperator (ctx->x_work_2, ctx->x_work_1, ctx->lambda, ls->step); 
		ierr = VecCopy (ctx->x_work_2, ctx->x_sol);							CHKERRQ (ierr);

		//Sufficient descent criterion
		ierr = TaoLineSearchComputeObjective (ls, ctx->x_work_2, f);	    CHKERRQ (ierr);
		ierr = VecAXPY (ctx->x_work_2, -1.0, x);							CHKERRQ (ierr);
		ierr = VecNorm (ctx->x_work_2, NORM_2, &norm);				    	CHKERRQ (ierr);

		if (*f <= f_old - 0.5 * ctx->sigma * (1.0 / ls->step) * norm * norm) {
			ls->reason = TAOLINESEARCH_SUCCESS;
			break;
		}
		ls->step *= 0.5;
	}

	if (ls->step < 0.25)
		ls->initstep = ls->step * 4.0; //start next linesearch at one order of magnitude higher
	else
		ls->initstep = 1.0;

	if (PetscIsInfOrNanReal (*f)) {
		ierr = PetscInfo (ls, "Function is inf or nan\n");				CHKERRQ (ierr);
		ls->reason = TAOLINESEARCH_FAILED_INFORNAN;
		PetscFunctionReturn (0);
	} else if (ls->step < ls->stepmin) {
		ierr = PetscInfo (ls, "Step length is below tolerance\n");		CHKERRQ (ierr);
		ls->reason = TAOLINESEARCH_HALTED_LOWERBOUND;
		PetscFunctionReturn (0);
	}

	ierr = VecCopy (x, ctx->x_work_2);									CHKERRQ (ierr);	//copy the old solution to work
																						//Used in convergence tests
	ierr = VecCopy (ctx->x_sol, x);										CHKERRQ (ierr);

	PetscFunctionReturn (0);
}


PetscErrorCode TaoSetup_ISTA (Tao tao) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	TaoCtx *ctx = (TaoCtx*) tao->data;

	ierr = VecDuplicate (tao->solution, &tao->gradient);				CHKERRQ (ierr);
	ierr = VecDuplicate (tao->solution, &tao->stepdirection);		    CHKERRQ (ierr);

	PetscFunctionReturn (0);
}


PetscErrorCode TaoView_ISTA (Tao tao, PetscViewer viewer) {
	//TODO
}

PetscErrorCode TaoDestroy_ISTA (Tao tao) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	TaoCtx *ctx = (TaoCtx*) tao->data;
	PetscFree (tao->data);
	tao->data = NULL;
	PetscFunctionReturn (0);
}

PetscErrorCode TaoSolve_ISTA (Tao tao) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	TaoCtx *ctx = (TaoCtx*) tao->data;
	Vec x = tao->solution;
	Vec g = tao->gradient;
	Vec s = tao->stepdirection;

	TaoLineSearchConvergedReason lsflag = TAOLINESEARCH_CONTINUE_ITERATING;
	TaoConvergedReason reason = TAO_CONTINUE_ITERATING;

	PetscReal f, gnorm, steplength = 0;
	PetscInt iter = 0;
	ierr = TaoComputeObjectiveAndGradient (tao, x, &f, g);						CHKERRQ (ierr);
	ierr = VecNorm (g, NORM_2, &gnorm); 										CHKERRQ(ierr);
	ierr = TaoLineSearchSetInitialStepLength (tao->linesearch, 1.0);			CHKERRQ (ierr);
	ierr = VecCopy (g, s);														CHKERRQ (ierr);

	while (1) {
		#if (PETSC_VERSION_MAJOR >= 3) && (PETSC_VERSION_MINOR >= 9)
			ierr = TaoMonitor (tao, iter, f, gnorm, 0.0, steplength);				CHKERRQ (ierr);
		#else
			ierr = TaoMonitor (tao, iter, f, gnorm, 0.0, steplength, &reason);		CHKERRQ (ierr);
		#endif
		if (reason != TAO_CONTINUE_ITERATING) 
			break;
		ierr = TaoComputeObjectiveAndGradient (tao, x, &f, g);								CHKERRQ (ierr);
		ierr = TaoLineSearchApply (tao->linesearch, x, &f, g, s, &steplength, &lsflag);		CHKERRQ (ierr);		//Perform linesearch and update function, solution and gradient values
		ierr = VecNorm (g, NORM_2, &gnorm);												    CHKERRQ (ierr);
		iter++;
		tao->niter = iter;   //For some reason, TaoMonitor does not do this: manually update
	}

	PetscFunctionReturn (0);
}