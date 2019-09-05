/*
Environment for creating new Tao solver for PDE-constrained L1 minimization
*/
#ifndef _TAOL1SOLVER_H
#define _TAOL1SOLVER_H

#include "Utils.h"
#include "petsc/private/taoimpl.h"
#include "petsc/private/taolinesearchimpl.h"
#undef __FUNCT__
#define __FUNCT__ __func__


namespace pglistr {
// Context to hold relevant parameters for the solve
struct TaoCtx {
	PetscReal f_tol;
	PetscReal x_tol;
};

PetscErrorCode TaoCreate_ISTA (Tao tao);	//Create new tao solver
PetscErrorCode TaoSetup_ISTA (Tao tao);		//Setup tao solver -- allocate memory for work vectors
// PetscErrorCode TaoSetFromOptions_ISTA (Tao tao, void *solver);    //Set thresholds and other algorithm-specific options
PetscErrorCode TaoView_ISTA (Tao tao, PetscViewer viewer);		  //Output statistics at end of tao solve
PetscErrorCode TaoDestroy_ISTA (Tao tao);				  //Destroy algorithm-specific data
PetscErrorCode TaoSolve_ISTA (Tao tao);		//Solver

//Linesearch routines
PetscErrorCode TaoLineSearchCreate_ISTA (TaoLineSearch ls);
PetscErrorCode TaoLineSearchDestroy_ISTA (TaoLineSearch ls);
PetscErrorCode TaoLineSearchApply_ISTA (TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s);

//Other
PetscErrorCode proximalOperator (Vec y, Vec x, double lambda, PetscReal step);
}

#endif
