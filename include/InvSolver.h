#ifndef INVSOLVER_H_
#define INVSOLVER_H_

#include "DerivativeOperators.h"

struct Ctx {
	DerivativeOperators *derivative_operators_;
	NMisc *n_misc_;
};

class InvSolver {
	public :
		InvSolver (PdeOperators pde_operators, NMisc *n_misc) {}

		Tao tao_;
		Mat A_;

		PetscErrorCode solve ();

		~InvSolver () {}
};

#endif