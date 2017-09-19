#ifndef INVSOLVER_H_
#define INVSOLVER_H_

#include "DerivativeOperators.h"

struct CtxInv {
	std::shared_ptr<DerivativeOperators> derivative_operators_;
	std::shared_ptr<NMisc> n_misc_;
};

class InvSolver {
	public :
		InvSolver (std::shared_ptr <DerivativeOperators> derivative_operators, std::shared_ptr <NMisc> n_misc);

		Tao tao_;
		Mat A_;

		PetscErrorCode solve ();

		~InvSolver ();
};

#endif