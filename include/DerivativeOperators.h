#ifndef DERIVATIVEOPERATORS_H_
#define DERIVATIVEOPERATORS_H_

#include "PdeOperators.h"

class DerivativeOperators {
	public :
		DerivativeOperators (std::shared_ptr <PdeOperators> pde_operators) : pde_operators_ (pde_operators) {}

		std::shared_ptr <PdeOperators> pde_operators_;

		virtual PetscErrorCode evaluateObjective (PetscReal *J, Vec x) = 0;
		virtual PetscErrorCode evaluateGradient (PetscReal *dJ, Vec x) = 0;
		virtual PetscErrorCode evaluateHessian (Vec x, Vec y) = 0;

		~DerivativeOperators () {}
};

class DerivativeOperatorsRD : public DerivativeOperators {
	public :
		DerivativeOperatorsRD (std::shared_ptr <PdeOperators> pde_operators) : PdeOperators (pde_operators) {}

		PetscErrorCode evaluateObjective (PetscReal *J, Vec x);
		PetscErrorCode evaluateGradient (PetscReal *dJ, Vec x);
		PetscErrorCode evaluateHessian (Vec x, Vec y);

		~DerivativeOperatorsRD () {}
};

#endif