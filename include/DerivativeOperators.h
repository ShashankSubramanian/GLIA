#ifndef DERIVATIVEOPERATORS_H_
#define DERIVATIVEOPERATORS_H_

#include "PdeOperators.h"

class DerivativeOperators {
	public :
		DerivativeOperators (std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc) 
				: pde_operators_ (pde_operators), n_misc_ (n_misc) {}

		std::shared_ptr <PdeOperators> pde_operators_;
		std::shared_ptr <NMisc> n_misc_;

		virtual PetscErrorCode evaluateObjective (PetscReal *J, Vec x) = 0;
		virtual PetscErrorCode evaluateGradient (Vec dJ, Vec x) = 0;
		virtual PetscErrorCode evaluateHessian (Vec x, Vec y) = 0;

		~DerivativeOperators () {}
};

class DerivativeOperatorsRD : public DerivativeOperators {
	public :
		DerivativeOperatorsRD (std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc) 
			 : DerivativeOperators (pde_operators, n_misc) {}

		PetscErrorCode evaluateObjective (PetscReal *J, Vec x);
		PetscErrorCode evaluateGradient (Vec dJ, Vec x);
		PetscErrorCode evaluateHessian (Vec x, Vec y);

		~DerivativeOperatorsRD () {}
};

#endif