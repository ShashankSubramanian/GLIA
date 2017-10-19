#ifndef DERIVATIVEOPERATORS_H_
#define DERIVATIVEOPERATORS_H_

#include "PdeOperators.h"

class DerivativeOperators {
	public :
		DerivativeOperators (std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc,
				std::shared_ptr<Tumor> tumor)
				: pde_operators_ (pde_operators), n_misc_ (n_misc), tumor_ (tumor) {
					VecDuplicate (tumor_->c_0_, &temp_);
					VecDuplicate (tumor_->p_, &ptemp_);
				}

		std::shared_ptr<PdeOperators> pde_operators_;
		std::shared_ptr<Tumor> tumor_;
		std::shared_ptr<NMisc> n_misc_;

		Vec temp_;
		Vec ptemp_;

		virtual PetscErrorCode evaluateObjective (PetscReal *J, Vec x, Vec data) = 0;
		virtual PetscErrorCode evaluateGradient (Vec dJ, Vec x, Vec data) = 0;
		virtual PetscErrorCode evaluateHessian (Vec y, Vec x) = 0;

		virtual ~DerivativeOperators () {}
};

class DerivativeOperatorsRD : public DerivativeOperators {
	public :
		DerivativeOperatorsRD (std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc,
				std::shared_ptr<Tumor> tumor)
			 : DerivativeOperators (pde_operators, n_misc, tumor) {}

		PetscErrorCode evaluateObjective (PetscReal *J, Vec x, Vec data);
		PetscErrorCode evaluateGradient (Vec dJ, Vec x, Vec data);
		PetscErrorCode evaluateHessian (Vec y, Vec x);

		~DerivativeOperatorsRD () { 
			VecDestroy (&temp_);
			VecDestroy (&ptemp_);
		}
};

#endif
