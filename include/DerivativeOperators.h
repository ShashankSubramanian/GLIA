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
                    VecDuplicate (tumor_->p_, &p_current_);
				}

		std::shared_ptr<PdeOperators> pde_operators_;
		std::shared_ptr<Tumor> tumor_;
		std::shared_ptr<NMisc> n_misc_;

		Vec temp_;
		Vec ptemp_;
        Vec p_current_; //Current solution vector in newton iteration                                

		virtual PetscErrorCode evaluateObjective (PetscReal *J, Vec x, Vec data) = 0;
		virtual PetscErrorCode evaluateGradient (Vec dJ, Vec x, Vec data) = 0;
		virtual PetscErrorCode evaluateHessian (Vec y, Vec x) = 0;

		virtual ~DerivativeOperators () { 
			VecDestroy (&temp_);
			VecDestroy (&ptemp_);
            VecDestroy (&p_current_);
		}
};

class DerivativeOperatorsRD : public DerivativeOperators {
	public :
		DerivativeOperatorsRD (std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc,
				std::shared_ptr<Tumor> tumor)
			 : DerivativeOperators (pde_operators, n_misc, tumor) {}

		PetscErrorCode evaluateObjective (PetscReal *J, Vec x, Vec data);
		PetscErrorCode evaluateGradient (Vec dJ, Vec x, Vec data);
		PetscErrorCode evaluateHessian (Vec y, Vec x);

		~DerivativeOperatorsRD () {}
};

class DerivativeOperatorsPos : public DerivativeOperators {
	public :
		DerivativeOperatorsPos (std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc,
				std::shared_ptr<Tumor> tumor)
			 : DerivativeOperators (pde_operators, n_misc, tumor) {
                VecDuplicate (temp_, &temp_phip_);
                VecDuplicate (temp_, &temp_phiptilde_);
             }

        Vec temp_phip_;
        Vec temp_phiptilde_;

		PetscErrorCode evaluateObjective (PetscReal *J, Vec x, Vec data);
		PetscErrorCode evaluateGradient (Vec dJ, Vec x, Vec data);
		PetscErrorCode evaluateHessian (Vec y, Vec x);

        PetscErrorCode sigmoid (Vec, Vec);

		~DerivativeOperatorsPos () {
            VecDestroy (&temp_phip_);
            VecDestroy (&temp_phiptilde_);
        }
};

#endif
