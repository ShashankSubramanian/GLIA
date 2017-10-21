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

    virtual PetscErrorCode setDistMeassureReferenceImage(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) = 0;
    virtual PetscErrorCode setDistMeassureTemplateImage(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) = 0;

		virtual ~DerivativeOperators () {
			VecDestroy (&temp_);
			VecDestroy (&ptemp_);
		}
};

class DerivativeOperatorsRD : public DerivativeOperators {
	public :
		DerivativeOperatorsRD (std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc,
				std::shared_ptr<Tumor> tumor)
			 : DerivativeOperators (pde_operators, n_misc, tumor) {}

		virtual PetscErrorCode evaluateObjective (PetscReal *J, Vec x, Vec data);
		virtual PetscErrorCode evaluateGradient (Vec dJ, Vec x, Vec data);
		virtual PetscErrorCode evaluateHessian (Vec y, Vec x);


		virtual ~DerivativeOperatorsRD () {}
};

class DerivativeOperatorsRDObj : public DerivativeOperators {
	public :
		DerivativeOperatorsRDObj (std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc,
				std::shared_ptr<Tumor> tumor)
			 : DerivativeOperators (pde_operators, n_misc, tumor) {}

		virtual PetscErrorCode evaluateObjective (PetscReal *J, Vec x, Vec data);
		virtual PetscErrorCode evaluateGradient (Vec dJ, Vec x, Vec data);
    virtual PetscErrorCode evaluateHessian (Vec y, Vec x);

		virtual PetscErrorCode setDistMeassureReferenceImage(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
			mR_wm_ = wm; mR_gm_ = gm; mR_csf_ = csf; mR_glm_ = glm; mR_bg_ = bg; PetscFunctionReturn(0);}
		virtual PetscErrorCode setDistMeassureTemplateImage(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
			mT_wm_ = wm; mT_gm_ = gm; mT_csf_ = csf; mT_glm_ = glm; mT_bg_ = bg; PetscFunctionReturn(0);}
	  virtual PetscErrorCode setGeometricCouplingAdjoint(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
		  xi_wm_ = wm; xi_gm_ = gm; xi_csf_ = csf; xi_glm_ = glm; xi_bg_ = bg; PetscFunctionReturn(0);}


		virtual ~DerivativeOperatorsRDObj () {}

	private:
		/// @brief reference image for brain difference measure || mR - mT||^2 (memory from outsie;)
		Vec mR_wm_, mR_gm_, mR_csf_, mR_glm_, mR_bg_;
		/// @brief tmeplate image for brain difference measure || mR - mT||^2 (memory from outsie;)
		Vec mT_wm_, mT_gm_, mT_csf_, mT_glm_, mT_bg_;
		/// @brief tmeplate image for brain difference measure || mR - mT||^2 (memory from outsie;)
		Vec xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_;
};

#endif
