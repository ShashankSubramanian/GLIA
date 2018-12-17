#ifndef DERIVATIVEOPERATORS_H_
#define DERIVATIVEOPERATORS_H_

#include "PdeOperators.h"
#include "Utils.h"

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
		virtual PetscErrorCode evaluateObjectiveAndGradient (PetscReal *J,Vec dJ, Vec x, Vec data) {};
		virtual PetscErrorCode evaluateHessian (Vec y, Vec x) = 0;
		virtual PetscErrorCode evaluateConstantHessianApproximation (Vec y, Vec x) {};

        virtual PetscErrorCode setDistMeassureSimulationGeoImages (Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {PetscFunctionReturn(0);}
        virtual PetscErrorCode setDistMeassureTargetDataImages (Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {PetscFunctionReturn(0);}
		virtual PetscErrorCode setDistMeassureDiffImages (Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {PetscFunctionReturn(0);}
        PetscErrorCode checkGradient (Vec p, Vec data);

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
			 : DerivativeOperators (pde_operators, n_misc, tumor) {
				 // tuMSGstd (" ----- Setting reaction-diffusion derivative operators --------");
			 }

		PetscErrorCode evaluateObjective (PetscReal *J, Vec x, Vec data);
		PetscErrorCode evaluateGradient (Vec dJ, Vec x, Vec data);
		PetscErrorCode evaluateObjectiveAndGradient (PetscReal *J,Vec dJ, Vec x, Vec data);
		PetscErrorCode evaluateHessian (Vec y, Vec x);
		virtual PetscErrorCode evaluateConstantHessianApproximation (Vec y, Vec x);
		~DerivativeOperatorsRD () {}

		//Vec work_np_;  // vector of size np to compute objective and part of gradient related to p
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

class DerivativeOperatorsRDObj : public DerivativeOperators {
	public :
		DerivativeOperatorsRDObj (std::shared_ptr <PdeOperators> pde_operators, std::shared_ptr <NMisc> n_misc,
				std::shared_ptr<Tumor> tumor) : DerivativeOperators (pde_operators, n_misc, tumor) {
					tuMSGstd (" ----- Setting RD derivative operators with modified objective --------");
				}

        PetscErrorCode evaluateObjective (PetscReal *J, Vec x, Vec data);
        PetscErrorCode evaluateGradient (Vec dJ, Vec x, Vec data);
		PetscErrorCode evaluateObjectiveAndGradient (PetscReal *J,Vec dJ, Vec x, Vec data);
        PetscErrorCode evaluateHessian (Vec y, Vec x);

        /** @brief: Sets the image vectors for the simulation geometry material properties
				 *  - MOVING PATIENT: mA(0) (= initial helathy atlas)
				 *  - MOVING ATLAS:   mA(1) (= initial helathy patient)
				 */
        PetscErrorCode setDistMeassureSimulationGeoImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
        	m_geo_wm_ = wm; m_geo_gm_ = gm; m_geo_csf_ = csf; m_geo_glm_ = glm; m_geo_bg_ = bg;
            nc_ = (wm != nullptr) + (gm != nullptr) + (csf != nullptr) + (glm != nullptr);
            PetscFunctionReturn(0);
        }

				/** @brief: Sets the image vectors for the target (patient) geometry material properties
				 *  - MOVING PATIENT: mP(1) (= advected patient)
				 *  - MOVING ATLAS:   mR    (= patient data)
				 */
        PetscErrorCode setDistMeassureTargetDataImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
        	m_data_wm_ = wm; m_data_gm_ = gm; m_data_csf_ = csf; m_data_glm_ = glm; m_data_bg_ = bg;
            PetscFunctionReturn(0);
        }

				/** @brief: Sets the image vectors for the distance measure difference
				 *  - MOVING PATIENT: || mA(0)(1-c(1)) - mP(1) ||^2
				 *  - MOVING ATLAS:   || mA(1)(1-c(1)) - mR    ||^2
				 */
        PetscErrorCode setDistMeassureDiffImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
            xi_wm_ = wm; xi_gm_ = gm; xi_csf_ = csf; xi_glm_ = glm; xi_bg_ = bg;
            PetscFunctionReturn(0);
        }

        ~DerivativeOperatorsRDObj () {}

	private :

		/** @brief: Image vectors for the simulation geometry material properties (memory from outsie;)
		 *  - MOVING PATIENT: mA(0) (= initial helathy atlas)      (reference image)
		 *  - MOVING ATLAS:   mA(1) (= initial helathy patient)    (template image)
		 */
		Vec m_geo_wm_, m_geo_gm_, m_geo_csf_, m_geo_glm_, m_geo_bg_;
		/** @brief: Image vectors for the target (patient) geometry material properties (memory from outsie;)
	 	 *  - MOVING PATIENT: mP(1) (= advected patient)           (template image)
		 *  - MOVING ATLAS:   mR    (= patient data)               (reference image)
		 */
		Vec m_data_wm_, m_data_gm_, m_data_csf_, m_data_glm_, m_data_bg_;

		/** @brief: Image vectors for the distance measure difference
		 *  - MOVING PATIENT: || mA(0)(1-c(1)) - mP(1) ||^2  (negative of the inner term)
		 *  - MOVING ATLAS:   || mA(1)(1-c(1)) - mR    ||^2  (negative of the inner term)
		 */
		Vec xi_wm_, xi_gm_, xi_csf_, xi_glm_, xi_bg_;
        // / number of components in objective function
		int nc_;
};


#endif
