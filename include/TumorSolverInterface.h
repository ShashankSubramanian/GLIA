#ifndef TUMORSOLVERINTERFACE_H_
#define TUMORSOLVERINTERFACE_H_

#include "Utils.h"
#include "Tumor.h"
#include "PdeOperators.h"
#include "DerivativeOperators.h"
#include "InvSolver.h"

class TumorSolverInterface {
	public :
		TumorSolverInterface (std::shared_ptr<NMisc> n_misc = {}, std::shared_ptr<Phi> phi = {}, std::shared_ptr<MatProp> mat_prop = {});
		/// @brief initializes the TumorSolverInterface
		PetscErrorCode initialize (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Phi> phi = {}, std::shared_ptr<MatProp> mat_prop = {});


    int npChangedResetComponents();
		PetscErrorCode setParams (Vec p, std::shared_ptr<TumorSettings> tumor_params);
		/** @brief Solves the forward tumor problem, given initial concentration
		*         and tumor parameters
		*  @param Vec c0  - initial tumor concentration
		*  @param Vec cT  - target tumor concentration after simulation
		*/
		PetscErrorCode solveForward (Vec cT, Vec c0);
		/** @brief Solves the inverse tumor problem using Tao, given target concentration
		 *
		 *  @param Vec d1     - tumor inverse target data
		 *  @param Vec p_rec, - reconstructed parameters for initial condition  c_rec = \Phi p_rec
		 */
		PetscErrorCode solveInverse (Vec prec, Vec d1, Vec d1g = {});
		/// @brief updates the initial guess for the inverse tumor solver
		PetscErrorCode setInitialGuess (Vec p);
		PetscErrorCode setInitialGuess(double d);
		PetscErrorCode resetTaoSolver();

		PetscErrorCode setDistMeassureSimulationGeoImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
      		return derivative_operators_->setDistMeassureSimulationGeoImages(wm, gm, csf, glm, bg);
		}
		PetscErrorCode setDistMeassureTargetDataImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
	    	return derivative_operators_->setDistMeassureTargetDataImages(wm, gm, csf, glm, bg);
		}
		PetscErrorCode setDistMeassureDiffImages(Vec wm, Vec gm, Vec csf, Vec glm, Vec bg) {
      		return derivative_operators_->setDistMeassureDiffImages(wm, gm, csf, glm, bg);
		}
		/** @brief updates the reaction and diffusion coefficients depending on
		 *         the probability maps for GRAY MATTER, WHITE MATTER and CSF.
		 *         A additional filter, that filters the admissable area for tumor
		 *         growth has to be passed (updates the \Phi filter)
		 */
		PetscErrorCode updateTumorCoefficients (Vec wm, Vec gm, Vec glm, Vec csf, Vec filter, std::shared_ptr<TumorSettings> tumor_params, bool use_nmisc = false);
		/// @brief evaluates gradient for given control variable p and data
		PetscErrorCode computeGradient(Vec dJ, Vec p, Vec data_gradeval);
		/// @brief true if TumorSolverInterface is initialized and ready to use

		bool isInitialized () {
			return initialized_;
		}
		int getNumberGaussians() {
			return n_misc_->np_;
		}
		/// @brief updates/sets the optimizer settings (copies settings)
		void setOptimizerSettings (std::shared_ptr<OptimizerSettings> optset);
		/// @brief sets the optimization feedback ptr in inv_solver, overrides old ptr
		void setOptimizerFeedback (std::shared_ptr<OptimizerFeedback> optfeed) {inv_solver_->setOptFeedback(optfeed);}
		// defines whether or not we have to update the reference gradeient for the inverse solve
		void updateReferenceGradient (bool b) {if (inv_solver_ != nullptr) inv_solver_->updateReferenceGradient(b);}
		/** @brief computes effect of varying/moving material properties, i.e.,
		 *  computes q = int_T dK / dm * (grad c)^T grad * \alpha + dRho / dm c(1-c) * \alpha dt
		 */
		PetscErrorCode computeTumorContributionRegistration(Vec q1, Vec q2, Vec q3, Vec q4) {
			PetscErrorCode ierr;
			if (pde_operators_ != nullptr) {
			  ierr = pde_operators_->computeTumorContributionRegistration(q1, q2, q3, q4); CHKERRQ(ierr);}
			PetscFunctionReturn(0);
		}
		//  ---------  getter functions -------------
		/// @brief returns the tumor shared ptr
		std::shared_ptr<Tumor> getTumor () {
			return tumor_;
		}
		/// @brief returns the context for the inverse tumor solver
		std::shared_ptr<CtxInv> getITctx () {return inv_solver_->getInverseSolverContext();}
		~TumorSolverInterface () {}

		std::shared_ptr<InvSolver> getInvSolver () {return inv_solver_;}

		PetscErrorCode solveInterpolation (Vec data, Vec p_rec, std::shared_ptr<Phi> phi, std::shared_ptr<NMisc> n_misc);

		std::vector<double> getSolverOutParams () {
			out_params_ = inv_solver_->getInvOutParams ();
			return out_params_;
		}

	private :
	  bool initialized_;
		bool optimizer_settings_changed_;
		std::shared_ptr<NMisc> n_misc_;
		std::shared_ptr<Tumor> tumor_;
		std::shared_ptr<PdeOperators> pde_operators_;
		std::shared_ptr<DerivativeOperators> derivative_operators_;
		std::shared_ptr<InvSolver> inv_solver_;

		std::vector<double> out_params_;
};

#endif
