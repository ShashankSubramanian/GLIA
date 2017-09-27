#ifndef TUMORSOLVERINTERFACE_H_
#define TUMORSOLVERINTERFACE_H_

#include "Utils.h"
#include "Tumor.h"
#include "PdeOperators.h"
#include "DerivativeOperators.h"
#include "InvSolver.h"

class TumorSolverInterface {
	public :
		TumorSolverInterface (std::shared_ptr<NMisc> n_misc = {});
		/// @brief initializes the TumorSolverInterface
		PetscErrorCode initialize (std::shared_ptr<NMisc> n_misc);

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
		/** @brief updates the reaction and diffusion coefficients depending on
		 *         the probability maps for GRAY MATTER, WHITE MATTER and CSF.
		 *         A additional filter, that filters the admissable area for tumor
		 *         growth has to be passed (updates the \Phi filter)
		 */
		PetscErrorCode updateTumorCoefficients (Vec wm, Vec gm, Vec glm, Vec csf, Vec filter, std::shared_ptr<TumorSettings> tumor_params);
		/// @brief evaluates gradient for given control variable p and data
		PetscErrorCode computeGradient(Vec dJ, Vec p, Vec data_gradeval);
		/// @brief true if TumorSolverInterface is initialized and ready to use
		bool isInitialized () {
			return initialized_;
		}
		/// @brief updates/sets the optimizer settings
		void setOptimizerSettings (std::shared_ptr<OptimizerSettings> optset);
		//  ---------  getter functions -------------
		/// @brief returns the tumor shared ptr
		std::shared_ptr<Tumor> getTumor () {
			return tumor_;
		}
		/// @brief returns the context for the inverse tumor solver
		std::shared_ptr<CtxInv> getITctx ();// {return itctx_;}
		~TumorSolverInterface () {}

	private :
	  bool initialized_;
		bool optimizer_settings_changed_;
		std::shared_ptr<NMisc> n_misc_;
		std::shared_ptr<Tumor> tumor_;
		std::shared_ptr<PdeOperators> pde_operators_;
		std::shared_ptr<DerivativeOperators> derivative_operators_;
		std::shared_ptr<InvSolver> inv_solver_;
};

#endif
