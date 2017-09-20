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

		/** @brief Solves the forward tumor problem, given initial concentration
	   *         and tumor parameters
	   *  @param Vec c0  - initial tumor concentration
	   *  @param Vec cT  - target tumor concentration after simulation
	   */
		PetscErrorCode solveForward (Vec c0, Vec cT);

		/** @brief Solves the inverse tumor problem using Tao, given target concentration
		 *
		 *  @param Vec d1     - tumor inverse target data
		 *  @param Vec p_rec, - reconstructed parameters for initial condition  c_rec = \Phi p_rec
		 */
		PetscErrorCode solveInverse (Vec d1, Vec p_rec);

		/// @brief updates the initial guess for the inverse tumor solver
		PetscErrorCode setInitialGuess(Vec p);

		/** @brief updates the reaction and diffusion coefficients depending on
		 *         the probability maps for GRAY MATTER, WHITE MATTER and CSF.
		 *         A additional filter, that filters the admissable area for tumor
		 *         growth has to be passed (updates the \Phi filter)
		 */
		PetscErrorCode updateTumorCoefficients(std::shared_ptr<MatProp> geometry, Vec p);

    /// @brief true if TumorSolverInterface is initialized and ready to use
		bool isInitialized() {return initialized_;}


    //  ---------  getter functions -------------
    /// @brief returns the tumor shared ptr
		std::shared_ptr<Tumor> getTumor() { return tumor_; }
		/// @brief returns the context for the inverse tumor solver
	  std::shared_ptr<CtxInv> getITctx();// {return itctx_;}

		~TumorSolverInterface () {}

	private :
	  bool initialized_;
		std::shared_ptr<NMisc> n_misc_;
		std::shared_ptr<Tumor> tumor_;
		std::shared_ptr<PdeOperators> pde_operators_;
		std::shared_ptr<DerivativeOperators> derivative_operators_;
		std::shared_ptr<InvSolver> inv_solver_;
};

#endif
