#ifndef PDEOPERATORS_H_
#define PDEOPERATORS_H_

#include "Utils.h"
#include "Tumor.h"
#include "DiffSolver.h"

#include <mpi.h>
#include <omp.h>

class PdeOperators {
	public:
		PdeOperators (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc) : tumor_(tumor), n_misc_(n_misc) {
			diff_solver_ = std::make_shared<DiffSolver> (n_misc, tumor->k_);
			nt_ = n_misc->nt_;
		}

		std::shared_ptr<Tumor> tumor_;
		std::shared_ptr<DiffSolver> diff_solver_;
		std::shared_ptr<NMisc> n_misc_;

		virtual PetscErrorCode solveState (int linearized) = 0;
		virtual PetscErrorCode solveAdjoint (int linearized) = 0;

		virtual ~PdeOperators () {}


	protected:
			/// @brief local copy of nt, bc if parameters change, pdeOperators needs to
			/// be re-constructed. However, the destructor has to use the nt value that
			/// was used upon construction of that object, not the changed value in nmisc
			int nt_;
};

class PdeOperatorsRD : public PdeOperators {
	public:
		PdeOperatorsRD (std::shared_ptr<Tumor> tumor, std::shared_ptr<NMisc> n_misc);

    // @brief time history of state variable
		Vec *c_;
		// @brief time history of adjoint variable
		Vec *p_;

		PetscErrorCode solveState (int linearized);
		PetscErrorCode reaction (int linearized, int i);
		PetscErrorCode reactionAdjoint (int linearized, int i);
		PetscErrorCode solveAdjoint (int linearized);

		/** @brief computes effect of varying/moving material properties, i.e.,
		 *  computes q = int_T dK / dm * (grad c)^T grad * \alpha + dRho / dm c(1-c) * \alpha dt
		 */
    PetscErrorCode computeVaryingMatProbContribution(Vec q);
		~PdeOperatorsRD ();

};


#endif
