#ifndef TUMORSOLVERINTERFACE_H_
#define TUMORSOLVERINTERFACE_H_

#include "Utils.h"
#include "Tumor.h"
#include "PdeOperators.h"
#include "DerivativeOperators.h"
#include "InvSolver.h"

class TumorSolverInterface {
	public :
		TumorSolverInterface (std::shared_ptr<NMisc> n_misc);

		PetscErrorCode solveForward ();
		PetscErrorCode solveInverse ();

		~TumorSolverInterface () {}

	private :
		std::shared_ptr<NMisc> n_misc_;
		std::shared_ptr<Tumor> tumor_;
		std::shared_ptr<PdeOperators> pde_operators_;
		std::shared_ptr<DerivativeOperators> derivative_operators_;
		std::shared_ptr<InvSolver> inv_solver_;
};

#endif