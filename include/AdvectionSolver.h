#ifndef ADVECTIONSOLVER_H_
#define ADVECTIONSOLVER_H_

#include "Utils.h"
#include "Tumor.h"
#include <mpi.h>
#include <omp.h>

struct CtxAdv {
	std::shared_ptr<NMisc> n_misc_;
	std::vector<Vec> temp_;
	std::vector<Vec> velocity_;
	double dt_;
};

class AdvectionSolver {
	public:
		AdvectionSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor);   //tumor is needed for its work vectors

		KSP ksp_;
		Mat A_;
		Vec rhs_;

		std::shared_ptr<CtxAdv> ctx_;

		virtual PetscErrorCode solve (Vec scalar, std::vector<Vec> velocity, double dt) = 0;

		virtual ~AdvectionSolver () {}

};

// Solve transport equations using Crank-Nicolson
class TrapezoidalSolver : public AdvectionSolver {
	public:
		TrapezoidalSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) : AdvectionSolver (n_misc, tumor) {}
		virtual PetscErrorCode solve (Vec scalar, std::vector<Vec> velocity, double dt);

		virtual ~TrapezoidalSolver ();
};

//Solve transport equations using semi-Lagrangian
class SemiLagrangianSolver : public AdvectionSolver {
	public:
		SemiLagrangianSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) : AdvectionSolver (n_misc, tumor) {}
		virtual PetscErrorCode solve (Vec scalar, std::vector<Vec> velocity, double dt);

		virtual ~SemiLagrangianSolver () {}
};

//Helper functions for KSP solve
PetscErrorCode operatorAdv (Mat A, Vec x, Vec y);

#endif
