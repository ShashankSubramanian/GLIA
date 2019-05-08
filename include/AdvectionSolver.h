#ifndef ADVECTIONSOLVER_H_
#define ADVECTIONSOLVER_H_

#include "Utils.h"
#include "Tumor.h"
#include <mpi.h>
#include <omp.h>

// #include "Interp.h"

struct CtxAdv {
	std::shared_ptr<NMisc> n_misc_;
	std::vector<Vec> temp_;
	std::shared_ptr<VecField> velocity_;
	double dt_;
};

class AdvectionSolver {
	public:
		AdvectionSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor);   //tumor is needed for its work vectors

		KSP ksp_;
		Mat A_;
		Vec rhs_;

		std::shared_ptr<CtxAdv> ctx_;

		int advection_mode_;						  // controls the source term of the advection equation

		virtual PetscErrorCode solve (Vec scalar, std::shared_ptr<VecField> velocity, double dt) = 0;

		virtual ~AdvectionSolver ();

};

// Solve transport equations using Crank-Nicolson
class TrapezoidalSolver : public AdvectionSolver {
	public:
		TrapezoidalSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) : AdvectionSolver (n_misc, tumor) {}
		virtual PetscErrorCode solve (Vec scalar, std::shared_ptr<VecField> velocity, double dt);

		virtual ~TrapezoidalSolver () {}
};

//Solve transport equations using semi-Lagrangian
// class SemiLagrangianSolver : public AdvectionSolver {
// 	public:
// 		SemiLagrangianSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor);

// 		int m_dofs_[2];  						  // controls number of interpolation plans: we need two
// 		Vec query_points_;						  // query point coordinates
// 		int n_ghost_;							  // ghost padding number = order of interpolation
// 		std::shared_ptr<InterpPlan> interp_plan_; // plan for interpolation
// 		double *scalar_field_ghost_;			  // local scalar field with ghost points
// 		double *vector_field_ghost_;			  // local vector field with ghost points
// 		std::shared_ptr<VecField> work_field_;	  // work vector field
// 		Vec *temp_;								  // temp vectors	
		
// 		virtual PetscErrorCode solve (Vec scalar, std::shared_ptr<VecField> velocity, double dt);	// solve transport equation
// 		PetscErrorCode computeTrajectories ();														// Computes RK2 trajectories and query points
// 		PetscErrorCode interpolate (Vec out, Vec in);												// Interpolated scalar field
// 		PetscErrorCode interpolate (std::shared_ptr<VecField> out, std::shared_ptr<VecField> in);	// Interpolates vector field

// 		virtual ~SemiLagrangianSolver ();
// };

//Helper functions for KSP solve
PetscErrorCode operatorAdv (Mat A, Vec x, Vec y);

#endif
