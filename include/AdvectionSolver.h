#ifndef ADVECTIONSOLVER_H_
#define ADVECTIONSOLVER_H_

#include "Utils.h"
#include "Tumor.h"
#include <mpi.h>
#include <omp.h>

#include "Interp.h"

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

// Solve transport equations using semi-Lagrangian
// Some class variables will remain null depending on the interpolation functions used and is controlled by preprocessor dirs CUDA
// and USEMPICUDA
class SemiLagrangianSolver : public AdvectionSolver {
	public:
		SemiLagrangianSolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor, std::shared_ptr<SpectralOperators> spec_ops);

		std::shared_ptr<SpectralOperators> spec_ops_; 
		int isize_g_[3], istart_g_[3];    		  			// local input sizes with ghost layers
		std::shared_ptr<VecField> work_field_;	  			// work vector field
		std::shared_ptr<VecField> coords_;		  			// x,y,z coordinates 
		Vec *temp_;								  			// temp vectors	
		Vec query_points_;						  			// query point coordinates

		int m_dofs_[2];  						  			// controls number of interpolation plans: we need two
		int n_ghost_;							  			// ghost padding number = order of interpolation
		int n_alloc_;							  			// allocation size with ghosts

		double *scalar_field_ghost_;			  			// local scalar field with ghost points
		double *vector_field_ghost_;			  			// local vector field with ghost points

		std::shared_ptr<InterpPlan> interp_plan_vector_;	// plans for interpolation on multi-GPU
		std::shared_ptr<InterpPlan> interp_plan_scalar_;    // plans for interpolation on multi-GPU

		std::shared_ptr<InterpPlan> interp_plan_; 			// plan for interpolation on the cpu

		float *temp_interpol1_;								// temporary floats for more efficient GPU interpolation
		float *temp_interpol2_;

		#ifdef CUDA
			cudaTextureObject_t m_texture_;			  			//  cuda texture object for interp - only defined in cuda header files
		#endif

		virtual PetscErrorCode solve (Vec scalar, std::shared_ptr<VecField> velocity, double dt);	// solve transport equation
		PetscErrorCode computeTrajectories ();														// Computes RK2 trajectories and query points
		PetscErrorCode interpolate (Vec out, Vec in);												// Interpolated scalar field
		PetscErrorCode interpolate (std::shared_ptr<VecField> out, std::shared_ptr<VecField> in);	// Interpolates vector field

		virtual ~SemiLagrangianSolver ();
};

//Helper functions for KSP solve
PetscErrorCode operatorAdv (Mat A, Vec x, Vec y);

#endif
