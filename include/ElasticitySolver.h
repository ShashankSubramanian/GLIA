#ifndef ELASTICITYSOLVER_H_
#define ELASTICITYSOLVER_H_

#include "Utils.h"
#include "Tumor.h"
#include <mpi.h>
#include <omp.h>

struct CtxElasticity {
	std::shared_ptr<NMisc> n_misc_;
	std::shared_ptr<Tumor> tumor_;

	PetscScalar mu_avg_, lam_avg_, screen_avg_;

	double computeMu (double E, double nu) {
		return (E / (2 * (1 + nu)));
	}

	double computeLam (double E, double nu) {
		return (nu * E / ((1 + nu) * (1 - 2 * nu)));
	}
};

class ElasticitySolver {
	public:
		ElasticitySolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor);   //tumor is needed for its work vectors

		KSP ksp_;
		Mat A_;
		Vec rhs_;

		std::shared_ptr<CtxElasticity> ctx_;

		virtual PetscErrorCode solve (std::vector<Vec> displacement, std::vector<Vec> rhs) = 0;

		virtual ~ElasticitySolver ();

};

class VariableLinearElasticitySolver : public ElasticitySolver {
	public:
		VariableLinearElasticitySolver (std::shared_ptr<NMisc> n_misc, std::shared_ptr<Tumor> tumor) : ElasticitySolver (n_misc, tumor) {}
		virtual PetscErrorCode solve (std::vector<Vec> displacement, std::vector<Vec> rhs);
		virtual ~VariableLinearElasticitySolver () {}
};

//Helper functions 
PetscErrorCode operatorConstantCoefficients (Mat A, Vec x, Vec y);

#endif
