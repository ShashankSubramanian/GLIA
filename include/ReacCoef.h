#ifndef REACCOEF_H_
#define REACCOEF_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>

class ReacCoef {
	public:
		ReacCoef (NMisc *n_misc);

		int smooth_flag_;

		double rho_scale_;
		Vec rho_vec_;

		PetscErrorCode setValues (double rho_scale, MatProp *mat_prop, NMisc *n_misc);
		PetscErrorCode smooth (NMisc *n_misc);

		~ReacCoef ();
};


#endif
