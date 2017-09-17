/*
Tumor class
*/

#ifndef TUMOR_H_
#define TUMOR_H_

#include "Utils.h"
#include "MatProp.h"
#include "DiffCoef.h"
#include "ReacCoef.h"
#include "Phi.h"
#include "Obs.h"

#include <mpi.h>
#include <omp.h>


class Tumor {
	public:
		Tumor (NMisc *n_misc);

		DiffCoef *k_;
		ReacCoef *rho_;
		Phi *phi_;
		Obs *obs_;

		MatProp *mat_prop_;

		Vec p_;
		Vec c_t_;
		Vec c_0_;

		PetscErrorCode setValues (double k, double rho, double *user_cm, Vec p, NMisc *n_misc);
		PetscErrorCode runForward (NMisc *n_misc); //TODO -- Time history class to be added; Is it necesary?

		~Tumor ();
};

#endif
