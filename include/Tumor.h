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
		Tumor (std::shared_ptr<NMisc> n_misc);

		std::shared_ptr<DiffCoef> k_;
		std::shared_ptr<ReacCoef> rho_;
		std::shared_ptr<Phi> phi_;
		std::shared_ptr<Obs> obs_;

		std::shared_ptr<MatProp> mat_prop_;

		Vec p_;
		Vec c_t_;
		Vec c_0_;

		//Adjoint Variables
		Vec p_t_;
		Vec p_0_;

		PetscErrorCode initialize (Vec p, std::shared_ptr<NMisc> n_misc);

		~Tumor ();
};

#endif
