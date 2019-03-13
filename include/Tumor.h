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

    	// parametrization
		Vec p_;
		Vec p_true_;
		// state variables
		Vec c_t_;
		Vec c_0_;
		// adjoint Variables
		Vec p_t_;
		Vec p_0_;
		// work vectors
    	Vec *work_;
    	// weights for w-l2
    	Vec weights_;

    	// mass effect parameters
    	// velocity
		std::vector<Vec> velocity_;
		std::vector<Vec> displacement_;
		std::vector<Vec> force_;

		PetscErrorCode initialize (Vec p, std::shared_ptr<NMisc> n_misc, std::shared_ptr<Phi> phi = {}, std::shared_ptr<MatProp> mat_prop = {});
		PetscErrorCode setParams (Vec p, std::shared_ptr<NMisc> n_misc, bool npchanged = false);
		PetscErrorCode setTrueP (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode setTrueP (std::shared_ptr<NMisc> n_misc, PetscScalar val);
		PetscErrorCode setTrueP (Vec p);

		~Tumor ();
};

#endif
