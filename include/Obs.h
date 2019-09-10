#ifndef OBS_H_
#define OBS_H_

#include "Utils.h"
#include <mpi.h>
#include <omp.h>

class Obs {
	public:
		Obs (std::shared_ptr<NMisc> n_misc);

		ScalarType threshold_;
		Vec filter_;
		std::shared_ptr<NMisc> n_misc_;

		PetscErrorCode setDefaultFilter (Vec data);
		PetscErrorCode setCustomFilter (Vec custom_filter);
		PetscErrorCode apply (Vec y, Vec x);
		// PetscErrorCode apply_transpose (ScalarType *y, ScalarType *x);


		~Obs ();
};


#endif