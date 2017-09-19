#ifndef OBS_H_
#define OBS_H_

#include "Utils.h"
#include <mpi.h>
#include <omp.h>

class Obs {
	public:
		Obs (std::shared_ptr<NMisc> n_misc, Vec data, double obs_thr);

		double threshold_;
		Vec filter_;

		PetscErrorCode setFilter (Vec &custom_filter);
		PetscErrorCode apply (Vec &y, Vec x);
		// PetscErrorCode apply_transpose (double *y, double *x);


		~Obs ();
};


#endif