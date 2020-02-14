#ifndef OBS_H_
#define OBS_H_

#include "Utils.h"
#include <mpi.h>
#include <omp.h>

class Obs {
	public:
		Obs (std::shared_ptr<NMisc> n_misc);

		ScalarType threshold_1_;
        ScalarType threshold_0_;
		Vec filter_1_;
        Vec filter_0_;
        bool two_snapshot_;
		std::shared_ptr<NMisc> n_misc_;

		PetscErrorCode setDefaultFilter (Vec data, int time_point);
		PetscErrorCode setCustomFilter (Vec custom_filter, int time_point);
		PetscErrorCode apply (Vec y, Vec x, int time_point);
		// PetscErrorCode apply_transpose (ScalarType *y, ScalarType *x);


		~Obs ();
};


#endif
