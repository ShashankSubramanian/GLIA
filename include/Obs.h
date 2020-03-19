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
        Vec one;
        bool two_snapshot_;
        bool low_res_data_;
        int isize_[3];
        int istart_[3];
	std::shared_ptr<NMisc> n_misc_;

	//PetscErrorCode setDefaultFilter (Vec data, int time_point=1);
        PetscErrorCode setDefaultFilter (Vec data, int time_point, ScalarType thr=-1);
	PetscErrorCode setCustomFilter (Vec custom_filter, int time_point=1);
	PetscErrorCode apply (Vec y, Vec x, int time_point=1, bool complement=false);
        PetscErrorCode applyT (Vec y, Vec x, int time_point=1, bool complement=false);
	// PetscErrorCode apply_transpose (ScalarType *y, ScalarType *x);


	~Obs ();
};


#endif
