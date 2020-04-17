#ifndef OBS_H_
#define OBS_H_

#include "Parameters.h"

class Obs {
	public:
	Obs (std::shared_ptr<Parameters> params);

	ScalarType threshold_1_;
  ScalarType threshold_0_;
	Vec filter_1_;
  Vec filter_0_;
  Vec one;
  bool two_snapshot_;
	std::shared_ptr<Parameters> params_;

  PetscErrorCode setDefaultFilter (Vec data, int time_point, ScalarType thr=-1);
	PetscErrorCode setCustomFilter (Vec custom_filter, int time_point=1);
	PetscErrorCode apply (Vec y, Vec x, int time_point=1, bool complement=false);
  PetscErrorCode applyT (Vec y, Vec x, int time_point=1, bool complement=false);
	// PetscErrorCode apply_transpose (ScalarType *y, ScalarType *x);

	~Obs ();
};


#endif