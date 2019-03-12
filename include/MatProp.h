#ifndef MATPROP_H_
#define MATPROP_H_

#include "Utils.h"
#include <mpi.h>
#include <omp.h>

class MatProp {
	public:
		MatProp (std::shared_ptr<NMisc> n_misc);

		Vec gm_;
		Vec wm_;
		Vec csf_;
		Vec glm_;
		Vec bg_;
		Vec filter_;

		double force_factor_;
		double edema_threshold_;	

		PetscErrorCode setValues (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode setValuesCustom (Vec gm, Vec wm, Vec glm, Vec csf, Vec bg, std::shared_ptr<NMisc> n_misc);

		~MatProp ();
};

#endif