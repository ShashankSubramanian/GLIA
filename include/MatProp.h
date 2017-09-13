#ifndef MATPROP_H_
#define MATPROP_H_

#include "Utils.h"
#include <mpi.h>
#include <omp.h>

class MatProp {
	public:
		MatProp (NMisc *n_misc);

		Vec gm_;
		Vec wm_;
		Vec csf_;
		Vec glm_;
		Vec filter_;

		double force_factor_;
		double edema_threshold_;	

		PetscErrorCode setValues (NMisc *n_misc);

		~MatProp ();
};

#endif