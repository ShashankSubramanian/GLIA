#ifndef PHI_H_
#define PHI_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>

class Phi {
	public:
		Phi (NMisc *n_misc);

		Vec *phi_vec_;

		double sigma_;
		double cm_[3];
		int np_, n_local_;

		PetscErrorCode setValues (double *user_cm, MatProp *mat_prop, NMisc *n_misc);
		PetscErrorCode phiMesh (double *center);
		PetscErrorCode initialize (double *out, NMisc *n_misc, double *center);
		PetscErrorCode apply (Vec out, Vec p, NMisc *n_misc);

		~Phi ();
};


#endif
