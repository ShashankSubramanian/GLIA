#ifndef PHI_H_
#define PHI_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>

class Phi {
	public:
		Phi (std::shared_ptr<NMisc> n_misc);

		Vec *phi_vec_;

		double sigma_;
		double cm_[3];
		int np_, n_local_;

		PetscErrorCode setValues (std::array<double, 3>& user_cm, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);
		PetscErrorCode phiMesh (double *center);
		PetscErrorCode initialize (double *out, std::shared_ptr<NMisc> n_misc, double *center);
		PetscErrorCode apply (Vec out, Vec p);
		PetscErrorCode applyTranspose (Vec pout, Vec in);

		~Phi ();
};


#endif
