#ifndef PHI_H_
#define PHI_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>

class Phi {
	public:
		Phi (std::shared_ptr<NMisc> n_misc);

		std::vector<Vec> phi_vec_;

		double sigma_;
		double spacing_factor_;
		double cm_[3];
		int np_, n_local_;

		std::shared_ptr<NMisc> n_misc_;
		std::vector<double> centers_;   //Vector of centers for the gaussians

		PetscErrorCode setGaussians (std::array<double, 3>& user_cm, double sigma, double spacing_factor); //, std::shared_ptr<NMisc> n_misc);   //Bounding box
		PetscErrorCode setGaussians (Vec data);																								//Adaptive phis
		PetscErrorCode setValues (std::shared_ptr<MatProp> mat_prop);
		PetscErrorCode phiMesh (double *center);
		PetscErrorCode initialize (double *out, std::shared_ptr<NMisc> n_misc, double *center);
		PetscErrorCode apply (Vec out, Vec p);
		PetscErrorCode applyTranspose (Vec pout, Vec in);

		~Phi ();
};


#endif
