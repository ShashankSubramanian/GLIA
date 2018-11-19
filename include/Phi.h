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
		std::vector<double> centers_temp_; // Keeps track of centers if phis ever need to be changed

		PetscErrorCode setGaussians (std::array<double, 3>& user_cm, double sigma, double spacing_factor, int np); //, std::shared_ptr<NMisc> n_misc);   //Bounding box
		PetscErrorCode setGaussians (Vec data);																								//Adaptive phis
		PetscErrorCode setValues (std::shared_ptr<MatProp> mat_prop);
		PetscErrorCode phiMesh (double *center);
		PetscErrorCode initialize (double *out, std::shared_ptr<NMisc> n_misc, double *center);
		PetscErrorCode apply (Vec out, Vec p);
		PetscErrorCode applyTranspose (Vec pout, Vec in);
		void modifyCenters (std::vector<int> support_idx);    	// Picks only the basis needed in the restricted subspace
		void resetCenters () {centers_ = centers_temp_; np_ = n_misc_->np_;};			// Resets centers to the older ones

		~Phi ();
};


#endif
