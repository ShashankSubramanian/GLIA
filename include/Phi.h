#ifndef PHI_H_
#define PHI_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>

class Phi {
	public:
		Phi (std::shared_ptr<NMisc> n_misc,  std::shared_ptr<SpectralOperators> spec_ops);

		std::vector<Vec> phi_vec_;

		Vec labels_;								// component labels for each voxel from connected components of data: memory managed from outside
		int num_components_;						// number of connected components of data
		std::vector<int> gaussian_labels_;			// gaussian labels set using labels_
		std::vector<ScalarType> component_weights_;		// weights for labels obtained using connected components
    	std::vector<ScalarType> component_centers_;		// weights for labels obtained using connected components

		ScalarType sigma_;
		ScalarType spacing_factor_;
		ScalarType cm_[3];
		int np_, n_local_;

		bool compute_;

		std::shared_ptr<MatProp> mat_prop_;
		std::shared_ptr<NMisc> n_misc_;
		std::shared_ptr<SpectralOperators> spec_ops_;
		std::vector<ScalarType> centers_;   //Vector of centers for the gaussians
		std::vector<ScalarType> centers_temp_; // Keeps track of centers if phis ever need to be changed

		PetscErrorCode setGaussians (std::array<ScalarType, 3>& user_cm, ScalarType sigma, ScalarType spacing_factor, int np); //, std::shared_ptr<NMisc> n_misc);   //Bounding box
		PetscErrorCode setGaussians (Vec data);		
		PetscErrorCode setGaussians (std::string file, bool read_comp_data = false);
    	PetscErrorCode setLabels (Vec labels) {labels_ = labels; PetscFunctionReturn(0);}																						//Adaptive phis
		PetscErrorCode setValues (std::shared_ptr<MatProp> mat_prop);
		PetscErrorCode phiMesh (ScalarType *center);
		PetscErrorCode initialize (ScalarType *out, std::shared_ptr<NMisc> n_misc, ScalarType *center);
		PetscErrorCode truncate (ScalarType *out, std::shared_ptr<NMisc> n_misc, ScalarType *center);
		PetscErrorCode apply (Vec out, Vec p);
		PetscErrorCode applyTranspose (Vec pout, Vec in);
		void modifyCenters (std::vector<int> support_idx);    	// Picks only the basis needed in the restricted subspace
		void resetCenters () {centers_.clear(); centers_ = centers_temp_; np_ = n_misc_->np_; 
							  if (!n_misc_->phi_store_) compute_ = true;};			// Resets centers to the older ones

		~Phi ();
};


#endif
