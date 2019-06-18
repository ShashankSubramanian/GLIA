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

		Vec labels_;								// component labels for each voxel from connected components of data: memory managed from outside
		int num_components_;						// number of connected components of data
		std::vector<int> gaussian_labels_;			// gaussian labels set using labels_
		std::vector<double> component_weights_;		// weights for labels obtained using connected components
    	std::vector<double> component_centers_;		// weights for labels obtained using connected components

		double sigma_;
		double spacing_factor_;
		double cm_[3];
		int np_, n_local_;

		std::shared_ptr<NMisc> n_misc_;
		std::vector<double> centers_;   																		      // Vector of centers for the gaussians
		std::vector<double> centers_temp_; 																			  // Keeps track of centers if phis ever need to be changed

		PetscErrorCode setGaussians (std::array<double, 3>& user_cm, double sigma, double spacing_factor, int np);    // Bounding box
		PetscErrorCode setGaussians (Vec data);																		  // Adaptive phis
    	PetscErrorCode setGaussians (std::string file);
    	PetscErrorCode setLabels (Vec labels) {labels_ = labels; PetscFunctionReturn(0);}
		PetscErrorCode setValues (std::shared_ptr<MatProp> mat_prop);
		PetscErrorCode phiMesh (double *center);
		PetscErrorCode initialize (double *out, std::shared_ptr<NMisc> n_misc, double *center);
    PetscErrorCode truncate (double *out, std::shared_ptr<NMisc> n_misc, double *center);
		PetscErrorCode apply (Vec out, Vec p);
		PetscErrorCode applyTranspose (Vec pout, Vec in);
		void modifyCenters (std::vector<int> support_idx);    														  // Picks only the basis needed in the restricted subspace
		void resetCenters () {centers_.clear(); centers_ = centers_temp_; np_ = n_misc_->np_;};						  // Resets centers to the original centers

		~Phi ();
};


#endif
