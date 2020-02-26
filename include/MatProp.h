#ifndef MATPROP_H_
#define MATPROP_H_

#include "Utils.h"
#include "SpectralOperators.h"

class MatProp {
	public:
		MatProp (std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops);

		Vec gm_;
		Vec wm_;
		Vec csf_;
		Vec glm_;
		Vec bg_;
		Vec filter_;

                // undeformed -- this is never changed; so use as pointers
                Vec gm_0_;
                Vec wm_0_;
                Vec csf_0_;
                Vec glm_0_;

		ScalarType force_factor_;
		ScalarType edema_threshold_;	

		std::shared_ptr<NMisc> n_misc_;
		std::shared_ptr<SpectralOperators> spec_ops_;

		PetscErrorCode setValues (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode setValuesCustom (Vec gm, Vec wm, Vec glm, Vec csf, Vec bg, std::shared_ptr<NMisc> n_misc);
		PetscErrorCode clipHealthyTissues ();
		PetscErrorCode filterBackgroundAndSmooth (Vec in);
		PetscErrorCode filterTumor (Vec c);

		PetscErrorCode setAtlas (Vec gm, Vec wm, Vec glm, Vec csf, Vec bg);
                PetscErrorCode resetValues (); 
                ~MatProp ();
};

#endif
