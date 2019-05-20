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

		double force_factor_;
		double edema_threshold_;	

		std::shared_ptr<NMisc> n_misc_;
		std::shared_ptr<SpectralOperators> spec_ops_;

		PetscErrorCode setValues (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode setValuesCustom (Vec gm, Vec wm, Vec glm, Vec csf, Vec bg, std::shared_ptr<NMisc> n_misc);

		PetscErrorCode filterBackgroundAndSmooth (Vec in);

		~MatProp ();
};

#endif