#ifndef REACCOEF_H_
#define REACCOEF_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>

class ReacCoef {
	public:
		ReacCoef (std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops);

		int smooth_flag_;

		ScalarType rho_scale_;
		ScalarType r_gm_wm_ratio_;
		ScalarType r_glm_wm_ratio_;
		Vec rho_vec_;

		std::shared_ptr<SpectralOperators> spec_ops_;

		PetscErrorCode setValues (ScalarType rho_scale, ScalarType r_gm_wm_ratio, ScalarType r_glm_wm_ratio, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);
		PetscErrorCode smooth (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode applydRdm(Vec x1, Vec x2, Vec x3, Vec x4, Vec input);
		PetscErrorCode updateIsotropicCoefficients (ScalarType rho_1, ScalarType rho_2, ScalarType rho_3, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);

		~ReacCoef ();
};


#endif
