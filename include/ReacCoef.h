#ifndef REACCOEF_H_
#define REACCOEF_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>

class ReacCoef {
	public:
		ReacCoef (std::shared_ptr<NMisc> n_misc);

		int smooth_flag_;

		double rho_scale_;
		double r_gm_wm_ratio_;
		double r_glm_wm_ratio_;
		Vec rho_vec_;

		PetscErrorCode setValues (double rho_scale, double r_gm_wm_ratio, double r_glm_wm_ratio, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);
		PetscErrorCode smooth (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode applydRdm(Vec x1, Vec x2, Vec x3, Vec x4, Vec input);
		PetscErrorCode updateIsotropicCoefficients (double rho_1, double rho_2, double rho_3, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);

		~ReacCoef ();
};


#endif
