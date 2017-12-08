#ifndef DIFFCOEF_H_
#define DIFFCOEF_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>

class DiffCoef {
	public:
		DiffCoef (std::shared_ptr<NMisc> n_misc);

		double k_scale_;         // (= k_f * k_wm),    k_wm  := 1 (fixed)      INVERSION for k_f, k_gm, k_glm
		double k_gm_wm_ratio_;   // (= k_f * k_gm),    k_gm  := ratio_gm_wm
		double k_glm_wm_ratio_;  // (= k_f * k_glm),   k_glm := ratio_glm_wm
		int smooth_flag_;

		Vec kxx_;
		Vec kxy_;
		Vec kxz_;
		Vec kyy_;
		Vec kyz_;
		Vec kzz_;

		double kxx_avg_;
		double kxy_avg_;
		double kxz_avg_;
		double kyy_avg_;
		double kyz_avg_;
		double kzz_avg_;
		double filter_avg_;

		Vec *temp_;
		double *temp_accfft_;

		PetscErrorCode setValues (double k_scale, double k_gm_wm_ratio, double k_glm_wm_ratio, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);
		PetscErrorCode setWorkVecs(Vec * workvecs);
		PetscErrorCode smooth (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode applyK (Vec x, Vec y, Vec z);
		PetscErrorCode applyD (Vec dc, Vec c, accfft_plan *plan);
		/** @brief computes x = k_bar (grad c)^T grad \alpha, where k_bar = dK/dm */
		PetscErrorCode compute_dKdm_gradc_gradp(Vec x1, Vec x2, Vec x3, Vec x4, Vec c, Vec p, accfft_plan *plan);
		~DiffCoef ();
};

#endif
