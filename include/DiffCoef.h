#ifndef DIFFCOEF_H_
#define DIFFCOEF_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>


/** Diffusion koefficient is defined as follows:
 *
 *  \mat{k} = \sum_i k_i * m_i * I + k_a * anisotropic_tensor
  * where k_i is defined below:
 *
 *  inversion for diffusivity: we invert for k_1, k_2, k_3 with
 *       k_1 = dk_dm_wm  = k_scale * 1;                     //WM
 *       k_2 = dk_dm_gm  = k_scale * k_gm_wm_ratio_;        //GM
 *       k_3 = dk_dm_glm = k_scale * k_glm_wm_ratio_;       //GLM
 */
class DiffCoef {
	public:
		DiffCoef (std::shared_ptr<NMisc> n_misc, std::shared_ptr<SpectralOperators> spec_ops);

		ScalarType k_scale_;         // (= k_f * k_wm),    k_wm  := 1 (fixed)      INVERSION for k_f, k_gm, k_glm
		ScalarType k_gm_wm_ratio_;   // (= k_f * k_gm),    k_gm  := ratio_gm_wm
		ScalarType k_glm_wm_ratio_;  // (= k_f * k_glm),   k_glm := ratio_glm_wm
		int smooth_flag_;

		std::shared_ptr<SpectralOperators> spec_ops_;

		Vec kxx_;
		Vec kxy_;
		Vec kxz_;
		Vec kyy_;
		Vec kyz_;
		Vec kzz_;

		ScalarType kxx_avg_;
		ScalarType kxy_avg_;
		ScalarType kxz_avg_;
		ScalarType kyy_avg_;
		ScalarType kyz_avg_;
		ScalarType kzz_avg_;
		ScalarType filter_avg_;

		Vec *temp_;
		ScalarType *temp_accfft_;

		ScalarType *work_cuda_;

		PetscErrorCode setValues (ScalarType k_scale, ScalarType k_gm_wm_ratio, ScalarType k_glm_wm_ratio, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);
		// needs to be called when we invert for diffusivity and use Gauss-Newton 
		PetscErrorCode setSecondaryCoefficients (ScalarType k1, ScalarType k2, ScalarType k3, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);
		// needs to be called when we invert for diffusivity (in every newton iteration, calls setValues())
		PetscErrorCode updateIsotropicCoefficients (ScalarType k1, ScalarType k2, ScalarType k3, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);
		PetscErrorCode setWorkVecs(Vec * workvecs);
		PetscErrorCode smooth (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode applyK (Vec x, Vec y, Vec z);
		PetscErrorCode applyD (Vec dc, Vec c);
		PetscErrorCode applyDWithSecondaryCoeffs (Vec dc, Vec c);
		/** @brief computes x = k_bar (grad c)^T grad \alpha, where k_bar = dK/dm */
		PetscErrorCode compute_dKdm_gradc_gradp(Vec x1, Vec x2, Vec x3, Vec x4, Vec c, Vec p, fft_plan *plan);
		~DiffCoef ();
};

// Cuda helpers
void initDiffCoefCudaConstants(int *n, int *ostart);
void precFactorDiffusionCuda (ScalarType *precfactor, ScalarType *work, int *sz);

#endif
