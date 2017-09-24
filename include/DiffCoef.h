#ifndef DIFFCOEF_H_
#define DIFFCOEF_H_

#include "Utils.h"
#include "MatProp.h"
#include <mpi.h>
#include <omp.h>

class DiffCoef {
	public:
		DiffCoef (std::shared_ptr<NMisc> n_misc);

		double k_scale_;
		double k_gm_wm_ratio_;
		double k_glm_wm_ratio_;
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
		PetscErrorCode smooth (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode applyK (Vec x, Vec y, Vec z);
		PetscErrorCode applyD (Vec dc, Vec c, accfft_plan *plan);

		~DiffCoef ();
};

#endif
