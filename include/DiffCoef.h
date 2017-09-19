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
		int smooth_flag_;

		Vec kxx_;
		Vec kxy_;
		Vec kxz_;
		Vec kyy_;
		Vec kyz_;
		Vec kzz_;

		Vec *temp_;

		PetscErrorCode setValues (double k_scale, std::shared_ptr<MatProp> mat_prop, std::shared_ptr<NMisc> n_misc);
		PetscErrorCode smooth (std::shared_ptr<NMisc> n_misc);
		PetscErrorCode applyK (Vec x, Vec y, Vec z);
		PetscErrorCode applyD (Vec dc, Vec c, accfft_plan *plan);

		~DiffCoef ();
};

#endif
