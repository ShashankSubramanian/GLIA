#ifndef _UTILS_H
#define _UTILS_H

#define BRAIN
#define ALIGNMENT 32
#define OMP_NUM_THREADS 16

#include <petsc.h>
#include <accfft.h>
#include <accfft_operators.h>
#include <glog/logging.h>
#include <math.h>

#define ParLOG if(procid==0) LOG(INFO)

class NMisc {
	public:
		int n_[3];
		int isize_[3];
		int istart_[3];
		double h_[3];

		int np_;
		double time_horizon_;
		double dt_;

		int64_t accfft_alloc_max_;
		int64_t n_local_;
		int64_t n_global_;

		accfft_plan *plan_;
		MPI_Comm c_comm_;

		NMisc (int *n, int *isize, int *istart, accfft_plan *plan, MPI_Comm c_comm) {
			memcpy (n_, n, 3 * sizeof(int));
			memcpy (isize_, isize, 3 * sizeof(int));
			memcpy (istart_, istart, 3 * sizeof(int));

			plan_ = plan;
			c_comm_ = c_comm;
			accfft_alloc_max_ = plan->alloc_max;

			dt_ = 0;
			time_horizon_ = 0;
			np_ = 0;
			n_global_ = n[0] * n[1] * n[2];
			n_local_ = isize[0] * isize[1] * isize[2];
			h_[0] = M_PI * 2 / n[0];
			h_[1] = M_PI * 2 / n[1];
		  	h_[2] = M_PI * 2 / n[2];
		}
};

int weierstrassSmoother (double *Wc, double *c, NMisc *n_misc, double sigma); //TODO: Clean up .cpp file

//Read/Write function prototypes

void dataIn (double *A, NMisc* n_misc, const char *fname);
void dataIn (Vec A, NMisc *n_misc, const char *fname);
void dataOut (double *A, NMisc *n_misc, const char *fname);
void dataOut (Vec A, NMisc *n_misc, const char *fname);

void accfft_grad (Vec grad_x, Vec grad_y, Vec grad_z, Vec x, accfft_plan *plan, std::bitset<3> *pXYZ, double *timers);
void accfft_divergence (Vec div, Vec dx, Vec dy, Vec dz, accfft_plan *plan, double *timers);

#endif
