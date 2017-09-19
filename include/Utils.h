#ifndef _UTILS_H
#define _UTILS_H

#define BRAIN
#define ALIGNMENT 32
#define OMP_NUM_THREADS 16

#include <petsc.h>
#include "petsctao.h"
#include <accfft.h>
#include <accfft_operators.h>
#include <glog/logging.h>
#include <math.h>
#include <memory>

#include <iostream>


class NMisc {
	public:
		NMisc (int *n, int *isize, int *istart, accfft_plan *plan, MPI_Comm c_comm)
				: rd_ (1)   //Reaction Diffusion
				, dt_ (0.02)
				, time_horizon_ (0.04)
				, np_ (8)
				, k_ (0.1)
				, rho_ (8)
				, p_scale_ (1.0)
				, beta_ (1e-2)	
				, writeOutput_ (1)	
		{
			user_cm_[0] = 4.0;
			user_cm_[1] = 2.03;
			user_cm_[2] = 2.07;

			memcpy (n_, n, 3 * sizeof(int));
			memcpy (isize_, isize, 3 * sizeof(int));
			memcpy (istart_, istart, 3 * sizeof(int));

			plan_ = plan;
			c_comm_ = c_comm;
			accfft_alloc_max_ = plan->alloc_max;

			n_global_ = n[0] * n[1] * n[2];
			n_local_ = isize[0] * isize[1] * isize[2];
			h_[0] = M_PI * 2 / n[0];
			h_[1] = M_PI * 2 / n[1];
		  	h_[2] = M_PI * 2 / n[2];
		}
		int n_[3];
		int isize_[3];
		int istart_[3];
		double h_[3];

		int np_;
		double time_horizon_;
		double dt_;

		int rd_;
		int writeOutput_;

		double k_;
		double rho_;
		double user_cm_[3];
		double p_scale_;
		double beta_;

		int64_t accfft_alloc_max_;
		int64_t n_local_;
		int64_t n_global_;

		accfft_plan *plan_;
		MPI_Comm c_comm_;

};

int weierstrassSmoother (double *Wc, double *c, std::shared_ptr<NMisc> n_misc, double sigma); //TODO: Clean up .cpp file

//Read/Write function prototypes
void dataIn (double *A, std::shared_ptr<NMisc> n_misc, const char *fname);
void dataIn (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname);
void dataOut (double *A, std::shared_ptr<NMisc> n_misc, const char *fname);
void dataOut (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname);

/* helper methods for print out to console */
PetscErrorCode tuMSG(std::string msg, int size = 98, bool parlog = false);
PetscErrorCode tuMSGstd(std::string msg, int size = 98, bool parlog = false);
PetscErrorCode tuMSGwarn(std::string msg, int size = 98, bool parlog = false);
PetscErrorCode _tuMSG(std::string msg, std::string color, int size, bool parlog);

/* accfft differential operators */
void accfft_grad (Vec grad_x, Vec grad_y, Vec grad_z, Vec x, accfft_plan *plan, std::bitset<3> *pXYZ, double *timers);
void accfft_divergence (Vec div, Vec dx, Vec dy, Vec dz, accfft_plan *plan, double *timers);

/* helper function for timer accumulation */
void accumulateTimers(double* tacc, double* tloc, double selfexec);


/* definition of tumor assert */
#ifndef NDEBUG
#   define TU_assert(Expr, Msg) \
    __TU_assert(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#   define TU_assert(Expr, Msg) ;
#endif
void __TU_assert(const char* expr_str, bool expr, const char* file, int line, const char* msg);


#endif // end _UTILS_H
