#include "petsctao.h"
#include "petsc.h"
#include "accfft.h"

#include "Utils.h"
#include "Tumor.h"
#include "PdeOperators.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

static char help[] = "ITP Edema";

PetscErrorCode setupNmisc (NMisc *n_misc);
PetscErrorCode setupTumor (Tumor *tumor, NMisc *n_misc);

int main (int argc, char** argv) {
    google::InitGoogleLogging (argv[0]);
    PetscErrorCode ierr;
    PetscInitialize (&argc, &argv, (char*) 0, help);

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  	int n[3];

	n[0] = 64;
	n[1] = 64;
	n[2] = 64;

	accfft_init();
    MPI_Comm c_comm;
	int c_dims[2] = { 0 };
	accfft_create_comm(MPI_COMM_WORLD, c_dims, &c_comm);
	
	int isize[3], osize[3], istart[3], ostart[3];
	int64_t alloc_max = accfft_local_size_dft_r2c (n, isize, istart, osize, ostart, c_comm);
	double *c_0 = (double*) accfft_alloc (alloc_max);
	Complex *c_hat = (Complex*) accfft_alloc (alloc_max);
	accfft_plan *plan = accfft_plan_dft_3d_r2c (n, c_0, (double*) c_hat, c_comm, ACCFFT_MEASURE);
    accfft_free (c_0);
    accfft_free (c_hat);

	NMisc *n_misc = new NMisc (n, isize, istart, plan, c_comm);
	ierr = setupNmisc (n_misc);

	Tumor *tumor = new Tumor (n_misc);
	ierr = setupTumor (tumor, n_misc);

	//Run forward solve
	tumor->runForward (n_misc);

}

PetscErrorCode setupNmisc ( NMisc *n_misc) {
	PetscErrorCode ierr = 0;
	double dt = 0.01;
	double time_horizon = 1;
	double np = 27;

	n_misc->dt_ = dt;
	n_misc->time_horizon_ = time_horizon;
	n_misc->np_ = np;

	return ierr;
}

PetscErrorCode setupTumor (Tumor *tumor, NMisc *n_misc) {
    PetscErrorCode ierr;
	double k = 0.0;
	double rho = 8.0;
	double user_cm[3];
	double p_scale = 0.8;
	user_cm[1] = 2.03;
	user_cm[2] = 2.07;
	user_cm[0] = 4.0;

	Vec p;
	ierr = VecCreate (PETSC_COMM_WORLD, &p); 							CHKERRQ (ierr);
	ierr = VecSetSizes (p, PETSC_DECIDE, n_misc->np_);					CHKERRQ (ierr);
	ierr = VecSetFromOptions (p);										CHKERRQ (ierr);

	ierr = VecSet (p, p_scale);											CHKERRQ (ierr);

	ierr = tumor->setValues (k, rho, user_cm, p, n_misc);		

	return ierr;
}
