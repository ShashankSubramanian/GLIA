#include "petsctao.h"
#include "petsc.h"
#include "accfft.h"

#include "Utils.h"
#include "Tumor.h"
#include "TumorSolverInterface.h"
#include "PdeOperators.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

static char help[] = "Forward Driver";

int main (int argc, char** argv) {
 /* ACCFFT, PETSC setup begin */
    google::InitGoogleLogging (argv[0]);
    PetscErrorCode ierr;
    PetscInitialize (&argc, &argv, (char*) 0, help);

    int procid, nprocs;
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank (MPI_COMM_WORLD, &procid);
  	int n[3];
  	int testcase = 0;
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
/* ACCFFT, PETSC setup end */

/* --------------------------------------------------------------------------------------------------------------*/

{
	std::shared_ptr<NMisc> n_misc =  std::make_shared<NMisc> (n, isize, osize, istart, ostart, plan, c_comm, testcase);   //This class contains all required parameters
	std::shared_ptr<TumorSolverInterface> solver_interface = std::make_shared<TumorSolverInterface> (n_misc);
	ierr = solver_interface->solveForward (nullptr , nullptr); // TODO fix that
}

/* --------------------------------------------------------------------------------------------------------------*/
	accfft_destroy_plan (plan);
	PetscFinalize ();
}
