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

	//Create IC
	Vec c_0, c_t;
	ierr = VecCreate (PETSC_COMM_WORLD, &c_t);                              CHKERRQ (ierr);
    ierr = VecSetSizes (c_t, n_misc->n_local_, n_misc->n_global_);          CHKERRQ (ierr);
    ierr = VecSetFromOptions (c_t);                                         CHKERRQ (ierr);
    ierr = VecDuplicate (c_t, &c_0);                                        CHKERRQ (ierr);

    ierr = VecSet (c_t, 0);                                                 CHKERRQ (ierr);
    ierr = VecSet (c_0, 0);                                                 CHKERRQ (ierr);

    std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
    ierr = tumor->setTrueP (n_misc->p_scale_true_, n_misc);
    ierr = tumor->phi_->apply (c_0, tumor->p_true_);
    #ifdef POSITIVITY
        ierr = enforcePositivity (c_0, n_misc);
    #endif
    double max, min;
    ierr = VecMax (c_0, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_0, NULL, &min);                                      CHKERRQ (ierr);

    PCOUT << "\nC Data IC Max and Min : " << max << " " << min << std::endl;
    PCOUT << "Forward solve begin" << std::endl;

	ierr = solver_interface->solveForward (c_t , c_0); 

	if (n_misc->writeOutput_)
        dataOut (c_t, n_misc, "results/data.nc");
    PCOUT << "Forward solve done" << std::endl;
    ierr = VecMax (c_t, NULL, &max);                                      CHKERRQ (ierr);
    ierr = VecMin (c_t, NULL, &min);                                      CHKERRQ (ierr);

    PCOUT << "\nC fwd solve IC Max and Min : " << max << " " << min << std::endl;
}

/* --------------------------------------------------------------------------------------------------------------*/
	accfft_destroy_plan (plan);
	PetscFinalize ();
}
