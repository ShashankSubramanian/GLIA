#include "Utils.h"
#include "TumorSolverInterface.h"

static char help[] = "Inverse Driver";

int main (int argc, char** argv) {
 /* ACCFFT, PETSC setup begin */
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
/* ACCFFT, PETSC setup end */
/* --------------------------------------------------------------------------------------------------------------*/

{
	std::shared_ptr<NMisc> n_misc =  std::make_shared<NMisc> (n, isize, istart, plan, c_comm);   //This class contains all required parameters
	std::shared_ptr<TumorSolverInterface> solver_interface = std::make_shared<TumorSolverInterface> (n_misc);

	Vec c_0, c_t;
	PetscErrorCode ierr = 0;
	ierr = generateSyntheticData (&c_0, &c_t, solver_interface);
	ierr = solver_interface->solveForward (nullptr , nullptr); // TODO fix that
}

/* --------------------------------------------------------------------------------------------------------------*/
	accfft_destroy_plan (plan);
	PetscFinalize ();
}

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, std::shared_ptr<TumorSolverInterface> solver_interface) {
	PetscFunctionBegin;
	std::shared_ptr<NMisc> n_misc = solver_interface->n_misc_;
	ierr = VecCreate (PETSC_COMM_WORLD, &c_t);								CHKERRQ (ierr);
	ierr = VecSetSizes (c_t, n_misc->n_local_, n_misc->n_global_);			CHKERRQ (ierr);
	ierr = VecSetFromOptions (c_t);											CHKERRQ (ierr);
	ierr = VecDuplicate (c_t, &c_0_);										CHKERRQ (ierr);

	ierr = VecSet (c_t, 0);													CHKERRQ (ierr);
	ierr = VecSet (c_0, 0);													CHKERRQ (ierr);

	std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
	ierr = tumor->setTrueP (n_misc->p_scale_true_);
	ierr = tumor->phi_->apply (c_0, tumor->p_true_);
	ierr = solver_interface->solveForward (c_0, c_t);
	PetscFunctioReturn (0);
}