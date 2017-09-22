#include "Utils.h"
#include "TumorSolverInterface.h"

static char help[] = "Inverse Driver";

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc);

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

	Vec c_0, c_t, p_rec;
	PetscErrorCode ierr = 0;
	PCOUT << "Generating Synthetic Data --->" << std::endl;
	ierr = generateSyntheticData (c_0, c_t, p_rec, solver_interface, n_misc);
	PCOUT << "Data Generated: Inverse solve begin --->" << std::endl;
	ierr = solver_interface->solveInverse (p_rec, c_t, nullptr);
}

/* --------------------------------------------------------------------------------------------------------------*/
	accfft_destroy_plan (plan);
	PetscFinalize ();
}

PetscErrorCode generateSyntheticData (Vec &c_0, Vec &c_t, Vec &p_rec, std::shared_ptr<TumorSolverInterface> solver_interface, std::shared_ptr<NMisc> n_misc) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	//Create p_rec
	ierr = VecCreate (PETSC_COMM_WORLD, &p_rec);							CHKERRQ (ierr);
	ierr = VecSetSizes (p_rec, PETSC_DECIDE, n_misc->np_);					CHKERRQ (ierr);
	ierr = VecSetFromOptions (p_rec);										CHKERRQ (ierr);

	ierr = VecCreate (PETSC_COMM_WORLD, &c_t);								CHKERRQ (ierr);
	ierr = VecSetSizes (c_t, n_misc->n_local_, n_misc->n_global_);			CHKERRQ (ierr);
	ierr = VecSetFromOptions (c_t);											CHKERRQ (ierr);
	ierr = VecDuplicate (c_t, &c_0);										CHKERRQ (ierr);

	ierr = VecSet (c_t, 0);													CHKERRQ (ierr);
	ierr = VecSet (c_0, 0);													CHKERRQ (ierr);

	std::shared_ptr<Tumor> tumor = solver_interface->getTumor ();
	ierr = tumor->setTrueP (n_misc->p_scale_true_);
	ierr = tumor->phi_->apply (c_0, tumor->p_true_);
	ierr = solver_interface->solveForward (c_t, c_0);

	//Smooth data
	double *c_t_ptr;
	double sigma_smooth = 2.0 * M_PI / n_misc->n_[0];
	ierr = VecGetArray (c_t, &c_t_ptr);										CHKERRQ (ierr);
	ierr = weierstrassSmoother (c_t_ptr, c_t_ptr, n_misc, sigma_smooth);
	ierr = VecRestoreArray (c_t, &c_t_ptr);								    CHKERRQ (ierr);
	PetscFunctionReturn (0);
}
