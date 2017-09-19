#include "Utils.h"

void accfft_grad (Vec grad_x, Vec grad_y, Vec grad_z, Vec x, accfft_plan *plan, std::bitset<3> *pXYZ, double *timers) {
	PetscErrorCode ierr = 0;
	double *grad_x_ptr, *grad_y_ptr, *grad_z_ptr, *x_ptr;
	ierr = VecGetArray (grad_x, &grad_x_ptr);
	ierr = VecGetArray (grad_y, &grad_y_ptr);
	ierr = VecGetArray (grad_z, &grad_z_ptr);
	ierr = VecGetArray (x, &x_ptr);

	accfft_grad (grad_x_ptr, grad_y_ptr, grad_z_ptr, x_ptr, plan, pXYZ, timers);

	ierr = VecRestoreArray (grad_x, &grad_x_ptr);
	ierr = VecRestoreArray (grad_y, &grad_y_ptr);
	ierr = VecRestoreArray (grad_z, &grad_z_ptr);
	ierr = VecRestoreArray (x, &x_ptr);
}

void accfft_divergence (Vec div, Vec dx, Vec dy, Vec dz, accfft_plan *plan, double *timers) {
	PetscErrorCode ierr = 0;
	double *div_ptr, *dx_ptr, *dy_ptr, *dz_ptr;
	ierr = VecGetArray (div, &div_ptr);
	ierr = VecGetArray (dx, &dx_ptr);
	ierr = VecGetArray (dy, &dy_ptr);
	ierr = VecGetArray (dz, &dz_ptr);

	accfft_divergence (div_ptr, dx_ptr, dy_ptr, dz_ptr, plan, timers);

	ierr = VecRestoreArray (div, &div_ptr);
	ierr = VecRestoreArray (dx, &dx_ptr);
	ierr = VecRestoreArray (dy, &dy_ptr);
	ierr = VecRestoreArray (dz, &dz_ptr);
}

void accumulateTimers(double* tacc, double* tloc, double selfexec) {
	tloc[5] = selfexec;
	tacc[0] += tloc[0];
	tacc[1] += tloc[1];
	tacc[2] += tloc[2];
	tacc[3] += tloc[3];
	tacc[4] += tloc[4];
	tacc[5] += tloc[5];
	tacc[6] += tloc[6];
}
