#include "Utils.h"

PetscErrorCode tuMSG(std::string msg, int size) {
	PetscFunctionBegin;
  PetscErrorCode ierr;
  std::string color = "\x1b[1;34;40m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode tuMSGstd(std::string msg, int size) {
  PetscErrorCode ierr;
  std::string color = "\x1b[37;40m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode tuMSGwarn(std::string msg, int size) {
  PetscErrorCode ierr;
  std::string color = "\x1b[1;31;40m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode _tuMSG(std::string msg, std::string color, int size) {
    PetscErrorCode ierr = 0;
    std::stringstream ss;
    PetscFunctionBegin;

    int procid, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    ss << std::left << std::setw(size)<< msg;
    msg = color+"[ "  + ss.str() + "]\x1b[0m\n";
    //msg = "\x1b[1;34;40m[ "  + ss.str() + "]\x1b[0m\n";

    // display message
    ierr = PetscPrintf(PETSC_COMM_WORLD,msg.c_str()); CHKERRQ(ierr);


    PetscFunctionReturn(0);
}

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

/* definition of tumor assert */
void __TU_assert(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
    if (!expr)
    {
        std::cerr << "Assert failed:\t" << msg << "\n"
            << "Expected:\t" << expr_str << "\n"
            << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

static bool isLittleEndian () {
	uint16_t number = 0x1;
	uint8_t *numPtr = (uint8_t*) &number;
	return (numPtr[0] == 1);
}

void dataIn (double *A, std::shared_ptr<NMisc> n_misc, const char *fname) {
	MPI_Comm c_comm = n_misc->c_comm_;
	int nprocs, procid;
	MPI_Comm_rank (c_comm, &procid);
	MPI_Comm_size (c_comm, &nprocs);

	if (A == NULL) {
		PCOUT << "Error in DataOut ---> Input data is null" << std::endl;
		return;
	}
	int *istart = n_misc->istart_;
	int *isize = n_misc->isize_;

	MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] };
	MPI_Offset isize_mpi[3] = { isize[0], isize[1], isize[2] };

	std::stringstream str;
	const char *prefix = "./brain_data/";
	str << prefix << "/" << n_misc->n_[0] << "/" << fname;
	read_pnetcdf(str.str().c_str(), istart_mpi, isize_mpi, c_comm, n_misc->n_, A);
	return;
}

void dataIn (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname) {
	double *a_ptr;
	PetscErrorCode ierr;
	ierr = VecGetArray (A, &a_ptr);
	dataIn(a_ptr, n_misc, fname);
	ierr = VecRestoreArray (A, &a_ptr);
}

void dataOut (double *A, std::shared_ptr<NMisc> n_misc, const char *fname) {
	MPI_Comm c_comm = n_misc->c_comm_;
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);

	if (A == NULL) {
		PCOUT << "Error in DataOut ---> Input data is null" << std::endl;
		return;
	}
	/* Write the output */
	int *istart = n_misc->istart_;
	int *isize = n_misc->isize_;

	std::string filename;
	MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] };
	MPI_Offset isize_mpi[3] = { isize[0], isize[1], isize[2] };
	filename = fname;
	write_pnetcdf(filename, istart_mpi, isize_mpi, c_comm, n_misc->n_, A);
	return;
}

void dataOut (Vec A, std::shared_ptr<NMisc> n_misc, const char *fname) {
	double *a_ptr;
	PetscErrorCode ierr;
	ierr = VecGetArray (A, &a_ptr);
	dataOut (a_ptr, n_misc, fname);
	ierr = VecRestoreArray (A, &a_ptr);
}

//TODO
//Rewrite variables according to standard conventions


int weierstrassSmoother (double * Wc, double *c, std::shared_ptr<NMisc> N_Misc, double sigma) {
	MPI_Comm c_comm = N_Misc->c_comm_;
	int nprocs, procid;
	MPI_Comm_rank(c_comm, &procid);
	MPI_Comm_size(c_comm, &nprocs);

	int *N = N_Misc->n_;
	int istart[3], isize[3], osize[3], ostart[3];
	int alloc_max = accfft_local_size_dft_r2c(N, isize, istart, osize, ostart,
			c_comm);

	double self_exec_time = -MPI_Wtime();

	const int Nx = N_Misc->n_[0], Ny = N_Misc->n_[1], Nz = N_Misc->n_[2];
	const double pi = M_PI, twopi = 2.0 * pi, factor = 1.0 / (Nx * Ny * Nz);
	const double hx = twopi / Nx, hy = twopi / Ny, hz = twopi / Nz;
	accfft_plan * plan = N_Misc->plan_;

	Complex *c_hat = (Complex*) accfft_alloc(alloc_max);
	Complex *f_hat = (Complex*) accfft_alloc(alloc_max);
	double *f = (double*) accfft_alloc(alloc_max);
	if ((c_hat == NULL) || (f_hat == NULL) || (f == NULL)) {
		printf("Proc %d: Error allocating array\n", procid);
		exit(-1);
	}


	//PCOUT<<"\033[1;32m weierstrass_smoother { "<<"\033[0m"<<std::endl;
	// Build the filter
	int num_th = omp_get_max_threads();
	double sum_th[num_th];
	for (int i = 0; i < num_th; i++)
		sum_th[i] = 0.;
	#pragma omp parallel num_threads(num_th)
	{
		int thid = omp_get_thread_num();
		double X, Y, Z, Xp, Yp, Zp;
		int64_t ptr;
	#pragma omp for
		for (int i = 0; i < isize[0]; i++)
			for (int j = 0; j < isize[1]; j++)
				for (int k = 0; k < isize[2]; k++) {
					X = (istart[0] + i) * hx;
					Xp = X - twopi;
					Y = (istart[1] + j) * hy;
					Yp = Y - twopi;
					Z = (istart[2] + k) * hz;
					Zp = Z - twopi;
					ptr = i * isize[1] * isize[2] + j * isize[2] + k;
					f[ptr] = std::exp((-X * X - Y * Y - Z * Z) / sigma / sigma / 2.0)
							+ std::exp((-Xp * Xp - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

					// f[ptr] += std::exp((-Xp * Xp - Y * Y - Z * Z) / sigma / sigma / 2.0)
					// 		+ std::exp((-X * X - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

					// f[ptr] += std::exp((-X * X - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
					// 		+ std::exp((-Xp * Xp - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

					// f[ptr] += std::exp((-Xp * Xp - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
					// 		+ std::exp((-X * X - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

					if (f[ptr] != f[ptr])
						f[ptr] = 0.; // To avoid Nan
					sum_th[thid] += f[ptr];
				}
	}

	// Normalize the Filter
	double sum_f_local = 0., sum_f = 0;
	for (int i = 0; i < num_th; i++)
		sum_f_local += sum_th[i];

	MPI_Allreduce(&sum_f_local, &sum_f, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	double normalize_factor = 1. / (sum_f * hx * hy * hz);

	#pragma omp parallel for
	for (int i = 0; i < isize[0] * isize[1] * isize[2]; i++)
		f[i] = f[i] * normalize_factor;
	//PCOUT<<"sum f= "<<sum_f<<std::endl;
	//PCOUT<<"normalize factor= "<<normalize_factor<<std::endl;

	/* Forward transform */
	accfft_execute_r2c(plan, f, f_hat);
	accfft_execute_r2c(plan, c, c_hat);

	// Perform the Hadamard Transform f_hat=f_hat.*c_hat
	std::complex<double>* cf_hat = (std::complex<double>*) (double*) f_hat;
	std::complex<double>* cc_hat = (std::complex<double>*) (double*) c_hat;
	#pragma omp parallel for
	for (int i = 0; i < osize[0] * osize[1] * osize[2]; i++)
		cf_hat[i] *= (cc_hat[i] * factor * hx * hy * hz);


	/* Backward transform */
	accfft_execute_c2r(plan, f_hat, Wc);

	accfft_free(f);
	accfft_free(f_hat);
	accfft_free(c_hat);

	//PCOUT<<"\033[1;32m weierstrass_smoother } "<<"\033[0m"<<std::endl;
	//self_exec_time+= MPI_Wtime();

	return 0;
}


PetscErrorCode geometricCouplingAdjoint(PetscScalar *sqrdl2norm,
	Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg,
	Vec mR_wm, Vec mR_gm, Vec mR_csf, Vec mR_glm, Vec mR_bg,
	Vec mT_wm, Vec mT_gm, Vec mT_csf, Vec mT_glm, Vec mT_bg) {
	PetscErrorCode ierr;
	PetscFunctionBegin;
	PetscScalar mis_wm = 0, mis_gm = 0, mis_csf = 0, mis_glm = 0;
	// \xi = mT - mR  (as opposed to mismatch ||mR - mT||)
	if(mR_wm_ != nullptr) {
		ierr = VecWAXPY (xi_wm_, -1.0, mR_wm_, mT_wm_);              CHKERRQ (ierr);
		ierr = VecDot (xi_wm_, xi_wm_, &mis_wm);                     CHKERRQ (ierr);
	}
	if(mR_gm_ != nullptr) {
		ierr = VecWAXPY (xi_gm_, -1.0, mR_gm_, mT_gm_);              CHKERRQ (ierr);
		ierr = VecDot (xi_gm_, xi_gm_, &mis_gm);                     CHKERRQ (ierr);
	}
	if(mR_csf_ != nullptr) {
		ierr = VecWAXPY (xi_csf_, -1.0, mR_csf_, mT_csf_);           CHKERRQ (ierr);
		ierr = VecDot (xi_csf_, xi_csf_, &mis_csf);                  CHKERRQ (ierr);
	}
	if(mR_glm_ != nullptr) {
		ierr = VecWAXPY (xi_glm_, -1.0, mR_glm_, mT_glm_);           CHKERRQ (ierr);
		ierr = VecDot (xi_glm_, xi_glm_, &mis_glm);                  CHKERRQ (ierr);
	}
	sqrdl2norm  = mis_wm + mis_gm + mis_csf + mis_glm;
	PetscPrintf(PETSC_COMM_WORLD," evaluateObjective mis(WM): %1.6e, mis(GM): %1.6e, mis(CSF): %1.6e, mis(GLM): %1.6e, ", 0.5*mis_wm, 0.5*mis_gm, 0.5* mis_csf, 0.5*mis_glm);
	PetscFunctionReturn(0);
}

PetscErrorCode geometricCoupling(Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg, Vec mR_wm, Vec mR_gm, Vec mR_csf, Vec mR_glm, Vec mR_bg, Vec c1, std::shared_ptr<NMisc> nmisc) {
	PetscErrorCode ierr;
	PetscFunctionBegin;
	PetscScalar *ptr_wm, *ptr_gm, *ptr_csf, *ptr_glm, *ptr_bg, *ptr_tu;
	PetscScalar *ptr_xiwm, *ptr_xigm, *ptr_xicsf, *ptr_xiglm, *ptr_xibg;
	PetscScalar sum = 0;
  if(mR_wm  != nullptr) {ierr = VecGetArray(mR_wm,  &ptr_wm);     CHKERRQ(ierr);}
	if(mR_gm  != nullptr) {ierr = VecGetArray(mR_gm,  &ptr_gm);     CHKERRQ(ierr);}
	if(mR_csf != nullptr) {ierr = VecGetArray(mR_csf, &ptr_csf);    CHKERRQ(ierr);}
	if(mR_glm != nullptr) {ierr = VecGetArray(mR_glm, &ptr_glm);    CHKERRQ(ierr);}
	if(mR_bg  != nullptr) {ierr = VecGetArray(mR_bg,  &ptr_bg);     CHKERRQ(ierr);}
	if(xi_wm  != nullptr) {ierr = VecGetArray(xi_wm,  &ptr_xiwm);   CHKERRQ(ierr);}
	if(xi_gm  != nullptr) {ierr = VecGetArray(xi_gm,  &ptr_xigm);   CHKERRQ(ierr);}
	if(xi_csf != nullptr) {ierr = VecGetArray(xi_csf, &ptr_xicsf);  CHKERRQ(ierr);}
	if(xi_glm != nullptr) {ierr = VecGetArray(xi_glm, &ptr_xiglm);  CHKERRQ(ierr);}
	if(xi_bg  != nullptr) {ierr = VecGetArray(xi_bg,  &ptr_xibg);   CHKERRQ(ierr);}
	if(c1     != nullptr) {ierr = VecGetArray(c1,     &ptr_tu);     CHKERRQ(ierr);}
  // m = m0(1-c(1))
	for (PetscInt j = 0; j < nmisc->n_local_; j++) {
		sum = 0;
    if(mR_gm   != nullptr) {ptr_xigm[j]  = ptr_gm[j]  * (1 - ptr_tu[j]); sum =+ ptr_gm[j];}
		if(mR_csf  != nullptr) {ptr_xicsf[j] = ptr_csf[j] * (1 - ptr_tu[j]); sum =+ ptr_csf[j];}
		if(mR_glm  != nullptr) {ptr_xiglm[j] = ptr_glm[j] * (1 - ptr_tu[j]); sum =+ ptr_glm[j];}
		if(mR_bg   != nullptr) {ptr_xibg[j]  = ptr_bg[j]  * (1 - ptr_tu[j]); sum =+ ptr_bg[j];}
		if(mR_wm   != nullptr) {ptr_xiwm[j]  = 1. - (sum + ptr_tu[j]);}
	}
	if(mR_wm  != nullptr) {ierr = VecRestoreArray(mR_wm,  &ptr_wm);    CHKERRQ(ierr);}
	if(mR_gm  != nullptr) {ierr = VecRestoreArray(mR_gm,  &ptr_gm);    CHKERRQ(ierr);}
	if(mR_csf != nullptr) {ierr = VecRestoreArray(mR_csf, &ptr_csf);   CHKERRQ(ierr);}
	if(mR_glm != nullptr) {ierr = VecRestoreArray(mR_glm, &ptr_glm);   CHKERRQ(ierr);}
	if(mR_bg  != nullptr) {ierr = VecRestoreArray(mR_bg,  &ptr_bg);    CHKERRQ(ierr);}
	if(xi_wm  != nullptr) {ierr = VecRestoreArray(xi_wm,  &ptr_xiwm);  CHKERRQ(ierr);}
	if(xi_gm  != nullptr) {ierr = VecRestoreArray(xi_gm,  &ptr_xigm);  CHKERRQ(ierr);}
	if(xi_csf != nullptr) {ierr = VecRestoreArray(xi_csf, &ptr_xicsf); CHKERRQ(ierr);}
	if(xi_glm != nullptr) {ierr = VecRestoreArray(xi_glm, &ptr_xiglm); CHKERRQ(ierr);}
	if(xi_bg  != nullptr) {ierr = VecRestoreArray(xi_bg,  &ptr_xibg);  CHKERRQ(ierr);}
	if(c1     != nullptr) {ierr = VecRestoreArray(c1,     &ptr_tu);    CHKERRQ(ierr);}
  // go home
	PetscFunctionReturn(0);
}
