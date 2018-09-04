#include "Utils.h"

PetscErrorCode tuMSG(std::string msg, int size) {
	PetscFunctionBegin;
  PetscErrorCode ierr;
  std::string color = "\x1b[1;34;m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode tuMSGstd(std::string msg, int size) {
  PetscErrorCode ierr;
  std::string color = "\x1b[37;m";
  ierr = _tuMSG(msg, color, size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode tuMSGwarn(std::string msg, int size) {
  PetscErrorCode ierr;
  std::string color = "\x1b[1;31;m";
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

PetscErrorCode TumorStatistics::print() {
	PetscErrorCode ierr;
	PetscFunctionBegin;
	std::stringstream s;
	ierr = tuMSG ("---- statistics                                                                            ----"); CHKERRQ(ierr);
	s << std::setw(8) << "     " << std::setw(8) << " #state " << std::setw(8) << " #adj " << std::setw(8) << " #obj "  << std::setw(8) << " #grad " << std::setw(8) << " #hess ";
	ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(std::string()); s.clear();
	s << std::setw(8) << " curr:" << std::setw(8) << nb_state_solves     << std::setw(8) << nb_adjoint_solves     << std::setw(8) << nb_obj_evals      << std::setw(8) << nb_grad_evals     << std::setw(8) << nb_hessian_evals;
	ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(std::string()); s.clear();
	s << std::setw(8) << " acc: " << std::setw(8) << nb_state_solves + nb_state_solves_acc << std::setw(8) << nb_adjoint_solves + nb_adjoint_solves_acc << std::setw(8) << nb_obj_evals + nb_obj_evals_acc  << std::setw(8) << nb_grad_evals + nb_grad_evals_acc << std::setw(8) << nb_hessian_evals + nb_hessian_evals_acc;
	ierr = tuMSGstd(s.str()); CHKERRQ(ierr); s.str(std::string()); s.clear();
	ierr = tuMSG ("----                                                                                        ----"); CHKERRQ(ierr);
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
	str << n_misc->readpath_.str().c_str() << fname;
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

	std::stringstream str;
	MPI_Offset istart_mpi[3] = { istart[0], istart[1], istart[2] };
	MPI_Offset isize_mpi[3] = { isize[0], isize[1], isize[2] };
	str << n_misc->writepath_.str().c_str() << fname;
	write_pnetcdf(str.str().c_str(), istart_mpi, isize_mpi, c_comm, n_misc->n_, A);
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

					 f[ptr] += std::exp((-Xp * Xp - Y * Y - Z * Z) / sigma / sigma / 2.0)
					 		+ std::exp((-X * X - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

					 f[ptr] += std::exp((-X * X - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
					 		+ std::exp((-Xp * Xp - Yp * Yp - Z * Z) / sigma / sigma / 2.0);

					 f[ptr] += std::exp((-Xp * Xp - Y * Y - Zp * Zp) / sigma / sigma / 2.0)
					 		+ std::exp((-X * X - Yp * Yp - Zp * Zp) / sigma / sigma / 2.0);

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

/// @brief computes difference diff = x - y
PetscErrorCode computeDifference(PetscScalar *sqrdl2norm,
	Vec diff_wm, Vec diff_gm, Vec diff_csf, Vec diff_glm, Vec diff_bg,
	Vec x_wm, Vec x_gm, Vec x_csf, Vec x_glm, Vec x_bg,
	Vec y_wm, Vec y_gm, Vec y_csf, Vec y_glm, Vec y_bg) {

	PetscErrorCode ierr;
	PetscFunctionBegin;
	PetscScalar mis_wm = 0, mis_gm = 0, mis_csf = 0, mis_glm = 0;
	// diff = x - y
	if(x_wm != nullptr) {
		ierr = VecWAXPY (diff_wm, -1.0, y_wm, x_wm);                 CHKERRQ (ierr);
		ierr = VecDot (diff_wm, diff_wm, &mis_wm);                   CHKERRQ (ierr);
	}
	if(x_gm != nullptr) {
		ierr = VecWAXPY (diff_gm, -1.0, y_gm, x_gm);                 CHKERRQ (ierr);
		ierr = VecDot (diff_gm, diff_gm, &mis_gm);                   CHKERRQ (ierr);
	}
	if(x_csf != nullptr) {
		ierr = VecWAXPY (diff_csf, -1.0, y_csf, x_csf);              CHKERRQ (ierr);
		ierr = VecDot (diff_csf, diff_csf, &mis_csf);                CHKERRQ (ierr);
	}
	if(x_glm != nullptr) {
		ierr = VecWAXPY (diff_glm, -1.0, y_glm, x_glm);              CHKERRQ (ierr);
		ierr = VecDot (diff_glm, diff_glm, &mis_glm);                CHKERRQ (ierr);
	}
	*sqrdl2norm  = mis_wm + mis_gm + mis_csf + mis_glm;
	//PetscPrintf(PETSC_COMM_WORLD," geometricCouplingAdjoint mis(WM): %1.6e, mis(GM): %1.6e, mis(CSF): %1.6e, mis(GLM): %1.6e, \n", 0.5*mis_wm, 0.5*mis_gm, 0.5* mis_csf, 0.5*mis_glm);
	PetscFunctionReturn(0);
	}

	/** @brief computes difference xi = m_data - m_geo
	 *  - function assumes that on input, xi = m_geo * (1-c(1))   */
PetscErrorCode geometricCouplingAdjoint(PetscScalar *sqrdl2norm,
	Vec xi_wm, Vec xi_gm, Vec xi_csf, Vec xi_glm, Vec xi_bg,
	Vec m_geo_wm, Vec m_geo_gm, Vec m_geo_csf, Vec m_geo_glm, Vec m_geo_bg,
	Vec m_data_wm, Vec m_data_gm, Vec m_data_csf, Vec m_data_glm, Vec m_data_bg) {
	PetscErrorCode ierr;
	PetscFunctionBegin;
	PetscScalar mis_wm = 0, mis_gm = 0, mis_csf = 0, mis_glm = 0;
	if(m_geo_wm != nullptr) {
		ierr = VecAXPY (xi_wm, -1.0, m_data_wm);                     CHKERRQ (ierr);
		ierr = VecScale (xi_wm, -1.0);                               CHKERRQ (ierr);
		ierr = VecDot (xi_wm, xi_wm, &mis_wm);                       CHKERRQ (ierr);
	}
	if(m_geo_gm != nullptr) {
		ierr = VecAXPY (xi_gm, -1.0, m_data_gm);                     CHKERRQ (ierr);
		ierr = VecScale (xi_gm, -1.0);                               CHKERRQ (ierr);
		ierr = VecDot (xi_gm, xi_gm, &mis_gm);                       CHKERRQ (ierr);
	}
	if(m_geo_csf != nullptr) {
		ierr = VecAXPY (xi_csf, -1.0, m_data_csf);                   CHKERRQ (ierr);
		ierr = VecScale (xi_csf, -1.0);                              CHKERRQ (ierr);
		ierr = VecDot (xi_csf, xi_csf, &mis_csf);                    CHKERRQ (ierr);
	}
	if(m_geo_glm != nullptr) {
		ierr = VecAXPY (xi_glm, -1.0, m_data_glm);                   CHKERRQ (ierr);
		ierr = VecScale (xi_glm, -1.0);                              CHKERRQ (ierr);
		ierr = VecDot (xi_glm, xi_glm, &mis_glm);                    CHKERRQ (ierr);
	}
	*sqrdl2norm  = mis_wm + mis_gm + mis_csf + mis_glm;
	//PetscPrintf(PETSC_COMM_WORLD," geometricCouplingAdjoint mis(WM): %1.6e, mis(GM): %1.6e, mis(CSF): %1.6e, mis(GLM): %1.6e, \n", 0.5*mis_wm, 0.5*mis_gm, 0.5* mis_csf, 0.5*mis_glm);
	PetscFunctionReturn(0);
}

//Hoyer measure for sparsity of a vector
PetscErrorCode vecSparsity (Vec x, double &sparsity) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;
	int size;
	ierr = VecGetSize (x, &size);									CHKERRQ (ierr);
	double norm_1, norm_inf;
	ierr = VecNorm (x, NORM_1, &norm_1);							CHKERRQ (ierr);
	ierr = VecNorm (x, NORM_INFINITY, &norm_inf);					CHKERRQ (ierr);

	if (norm_inf == 0) {
		sparsity = 1.0;
		PetscFunctionReturn (0);
	}

	sparsity = (size - (norm_1 / norm_inf)) / (size - 1);

	PetscFunctionReturn (0);
}

/// @brief computes geometric tumor coupling m1 = m0(1-c(1))
PetscErrorCode geometricCoupling(
	Vec m1_wm, Vec m1_gm, Vec m1_csf, Vec m1_glm, Vec m1_bg,
	Vec m0_wm, Vec m0_gm, Vec m0_csf, Vec m0_glm, Vec m0_bg,
	Vec c1, std::shared_ptr<NMisc> nmisc) {
	PetscErrorCode ierr;
	PetscFunctionBegin;
	PetscScalar *ptr_wm, *ptr_gm, *ptr_csf, *ptr_glm, *ptr_bg, *ptr_tu;
	PetscScalar *ptr_m1_wm, *ptr_m1_gm, *ptr_m1_csf, *ptr_m1_glm, *ptr_m1_bg;
	PetscScalar sum = 0;
  if(m0_wm  != nullptr) {ierr = VecGetArray(m0_wm,  &ptr_wm);     CHKERRQ(ierr);}
	if(m0_gm  != nullptr) {ierr = VecGetArray(m0_gm,  &ptr_gm);     CHKERRQ(ierr);}
	if(m0_csf != nullptr) {ierr = VecGetArray(m0_csf, &ptr_csf);    CHKERRQ(ierr);}
	if(m0_glm != nullptr) {ierr = VecGetArray(m0_glm, &ptr_glm);    CHKERRQ(ierr);}
	if(m0_bg  != nullptr) {ierr = VecGetArray(m0_bg,  &ptr_bg);     CHKERRQ(ierr);}
	if(m1_wm  != nullptr) {ierr = VecGetArray(m1_wm,  &ptr_m1_wm);  CHKERRQ(ierr);}
	if(m1_gm  != nullptr) {ierr = VecGetArray(m1_gm,  &ptr_m1_gm);  CHKERRQ(ierr);}
	if(m1_csf != nullptr) {ierr = VecGetArray(m1_csf, &ptr_m1_csf); CHKERRQ(ierr);}
	if(m1_glm != nullptr) {ierr = VecGetArray(m1_glm, &ptr_m1_glm); CHKERRQ(ierr);}
	if(m1_bg  != nullptr) {ierr = VecGetArray(m1_bg,  &ptr_m1_bg);  CHKERRQ(ierr);}
	if(c1     != nullptr) {ierr = VecGetArray(c1,     &ptr_tu);     CHKERRQ(ierr);}
  // m = m0(1-c(1))
	for (PetscInt j = 0; j < nmisc->n_local_; j++) {
		sum = 0;
    if(m0_gm   != nullptr) {ptr_m1_gm[j]  = ptr_gm[j]  * (1 - ptr_tu[j]); sum += ptr_m1_gm[j];}
		if(m0_csf  != nullptr) {ptr_m1_csf[j] = ptr_csf[j] * (1 - ptr_tu[j]); sum += ptr_m1_csf[j];}
		if(m0_glm  != nullptr) {ptr_m1_glm[j] = ptr_glm[j] * (1 - ptr_tu[j]); sum += ptr_m1_glm[j];}
		if(m0_bg   != nullptr) {ptr_m1_bg[j]  = ptr_bg[j]  * (1 - ptr_tu[j]); sum += ptr_m1_bg[j];}
		if(m0_wm   != nullptr) {ptr_m1_wm[j]  = 1. - (sum + ptr_tu[j]);}
	}
	if(m0_wm  != nullptr) {ierr = VecRestoreArray(m0_wm,  &ptr_wm);    CHKERRQ(ierr);}
	if(m0_gm  != nullptr) {ierr = VecRestoreArray(m0_gm,  &ptr_gm);    CHKERRQ(ierr);}
	if(m0_csf != nullptr) {ierr = VecRestoreArray(m0_csf, &ptr_csf);   CHKERRQ(ierr);}
	if(m0_glm != nullptr) {ierr = VecRestoreArray(m0_glm, &ptr_glm);   CHKERRQ(ierr);}
	if(m0_bg  != nullptr) {ierr = VecRestoreArray(m0_bg,  &ptr_bg);    CHKERRQ(ierr);}
	if(m1_wm  != nullptr) {ierr = VecRestoreArray(m1_wm,  &ptr_m1_wm); CHKERRQ(ierr);}
	if(m1_gm  != nullptr) {ierr = VecRestoreArray(m1_gm,  &ptr_m1_gm); CHKERRQ(ierr);}
	if(m1_csf != nullptr) {ierr = VecRestoreArray(m1_csf, &ptr_m1_csf);CHKERRQ(ierr);}
	if(m1_glm != nullptr) {ierr = VecRestoreArray(m1_glm, &ptr_m1_glm);CHKERRQ(ierr);}
	if(m1_bg  != nullptr) {ierr = VecRestoreArray(m1_bg,  &ptr_m1_bg); CHKERRQ(ierr);}
	if(c1     != nullptr) {ierr = VecRestoreArray(c1,     &ptr_tu);    CHKERRQ(ierr);}
  // go home
	PetscFunctionReturn(0);
}

PetscErrorCode vecSign (Vec x) {
	PetscFunctionBegin;
	PetscErrorCode ierr = 0;

	double *x_ptr;
	int size;
	ierr = VecGetSize (x, &size);		CHKERRQ (ierr);
	ierr = VecGetArray (x, &x_ptr);		CHKERRQ (ierr);

	for (int i = 0; i < size; i++) {
		if (x_ptr[i] > 0) x_ptr[i] = 1.0;
		else if (x_ptr[i] == 0) x_ptr[i] = 0.0;
		else x_ptr[i] = -1.0;
	}

	ierr = VecRestoreArray (x, &x_ptr);	CHKERRQ (ierr);

	PetscFunctionReturn (0);
}
