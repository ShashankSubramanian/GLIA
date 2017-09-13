/**
 *  Description: This code performs a Weierstrass smoothing.
 *  Copyright (c) 2015-2016.
 *  All rights reserved.
 *  This file is part of PGLISTR library.
 *
 *  PGLISTR is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  PGLISTR is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with PGLISTR.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include "Utils.h"
#include <mpi.h>
#include <omp.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <accfft.h>
#include <accfft_utils.h>


//TODO
//Rewrite variables according to standard conventions


int weierstrassSmoother(double * Wc, double *c, NMisc* N_Misc, double sigma) {
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

