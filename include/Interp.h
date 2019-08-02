#ifndef _INTERP_H_
#define _INTERP_H_

//#define USEMPICUDA

#ifdef USEMPICUDA

	#include "petsc.h"
	#include <accfft.h>
	#include <accfftf.h>
	#include "petsccuda.h"
	#include <accfft_gpu.h>
	#include <accfft_operators_gpu.h>
	#include <cuda.h>
	#include <cuda_runtime_api.h>
	#include <mpi.h>
	#include <vector>

	#include <compact_mem_mgr.hpp>

	#define INTERP_PINNED // if defined will use pinned memory for GPU

	typedef double Real;
	#define MPI_T MPI_DOUBLE

	#define COORD_DIM 3

	struct InterpPlan {
	  public:
	  InterpPlan(size_t g_alloc_max);
	  Real * query_points;
	  void allocate (int N_pts, int data_dof);
	  void slow_run( Real* ghost_reg_grid_vals, int data_dof,
	              int* N_reg, int * isize, int* istart, const int N_pts, const int g_size, Real* query_points_in,
	              Real* query_values,int* c_dims, MPI_Comm c_comm);
	  void scatter(int data_dof,
	              int* N_reg, int * isize, int* istart, const int N_pts, const int g_size, Real* query_points_in,
	              int* c_dims, MPI_Comm c_comm, double * timings);
	  void interpolate( Real* ghost_reg_grid_vals, int data_dof,
	              int* N_reg, int * isize, int* istart, const int N_pts, const int g_size,
	              Real* query_values,int* c_dims, MPI_Comm c_comm,double * timings);

	  size_t g_alloc_max; // size in bytes of the ghost input
	  int N_reg_g[3];
	  int isize_g[3];
	  int total_query_points;
	  int data_dof;
	  size_t all_query_points_allocation;
	  MPI_Datatype *stype,*rtype;

	  Real * all_query_points;
	  Real* all_f_cubic;
	  Real * f_cubic_unordered;
	  int* f_index_procs_others_offset; // offset in the all_query_points array
	  int* f_index_procs_self_offset  ; // offset in the query_outside array
	  int* f_index_procs_self_sizes   ; // sizes of the number of interpolations that need to be sent to procs
	  int* f_index_procs_others_sizes ; // sizes of the number of interpolations that need to be received from procs

	  MPI_Request * s_request;
	  MPI_Request * request;

	  std::vector <int> *f_index;
	  std::vector <Real> *query_outside;

	  bool allocate_baked;
	  bool scatter_baked;


	  Real* all_f_cubic_d;
	  Real* all_query_points_d;
	  Real* ghost_reg_grid_vals_d;

	  ~InterpPlan();

	};

	void par_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			int * isize, int* istart, const int N_pts, int g_size,
			Real* query_points, Real* query_values, int* c_dims, MPI_Comm c_comm,
			InterpPlan* interp_plan);

	void gpu_par_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			int * isize, int* istart, const int N_pts, int g_size,
			Real* query_points, Real* query_values, int* c_dims, MPI_Comm c_comm);

	void gpu_interp3_p( Real* reg_grid_vals, int data_dof,
	              int* N_reg, const int N_pts, Real* query_points,
	              Real* query_values);

	void gpu_interp3_ghost_xyz_p( Real* reg_grid_vals_d, int data_dof,
	    int* N_reg, int* istart, int * isize, const int N_pts, const int g_size, Real* query_points_d,
	    Real* query_values_d,bool query_values_already_scaled=false);

	// __global__ void gpu_interp3_ghost_xyz_p_kernel( Real* reg_grid_vals, int data_dof,
	//     int* N_reg, int* isize,int* istart, const int N_pts, const int g_size, Real* query_points,
	//     Real* query_values,bool query_values_already_scaled=false);

	size_t accfft_ghost_local_size_dft_r2c(accfft_plan_gpu* plan, int g_size,
	    int * isize_g, int* istart_g);
	void accfft_get_ghost(accfft_plan_gpu* plan, int g_size, int* isize_g, Real* data,
	    Real* ghost_data);
	size_t accfft_ghost_xyz_local_size_dft_r2c(accfft_plan_gpu* plan, int g_size,
	    int * isize_g, int* istart_g);
	void accfft_get_ghost_xyz(accfft_plan_gpu* plan, int g_size, int* isize_g,
	    Real* data, Real* ghost_data);
	void rescale_xyz(const int g_size, int* N_reg, int* N_reg_g, int* istart,
			int* isize, int* isize_g, const int N_pts, Real* Q_);
	void rescale_xyzgrid(const int g_size, int* N_reg, int* N_reg_g, int* istart,
			int* isize, int* isize_g, const int N_pts, pvfmm::Iterator<Real> Q_);

#else
	#include "UtilsCuda.h"
	#include "petsc.h"
	#include <accfft.h>
	#include <accfftf.h>
	#define FAST_INTERP

	#if defined(PETSC_USE_REAL_SINGLE)
		#define FAST_INTERPV // enable ONLY for single precision
	#endif

	#define FAST_INTERP_BINNING
	//#define HASWELL
	//#define KNL

	#if defined(KNL)
		#define INTERP_USE_MORE_MEM_L1
	#endif


	#if defined(PETSC_USE_REAL_SINGLE)
		typedef float Real;
		#define MPI_T MPI_FLOAT
		#define TC Complexf
		#define PL fftwf_plan
	#else
		typedef double Real;
		#define MPI_T MPI_DOUBLE
		#define TC Complex
		#define PL fftw_plan
	#endif

	#ifndef __INTEL_COMPILER
		#undef FAST_INTERPV
	#endif

	#define COORD_DIM 3
	#include <mpi.h>
	#include <vector>
	#include <set>
	#include <compact_mem_mgr.hpp>

	#ifdef CUDA
		// use cuda routines
		#include "petsccuda.h"
		#include <cuda.h>
		#include <cuda_runtime.h>
		void gpuInterp3D(PetscScalar* yi, const PetscScalar* xq1, const PetscScalar* xq2, const PetscScalar* xq3, PetscScalar* yo, int* nx,
						cudaTextureObject_t yi_tex, int iporder, PetscScalar* interp_time);

		void gpuInterpVec3D(PetscScalar* yi1, PetscScalar* yi2, PetscScalar* yi3, 
    					const PetscScalar* xq1, const PetscScalar* xq2, const PetscScalar* xq3, 
    					PetscScalar* yo1, PetscScalar* yo2, PetscScalar* yo3, 
    					int* nx, cudaTextureObject_t yi_tex, int iporder, PetscScalar* interp_time);

		extern "C" cudaTextureObject_t gpuInitEmptyTexture(int* nx);

		void interp0(float* m, float* q1, float *q2, float *q3, float *q, int nx[3]);
	#endif

	void rescale_xyz(const int g_size, int* N_reg, int* N_reg_g, int* istart,
			int* isize, int* isize_g, const int N_pts, Real* Q_);
	void rescale_xyzgrid(const int g_size, int* N_reg, int* N_reg_g, int* istart,
			int* isize, int* isize_g, const int N_pts, pvfmm::Iterator<Real> Q_);
	void interp3_p_col(Real* reg_grid_vals, int data_dof, int N_reg,
			const int N_pts, Real* query_points, Real* query_values);
	void interp3_p(Real* reg_grid_vals, int data_dof, int N_reg, const int N_pts,
			Real* query_points, Real* query_values);

	void interp3_p(Real* reg_grid_vals, int data_dof, int* N_reg, const int N_pts,
			Real* query_points, Real* query_values);

	void vectorized_interp3_ghost_xyz_p(__restrict Real* reg_grid_vals, int data_dof, const int* __restrict N_reg,
			const int* __restrict N_reg_g, const int * __restrict isize_g, const int* __restrict istart, const int N_pts,
			const int g_size, Real* __restrict query_points, Real* __restrict query_values,
			bool query_values_already_scaled = false);

	void optimized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			int * N_reg_g, int* isize_g, int* istart, const int N_pts, int g_size,
			Real* query_points, Real* query_values,
			bool query_values_already_scaled = false); // cubic interpolation

	void interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			int * N_reg_g, int* isize_g, int* istart, const int N_pts, int g_size,
			Real* query_points, Real* query_values,
			bool query_values_already_scaled = false); // cubic interpolation

	void interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			int * N_reg_g, int* isize_g, int* istart, const int N_pts, int g_size,
			Real* query_points, Real* query_values, int interp_order,
			bool query_values_already_scaled = false); // higher order interpolation

	void interp3_ghost_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			int * N_reg_g, int* isize_g, int* istart, const int N_pts, int g_size,
			Real* query_points, Real* query_values);

	void par_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			int * isize, int* istart, const int N_pts, int g_size,
			Real* query_points, Real* query_values, int* c_dims, MPI_Comm c_comm);

	struct InterpPlan {
		public:
			InterpPlan();
		  	pvfmm::Iterator<Real> query_points;
			void allocate(int N_pts, int* data_dofs = NULL,int nplans = 1);
			void slow_run(Real* ghost_reg_grid_vals, int data_dof, int* N_reg,
					int * isize, int* istart, const int N_pts, const int g_size,
					Real* query_points_in, Real* query_values, int* c_dims,
					MPI_Comm c_comm);
			void fast_scatter(int* N_reg, int * isize, int* istart,
					const int N_pts, const int g_size, Real* query_points_in,
					int* c_dims, MPI_Comm c_comm, double * timings);
			void scatter(int* N_reg, int * isize, int* istart,
					const int N_pts, const int g_size, Real* query_points_in,
					int* c_dims, MPI_Comm c_comm, double * timings);
			// void interpolate(Real* ghost_reg_grid_vals, int data_dof, int* N_reg,
			// 		int * isize, int* istart, const int N_pts, const int g_size,
			// 		Real* query_values, int* c_dims, MPI_Comm c_comm, double * timings);
		  	void interpolate(Real* __restrict ghost_reg_grid_vals,
				int*__restrict N_reg, int *__restrict isize, int*__restrict istart, const int N_pts, const int g_size,
				Real*__restrict query_values, int*__restrict c_dims, MPI_Comm c_comm, double *__restrict timings, int version =0);
			void high_order_interpolate(Real* ghost_reg_grid_vals, int data_dof, int* N_reg,
					int * isize, int* istart, const int N_pts, const int g_size,
					Real* query_values, int* c_dims, MPI_Comm c_comm, double * timings, int interp_order);

			int N_reg_g[3];
			int isize_g[3];
			int total_query_points;
			int data_dof_max;
			int nplans_;
			pvfmm::Iterator<int> data_dofs_;
			size_t all_query_points_allocation;
			pvfmm::Iterator<MPI_Datatype> stypes, rtypes;

			pvfmm::Iterator<Real> all_query_points;
			pvfmm::Iterator<Real> all_f_cubic;
			pvfmm::Iterator<Real> f_cubic_unordered;

			pvfmm::Iterator<int> f_index_procs_others_offset; // offset in the all_query_points array
			pvfmm::Iterator<int> f_index_procs_self_offset; // offset in the query_outside array
			pvfmm::Iterator<int> f_index_procs_self_sizes; // sizes of the number of interpolations that need to be sent to procs
			pvfmm::Iterator<int> f_index_procs_others_sizes; // sizes of the number of interpolations that need to be received from procs

			pvfmm::Iterator<MPI_Request> s_request;
			pvfmm::Iterator<MPI_Request> request;

			std::vector<int> *f_index;
			std::vector<Real> *query_outside;

			bool allocate_baked;
			bool scatter_baked;

			std::vector<int> procs_i_send_to_; // procs who i have to send my q
			std::vector<int> procs_i_recv_from_; // procs whose q I have to recv
			int procs_i_send_to_size_, procs_i_recv_from_size_;

			~InterpPlan();

	};

	void par_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			int * isize, int* istart, const int N_pts, int g_size,
			Real* query_points, Real* query_values, int* c_dims, MPI_Comm c_comm,
			InterpPlan* interp_plan);

	void gpu_interp3_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			const int N_pts, Real* query_points, Real* query_values);

	void gpu_interp3_ghost_xyz_p(Real* reg_grid_vals_d, int data_dof, int* N_reg,
			int* istart, int * isize, const int N_pts, const int g_size,
			Real* query_points_d, Real* query_values_d);

	void gpu_par_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
			int * isize, int* istart, const int N_pts, int g_size,
			Real* query_points, Real* query_values, int* c_dims, MPI_Comm c_comm);

	// GHOST FUNCTIONS

	size_t accfft_ghost_local_size_dft_r2c(accfft_plan_t<Real, TC, PL>* plan, int g_size,
			int * isize_g, int* istart_g);
	//size_t accfft_ghost_local_size_dft_r2c(accfft_plan_gpu* plan, int g_size,
	//		int * isize_g, int* istart_g);
	void accfft_get_ghost(accfft_plan_t<Real, TC, PL>* plan, int g_size, int* isize_g, Real* data,
			Real* ghost_data);
	//void accfft_get_ghost(accfft_plan_gpu* plan, int g_size, int* isize_g, Real* data,
	//		Real* ghost_data);

	size_t accfft_ghost_xyz_local_size_dft_r2c(accfft_plan_t<Real, TC, PL>* plan, int g_size,
			int * isize_g, int* istart_g);
	//size_t accfft_ghost_xyz_local_size_dft_r2c(accfft_plan_gpu* plan, int g_size,
	//		int * isize_g, int* istart_g);

	void accfft_get_ghost_xyz(accfft_plan_t<Real, TC, PL>* plan, int g_size, int* isize_g,
			Real* data, Real* ghost_data);
	//void accfft_get_ghost_xyz(accfft_plan_gpu* plan, int g_size, int* isize_g,
	//		Real* data, Real* ghost_data);

	#endif

#endif
