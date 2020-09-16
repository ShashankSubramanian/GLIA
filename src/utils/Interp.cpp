// This function performs a 3D cubic interpolation.

#include "Interp.h"


#ifdef MPICUDA

  #define _mm256_set_m128(va, vb) \
            _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)
  #define _mm512_set_m256(va, vb) \
            _mm512_insertf32x8(_mm512_castps256_ps512(vb), va, 1)
  #include <immintrin.h>

  #include <stdint.h>
  #include <limits.h>


  #include <string.h>
  #include <stdlib.h>
  #include <vector>
  #include <iostream>
  #include <cuda.h>
  #include <cufft.h>
  #include <cuda_runtime_api.h>
  #include <cmath>

  #include <algorithm>
  #include <iostream>


  #ifndef ACCFFT_CHECKCUDA_H
  #define ACCFFT_CHECKCUDA_H
  inline cudaError_t checkCuda_accfft(cudaError_t result)
  {
  #if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
      assert(result == cudaSuccess);
    }
  #endif
    return result;
  }
  inline cufftResult checkCuda_accfft(cufftResult result)
  {
  #if defined(DEBUG) || defined(_DEBUG)
    if (result != CUFFT_SUCCESS) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", result);
      assert(result == CUFFT_SUCCESS);
    }
  #endif
    return result;
  }
  #endif


  class Trip_GPU{
    public:
      Trip_GPU(){};
      ScalarType x;
      ScalarType y;
      ScalarType z;
      int ind;
      int N[3];
      ScalarType h[3];

  };
  static bool ValueCmp(Trip_GPU const & a, Trip_GPU const & b)
  {
      return a.z + a.y/a.h[1]*a.N[2] + a.x/a.h[0]* a.N[1]*a.N[2]<b.z + b.y/b.h[1]*b.N[2] + b.x/b.h[0]* b.N[1]*b.N[2] ;
  }
  static void sort_queries(std::vector<Real>* query_outside,std::vector<int>* f_index,int* N_reg,Real* h,MPI_Comm c_comm){

    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);
    for(int proc=0;proc<nprocs;++proc){
      int qsize=query_outside[proc].size()/COORD_DIM;
      Trip_GPU* trip=new Trip_GPU[qsize];

      for(int i=0;i<qsize;++i){
        trip[i].x=query_outside[proc][i*COORD_DIM+0];
        trip[i].y=query_outside[proc][i*COORD_DIM+1];
        trip[i].z=query_outside[proc][i*COORD_DIM+2];
        trip[i].ind=f_index[proc][i];
        trip[i].N[0]=N_reg[0];
        trip[i].N[1]=N_reg[1];
        trip[i].N[2]=N_reg[2];
        trip[i].h[0]=h[0];
        trip[i].h[1]=h[1];
        trip[i].h[2]=h[2];
      }

      std::sort(trip, trip + qsize, ValueCmp);

      query_outside[proc].clear();
      f_index[proc].clear();

      for(int i=0;i<qsize;++i){
        query_outside[proc].push_back(trip[i].x);
        query_outside[proc].push_back(trip[i].y);
        query_outside[proc].push_back(trip[i].z);
        f_index[proc].push_back(trip[i].ind);
      }
      delete(trip);
    }
    return;
  }



  InterpPlan::InterpPlan (size_t g_alloc_max) {
    this->g_alloc_max=g_alloc_max;
    this->allocate_baked=false;
    this->scatter_baked=false;
  }

  void InterpPlan::allocate (int N_pts, int data_dof)
  {
    int nprocs, procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    query_points=(Real*) malloc(N_pts*COORD_DIM*sizeof(Real));

    f_index_procs_others_offset=(int*)malloc(nprocs*sizeof(int)); // offset in the all_query_points array
    f_index_procs_self_offset  =(int*)malloc(nprocs*sizeof(int)); // offset in the query_outside array
    f_index_procs_self_sizes   =(int*)malloc(nprocs*sizeof(int)); // sizes of the number of interpolations that need to be sent to procs
    f_index_procs_others_sizes =(int*)malloc(nprocs*sizeof(int)); // sizes of the number of interpolations that need to be received from procs

    s_request= new MPI_Request[nprocs];
    request= new MPI_Request[nprocs];

    f_index = new std::vector<int> [nprocs];
    query_outside=new std::vector<Real> [nprocs];

    f_cubic_unordered=(Real*) malloc(N_pts*sizeof(Real)*data_dof); // The reshuffled semi-final interpolated values are stored here

    //ScalarType time=0;
    //time=-MPI_Wtime();
  #ifdef INTERP_PINNED
    //cudaMallocHost((void**)&this->ghost_reg_grid_vals_d,g_alloc_max*data_dof);
    cudaMalloc((void**)&this->ghost_reg_grid_vals_d, g_alloc_max*data_dof);
  #else
    cudaMalloc((void**)&this->ghost_reg_grid_vals_d, g_alloc_max*data_dof);
  #endif

    //time+=MPI_Wtime();
    //if(procid==0)
    //  std::cout<<"malloc time="<<time<<std::endl;

    stype= new MPI_Datatype[nprocs];
    rtype= new MPI_Datatype[nprocs];
    this->data_dof=data_dof;
    this->allocate_baked=true;
  }

  InterpPlan::~InterpPlan ()
  {
    int nprocs, procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(this->allocate_baked){
      free(query_points);

      free(f_index_procs_others_offset);
      free(f_index_procs_self_offset  );
      free(f_index_procs_self_sizes   );
      free(f_index_procs_others_sizes );

      delete(s_request);
      delete(request);
      //vectors
      for(int proc=0;proc<nprocs;++proc)
      {
        std::vector<int>().swap(f_index[proc]);
        std::vector<Real>().swap(query_outside[proc]);
      }
      free(f_cubic_unordered);

    }

    if(this->scatter_baked){
      free(all_query_points);
      free(all_f_cubic);

  #ifdef INTERP_PINNED
      //cudaFreeHost(ghost_reg_grid_vals_d);
      cudaFree(ghost_reg_grid_vals_d);
  #else
      cudaFree(ghost_reg_grid_vals_d);
  #endif




  #ifdef INTERP_PINNED
      cudaFreeHost(all_f_cubic_d);
      cudaFreeHost(all_query_points_d);
  #else
      cudaFree(all_f_cubic_d);
      cudaFree(all_query_points_d);
  #endif

      for(int i=0;i<nprocs;++i){
        MPI_Type_free(&stype[i]);
        MPI_Type_free(&rtype[i]);
      }

    }

    if(this->allocate_baked){
      delete(stype);
      delete(rtype);
    }
    return;
  }

  void rescale_xyz(const int g_size,  int* N_reg, int* N_reg_g, int* istart, const int N_pts, Real* query_points);


  /*
   * Phase 1 of the parallel interpolation: This function computes which query_points needs to be sent to
   * other processors and which ones can be interpolated locally. Then a sparse alltoall is performed and
   * all the necessary information is sent/received including the coordinates for the query_points.
   * At the end, each process has the coordinates for interpolation of its own data and those of the others.
   *
   * IMPORTANT: This function must be called just once for a specific query_points. The reason is because of the
   * optimizations performed which assumes that the query_points do not change. For repeated interpolation you should
   * just call this function once, and instead repeatedly call Interp3_Plan::interpolate function.
   */
  void InterpPlan::scatter( int data_dof,
      int* N_reg, int * isize, int* istart, const int N_pts, const int g_size, Real* query_points_in,
      int* c_dims, MPI_Comm c_comm, double * timings)
  {
    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);
    if(this->allocate_baked==false){
      std::cout<<"ERROR InterpPlan Scatter called before calling allocate.\n";
      return;
    }
    if(this->scatter_baked==true){
      for(int proc=0;proc<nprocs;++proc)
      {
        std::vector<int>().swap(f_index[proc]);
        std::vector<Real>().swap(query_outside[proc]);
      }
    }
    all_query_points_allocation=0;

    {

      //int N_reg_g[3], isize_g[3];
      N_reg_g[0]=N_reg[0]+2*g_size;
      N_reg_g[1]=N_reg[1]+2*g_size;
      N_reg_g[2]=N_reg[2]+2*g_size;

      isize_g[0]=isize[0]+2*g_size;
      isize_g[1]=isize[1]+2*g_size;
      isize_g[2]=isize[2]+2*g_size;

      Real h[3]; // original grid size along each axis
      h[0]=1./N_reg[0];
      h[1]=1./N_reg[1];
      h[2]=1./N_reg[2];

      // We copy query_points_in to query_points to aviod overwriting the input coordinates
      Real* query_points=(Real*) malloc(N_pts*COORD_DIM*sizeof(Real));
      memcpy(query_points,query_points_in,N_pts*COORD_DIM*sizeof(Real));
      // Enforce periodicity
      for(int i=0;i<N_pts;i++){
        while(query_points[i*COORD_DIM+0]<=-h[0]) {query_points[i*COORD_DIM+0]=query_points[i*COORD_DIM+0]+1;}
        while(query_points[i*COORD_DIM+1]<=-h[1]) {query_points[i*COORD_DIM+1]=query_points[i*COORD_DIM+1]+1;}
        while(query_points[i*COORD_DIM+2]<=-h[2]) {query_points[i*COORD_DIM+2]=query_points[i*COORD_DIM+2]+1;}

        while(query_points[i*COORD_DIM+0]>=1) {query_points[i*COORD_DIM+0]=query_points[i*COORD_DIM+0]-1;}
        while(query_points[i*COORD_DIM+1]>=1) {query_points[i*COORD_DIM+1]=query_points[i*COORD_DIM+1]-1;}
        while(query_points[i*COORD_DIM+2]>=1) {query_points[i*COORD_DIM+2]=query_points[i*COORD_DIM+2]-1;}
      }


      // Compute the start and end coordinates that this processor owns
      Real iX0[3],iX1[3];
      for (int j=0;j<3;j++){
        iX0[j]=istart[j]*h[j];
        iX1[j]=iX0[j]+(isize[j]-1)*h[j];
      }

      // Now march through the query points and split them into nprocs parts.
      // These are stored in query_outside which is an array of vectors of size nprocs.
      // That is query_outside[i] is a vector that contains the query points that need to
      // be sent to process i. Obviously for the case of query_outside[procid], we do not
      // need to send it to any other processor, as we own the necessary information locally,
      // and interpolation can be done locally.
      int Q_local=0, Q_outside=0;

      // This is needed for one-to-one correspondence with output f. This is becaues we are reshuffling
      // the data according to which processor it land onto, and we need to somehow keep the original
      // index to write the interpolation data back to the right location in the output.

      // This is necessary because when we want to compute dproc0 and dproc1 we have to divide by
      // the max isize. If the proc grid is unbalanced, the last proc's isize will be different
      // than others. With this approach we always use the right isize0 for all procs.
      int isize0=std::ceil(N_reg[0]*1./c_dims[0]);
      int isize1=std::ceil(N_reg[1]*1./c_dims[1]);
      for(int i=0;i<N_pts;i++){
        // The if condition check whether the query points fall into the locally owned domain or not
        if(
            iX0[0]-h[0]<=query_points[i*COORD_DIM+0] && query_points[i*COORD_DIM+0]<=iX1[0]+h[0] &&
            iX0[1]-h[1]<=query_points[i*COORD_DIM+1] && query_points[i*COORD_DIM+1]<=iX1[1]+h[1] &&
            iX0[2]-h[2]<=query_points[i*COORD_DIM+2] && query_points[i*COORD_DIM+2]<=iX1[2]+h[2]
          ){
          query_outside[procid].push_back(query_points[i*COORD_DIM+0]);
          query_outside[procid].push_back(query_points[i*COORD_DIM+1]);
          query_outside[procid].push_back(query_points[i*COORD_DIM+2]);
          f_index[procid].push_back(i);
          Q_local++;
          //PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
          continue;
        }
        else{
          // If the point does not reside in the processor's domain then we have to
          // first compute which processor owns the point. After computing that
          // we add the query point to the corresponding vector.
          int dproc0=(int)(query_points[i*COORD_DIM+0]/h[0])/isize0;
          int dproc1=(int)(query_points[i*COORD_DIM+1]/h[1])/isize1;
          int proc=dproc0*c_dims[1]+dproc1; // Compute which proc has to do the interpolation
          //if(proc>=nprocs) std::cout<<"dp0="<<dproc0<<" dp1="<<dproc1<<" proc="<<proc<<" q[0]="<<query_points[i*COORD_DIM+0]<<" h[0]="<<h[0]<< " div="<<(query_points[i*COORD_DIM+0]/h[0])<<" "<<isize[0]<<std::endl;
          //PCOUT<<"proc="<<proc<<std::endl;
          query_outside[proc].push_back(query_points[i*COORD_DIM+0]);
          query_outside[proc].push_back(query_points[i*COORD_DIM+1]);
          query_outside[proc].push_back(query_points[i*COORD_DIM+2]);
          f_index[proc].push_back(i);
          Q_outside++;
          //PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
          continue;
        }

      }


      // Now sort the query points in zyx order
  #ifdef SORT_QUERIES
      timings[3]+=-MPI_Wtime();
      sort_queries(query_outside,f_index,N_reg,h,c_comm);
      timings[3]+=+MPI_Wtime();
      //if(procid==0) std::cout<<"Sorting Queries\n";
  #endif

      // Now we need to send the query_points that land onto other processor's domain.
      // This done using a sparse alltoallv.
      // Right now each process knows how much data to send to others, but does not know
      // how much data it should receive. This is a necessary information both for the MPI
      // command as well as memory allocation for received data.
      // So we first do an alltoall to get the f_index[proc].size from all processes.


      for (int proc=0;proc<nprocs;proc++){
        if(!f_index[proc].empty())
          f_index_procs_self_sizes[proc]=f_index[proc].size();
        else
          f_index_procs_self_sizes[proc]=0;
      }
      timings[0]+=-MPI_Wtime();
      MPI_Alltoall(f_index_procs_self_sizes,1, MPI_INT,
          f_index_procs_others_sizes,1, MPI_INT,
          c_comm);
      timings[0]+=+MPI_Wtime();


      // Now we need to allocate memory for the receiving buffer of all query
      // points including ours. This is simply done by looping through
      // f_index_procs_others_sizes and adding up all the sizes.
      // Note that we would also need to know the offsets.
      f_index_procs_others_offset[0]=0;
      f_index_procs_self_offset[0]=0;
      for (int proc=0;proc<nprocs;++proc){
        // The reason we multiply by COORD_DIM is that we have three coordinates per interpolation request
        all_query_points_allocation+=f_index_procs_others_sizes[proc]*COORD_DIM;
        if(proc>0){
          f_index_procs_others_offset[proc]=f_index_procs_others_offset[proc-1]+f_index_procs_others_sizes[proc-1];
          f_index_procs_self_offset[proc]=f_index_procs_self_offset[proc-1]+f_index_procs_self_sizes[proc-1];
        }
      }
      total_query_points = all_query_points_allocation / COORD_DIM;

      // This if condition is to allow multiple calls to scatter fucntion with different query points
      // without having to create a new plan
      if(this->scatter_baked==true){
        free(this->all_query_points);
        free(this->all_f_cubic);
        all_query_points=(Real*) malloc(all_query_points_allocation*sizeof(Real));
        all_f_cubic=(Real*)malloc(total_query_points*sizeof(Real)*data_dof);
      }
      else{
        all_query_points=(Real*) malloc(all_query_points_allocation*sizeof(Real));
        all_f_cubic=(Real*)malloc(total_query_points*sizeof(Real)*data_dof);
      }

      // Now perform the allotall to send/recv query_points
      timings[0]+=-MPI_Wtime();
      {
        int dst_r,dst_s;
        for (int i=0;i<nprocs;++i){
          dst_r=i;//(procid+i)%nprocs;
          dst_s=i;//(procid-i+nprocs)%nprocs;
          s_request[dst_s]=MPI_REQUEST_NULL;
          request[dst_r]=MPI_REQUEST_NULL;
          int roffset=f_index_procs_others_offset[dst_r]*COORD_DIM; // notice that COORD_DIM is needed because query_points are 3 times f
          int soffset=f_index_procs_self_offset[dst_s]*COORD_DIM;
          if(f_index_procs_others_sizes[dst_r]!=0)
            MPI_Irecv(&all_query_points[roffset],f_index_procs_others_sizes[dst_r]*COORD_DIM,MPI_T, dst_r,
                0, c_comm, &request[dst_r]);
          if(!query_outside[dst_s].empty())
            MPI_Isend(&query_outside[dst_s][0],f_index_procs_self_sizes[dst_s]*COORD_DIM,MPI_T,dst_s,
                0, c_comm, &s_request[dst_s]);
        }
        // Wait for all the communication to finish
        MPI_Status ierr;
        for (int proc=0;proc<nprocs;++proc){
          if(request[proc]!=MPI_REQUEST_NULL)
            MPI_Wait(&request[proc], &ierr);
          if(s_request[proc]!=MPI_REQUEST_NULL)
            MPI_Wait(&s_request[proc], &ierr);
        }
      }
      timings[0]+=+MPI_Wtime();

      // Now perform the interpolation on all query points including those that need to
      // be sent to other processors and store them into all_f_cubic
      free(query_points);
    }

    for(int i=0;i<nprocs;++i){
      MPI_Type_vector(data_dof,f_index_procs_self_sizes[i],N_pts, MPI_T, &rtype[i]);
      MPI_Type_vector(data_dof,f_index_procs_others_sizes[i],total_query_points, MPI_T, &stype[i]);
      MPI_Type_commit(&stype[i]);
      MPI_Type_commit(&rtype[i]);
    }

    rescale_xyz(g_size,N_reg,N_reg_g,istart,total_query_points,all_query_points); //SNAFU

    // This if condition is to allow multiple calls to scatter fucntion with different query points
    // without having to create a new plan
    if(this->scatter_baked==true){
  #ifdef INTERP_PINNED
      cudaFreeHost(this->all_query_points_d);
      cudaFreeHost(this->all_f_cubic_d);
      cudaMallocHost((void**)&all_query_points_d,all_query_points_allocation*sizeof(Real) );
      cudaMallocHost((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof);
  #else
      cudaFree(this->all_query_points_d);
      cudaFree(this->all_f_cubic_d);
      cudaMalloc((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof);
      cudaMalloc((void**)&all_query_points_d,all_query_points_allocation*sizeof(Real) );
  #endif
    }
    else{
  #ifdef INTERP_PINNED
      cudaMallocHost((void**)&all_query_points_d,all_query_points_allocation*sizeof(Real) );
      cudaMallocHost((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof);
  #else
      cudaMalloc((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof);
      cudaMalloc((void**)&all_query_points_d,all_query_points_allocation*sizeof(Real) );
  #endif
    }

    timings[2]+=-MPI_Wtime();
    cudaMemcpy(all_query_points_d,all_query_points,all_query_points_allocation*sizeof(Real),cudaMemcpyHostToDevice);
    timings[2]+=+MPI_Wtime();


    this->scatter_baked=true;
    return;
  }

  /*
   * Phase 2 of the parallel interpolation: This function must be called after the scatter function is called.
   * It performs local interpolation for all the points that the processor has for itself, as well as the interpolations
   * that it has to send to other processors. After the local interpolation is performed, a sparse
   * alltoall is performed so that all the interpolated results are sent/received.
   *
   */
  void InterpPlan::interpolate( Real* ghost_reg_grid_vals, int data_dof,
      int* N_reg, int * isize, int* istart, const int N_pts, const int g_size,
      Real* query_values,int* c_dims, MPI_Comm c_comm,double * timings)
  {

    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);
    if(this->allocate_baked==false){
      std::cout<<"ERROR InterpPlan interpolate called before calling allocate.\n";
      return;
    }
    if(this->scatter_baked==false){
      std::cout<<"ERROR InterpPlan interpolate called before calling scatter.\n";
      return;
    }


    timings[2]+=-MPI_Wtime();
    cudaMemcpy(ghost_reg_grid_vals_d,ghost_reg_grid_vals,g_alloc_max*data_dof,cudaMemcpyHostToDevice);
    timings[2]+=+MPI_Wtime();

    timings[1]+=-MPI_Wtime();
    gpu_interp3_ghost_xyz_p(ghost_reg_grid_vals_d, data_dof, N_reg, isize,istart,total_query_points,g_size,all_query_points_d, all_f_cubic_d,true);
    timings[1]+=+MPI_Wtime();

    timings[2]+=-MPI_Wtime();
    cudaMemcpy(all_f_cubic,all_f_cubic_d, total_query_points*sizeof(Real)*data_dof ,cudaMemcpyDeviceToHost);
    timings[2]+=+MPI_Wtime();


    // Now we have to do an alltoall to distribute the interpolated data from all_f_cubic to
    // f_cubic_unordered.
    timings[0]+=-MPI_Wtime();
    {
      int dst_r,dst_s;
      for (int i=0;i<nprocs;++i){
        dst_r=i;//(procid+i)%nprocs;
        dst_s=i;//(procid-i+nprocs)%nprocs;
        s_request[dst_s]=MPI_REQUEST_NULL;
        request[dst_r]=MPI_REQUEST_NULL;
        // Notice that this is the adjoint of the first comm part
        // because now you are sending others f and receiving your part of f
        int soffset=f_index_procs_others_offset[dst_r];
        int roffset=f_index_procs_self_offset[dst_s];
        if(f_index_procs_self_sizes[dst_r]!=0)
          MPI_Irecv(&f_cubic_unordered[roffset],1,rtype[i], dst_r,
              0, c_comm, &request[dst_r]);
        if(f_index_procs_others_sizes[dst_s]!=0)
          MPI_Isend(&all_f_cubic[soffset],1,stype[i],dst_s,
              0, c_comm, &s_request[dst_s]);
      }
      MPI_Status ierr;
      for (int proc=0;proc<nprocs;++proc){
        if(request[proc]!=MPI_REQUEST_NULL)
          MPI_Wait(&request[proc], &ierr);
        if(s_request[proc]!=MPI_REQUEST_NULL)
          MPI_Wait(&s_request[proc], &ierr);
      }
    }
    timings[0]+=+MPI_Wtime();

    // Now copy back f_cubic_unordered to f_cubic in the correct f_index
    for(int dof=0;dof<data_dof;++dof){
      for(int proc=0;proc<nprocs;++proc){
        if(!f_index[proc].empty())
          for(int i=0;i<f_index[proc].size();++i){
            int ind=f_index[proc][i];
            query_values[ind+dof*N_pts]=f_cubic_unordered[f_index_procs_self_offset[proc]+i+dof*N_pts];
          }
      }
    }

    return;
  }


  /*
   * A dummy function that performs the whole process of scatter and interpolation without any planning or
   * prior allocation. Use only for debugging.
   */
  void InterpPlan::slow_run( Real* ghost_reg_grid_vals_d, int data_dof,
      int* N_reg, int * isize, int* istart, const int N_pts, const int g_size, Real* query_points_in,
      Real* query_values,int* c_dims, MPI_Comm c_comm)
  {
    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);

    //printf("====== isize[0]=%d\n",isize[0]);
    int N_reg_g[3], isize_g[3];
    N_reg_g[0]=N_reg[0]+2*g_size;
    N_reg_g[1]=N_reg[1]+2*g_size;
    N_reg_g[2]=N_reg[2]+2*g_size;

    isize_g[0]=isize[0]+2*g_size;
    isize_g[1]=isize[1]+2*g_size;
    isize_g[2]=isize[2]+2*g_size;

    Real h[3]; // original grid size along each axis
    h[0]=1./N_reg[0];
    h[1]=1./N_reg[1];
    h[2]=1./N_reg[2];

    // We copy query_points_in to query_points to aviod overwriting the input coordinates
    Real* query_points=(Real*) malloc(N_pts*COORD_DIM*sizeof(Real));
    memcpy(query_points,query_points_in,N_pts*COORD_DIM*sizeof(Real));
    // Enforce periodicity
    for(int i=0;i<N_pts;i++){
      while(query_points[i*COORD_DIM+0]<=-h[0]) {query_points[i*COORD_DIM+0]=query_points[i*COORD_DIM+0]+1;}
      while(query_points[i*COORD_DIM+1]<=-h[1]) {query_points[i*COORD_DIM+1]=query_points[i*COORD_DIM+1]+1;}
      while(query_points[i*COORD_DIM+2]<=-h[2]) {query_points[i*COORD_DIM+2]=query_points[i*COORD_DIM+2]+1;}

      while(query_points[i*COORD_DIM+0]>=1) {query_points[i*COORD_DIM+0]=query_points[i*COORD_DIM+0]-1;}
      while(query_points[i*COORD_DIM+1]>=1) {query_points[i*COORD_DIM+1]=query_points[i*COORD_DIM+1]-1;}
      while(query_points[i*COORD_DIM+2]>=1) {query_points[i*COORD_DIM+2]=query_points[i*COORD_DIM+2]-1;}
    }


    // Compute the start and end coordinates that this processor owns
    Real iX0[3],iX1[3];
    for (int j=0;j<3;j++){
      iX0[j]=istart[j]*h[j];
      iX1[j]=iX0[j]+(isize[j]-1)*h[j];
    }

    // Now march through the query points and split them into nprocs parts.
    // These are stored in query_outside which is an array of vectors of size nprocs.
    // That is query_outside[i] is a vector that contains the query points that need to
    // be sent to process i. Obviously for the case of query_outside[procid], we do not
    // need to send it to any other processor, as we own the necessary information locally,
    // and interpolation can be done locally.
    int Q_local=0, Q_outside=0;

    // This is needed for one-to-one correspondence with output f. This is becaues we are reshuffling
    // the data according to which processor it land onto, and we need to somehow keep the original
    // index to write the interpolation data back to the right location in the output.
    std::vector <int> f_index[nprocs];
    std::vector <Real> query_outside[nprocs];
    for(int i=0;i<N_pts;i++){
      // The if condition check whether the query points fall into the locally owned domain or not
      if(
          iX0[0]-h[0]<=query_points[i*COORD_DIM+0] && query_points[i*COORD_DIM+0]<=iX1[0]+h[0] &&
          iX0[1]-h[1]<=query_points[i*COORD_DIM+1] && query_points[i*COORD_DIM+1]<=iX1[1]+h[1] &&
          iX0[2]-h[2]<=query_points[i*COORD_DIM+2] && query_points[i*COORD_DIM+2]<=iX1[2]+h[2]
        ){
        query_outside[procid].push_back(query_points[i*COORD_DIM+0]);
        query_outside[procid].push_back(query_points[i*COORD_DIM+1]);
        query_outside[procid].push_back(query_points[i*COORD_DIM+2]);
        f_index[procid].push_back(i);
        Q_local++;
        //PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
        continue;
      }
      else{
        // If the point does not reside in the processor's domain then we have to
        // first compute which processor owns the point. After computing that
        // we add the query point to the corresponding vector.
        int dproc0=(int)(query_points[i*COORD_DIM+0]/h[0])/isize[0];
        int dproc1=(int)(query_points[i*COORD_DIM+1]/h[1])/isize[1];
        int proc=dproc0*c_dims[1]+dproc1; // Compute which proc has to do the interpolation
        //PCOUT<<"proc="<<proc<<std::endl;
        query_outside[proc].push_back(query_points[i*COORD_DIM+0]);
        query_outside[proc].push_back(query_points[i*COORD_DIM+1]);
        query_outside[proc].push_back(query_points[i*COORD_DIM+2]);
        f_index[proc].push_back(i);
        Q_outside++;
        //PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
        continue;
      }

    }

    // Now we need to send the query_points that land onto other processor's domain.
    // This done using a sparse alltoallv.
    // Right now each process knows how much data to send to others, but does not know
    // how much data it should receive. This is a necessary information both for the MPI
    // command as well as memory allocation for received data.
    // So we first do an alltoall to get the f_index[proc].size from all processes.
    int f_index_procs_self_sizes[nprocs]; // sizes of the number of interpolations that need to be sent to procs
    int f_index_procs_others_sizes[nprocs]; // sizes of the number of interpolations that need to be received from procs

    for (int proc=0;proc<nprocs;proc++){
      if(!f_index[proc].empty())
        f_index_procs_self_sizes[proc]=f_index[proc].size();
      else
        f_index_procs_self_sizes[proc]=0;
    }
    MPI_Alltoall(f_index_procs_self_sizes,1, MPI_INT,
        f_index_procs_others_sizes,1, MPI_INT,
        c_comm);

  #ifdef VERBOSE2
    sleep(1);
    if(procid==0){
      std::cout<<"procid="<<procid<<std::endl;
      std::cout<<"f_index_procs_self[0]="<<f_index_procs_self_sizes[0]<<" [1]= "<<f_index_procs_self_sizes[1]<<std::endl;
      std::cout<<"f_index_procs_others[0]="<<f_index_procs_others_sizes[0]<<" [1]= "<<f_index_procs_others_sizes[1]<<std::endl;
    }
    sleep(1);
    if(procid==1){
      std::cout<<"procid="<<procid<<std::endl;
      std::cout<<"f_index_procs_self[0]="<<f_index_procs_self_sizes[0]<<" [1]= "<<f_index_procs_self_sizes[1]<<std::endl;
      std::cout<<"f_index_procs_others[0]="<<f_index_procs_others_sizes[0]<<" [1]= "<<f_index_procs_others_sizes[1]<<std::endl;
    }
  #endif


    // Now we need to allocate memory for the receiving buffer of all query
    // points including ours. This is simply done by looping through
    // f_index_procs_others_sizes and adding up all the sizes.
    // Note that we would also need to know the offsets.
    size_t all_query_points_allocation=0;
    int f_index_procs_others_offset[nprocs]; // offset in the all_query_points array
    int f_index_procs_self_offset[nprocs]; // offset in the query_outside array
    f_index_procs_others_offset[0]=0;
    f_index_procs_self_offset[0]=0;
    for (int proc=0;proc<nprocs;++proc){
      // The reason we multiply by COORD_DIM is that we have three coordinates per interpolation request
      all_query_points_allocation+=f_index_procs_others_sizes[proc]*COORD_DIM;
      if(proc>0){
        f_index_procs_others_offset[proc]=f_index_procs_others_offset[proc-1]+f_index_procs_others_sizes[proc-1];
        f_index_procs_self_offset[proc]=f_index_procs_self_offset[proc-1]+f_index_procs_self_sizes[proc-1];
      }
    }
    int total_query_points=all_query_points_allocation/COORD_DIM;
    Real * all_query_points=(Real*) malloc(all_query_points_allocation*sizeof(Real));
  #ifdef VERBOSE2
    if(procid==0){
      std::cout<<"procid="<<procid<<std::endl;
      for (int proc=0;proc<nprocs;++proc)
        std::cout<<"proc= "<<proc<<" others_offset= "<<f_index_procs_others_offset[proc]<<" others_sizes= "<<f_index_procs_others_sizes[proc]<<std::endl;
      for (int proc=0;proc<nprocs;++proc)
        std::cout<<"proc= "<<proc<<" self_offset= "<<f_index_procs_self_offset[proc]<<" self_sizes= "<<f_index_procs_self_sizes[proc]<<std::endl;
    }
  #endif

    MPI_Request * s_request= new MPI_Request[nprocs];
    MPI_Request * request= new MPI_Request[nprocs];

    // Now perform the allotall to send/recv query_points
    {
      int dst_r,dst_s;
      for (int i=0;i<nprocs;++i){
        dst_r=i;//(procid+i)%nprocs;
        dst_s=i;//(procid-i+nprocs)%nprocs;
        s_request[dst_s]=MPI_REQUEST_NULL;
        request[dst_r]=MPI_REQUEST_NULL;
        int roffset=f_index_procs_others_offset[dst_r]*COORD_DIM; // notice that COORD_DIM is needed because query_points are 3 times f
        int soffset=f_index_procs_self_offset[dst_s]*COORD_DIM;
        if(f_index_procs_others_sizes[dst_r]!=0)
          MPI_Irecv(&all_query_points[roffset],f_index_procs_others_sizes[dst_r]*COORD_DIM,MPI_T, dst_r,
              0, c_comm, &request[dst_r]);
        if(!query_outside[dst_s].empty())
          MPI_Isend(&query_outside[dst_s][0],f_index_procs_self_sizes[dst_s]*COORD_DIM,MPI_T,dst_s,
              0, c_comm, &s_request[dst_s]);
        //if(procid==1){
        //std::cout<<"soffset="<<soffset<<" roffset="<<roffset<<std::endl;
        //std::cout<<"f_index_procs_self_sizes[0]="<<f_index_procs_self_sizes[0]<<std::endl;
        //std::cout<<"f_index_procs_others_sizes[0]="<<f_index_procs_others_sizes[0]<<std::endl;
        //std::cout<<"q_outside["<<dst_s<<"]="<<query_outside[dst_s][0]<<std::endl;
        //}
      }
      // Wait for all the communication to finish
      MPI_Status ierr;
      for (int proc=0;proc<nprocs;++proc){
        if(request[proc]!=MPI_REQUEST_NULL)
          MPI_Wait(&request[proc], &ierr);
        if(s_request[proc]!=MPI_REQUEST_NULL)
          MPI_Wait(&s_request[proc], &ierr);
      }
    }


    // Now perform the interpolation on all query points including those that need to
    // be sent to other processors and store them into all_f_cubic
    Real* all_f_cubic_d;
    Real* all_query_points_d;
    cudaMalloc((void**)&all_f_cubic_d, total_query_points*sizeof(Real)*data_dof);
    cudaMalloc((void**)&all_query_points_d,all_query_points_allocation*sizeof(Real) );
    cudaMemcpy(all_query_points_d,all_query_points,all_query_points_allocation*sizeof(Real),cudaMemcpyHostToDevice);

    gpu_interp3_ghost_xyz_p(ghost_reg_grid_vals_d, data_dof, N_reg, isize,istart,total_query_points,g_size,all_query_points_d, all_f_cubic_d);
    Real* all_f_cubic=(Real*)malloc(total_query_points*sizeof(Real)*data_dof);
    cudaMemcpy(all_f_cubic,all_f_cubic_d, total_query_points*sizeof(Real)*data_dof ,cudaMemcpyDeviceToHost);


    // Now we have to do an alltoall to distribute the interpolated data from all_f_cubic to
    // f_cubic_unordered.
    Real * f_cubic_unordered=(Real*) malloc(N_pts*sizeof(Real)*data_dof); // The reshuffled semi-final interpolated values are stored here
    {
      //PCOUT<<"total_query_points="<<total_query_points<<" N_pts="<<N_pts<<std::endl;
      int dst_r,dst_s;
      MPI_Datatype stype[nprocs],rtype[nprocs];
      for(int i=0;i<nprocs;++i){
        MPI_Type_vector(data_dof,f_index_procs_self_sizes[i],N_pts, MPI_T, &rtype[i]);
        MPI_Type_vector(data_dof,f_index_procs_others_sizes[i],total_query_points, MPI_T, &stype[i]);
        MPI_Type_commit(&stype[i]);
        MPI_Type_commit(&rtype[i]);
      }
      for (int i=0;i<nprocs;++i){
        dst_r=i;//(procid+i)%nprocs;
        dst_s=i;//(procid-i+nprocs)%nprocs;
        s_request[dst_s]=MPI_REQUEST_NULL;
        request[dst_r]=MPI_REQUEST_NULL;
        // Notice that this is the adjoint of the first comm part
        // because now you are sending others f and receiving your part of f
        int soffset=f_index_procs_others_offset[dst_r];
        int roffset=f_index_procs_self_offset[dst_s];
        //if(procid==0)
        //  std::cout<<"procid="<<procid<<" dst_s= "<<dst_s<<" soffset= "<<soffset<<" s_size="<<f_index_procs_others_sizes[dst_s]<<" dst_r= "<<dst_r<<" roffset="<<roffset<<" r_size="<<f_index_procs_self_sizes[dst_r]<<std::endl;
        //if(f_index_procs_self_sizes[dst_r]!=0)
        //  MPI_Irecv(&f_cubic_unordered[roffset],f_index_procs_self_sizes[dst_r],rtype, dst_r,
        //      0, c_comm, &request[dst_r]);
        //if(f_index_procs_others_sizes[dst_s]!=0)
        //  MPI_Isend(&all_f_cubic[soffset],f_index_procs_others_sizes[dst_s],stype,dst_s,
        //      0, c_comm, &s_request[dst_s]);
        //
        if(f_index_procs_self_sizes[dst_r]!=0)
          MPI_Irecv(&f_cubic_unordered[roffset],1,rtype[i], dst_r,
              0, c_comm, &request[dst_r]);
        if(f_index_procs_others_sizes[dst_s]!=0)
          MPI_Isend(&all_f_cubic[soffset],1,stype[i],dst_s,
              0, c_comm, &s_request[dst_s]);
      }
      MPI_Status ierr;
      for (int proc=0;proc<nprocs;++proc){
        if(request[proc]!=MPI_REQUEST_NULL)
          MPI_Wait(&request[proc], &ierr);
        if(s_request[proc]!=MPI_REQUEST_NULL)
          MPI_Wait(&s_request[proc], &ierr);
      }
      for(int i=0;i<nprocs;++i){
        MPI_Type_free(&stype[i]);
        MPI_Type_free(&rtype[i]);
      }
    }

    // Now copy back f_cubic_unordered to f_cubic in the correct f_index
    for(int dof=0;dof<data_dof;++dof){
      for(int proc=0;proc<nprocs;++proc){
        if(!f_index[proc].empty())
          for(int i=0;i<f_index[proc].size();++i){
            int ind=f_index[proc][i];
            //f_cubic[ind]=all_f_cubic[f_index_procs_others_offset[proc]+i];
            query_values[ind+dof*N_pts]=f_cubic_unordered[f_index_procs_self_offset[proc]+i+dof*N_pts];
          }
      }
    }

    free(query_points);
    free(all_query_points);
    free(all_f_cubic);
    cudaFree(all_f_cubic_d);
    cudaFree(all_query_points_d);
    free(f_cubic_unordered);
    delete(s_request);
    delete(request);
    //vector
    for(int proc=0;proc<nprocs;++proc)
    {
      std::vector<int>().swap(f_index[proc]);
      std::vector<Real>().swap(query_outside[proc]);
    }
    return;
  }

  /*
   * Get the left right ghost cells.
   *
   * @param[out] padded_data: The output of the function which pads the input array data, with the ghost
   * cells from left and right neighboring processors.
   * @param[in] data: Input data to be padded
   * @param[in] g_size: The size of the ghost cell padding. Note that it cannot exceed the neighboring processor's
   * local data size
   * @param[in] plan: AccFFT R2C plan
   */
  void ghost_left_right(pvfmm::Iterator<Real> padded_data, Real* data, int g_size,
      accfft_plan_gpu * plan) {
    int nprocs, procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  //  MPI_Comm c_comm = plan->c_comm;

    /* Get the local pencil size and the allocation size */
    //int isize[3],osize[3],istart[3],ostart[3];
    int * isize = plan->isize;
  //  int * osize = plan->isize;
  //  int * istart = plan->istart;
  //  int * ostart = plan->ostart;
  //  int alloc_max = plan->alloc_max;

    MPI_Comm row_comm = plan->row_comm;
    int nprocs_r, procid_r;
    MPI_Comm_rank(row_comm, &procid_r);
    MPI_Comm_size(row_comm, &nprocs_r);
    /* Halo Exchange along y axis
     * Phase 1: Write local data to be sent to the right process to RS
     */
  #ifdef VERBOSE2
    PCOUT<<"\nGL Row Communication\n";
  #endif

    int rs_buf_size = g_size * isize[2] * isize[0];
    Real *RS = (Real*) accfft_alloc(rs_buf_size * sizeof(Real)); // Stores local right ghost data to be sent
    Real *GL = (Real*) accfft_alloc(rs_buf_size * sizeof(Real)); // Left Ghost cells to be received

    for (int x = 0; x < isize[0]; ++x)
      memcpy(&RS[x * g_size * isize[2]],
          &data[x * isize[2] * isize[1] + (isize[1] - g_size) * isize[2]],
          g_size * isize[2] * sizeof(Real));

    /* Phase 2: Send your data to your right process
     * First question is who is your right process?
     */
    int dst_s = (procid_r + 1) % nprocs_r;
    int dst_r = (procid_r - 1) % nprocs_r;
    if (procid_r == 0)
      dst_r = nprocs_r - 1;
    MPI_Request rs_s_request, rs_r_request;
    MPI_Status ierr;
    MPI_Isend(RS, rs_buf_size, MPI_T, dst_s, 0, row_comm, &rs_s_request);
    MPI_Irecv(GL, rs_buf_size, MPI_T, dst_r, 0, row_comm, &rs_r_request);
    MPI_Wait(&rs_s_request, &ierr);
    MPI_Wait(&rs_r_request, &ierr);

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"procid="<<procid<<" data\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1];++j)
        std::cout<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<g_size;++j)
        std::cout<<RS[(i*g_size+j)*isize[2]]<<" ";
        //PCOUT<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
    sleep(1);
    if(procid==1) {
      std::cout<<"procid="<<procid<<" GL data\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<g_size;++j)
        std::cout<<GL[(i*g_size+j)*isize[2]]<<" ";
        //PCOUT<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
    PCOUT<<"\nGR Row Communication\n";
  #endif

    /* Phase 3: Now do the exact same thing for the right ghost side */
    int ls_buf_size = g_size * isize[2] * isize[0];
    Real *LS = (Real*) accfft_alloc(ls_buf_size * sizeof(Real)); // Stores local right ghost data to be sent
    Real *GR = (Real*) accfft_alloc(ls_buf_size * sizeof(Real)); // Left Ghost cells to be received
    for (int x = 0; x < isize[0]; ++x)
      memcpy(&LS[x * g_size * isize[2]], &data[x * isize[2] * isize[1]],
          g_size * isize[2] * sizeof(Real));

    /* Phase 4: Send your data to your right process
     * First question is who is your right process?
     */
    dst_s = (procid_r - 1) % nprocs_r;
    dst_r = (procid_r + 1) % nprocs_r;
    if (procid_r == 0)
      dst_s = nprocs_r - 1;
    MPI_Isend(LS, ls_buf_size, MPI_T, dst_s, 0, row_comm, &rs_s_request);
    MPI_Irecv(GR, ls_buf_size, MPI_T, dst_r, 0, row_comm, &rs_r_request);
    MPI_Wait(&rs_s_request, &ierr);
    MPI_Wait(&rs_r_request, &ierr);

  #ifdef VERBOSE2
    if(procid==1) {
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1];++j)
        std::cout<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
      std::cout<<"1 sending to dst_s="<<dst_s<<std::endl;
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<g_size;++j)
        std::cout<<LS[(i*g_size+j)*isize[2]]<<" ";
        //PCOUT<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
      std::cout<<"\n";
    }
    sleep(1);
    if(procid==0) {
      std::cout<<"0 receiving from dst_r="<<dst_r<<std::endl;
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<g_size;++j)
        std::cout<<GR[(i*g_size+j)*isize[2]]<<" ";
        //PCOUT<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
  #endif

    // Phase 5: Pack the data GL+ data + GR
    for (int i = 0; i < isize[0]; ++i) {
      memcpy(&padded_data[i * isize[2] * (isize[1] + 2 * g_size)],
          &GL[i * g_size * isize[2]], g_size * isize[2] * sizeof(Real));
      memcpy(
          &padded_data[i * isize[2] * (isize[1] + 2 * g_size)
              + g_size * isize[2]], &data[i * isize[2] * isize[1]],
          isize[1] * isize[2] * sizeof(Real));
      memcpy(
          &padded_data[i * isize[2] * (isize[1] + 2 * g_size)
              + g_size * isize[2] + isize[2] * isize[1]],
          &GR[i * g_size * isize[2]], g_size * isize[2] * sizeof(Real));
    }

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"procid= "<<procid<<" padded_array=\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<padded_data[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
  #endif

    accfft_free(LS);
    accfft_free(GR);
    accfft_free(RS);
    accfft_free(GL);

  }

  /*
   * Get the top bottom ghost cells AFTER getting the left right ones.
   *
   * @param[out] ghost_data: The output of the function which pads the input array data, with the ghost
   * cells from neighboring processors.
   * @param[in] padded_data: Input data that is already padded with ghost cells from left and right.
   * @param[in] g_size: The size of the ghost cell padding. Note that it cannot exceed the neighboring processor's
   * local data size
   * @param[in] plan: AccFFT R2C plan
   */
  void ghost_top_bottom(pvfmm::Iterator<Real> ghost_data, pvfmm::Iterator<Real> padded_data, int g_size,
      accfft_plan_gpu * plan) {
    int nprocs, procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  //  MPI_Comm c_comm = plan->c_comm;

    /* Get the local pencil size and the allocation size */
    //int isize[3],osize[3],istart[3],ostart[3];
    int * isize = plan->isize;
  //  int * osize = plan->isize;
  //  int * istart = plan->istart;
  //  int * ostart = plan->ostart;
  //  int alloc_max = plan->alloc_max;

    MPI_Comm col_comm = plan->col_comm;
    int nprocs_c, procid_c;
    MPI_Comm_rank(col_comm, &procid_c);
    MPI_Comm_size(col_comm, &nprocs_c);

    /* Halo Exchange along x axis
     * Phase 1: Write local data to be sent to the bottom process
     */
  #ifdef VERBOSE2
    PCOUT<<"\nGB Col Communication\n";
  #endif
    int bs_buf_size = g_size * isize[2] * (isize[1] + 2 * g_size); // isize[1] now includes two side ghost cells
    //Real *BS=(Real*)accfft_alloc(bs_buf_size*sizeof(Real)); // Stores local right ghost data to be sent
    pvfmm::Iterator<Real> GT = pvfmm::aligned_new<Real>(bs_buf_size); // Left Ghost cells to be received
    // snafu: not really necessary to do memcpy, you can simply use padded_data directly
    //memcpy(BS,&padded_data[(isize[0]-g_size)*isize[2]*(isize[1]+2*g_size)],bs_buf_size*sizeof(Real));
    Real* BS = &padded_data[(isize[0] - g_size) * isize[2]
                              * (isize[1] + 2 * g_size)];
    /* Phase 2: Send your data to your bottom process
     * First question is who is your bottom process?
     */
    int dst_s = (procid_c + 1) % nprocs_c;
    int dst_r = (procid_c - 1) % nprocs_c;
    if (procid_c == 0)
      dst_r = nprocs_c - 1;
    MPI_Request bs_s_request, bs_r_request;
    MPI_Status ierr;
    MPI_Isend(&BS[0], bs_buf_size, MPI_T, dst_s, 0, col_comm, &bs_s_request);
    MPI_Irecv(&GT[0], bs_buf_size, MPI_T, dst_r, 0, col_comm, &bs_r_request);
    MPI_Wait(&bs_s_request, &ierr);
    MPI_Wait(&bs_r_request, &ierr);

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"procid= "<<procid<<" padded_array=\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<padded_data[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }

      std::cout<<"procid= "<<procid<<" dst_s="<<dst_s<<" BS_array=\n";
      for (int i=0;i<g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<BS[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
    sleep(1);
    if(procid==2) {
      std::cout<<"procid= "<<procid<<" dst_r="<<dst_r<<" GT=\n";
      for (int i=0;i<g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<GT[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }

    PCOUT<<"\nGB Col Communication\n";
  #endif

    /* Phase 3: Now do the exact same thing for the right ghost side */
    int ts_buf_size = g_size * isize[2] * (isize[1] + 2 * g_size); // isize[1] now includes two side ghost cells
    //Real *TS=(Real*)accfft_alloc(ts_buf_size*sizeof(Real)); // Stores local right ghost data to be sent
    pvfmm::Iterator<Real> GB = pvfmm::aligned_new<Real>(ts_buf_size); // Left Ghost cells to be received
    // snafu: not really necessary to do memcpy, you can simply use padded_data directly
    //memcpy(TS,padded_data,ts_buf_size*sizeof(Real));
    Real *TS = &padded_data[0];

    /* Phase 4: Send your data to your right process
     * First question is who is your right process?
     */
    MPI_Request ts_s_request, ts_r_request;
    dst_s = (procid_c - 1) % nprocs_c;
    dst_r = (procid_c + 1) % nprocs_c;
    if (procid_c == 0)
      dst_s = nprocs_c - 1;
    MPI_Isend(&TS[0], ts_buf_size, MPI_T, dst_s, 0, col_comm, &ts_s_request);
    MPI_Irecv(&GB[0], ts_buf_size, MPI_T, dst_r, 0, col_comm, &ts_r_request);
    MPI_Wait(&ts_s_request, &ierr);
    MPI_Wait(&ts_r_request, &ierr);

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"procid= "<<procid<<" padded_array=\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<padded_data[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }

      std::cout<<"procid= "<<procid<<" dst_s="<<dst_s<<" BS_array=\n";
      for (int i=0;i<g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<TS[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
    sleep(1);
    if(procid==2) {
      std::cout<<"procid= "<<procid<<" dst_r="<<dst_r<<" GB=\n";
      for (int i=0;i<g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<GB[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
  #endif

    // Phase 5: Pack the data GT+ padded_data + GB
    memcpy(&ghost_data[0], &GT[0],
        g_size * isize[2] * (isize[1] + 2 * g_size) * sizeof(Real));
    memcpy(&ghost_data[g_size * isize[2] * (isize[1] + 2 * g_size)],
        &padded_data[0],
        isize[0] * isize[2] * (isize[1] + 2 * g_size) * sizeof(Real));
    memcpy(
        &ghost_data[g_size * isize[2] * (isize[1] + 2 * g_size)
            + isize[0] * isize[2] * (isize[1] + 2 * g_size)], &GB[0],
        g_size * isize[2] * (isize[1] + 2 * g_size) * sizeof(Real));

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"\n final ghost data\n";
      for (int i=0;i<isize[0]+2*g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<ghost_data[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
  #endif

    //accfft_free(TS);
    pvfmm::aligned_delete<Real>(GB);
    //accfft_free(BS);
    //accfft_free(GT);
    pvfmm::aligned_delete<Real>(GT);
  }

  /*
   * Perform a periodic z padding of size g_size. This function must be called after the ghost_top_bottom.
   * The idea is to have a symmetric periodic padding in all directions not just x, and y.
   *
   * @param[out] ghost_data_z: The output of the function which pads the input array data, with the ghost
   * cells from neighboring processors.
   * @param[in] ghost_data: Input data that is already padded in x, and y direction with ghost cells
   * @param[in] g_size: The size of the ghost cell padding. Note that it cannot exceed the neighboring processor's
   * local data size
   * @param[in] isize_g: An integer array specifying ghost cell padded local sizes.
   * @param[in] plan: AccFFT R2C plan
   */
  void ghost_z(Real *ghost_data_z, pvfmm::Iterator<Real> ghost_data, int g_size, int* isize_g,
      accfft_plan_gpu* plan) {

    int * isize = plan->isize;
    for (int i = 0; i < isize_g[0]; ++i)
      for (int j = 0; j < isize_g[1]; ++j) {
        memcpy(&ghost_data_z[(i * isize_g[1] + j) * isize_g[2]],
            &ghost_data[(i * isize_g[1] + j) * isize[2] + isize[2]
                - g_size], g_size * sizeof(Real));
        memcpy(&ghost_data_z[(i * isize_g[1] + j) * isize_g[2] + g_size],
            &ghost_data[(i * isize_g[1] + j) * isize[2]],
            isize[2] * sizeof(Real));
        memcpy(
            &ghost_data_z[(i * isize_g[1] + j) * isize_g[2] + g_size
                + isize[2]],
            &ghost_data[(i * isize_g[1] + j) * isize[2]],
            g_size * sizeof(Real));
      }
    return;
  }


  /*
   * Returns the necessary memory allocation in Bytes for the ghost data, as well
   * the local ghost sizes when ghost cell padding is desired only in x and y directions (and not z direction
   * which is locally owned).
   * @param[in] plan: AccFFT plan
   * @param[in] g_size: The number of ghost cells desired. Note that g_size cannot be bigger than
   * the minimum isize in each dimension among all processors. This means that you cannot get ghost
   * cells from a processor that is not a neighbor of the calling processor.
   * @param[out] isize_g: The new local sizes after getting the ghost cells.
   * @param[out] istart_g: Returns the new istart after getting the ghost cells. Note that this is the global
   * istart of the ghost cells. So for example, if processor zero gets 3 ghost cells, the left ghost cells will
   * come from the last process because of the periodicity. Then the istart_g would be the index of those elements
   * (that originally resided in the last processor).
   */

  size_t accfft_ghost_local_size_dft_r2c(accfft_plan_gpu* plan, int g_size,
      int * isize_g, int* istart_g) {

    size_t alloc_max = plan->alloc_max;
    int *isize = plan->isize;
    int *istart = plan->istart;
    int* n = plan->N;
    istart_g[2] = istart[2];
    isize_g[2] = isize[2];

    istart_g[0] = istart[0] - g_size;
    istart_g[1] = istart[1] - g_size;

    if (istart_g[0] < 0)
      istart_g[0] += n[0];
    if (istart_g[1] < 0)
      istart_g[1] += n[1];

    isize_g[0] = isize[0] + 2 * g_size;
    isize_g[1] = isize[1] + 2 * g_size;
    return (alloc_max + 2 * g_size * isize[2] * isize[0] * sizeof(Real)
        + 2 * g_size * isize[2] * (isize[1] + 2 * g_size) * sizeof(Real));
  }

  /*size_t accfft_ghost_local_size_dft_r2c(accfft_plan_gpu* plan, int g_size,
      int * isize_g, int* istart_g) {
    return accfft_ghost_local_size_dft_r2c((accfft_plan_gpu*)plan, g_size, isize_g, istart_g);
  }*/
  /*
   * Gather the ghost cells for a real input when ghost cell padding is desired only in x and y
   * directions (and not z direction which is locally owned). This function currently has the following limitations:
   *  - The AccFFT plan has to be outplace. Note that for inplace R2C plans, the input array has to be
   *  padded, which would slightly change the pattern of communicating ghost cells.
   *  - The number of ghost cells needed has to be the same in both x and y directions (note that each
   *  processor owns the whole z direction locally, so typically you would not need ghost cells. You can
   *  call accfft_get_ghost_xyz which would actually get ghost cells in z direction as well, but that is
   *  like a periodic padding in z direction).
   *  - The number of ghost cells requested cannot be bigger than the minimum isize of all processors. This
   *  means that a process cannot get a ghost element that does not belong to its immediate neighbors. This
   *  is a limitation that should barely matter, as typically ghost_size is a small integer while the global
   *  array sizes are very large. In mathematical terms g_size< min(isize[0],isize[1]) among isize of all
   *  processors.
   *
   * @param[in] plan: AccFFT plan
   * @param[in] g_size: The number of ghost cells desired. Note that g_size cannot be bigger than
   * the minimum isize in each dimension among all processors. This means that you cannot get ghost
   * cells from a processor that is not a neighbor of the calling processor.
   * @param[in] isize_g: An integer array specifying ghost cell padded local sizes.
   * @param[in] data: The local data whose ghost cells from other processors are sought.
   * @param[out] ghost_data: An array that is the ghost cell padded version of the input data.
   */
  void accfft_get_ghost(accfft_plan_gpu* plan, int g_size, int* isize_g, Real* data,
      Real* ghost_data) {
    int nprocs, procid;
    MPI_Comm_rank(plan->c_comm, &procid);
    MPI_Comm_size(plan->c_comm, &nprocs);

    if (plan->inplace == true) {
      PCOUT << "accfft_get_ghost_r2c does not support inplace transforms."
          << std::endl;
      return;
    }

    if (g_size == 0) {
      memcpy(ghost_data, data, plan->alloc_max);
      return;
    }

    int *isize = plan->isize;
    //int *istart = plan->istart;
    //int *n = plan->N;
    if (g_size > isize[0] || g_size > isize[1]) {
      std::cout
          << "accfft_get_ghost_r2c does not support g_size greater than isize."
          << std::endl;
      return;
    }

    pvfmm::Iterator<Real> padded_data = pvfmm::aligned_new<Real>
      (plan->alloc_max + 2 * g_size * isize[2] * isize[0]);
    ghost_left_right(padded_data, data, g_size, plan);
    ghost_top_bottom(ghost_data, padded_data, g_size, plan);
    pvfmm::aligned_delete<Real>(padded_data);
    return;

  }

  /*void accfft_get_ghost(accfft_plan* plan, int g_size, int* isize_g, Real* data,
      Real* ghost_data) {
    accfft_get_ghost((accfft_plan_gpu*)plan, g_size, isize_g, data, ghost_data);
  }*/
  /*
   * Returns the necessary memory allocation in Bytes for the ghost data, as well
   * the local ghost sizes when padding in all directions (including z direction that is locally owned by each process).
   * @param[in] plan: AccFFT plan
   * @param[in] g_size: The number of ghost cells desired. Note that g_size cannot be bigger than
   * the minimum isize in each dimension among all processors. This means that you cannot get ghost
   * cells from a processor that is not a neighbor of the calling processor.
   * @param[out] isize_g: The new local sizes after getting the ghost cells.
   * @param[out] istart_g: Returns the new istart after getting the ghost cells. Note that this is the global
   * istart of the ghost cells. So for example, if processor zero gets 3 ghost cells, the left ghost cells will
   * come from the last process because of the periodicity. Then the istart_g would be the index of those elements
   * (that originally resided in the last processor).
   */

  size_t accfft_ghost_xyz_local_size_dft_r2c(accfft_plan_gpu* plan, int g_size,
      int * isize_g, int* istart_g) {

    size_t alloc_max = plan->alloc_max;
    int *isize = plan->isize;
    int *istart = plan->istart;
    int* n = plan->N;

    istart_g[0] = istart[0] - g_size;
    istart_g[1] = istart[1] - g_size;
    istart_g[2] = istart[2] - g_size;

    if (istart_g[0] < 0)
      istart_g[0] += n[0];
    if (istart_g[1] < 0)
      istart_g[1] += n[1];
    if (istart_g[2] < 0)
      istart_g[2] += n[2];

    isize_g[0] = isize[0] + 2 * g_size;
    isize_g[1] = isize[1] + 2 * g_size;
    isize_g[2] = isize[2] + 2 * g_size;
    size_t alloc_max_g = alloc_max + 2 * g_size * isize[2] * isize[0] * sizeof(Real)
        + 2 * g_size * isize[2] * isize_g[1] * sizeof(Real)
        + 2 * g_size * isize_g[0] * isize_g[1] * sizeof(Real);
    alloc_max_g += (16*isize_g[2]*isize_g[1]+16*isize_g[1]+16)*sizeof(Real); // to account for padding required for peeled loop in interp
    return alloc_max_g;
  }

  /*
   * Gather the ghost cells for a real input when ghost cell padding is desired in all directions including z direction
   * (which is locally owned). This function currently has the following limitations:
   *  - The AccFFT plan has to be outplace. Note that for inplace R2C plans, the input array has to be
   *  padded, which would slightly change the pattern of communicating ghost cells.
   *  - The number of ghost cells needed has to be the same in both x and y directions (note that each
   *  processor owns the whole z direction locally, so typically you would not need ghost cells. You can
   *  call accfft_get_ghost_xyz which would actually get ghost cells in z direction as well, but that is
   *  like a periodic padding in z direction).
   *  - The number of ghost cells requested cannot be bigger than the minimum isize of all processors. This
   *  means that a process cannot get a ghost element that does not belong to its immediate neighbors. This
   *  is a limitation that should barely matter, as typically ghost_size is a small integer while the global
   *  array sizes are very large. In mathematical terms g_size< min(isize[0],isize[1],isize[2]) among isize of all
   *  processors.
   *
   * @param[in] plan: AccFFT plan
   * @param[in] g_size: The number of ghost cells desired. Note that g_size cannot be bigger than
   * the minimum isize in each dimension among all processors. This means that you cannot get ghost
   * cells from a processor that is not a neighbor of the calling processor.
   * @param[in] isize_g: An integer array specifying ghost cell padded local sizes.
   * @param[in] data: The local data whose ghost cells from other processors are sought.
   * @param[out] ghost_data: An array that is the ghost cell padded version of the input data.
   */
  void accfft_get_ghost_xyz(accfft_plan_gpu* plan, int g_size, int* isize_g,
      Real* data, Real* ghost_data) {
    int nprocs, procid;
    MPI_Comm_rank(plan->c_comm, &procid);
    MPI_Comm_size(plan->c_comm, &nprocs);

    if (plan->inplace == true) {
      PCOUT << "accfft_get_ghost_r2c does not support inplace transforms."
          << std::endl;
      return;
    }

    if (g_size == 0) {
      memcpy(ghost_data, data, plan->alloc_max);
      return;
    }

    int *isize = plan->isize;
  //  int *istart = plan->istart;
  //  int *n = plan->N;
    if (g_size > isize[0] || g_size > isize[1]) {
      std::cout
          << "accfft_get_ghost_r2c does not support g_size greater than isize."
          << std::endl;
      return;
    }

    pvfmm::Iterator<Real> padded_data = pvfmm::aligned_new<Real>
      (plan->alloc_max + 2 * g_size * isize[2] * isize[0]);
    pvfmm::Iterator<Real> ghost_data_xy = pvfmm::aligned_new<Real>(
        plan->alloc_max + 2 * g_size * isize[2] * isize[0]
            + 2 * g_size * isize[2] * isize_g[1]);

    ghost_left_right(padded_data, data, g_size, plan);
    ghost_top_bottom(ghost_data_xy, padded_data, g_size, plan);
    ghost_z(&ghost_data[0], ghost_data_xy, g_size, isize_g, plan);

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"\n final ghost data\n";
      for (int i=0;i<isize_g[0];++i) {
        for (int j=0;j<isize_g[1];++j)
        std::cout<<ghost_data[(i*isize_g[1]+j)*isize_g[2]]<<" ";
        std::cout<<"\n";
      }

      std::cout<<"\n a random z\n";
      int i=3+0*isize_g[0]/2;
      int j=3+0*isize_g[1]/2;
      for(int k=0;k<isize_g[2];++k)
      std::cout<<ghost_data[(i*isize_g[1]+j)*isize_g[2]+k]<<" ";
      std::cout<<"\n";
    }
  #endif

    pvfmm::aligned_delete<Real>(padded_data);
    pvfmm::aligned_delete<Real>(ghost_data_xy);
    return;
  }

  void rescale_xyz(const int g_size, int* N_reg, int* N_reg_g, int* istart,
      const int N_pts, Real* Q_) {

    if (g_size == 0)
      return;
    Real hp[3];
    Real h[3];
    hp[0] = 1. / N_reg_g[0]; // New mesh size
    hp[1] = 1. / N_reg_g[1]; // New mesh size
    hp[2] = 1. / N_reg_g[2]; // New mesh size

    h[0] = 1. / (N_reg[0]); // old mesh size
    h[1] = 1. / (N_reg[1]); // old mesh size
    h[2] = 1. / (N_reg[2]); // old mesh size

    const Real factor0 = (1. - (2. * g_size + 1.) * hp[0]) / (1. - h[0]);
    const Real factor1 = (1. - (2. * g_size + 1.) * hp[1]) / (1. - h[1]);
    const Real factor2 = (1. - (2. * g_size + 1.) * hp[2]) / (1. - h[2]);
    const Real iX0 = istart[0]*h[0];
    const Real iX1 = istart[1]*h[1];
    const Real iX2 = istart[2]*h[2];

    for (int i = 0; i < N_pts; i++) {
      Q_[0 + COORD_DIM * i] = (Q_[0 + COORD_DIM * i]
          - iX0) * factor0 + g_size * hp[0];
      Q_[1 + COORD_DIM * i] = (Q_[1 + COORD_DIM * i]
          - iX1) * factor1 + g_size * hp[1];
      Q_[2 + COORD_DIM * i] = (Q_[2 + COORD_DIM * i]
          - iX2) * factor2 + g_size * hp[2];
    }
    return;
  } // end of rescale_xyz


  /*
   * multiply and shift query points based on each proc's domain
   */
  void rescale_xyz(const int g_size, int* N_reg, int* N_reg_g, int* istart,
      int* isize, int* isize_g, const int N_pts, Real* Q_) {

    if (g_size == 0)
      return;
    Real h[3];

    h[0] = 1. / (N_reg[0]);
    h[1] = 1. / (N_reg[1]);
    h[2] = 1. / (N_reg[2]);

    const Real iX0 = istart[0]*h[0];
    const Real iX1 = istart[1]*h[1];
    const Real iX2 = istart[2]*h[2];
    const Real N0 = N_reg[0];
    const Real N1 = N_reg[1];
    const Real N2 = N_reg[2];

  #pragma omp parallel for
    for (int i = 0; i < N_pts; i++) {
      Real* Q_ptr = &Q_[COORD_DIM * i];
      Q_ptr[0] = (Q_ptr[0]-iX0)*N0+g_size;
      Q_ptr[1] = (Q_ptr[1]-iX1)*N1+g_size;
      Q_ptr[2] = (Q_ptr[2]-iX2)*N2+g_size;
    }

    //std::cout <<std::floor(N_pts/16.0)*16<< '\t' <<std::ceil(N_pts/16.0)*16 << std::endl;
    // set Q to be one so that for the peeled loop we only access grid indxx =0
    if(N_pts%16 != 0)
    for (int i = std::floor(N_pts/16.0)*16+N_pts%16; i < std::ceil(N_pts/16.0)*16; i++) {
      Real* Q_ptr = &Q_[COORD_DIM * i];
      Q_ptr[0] = 1;
      Q_ptr[1] = 1;
      Q_ptr[2] = 1;
    }
    return;
  } // end of rescale_xyz

  void rescale_xyzgrid(const int g_size, int* N_reg, int* N_reg_g, int* istart,
      int* isize, int* isize_g, const int N_pts, pvfmm::Iterator<Real> Q_) {

    if (g_size == 0)
      return;
    Real h[3];

    h[0] = 1. / (N_reg[0]);
    h[1] = 1. / (N_reg[1]);
    h[2] = 1. / (N_reg[2]);

    const Real iX0 = istart[0]*h[0];
    const Real iX1 = istart[1]*h[1];
    const Real iX2 = istart[2]*h[2];
    const Real Nx = N_reg[0];
    const Real Ny = N_reg[1];
    const Real Nz = N_reg[2];
    const int isize_g2 = isize_g[2];
    const int isize_g1g2 = isize_g2 * isize_g[1];
    pvfmm::Iterator<Real> tmp = pvfmm::aligned_new<Real>(N_pts*3);
    pvfmm::memcopy(tmp, Q_, N_pts*3);

  #pragma omp parallel for
    for (int i = 0; i < N_pts; i++) {
      Real* Q_ptr = &Q_[4 * i];
      Real* tmp_ptr = &tmp[3 * i];
      // std::cout << tmp_ptr[0] << '\t' << tmp_ptr[1] << '\t' << tmp_ptr[2] << std::endl;
      Q_ptr[0] = (tmp_ptr[0]-iX0)*Nx+g_size;
      Q_ptr[1] = (tmp_ptr[1]-iX1)*Ny+g_size;
      Q_ptr[2] = (tmp_ptr[2]-iX2)*Nz+g_size;

      const int grid_indx0 = ((int)(Q_ptr[0])) - 1;
      Q_ptr[0] -= grid_indx0;
      const int grid_indx1 = ((int)(Q_ptr[1])) - 1;
      Q_ptr[1] -= grid_indx1;
      const int grid_indx2 = ((int)(Q_ptr[2])) - 1;
      Q_ptr[2] -= grid_indx2;
      const int indxx = isize_g1g2 * grid_indx0 + grid_indx2 + isize_g2 * grid_indx1 ;
      Q_ptr[3] = (Real)indxx;
      // std::cout << grid_indx0 << '\t'
      // << grid_indx1 << '\t'
      // << grid_indx2 << std::endl;
    }
    pvfmm::aligned_delete(tmp);

    //std::cout <<std::floor(N_pts/16.0)*16<< '\t' <<std::ceil(N_pts/16.0)*16 << std::endl;
    // set Q to be one so that for the peeled loop we only access grid indxx =0
    if(N_pts%16 != 0)
    for (int i = std::floor(N_pts/16.0)*16+N_pts%16; i < std::ceil(N_pts/16.0)*16; i++) {
      Real* Q_ptr = &Q_[4 * i];
      Q_ptr[0] = 1;
      Q_ptr[1] = 1;
      Q_ptr[2] = 1;
      Q_ptr[3] = 0;
    }
    return;
  } // end of rescale_xyz
  // acknowledgemet to http://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
  // x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
  float sum8(__m256 x) {
      // hiQuad = ( x7, x6, x5, x4 )
      const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
      // loQuad = ( x3, x2, x1, x0 )
      const __m128 loQuad = _mm256_castps256_ps128(x);
      // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
      const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
      // loDual = ( -, -, x1 + x5, x0 + x4 )
      const __m128 loDual = sumQuad;
      // hiDual = ( -, -, x3 + x7, x2 + x6 )
      const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
      // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
      const __m128 sumDual = _mm_add_ps(loDual, hiDual);
      // lo = ( -, -, -, x0 + x2 + x4 + x6 )
      const __m128 lo = sumDual;
      // hi = ( -, -, -, x1 + x3 + x5 + x7 )
      const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
      // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
      const __m128 sum = _mm_add_ss(lo, hi);
      return _mm_cvtss_f32(sum);
  }


//------------ NO CUDA -----------
#elif (!CUDA)

  #define _mm256_set_m128(va, vb) \
            _mm256_insertf128_ps(_mm256_castps128_ps256(vb), va, 1)
  #define _mm512_set_m256(va, vb) \
            _mm512_insertf32x8(_mm512_castps256_ps512(vb), va, 1)
  #include <cmath>
  #include <mpi.h>
  #include <stdlib.h>
  #include <iostream>
  #include <string.h>
  #include <vector>

  #include <immintrin.h>
  #define COORD_DIM 3

  #include <cmath>
  #include <algorithm>
  #include <stdint.h>
  #include <limits.h>
  #ifdef __unix__
  # include <unistd.h>
  #elif defined _WIN32
  # include <windows.h>
  #define sleep(x) Sleep(1000 * x)
  #endif

  #ifdef INTERP_DEBUG
  template <typename T>
  void parallel_print(T in, const char* prefix) {

    int nprocs, procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    PCOUT << prefix << std::endl;
    for(int proc = 0; proc < nprocs; ++proc) {
      if(procid == proc)
        std::cout << "proc = " << proc << "\t" << in << std::endl;
      else
        sleep(.8);
    }

  }
  #endif

  InterpPlan::InterpPlan() {
    this->allocate_baked = false;
    this->scatter_baked = false;
    procs_i_recv_from_size_ = 0;
    procs_i_send_to_size_ = 0;
  }

  void InterpPlan::allocate(int N_pts, int* data_dofs, int nplans) {
    int nprocs, procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  #ifdef INTERP_DEBUG
    PCOUT << "entered allocate\n";
  #endif
    query_points = pvfmm::aligned_new<Real>(N_pts * COORD_DIM);
    pvfmm::memset(query_points,0, N_pts*COORD_DIM);

    f_index_procs_others_offset = pvfmm::aligned_new<int>(nprocs); // offset in the all_query_points array
    f_index_procs_self_offset   = pvfmm::aligned_new<int>(nprocs); // offset in the query_outside array
    f_index_procs_self_sizes    = pvfmm::aligned_new<int>(nprocs); // sizes of the number of interpolations that need to be sent to procs
    f_index_procs_others_sizes  = pvfmm::aligned_new<int>(nprocs); // sizes of the number of interpolations that need to be received from procs

    s_request = pvfmm::aligned_new<MPI_Request>(nprocs);
    request = pvfmm::aligned_new<MPI_Request>(2*nprocs);

    f_index = new std::vector<int>[nprocs];
    query_outside = new std::vector<Real>[nprocs];


    this->nplans_ = nplans; // number of reuses of the plan with the same scatter points
    this->data_dofs_ = pvfmm::aligned_new<int>(nplans);

    int max =0;
    for(int i = 0; i < nplans_; ++i) {
      max = std::max(max, data_dofs[i]);
      this->data_dofs_[i] = data_dofs[i];
    }
    this->data_dof_max = max;

    f_cubic_unordered = pvfmm::aligned_new<Real>(N_pts * data_dof_max); // The reshuffled semi-final interpolated values are stored here
    memset(&f_cubic_unordered[0],0, N_pts * sizeof(Real) * data_dof_max);

    stypes = pvfmm::aligned_new<MPI_Datatype>(nprocs*nplans_); // strided for multiple plan calls
    rtypes = pvfmm::aligned_new<MPI_Datatype>(nprocs*nplans_);
    this->allocate_baked = true;
  #ifdef INTERP_DEBUG
    PCOUT << "allocate done\n";
  #endif
  }

  class Trip {
  public:
    Trip() {
    }
    ;
    Real x;
    Real y;
    Real z;
    int ind;
    int* N;
    Real* h;

  };

  #ifdef SORT_QUERIES
  static bool ValueCmp(Trip const & a, Trip const & b) {
    return a.z + a.y / a.h[1] * a.N[2] + a.x / a.h[0] * a.N[1] * a.N[2]
        < b.z + b.y / b.h[1] * b.N[2] + b.x / b.h[0] * b.N[1] * b.N[2];
  }
  #endif

  #ifdef SORT_QUERIES
  static void sort_queries(std::vector<Real>* query_outside,
      std::vector<int>* f_index, int* N_reg, Real* h, MPI_Comm c_comm) {

    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);
    for (int proc = 0; proc < nprocs; ++proc) {
      int qsize = query_outside[proc].size() / COORD_DIM;
      Trip* trip = new Trip[qsize];

      for (int i = 0; i < qsize; ++i) {
        trip[i].x = query_outside[proc][i * COORD_DIM + 0];
        trip[i].y = query_outside[proc][i * COORD_DIM + 1];
        trip[i].z = query_outside[proc][i * COORD_DIM + 2];
        trip[i].ind = f_index[proc][i];
        trip[i].N = N_reg;
        trip[i].h = h;
      }

      std::sort(trip, trip + qsize, ValueCmp);

      query_outside[proc].clear();
      f_index[proc].clear();

      for (int i = 0; i < qsize; ++i) {
        query_outside[proc].push_back(trip[i].x);
        query_outside[proc].push_back(trip[i].y);
        query_outside[proc].push_back(trip[i].z);
        f_index[proc].push_back(trip[i].ind);
      }
      delete[] trip;
    }
    return;
  }
  #endif

  #include "libmorton/libmorton/include/morton.h"

  class zTrip {
  public:
    zTrip() {
    }
    ;
    int mortid_;
    int i_;// bin index

  };
  inline uint64_t mortonEncode_for(unsigned int x, unsigned int y, unsigned int z){
      uint64_t answer = 0;
      for (uint64_t i = 0; i < (sizeof(uint64_t)* CHAR_BIT)/3; ++i) {
          answer |= ((x & ((uint64_t)1 << i)) << 2*i) | ((y & ((uint64_t)1 << i)) << (2*i + 1)) | ((z & ((uint64_t)1 << i)) << (2*i + 2));
      }
      return answer;
  }

  #ifdef SORT_QUERIES
  static bool zValueCmp(zTrip const & a, zTrip const & b) {
    return (a.mortid_ < b.mortid_);
  }
  #endif


  #ifdef SORT_QUERIES
  static void zsort_queries(std::vector<Real>* query_outside,
      std::vector<int>* f_index, int* N_reg, Real* h, MPI_Comm c_comm) {

    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);
    const Real h0 = h[0];
    const Real h1 = h[1];
    const Real h2 = h[2];

  #ifdef FAST_INTERP_BINNING
    int bsize_xyz[COORD_DIM];
    const int bsize = 16;
    bsize_xyz[0] = std::ceil(N_reg[0] / (Real)bsize);
    bsize_xyz[1] = std::ceil(N_reg[1] / (Real)bsize);
    bsize_xyz[2] = std::ceil(N_reg[2] / (Real)bsize);
    const size_t total_bsize = bsize_xyz[0] * bsize_xyz[1] * bsize_xyz[2];
    pvfmm::Iterator<zTrip> trip = pvfmm::aligned_new<zTrip>(total_bsize);
    for(int i = 0; i < bsize_xyz[0]; ++i) {
      for(int j = 0; j < bsize_xyz[1]; ++j) {
        for(int k = 0; k < bsize_xyz[2]; ++k) {
          int indx = k + j * bsize_xyz[2] + i * bsize_xyz[2] * bsize_xyz[1];
          trip[indx].mortid_ =  morton3D_32_encode(i, j, k);
          trip[indx].i_ = indx;
        }
      }
    }
    std::sort(trip, trip + total_bsize, zValueCmp);
    for (int proc = 0; proc < nprocs; ++proc) {
      const int qsize = query_outside[proc].size() / COORD_DIM;
      //std::cout << "------------------ total bin size = " << total_bsize << std::endl;

      // ScalarType time = -MPI_Wtime();
      std::vector<Real> bins_Q[total_bsize];
      std::vector<int> bins_f[total_bsize];
      Real* x_ptr = &query_outside[proc][0];

      for (int i = 0; i < qsize; ++i) {
        const int x = (int) std::abs(std::floor(x_ptr[0] / h0) / bsize);
        const int y = (int) std::abs(std::floor(x_ptr[1] / h1) / bsize);
        const int z = (int) std::abs(std::floor(x_ptr[2] / h2) / bsize);
        const int indx = z + y * bsize_xyz[2] + x * bsize_xyz[2] * bsize_xyz[1];
        bins_Q[indx].push_back(x_ptr[0]);
        bins_Q[indx].push_back(x_ptr[1]);
        bins_Q[indx].push_back(x_ptr[2]);
        bins_f[indx].push_back(f_index[proc][i]);
        x_ptr += 3;
      }


      query_outside[proc].clear();
      f_index[proc].clear();
      for (int i = 0; i < (int)total_bsize; ++i) {
        int bindx = trip[i].i_;
        if(!bins_Q[bindx].empty()){
          query_outside[proc].insert(query_outside[proc].end(), bins_Q[bindx].begin(), bins_Q[bindx].end());
          f_index[proc].insert(f_index[proc].end(), bins_f[bindx].begin(), bins_f[bindx].end());
        }

      }
      // time+=MPI_Wtime();
      // std::cout << "*** time = " << time << std::endl;
    }
      // pvfmm::aligned_delete(bins_Q);
      // pvfmm::aligned_delete(bins_f);
    pvfmm::aligned_delete<zTrip>(trip);
    // delete[] trip;
  #else
    for (int proc = 0; proc < nprocs; ++proc) {
      int qsize = query_outside[proc].size() / COORD_DIM;
      zTrip* trip = new zTrip[qsize];
      std::vector<Real> tmp_query(query_outside[proc]); // tol hold xyz coordinates
      std::vector<int> tmp_f_index(f_index[proc]); // tol hold xyz coordinates

      //ScalarType* x_ptr = &query_outside[proc][i * COORD_DIM + 0];
      Real* x_ptr = &query_outside[proc][0];
      for (int i = 0; i < qsize; ++i) {
        int x = (int) std::abs(std::floor(x_ptr[0] / h0));
        int y = (int) std::abs(std::floor(x_ptr[1] / h1));
        int z = (int) std::abs(std::floor(x_ptr[2] / h2));
        if(0){
          trip[i].mortid_ =  mortonEncode_for(x, y, z);
        }
        else{
          //trip[i].mortid_ =  mortonEncode_LUT(x, y, z);
          trip[i].mortid_ =  morton3D_32_encode(x, y, z);
          //trip[i].mortid_ =  mortonEncode_LUT(x, y, z);
        }
        trip[i].i_ = i;
        x_ptr += 3;
      }

      std::sort(trip, trip + qsize, zValueCmp);

      query_outside[proc].clear();
      f_index[proc].clear();

      for (int i = 0; i < qsize; ++i) {
        // std::cout << "bindx = " << trip[i].i_ << " mid = " << trip[i].mortid_<< std::endl;
        query_outside[proc].push_back(tmp_query[trip[i].i_ * COORD_DIM + 0]);
        query_outside[proc].push_back(tmp_query[trip[i].i_ * COORD_DIM + 1]);
        query_outside[proc].push_back(tmp_query[trip[i].i_ * COORD_DIM + 2]);
        f_index[proc].push_back(tmp_f_index[trip[i].i_]);
      }
      delete[] trip;
    }
  #endif
    return;
  }
  #endif

  /*
   * Phase 1 of the parallel interpolation: This function computes which query_points needs to be sent to
   * other processors and which ones can be interpolated locally. Then a sparse alltoall is performed and
   * all the necessary information is sent/received including the coordinates for the query_points.
   * At the end, each process has the coordinates for interpolation of its own data and those of the others.
   *
   * IMPORTANT: This function must be called just once for a specific query_points. The reason is because of the
   * optimizations performed which assumes that the query_points do not change. For repeated interpolation you should
   * just call this function once, and instead repeatedly call InterpPlan::interpolate function.
   */
  void InterpPlan::fast_scatter(int* N_reg, int * isize, int* istart,
      const int N_pts, const int g_size, Real* query_points_in, int* c_dims,
      MPI_Comm c_comm, double * timings) {
    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);

  #ifdef INTERP_DEBUG
    PCOUT << "entered fast scatter\n";
  #endif
    if (this->allocate_baked == false) {
      std::cout
          << "ERROR InterpPlan Scatter called before calling allocate.\n";
      return;
    }
    if (this->scatter_baked == true) {
      for (int proc = 0; proc < nprocs; ++proc) {
        std::vector<int>().swap(f_index[proc]);
        std::vector<Real>().swap(query_outside[proc]);
      }
    }
    all_query_points_allocation = 0;

    {
      ScalarType time = -MPI_Wtime();

      //int N_reg_g[3], isize_g[3];
      N_reg_g[0] = N_reg[0] + 2 * g_size;
      N_reg_g[1] = N_reg[1] + 2 * g_size;
      N_reg_g[2] = N_reg[2] + 2 * g_size;

      isize_g[0] = isize[0] + 2 * g_size;
      isize_g[1] = isize[1] + 2 * g_size;
      isize_g[2] = isize[2] + 2 * g_size;

      Real h[3]; // original grid size along each axis
      h[0] = 1. / N_reg[0];
      h[1] = 1. / N_reg[1];
      h[2] = 1. / N_reg[2];

      // We copy query_points_in to query_points to aviod overwriting the input coordinates
      memcpy(&query_points[0], query_points_in, N_pts * COORD_DIM * sizeof(Real));
  #ifdef INTERP_DEBUG
    PCOUT << "enforcing periodicity\n";
  #endif
      // Enforce periodicity
  #pragma omp parallel for
      for (int i = 0; i < N_pts; i++) {
        pvfmm::Iterator<Real> Q_ptr = query_points+(i * COORD_DIM);
        //Real* Q_ptr = &query_points[i * COORD_DIM];
        while (Q_ptr[0] <= -h[0]) {
          Q_ptr[0] +=  1;
        }
        while (Q_ptr[1] <= -h[1]) {
          Q_ptr[1] +=  1;
        }
        while (Q_ptr[2] <= -h[2]) {
          Q_ptr[2] +=  1;
        }
        while (Q_ptr[0] >= 1) {
          Q_ptr[0] += - 1;
        }
        while (Q_ptr[1] >= 1) {
          Q_ptr[1] += - 1;
        }
        while (Q_ptr[2] >= 1) {
          Q_ptr[2] += - 1;
        }
      }

      // Compute the start and end coordinates that this processor owns
      Real iX0[3], iX1[3];
      for (int j = 0; j < 3; j++) {
        iX0[j] = istart[j] * h[j];
        iX1[j] = iX0[j] + (isize[j] - 1) * h[j];
      }

      // Now march through the query points and split them into nprocs parts.
      // These are stored in query_outside which is an array of vectors of size nprocs.
      // That is query_outside[i] is a vector that contains the query points that need to
      // be sent to process i. Obviously for the case of query_outside[procid], we do not
      // need to send it to any other processor, as we own the necessary information locally,
      // and interpolation can be done locally.
      int Q_local = 0, Q_outside = 0;

      // This is needed for one-to-one correspondence with output f. This is becaues we are reshuffling
      // the data according to which processor it land onto, and we need to somehow keep the original
      // index to write the interpolation data back to the right location in the output.

      // This is necessary because when we want to compute dproc0 and dproc1 we have to divide by
      // the max isize. If the proc grid is unbalanced, the last proc's isize will be different
      // than others. With this approach we always use the right isize0 for all procs.
      int isize0 = std::ceil(N_reg[0] * 1. / c_dims[0]);
      int isize1 = std::ceil(N_reg[1] * 1. / c_dims[1]);
  #ifdef INTERP_DEBUG
    MPI_Barrier(c_comm);
    PCOUT << "sorting\n";
  #endif

      //int bsize_xyz[COORD_DIM];
      //const int bsize = 16;
      //bsize_xyz[0] = std::ceil(N_reg[0] / (Real)bsize);
      //bsize_xyz[1] = std::ceil(N_reg[1] / (Real)bsize);
      //bsize_xyz[2] = std::ceil(N_reg[2] / (Real)bsize);
      //const size_t total_bsize = bsize_xyz[0] * bsize_xyz[1] * bsize_xyz[2];
      //pvfmm::Iterator<zTrip> trip = pvfmm::aligned_new<zTrip>(total_bsize);
      //pvfmm::Iterator<std::vector<Real>> bins_Q =
      //  pvfmm::aligned_new<std::vector<Real>>(total_bsize*nprocs);
      //pvfmm::Iterator<std::vector<int>> bins_f =
      //  pvfmm::aligned_new<std::vector<int>>(total_bsize*nprocs);
      //std::vector<Real> bins_Q[nprocs][total_bsize];
      //std::vector<int> bins_f[nprocs][total_bsize];
      //for(int i = 0; i < bsize_xyz[0]; ++i) {
      //for(int j = 0; j < bsize_xyz[1]; ++j) {
      //for(int k = 0; k < bsize_xyz[2]; ++k) {
      //  int indx = k + j * bsize_xyz[2] + i * bsize_xyz[2] * bsize_xyz[1];
      //  trip[indx].mortid_ =  morton3D_32_encode(i, j, k);
      //  trip[indx].i_ = indx;
      //}
      //}
      //}
      //std::sort(trip, trip + total_bsize, zValueCmp);

      for (int i = 0; i < N_pts; i++) {
        // Real* Q_ptr = &query_points[i * COORD_DIM];
        pvfmm::Iterator<Real> Q_ptr = query_points+(i * COORD_DIM);
        //const int x = (int) std::abs(std::floor(Q_ptr[0] / h[0]) / bsize);
        //const int y = (int) std::abs(std::floor(Q_ptr[1] / h[1]) / bsize);
        //const int z = (int) std::abs(std::floor(Q_ptr[2] / h[2]) / bsize);
        //const int indx = z + y * bsize_xyz[2] + x * bsize_xyz[2] * bsize_xyz[1];
        // The if condition checks whether the query points fall into the locally owned domain or not
        //if (iX0[0] - h[0] <= Q_ptr[0]
        //    && Q_ptr[0] <= iX1[0] + h[0]
        //    && iX0[1] - h[1] <= Q_ptr[1]
        //    && Q_ptr[1] <= iX1[1] + h[1]
        //    && iX0[2] - h[2] <= Q_ptr[2]
        //    && Q_ptr[2] <= iX1[2] + h[2]) {
        if (iX0[0] - h[0] > Q_ptr[0]
            || Q_ptr[0] > iX1[0] + h[0]
            || iX0[1] - h[1] > Q_ptr[1]
            || Q_ptr[1] > iX1[1] + h[1]
            || iX0[2] - h[2] > Q_ptr[2]
            || Q_ptr[2] > iX1[2] + h[2]) {
          // todo create a set for procs_i_communicate
          // If the point does not reside in the processor's domain then we have to
          // first compute which processor owns the point. After computing that
          // we add the query point to the corresponding vector.
          int dproc0 = (int) (Q_ptr[0] / h[0]) / isize0;
          int dproc1 = (int) (Q_ptr[1] / h[1]) / isize1;
          int proc = dproc0 * c_dims[1] + dproc1; // Compute which proc has to do the interpolation
          //PCOUT<<"proc="<<proc<<std::endl;
          query_outside[proc].push_back(Q_ptr[0]);
          query_outside[proc].push_back(Q_ptr[1]);
          query_outside[proc].push_back(Q_ptr[2]);
          f_index[proc].push_back(i);

          //bins_Q[proc*total_bsize+indx].push_back(Q_ptr[0]);
          //bins_Q[proc*total_bsize+indx].push_back(Q_ptr[1]);
          //bins_Q[proc*total_bsize+indx].push_back(Q_ptr[2]);
          //bins_f[proc*total_bsize+indx].push_back(i);
          // bins_Q[proc][indx].push_back(Q_ptr[0]);
          // bins_Q[proc][indx].push_back(Q_ptr[1]);
          // bins_Q[proc][indx].push_back(Q_ptr[2]);
          // bins_f[proc][indx].push_back(i);
          ++Q_outside;
          //PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
          continue;
        } else {
          query_outside[procid].push_back(Q_ptr[0]);
          query_outside[procid].push_back(Q_ptr[1]);
          query_outside[procid].push_back(Q_ptr[2]);
          f_index[procid].push_back(i);

          //bins_Q[procid*total_bsize+indx].push_back(Q_ptr[0]);
          //bins_Q[procid*total_bsize+indx].push_back(Q_ptr[1]);
          //bins_Q[procid*total_bsize+indx].push_back(Q_ptr[2]);
          //bins_f[procid*total_bsize+indx].push_back(i);
          // bins_Q[procid][indx].push_back(Q_ptr[0]);
          // bins_Q[procid][indx].push_back(Q_ptr[1]);
          // bins_Q[procid][indx].push_back(Q_ptr[2]);
          // bins_f[procid][indx].push_back(i);

          ++Q_local;
          //PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
          continue;
        }

      }

      //for(int proc = 0; proc < nprocs; ++proc){
      //  query_outside[proc].clear();
      //  f_index[proc].clear();
      //  for (int i = 0; i < total_bsize; ++i) {
      //    int bindx = trip[i].i_;
      //    //if(!bins_Q[proc][bindx].empty()){
      //    if(!bins_Q[proc*total_bsize+bindx].empty()){
      //      query_outside[proc].insert(query_outside[proc].end(),
      //          bins_Q[proc*total_bsize+bindx].begin(),
      //          bins_Q[proc*total_bsize+bindx].end());
      //      f_index[proc].insert(f_index[proc].end(),
      //          bins_f[proc*total_bsize+bindx].begin(),
      //          bins_f[proc*total_bsize+bindx].end());
      //    }

      //  }
      //}
      // pvfmm::aligned_delete<zTrip>(trip);
      // pvfmm::aligned_delete(bins_Q);
      // pvfmm::aligned_delete(bins_f);
      //pvfmm::aligned_delete<Real>(query_points);
      // Now sort the query points in zyx order
  #ifdef SORT_QUERIES
      timings[3]+=-MPI_Wtime();
      zsort_queries(query_outside,f_index,N_reg,h,c_comm);
      timings[3]+=+MPI_Wtime();
      //if(procid==0) std::cout<<"Sorting time="<<s_time<<std::endl;;
      //if(procid==0) std::cout<<"Sorting Queries\n";
  #endif

      // Now we need to send the query_points that land onto other processor's domain.
      // This is done using a sparse alltoallv.
      // Right now each process knows how much data to send to others, but does not know
      // how much data it should receive. This is a necessary information both for the MPI
      // command as well as memory allocation for received data.
      // So we first do an alltoall to get the f_index[proc].size from all processes.

  #ifdef INTERP_DEBUG
    PCOUT << "Communicating sizes\n";
  #endif
      for (int proc = 0; proc < nprocs; proc++) {
        if (!f_index[proc].empty()){
          f_index_procs_self_sizes[proc] = f_index[proc].size();
        }
        else
          f_index_procs_self_sizes[proc] = 0;
      }
      timings[0] += -MPI_Wtime();
      MPI_Alltoall(&f_index_procs_self_sizes[0], 1, MPI_INT,
          &f_index_procs_others_sizes[0], 1, MPI_INT, c_comm);
      timings[0] += +MPI_Wtime();

      for (int proc = 0; proc < nprocs; proc++) {
        if (f_index_procs_self_sizes[proc] > 0){
          procs_i_send_to_.push_back(proc);
        }
        if (f_index_procs_others_sizes[proc] > 0 ){
          procs_i_recv_from_.push_back(proc);
        }
      }
      if(!procs_i_recv_from_.empty())
        procs_i_recv_from_size_ = procs_i_recv_from_.size();
      else
        procs_i_recv_from_size_ = 0;
      if(!procs_i_send_to_.empty())
        procs_i_send_to_size_ = procs_i_send_to_.size();
      else
        procs_i_send_to_size_ = 0;

      // Now we need to allocate memory for the receiving buffer of all query
      // points including ours. This is simply done by looping through
      // f_index_procs_others_sizes and adding up all the sizes.
      // Note that we would also need to know the offsets.
      f_index_procs_others_offset[0] = 0;
      f_index_procs_self_offset[0] = 0;
      for (int proc = 0; proc < nprocs; ++proc) {
        // The reason we multiply by COORD_DIM is that we have three coordinates per interpolation request
        all_query_points_allocation += f_index_procs_others_sizes[proc]
            * COORD_DIM;
        if(proc >0){
          f_index_procs_others_offset[proc] =
              f_index_procs_others_offset[proc - 1]
                  + f_index_procs_others_sizes[proc - 1];
          f_index_procs_self_offset[proc] = f_index_procs_self_offset[proc
              - 1] + f_index_procs_self_sizes[proc - 1];
      }
      }
      total_query_points = all_query_points_allocation / COORD_DIM;

      // This if condition is to allow multiple calls to scatter fucntion with different query points
      // without having to create a new plan
      if (this->scatter_baked == true) {
        pvfmm::aligned_delete<Real>(this->all_query_points);
        pvfmm::aligned_delete<Real>(this->all_f_cubic);
      }
  #ifdef INTERP_USE_MORE_MEM_L1
      all_query_points = pvfmm::aligned_new<Real>(
          (total_query_points+16)*(COORD_DIM+1)); // 16 for blocking in interp
  #else
      all_query_points = pvfmm::aligned_new<Real>(
          (total_query_points+16)*(COORD_DIM)); // 16 for blocking in interp
  #endif
      all_f_cubic = pvfmm::aligned_new<Real>(
          total_query_points * data_dof_max + (16*isize_g[2]*isize_g[1]+16*isize_g[1]+16));

  #ifdef INTERP_DEBUG
      //parallel_print(total_query_points, "total_q_points");
      //parallel_print(procs_i_send_to_size_, "procs_i_send_to_size_");
      //parallel_print(procs_i_recv_from_size_, "procs_i_recv_from_size_");
    PCOUT << "communicating query points\n";
  #endif
      // Now perform the allotall to send/recv query_points
      timings[0] += -MPI_Wtime();
      {
        int dst_r, dst_s;
        for (int i = 0; i < procs_i_recv_from_size_; ++i) {
          dst_r = procs_i_recv_from_[i];    //(procid+i)%nprocs;
          request[dst_r] = MPI_REQUEST_NULL;
          int roffset = f_index_procs_others_offset[dst_r] * COORD_DIM; // notice that COORD_DIM is needed because query_points are 3 times f
          MPI_Irecv(&all_query_points[roffset],
              f_index_procs_others_sizes[dst_r] * COORD_DIM,
              MPI_T, dst_r, 0, c_comm, &request[dst_r]);
        }
        for (int i = 0; i < procs_i_send_to_size_; ++i) {
          dst_s = procs_i_send_to_[i];    //(procid-i+nprocs)%nprocs;
          s_request[dst_s] = MPI_REQUEST_NULL;
          //int soffset = f_index_procs_self_offset[dst_s] * COORD_DIM;
          MPI_Isend(&query_outside[dst_s][0],
              f_index_procs_self_sizes[dst_s] * COORD_DIM, MPI_T,
              dst_s, 0, c_comm, &s_request[dst_s]);
        }
        for (int i = 0; i < procs_i_recv_from_size_; ++i) {
          int proc = procs_i_recv_from_[i];    //(procid+i)%nprocs;
          if (request[proc] != MPI_REQUEST_NULL)
            MPI_Wait(&request[proc], MPI_STATUS_IGNORE);
       }
        for (int i = 0; i < procs_i_send_to_size_; ++i) {
          int proc = procs_i_send_to_[i];    //(procid+i)%nprocs;
          if (s_request[proc] != MPI_REQUEST_NULL)
            MPI_Wait(&s_request[proc], MPI_STATUS_IGNORE);
       }
        //for (int i = 0; i < nprocs; ++i) {
        //  dst_r = i;    //(procid+i)%nprocs;
        //  dst_s = i;    //(procid-i+nprocs)%nprocs;
        //  s_request[dst_s] = MPI_REQUEST_NULL;
        //  request[dst_r] = MPI_REQUEST_NULL;
        //  int roffset = f_index_procs_others_offset[dst_r] * COORD_DIM; // notice that COORD_DIM is needed because query_points are 3 times f
        //  int soffset = f_index_procs_self_offset[dst_s] * COORD_DIM;
        //  if (f_index_procs_others_sizes[dst_r] != 0)
        //    MPI_Irecv(&all_query_points[roffset],
        //        f_index_procs_others_sizes[dst_r] * COORD_DIM,
        //        MPI_T, dst_r, 0, c_comm, &request[dst_r]);
        //  if (!query_outside[dst_s].empty())
        //    MPI_Isend(&query_outside[dst_s][0],
        //        f_index_procs_self_sizes[dst_s] * COORD_DIM, MPI_T,
        //        dst_s, 0, c_comm, &s_request[dst_s]);
        //}
        //// Wait for all the communication to finish
        //MPI_Status ierr;
        //for (int proc = 0; proc < nprocs; ++proc) {
        //  if (request[proc] != MPI_REQUEST_NULL)
        //    MPI_Wait(&request[proc], &ierr);
        //  if (s_request[proc] != MPI_REQUEST_NULL)
        //    MPI_Wait(&s_request[proc], &ierr);
        //}
      }
      timings[0] += +MPI_Wtime();
      time+=MPI_Wtime();
      //std::cout << "**** time = " << time << std::endl;

    }

  #ifdef INTERP_DEBUG
    PCOUT << "done with comm\n";
  #endif
    // ParLOG << "nplans_ = " << nplans_ << " data_dof_max = " << data_dof_max << std::endl;
    // ParLOG << "data_dofs[0] = " << data_dofs_[0] << " [1] = " << data_dofs_[1] << std::endl;
    for(int ver = 0; ver < nplans_; ++ver){
    for (int i = 0; i < nprocs; ++i) {
      MPI_Type_vector(data_dofs_[ver], f_index_procs_self_sizes[i], N_pts, MPI_T,
          &rtypes[i+ver*nprocs]);
      MPI_Type_vector(data_dofs_[ver], f_index_procs_others_sizes[i],
          total_query_points, MPI_T, &stypes[i+ver*nprocs]);
      MPI_Type_commit(&stypes[i+ver*nprocs]);
      MPI_Type_commit(&rtypes[i+ver*nprocs]);
    }
    }
  #ifdef INTERP_USE_MORE_MEM_L1
    if(total_query_points !=0)
    rescale_xyzgrid(g_size, N_reg, N_reg_g, istart, isize, isize_g, total_query_points,
        all_query_points);
  #else
    if(total_query_points !=0)
    rescale_xyz(g_size, N_reg, N_reg_g, istart, isize, isize_g, total_query_points,
        &all_query_points[0]);
  #endif

      if (procs_i_recv_from_.size() != 0) procs_i_recv_from_.clear();
      if (procs_i_send_to_.size() != 0) procs_i_send_to_.clear();

    this->scatter_baked = true;
  #ifdef INTERP_DEBUG
    PCOUT << "scatter DONE\n";
  #endif
    return;
  }


  /*
   * Phase 2 of the parallel interpolation: This function must be called after the scatter function is called.
   * It performs local interpolation for all the points that the processor has for itself, as well as the interpolations
   * that it has to send to other processors. After the local interpolation is performed, a sparse
   * alltoall is performed so that all the interpolated results are sent/received.
   * version is a number between zero and nplans_ specifying which data_dof to use
   *
   */
  void InterpPlan::interpolate(Real* __restrict ghost_reg_grid_vals,
      int*__restrict N_reg, int *__restrict isize, int*__restrict istart, const int N_pts, const int g_size,
      Real*__restrict query_values, int*__restrict c_dims, MPI_Comm c_comm, double *__restrict timings, int version) {
    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);
  #ifdef INTERP_DEBUG
    PCOUT << "entered interpolate\n";
  #endif
    if (this->allocate_baked == false) {
      std::cout
          << "ERROR InterpPlan interpolate called before calling allocate.\n";
      return;
    }
    if (this->scatter_baked == false) {
      std::cout
          << "ERROR InterpPlan interpolate called before calling scatter.\n";
      return;
    }

    timings[1] += -MPI_Wtime();
  #ifdef FAST_INTERP
  #ifdef FAST_INTERPV
    const int N_reg3 = isize_g[0] * isize_g[1] * isize_g[2];
    const int* N_reg_c = N_reg;
    const int* N_reg_g_c = N_reg_g;
    const int* istart_c = istart;
    const int* isize_g_c = isize_g;
    const int total_query_points_c = total_query_points;
    if(total_query_points!=0)
      for (int k = 0; k < data_dofs_[version]; ++k){
        vectorized_interp3_ghost_xyz_p(&ghost_reg_grid_vals[k*N_reg3], 1, N_reg_c, N_reg_g_c, isize_g_c,
          istart_c, total_query_points_c, g_size, &all_query_points[0], &all_f_cubic[k*total_query_points],
          true);
        //std::cout << "data_dofs_[version ] = " << data_dofs_[version] << std::endl;
        //do{}while(1);
      }
  #else
    const int N_reg3 = isize_g[0] * isize_g[1] * isize_g[2];
    if(total_query_points!=0)
      for (int k = 0; k < data_dofs_[version]; ++k)
      optimized_interp3_ghost_xyz_p(&ghost_reg_grid_vals[k*N_reg3], 1, N_reg, N_reg_g, isize_g,
        istart, total_query_points, g_size, &all_query_points[0], &all_f_cubic[k*total_query_points],
        true);
  #endif
  #else
    if(total_query_points!=0)
     interp3_ghost_xyz_p(ghost_reg_grid_vals, data_dofs_[version], N_reg, N_reg_g, isize_g,
        istart, total_query_points, g_size, &all_query_points[0], &all_f_cubic[0],
        true);
  #endif
    timings[1] += +MPI_Wtime();

    // Now we have to do an alltoall to distribute the interpolated data from all_f_cubic to
    // f_cubic_unordered.
  #ifdef INTERP_DEBUG
    PCOUT << "finished interpolation, starting comm\n";
  #endif
    ScalarType shuffle_time =0;
    timings[0] += -MPI_Wtime();
    {
      int dst_r, dst_s;

      //for (int i = 0; i < procs_i_send_to_size_; ++i) {
      for (int i = procs_i_send_to_size_-1; i >=0; --i) {
        //dst_r = (procid+i)%nprocs;
        dst_r = procs_i_send_to_[i];    //(procid-i+nprocs)%nprocs;
        request[dst_r] = MPI_REQUEST_NULL; //recv
        int roffset = f_index_procs_self_offset[dst_r];

        MPI_Irecv(&f_cubic_unordered[roffset], 1, rtypes[dst_r+version*nprocs], dst_r, 0,
            c_comm, &request[dst_r]);
      }
      for (int i = 0; i < procs_i_recv_from_size_; ++i) {
        // dst_s = (procid-i+nprocs)%nprocs;
        dst_s = procs_i_recv_from_[i];    //(procid+i)%nprocs;
        s_request[dst_s] = MPI_REQUEST_NULL; //send
        int soffset = f_index_procs_others_offset[dst_s];

        MPI_Isend(&all_f_cubic[soffset], 1, stypes[dst_s+version*nprocs], dst_s, 0, c_comm,
            &s_request[dst_s]);
      }
      // wait to receive your part
        for (int i = 0; i < procs_i_send_to_size_; ++i) {
          int proc = procs_i_send_to_[i];    //(procid+i)%nprocs;
          if (request[proc] != MPI_REQUEST_NULL)
            MPI_Wait(&request[proc], MPI_STATUS_IGNORE);
            shuffle_time += -MPI_Wtime();
            for (int dof = 0; dof < data_dofs_[version]; ++dof) {
              Real* ptr = &f_cubic_unordered[f_index_procs_self_offset[proc]+dof*N_pts];
  #pragma omp parallel for
                  for (int i = 0; i < (int)f_index[proc].size(); ++i) {
                    int ind = f_index[proc][i];
                    query_values[ind + dof * N_pts] =ptr[i];
                  }
            }
            shuffle_time += +MPI_Wtime();
       }
     // wait for send
        for (int i = 0; i < procs_i_recv_from_size_; ++i) {
          int proc = procs_i_recv_from_[i];    //(procid+i)%nprocs;
          if (s_request[proc] != MPI_REQUEST_NULL)
            MPI_Wait(&s_request[proc], MPI_STATUS_IGNORE);
       }

      // for (int i = 0; i < nprocs; ++i) {
      //  dst_r = (procid+i)%nprocs;
      //  dst_s = (procid-i+nprocs)%nprocs;
      //  // s_request[dst_s] = MPI_REQUEST_NULL;
      //  request[2*dst_s] = MPI_REQUEST_NULL; //send
      //  request[2*dst_r+1] = MPI_REQUEST_NULL; //recv
      //  // Notice that this is the adjoint of the first comm part
      //  // because now you are sending others f and receiving your part of f
      //  int soffset = f_index_procs_others_offset[dst_s];
      //  int roffset = f_index_procs_self_offset[dst_r];

      //  if (f_index_procs_self_sizes[dst_r] != 0)
      //    MPI_Irecv(&f_cubic_unordered[roffset], 1, rtype[dst_r], dst_r, 0,
      //        c_comm, &request[2*dst_r+1]);
      //  if (f_index_procs_others_sizes[dst_s] != 0)
      //    MPI_Isend(&all_f_cubic[soffset], 1, stype[dst_s], dst_s, 0, c_comm,
      //        &request[2*dst_s]);
      // }
      // MPI_Waitall(2*nprocs, request, MPI_STATUSES_IGNORE);
    }
    timings[0] += +MPI_Wtime();
    timings[0] -= shuffle_time;

    // Now copy back f_cubic_unordered to f_cubic in the correct f_index
    //for (int dof = 0; dof < data_dof; ++dof) {
    //  for (int proc = 0; proc < nprocs; ++proc) {
    //    if (!f_index[proc].empty())
    //      for (int i = 0; i < f_index[proc].size(); ++i) {
    //        int ind = f_index[proc][i];
    //        //f_cubic[ind]=all_f_cubic[f_index_procs_others_offset[proc]+i];
    //        query_values[ind + dof * N_pts] =
    //            f_cubic_unordered[f_index_procs_self_offset[proc]
    //                + i + dof * N_pts];
    //      }
    //  }
    //}

  #ifdef INTERP_DEBUG
    PCOUT << "interpolation done\n";
  #endif
    return;
  }

  InterpPlan::~InterpPlan() {
    int nprocs, procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (this->allocate_baked) {
      //if(this->scatter_baked == false)
        pvfmm::aligned_delete<Real>(query_points);

      pvfmm::aligned_delete<int>(f_index_procs_others_offset);
      pvfmm::aligned_delete<int>(f_index_procs_self_offset);
      pvfmm::aligned_delete<int>(f_index_procs_self_sizes);
      pvfmm::aligned_delete<int>(f_index_procs_others_sizes);

      pvfmm::aligned_delete<MPI_Request>(s_request);
      pvfmm::aligned_delete<MPI_Request>(request);
      //vectors
      for (int proc = 0; proc < nprocs; ++proc) {
        std::vector<int>().swap(f_index[proc]);
        std::vector<Real>().swap(query_outside[proc]);
      }
      pvfmm::aligned_delete<Real>(f_cubic_unordered);

    }

    if (this->scatter_baked) {
      for (int ver = 0; ver < nplans_; ++ver)
      for (int i = 0; i < nprocs; ++i) {
        MPI_Type_free(&stypes[i+ver*nprocs]);
        MPI_Type_free(&rtypes[i+ver*nprocs]);
      }
      pvfmm::aligned_delete<Real>(all_query_points);
      pvfmm::aligned_delete<Real>(all_f_cubic);
    }

    if (this->allocate_baked) {
      pvfmm::aligned_delete<MPI_Datatype>(rtypes);
      pvfmm::aligned_delete<MPI_Datatype>(stypes);
      pvfmm::aligned_delete<int>(data_dofs_);
    }

    return;
  }

  //void InterpPlan::high_order_interpolate(Real* ghost_reg_grid_vals, int data_dof,
  //    int* N_reg, int * isize, int* istart, const int N_pts, const int g_size,
  //    Real* query_values, int* c_dims, MPI_Comm c_comm, double * timings, int interp_order) {
  //  int nprocs, procid;
  //  MPI_Comm_rank(c_comm, &procid);
  //  MPI_Comm_size(c_comm, &nprocs);
  //  if (this->allocate_baked == false) {
  //    std::cout
  //        << "ERROR InterpPlan interpolate called before calling allocate.\n";
  //    return;
  //  }
  //  if (this->scatter_baked == false) {
  //    std::cout
  //        << "ERROR InterpPlan interpolate called before calling scatter.\n";
  //    return;
  //  }
  //
  //  timings[1] += -MPI_Wtime();
  //  interp3_ghost_xyz_p(ghost_reg_grid_vals, data_dof, N_reg, N_reg_g, isize_g,
  //      istart, total_query_points, g_size, all_query_points, all_f_cubic, interp_order,
  //      true);
  //  timings[1] += +MPI_Wtime();
  //
  //  // Now we have to do an alltoall to distribute the interpolated data from all_f_cubic to
  //  // f_cubic_unordered.
  //  timings[0] += -MPI_Wtime();
  //  {
  //    int dst_r, dst_s;
  //    for (int i = 0; i < nprocs; ++i) {
  //      dst_r = i;  //(procid+i)%nprocs;
  //      dst_s = i;  //(procid-i+nprocs)%nprocs;
  //      s_request[dst_s] = MPI_REQUEST_NULL;
  //      request[dst_r] = MPI_REQUEST_NULL;
  //      // Notice that this is the adjoint of the first comm part
  //      // because now you are sending others f and receiving your part of f
  //      int soffset = f_index_procs_others_offset[dst_r];
  //      int roffset = f_index_procs_self_offset[dst_s];
  //
  //      if (f_index_procs_self_sizes[dst_r] != 0)
  //        MPI_Irecv(&f_cubic_unordered[roffset], 1, rtype[i], dst_r, 0,
  //            c_comm, &request[dst_r]);
  //      if (f_index_procs_others_sizes[dst_s] != 0)
  //        MPI_Isend(&all_f_cubic[soffset], 1, stype[i], dst_s, 0, c_comm,
  //            &s_request[dst_s]);
  //    }
  //    MPI_Status ierr;
  //    for (int proc = 0; proc < nprocs; ++proc) {
  //      if (request[proc] != MPI_REQUEST_NULL)
  //        MPI_Wait(&request[proc], &ierr);
  //      if (s_request[proc] != MPI_REQUEST_NULL)
  //        MPI_Wait(&s_request[proc], &ierr);
  //    }
  //  }
  //  timings[0] += +MPI_Wtime();
  //
  //  // Now copy back f_cubic_unordered to f_cubic in the correct f_index
  //  for (int dof = 0; dof < data_dof; ++dof) {
  //    for (int proc = 0; proc < nprocs; ++proc) {
  //      if (!f_index[proc].empty())
  //        for (int i = 0; i < f_index[proc].size(); ++i) {
  //          int ind = f_index[proc][i];
  //          //f_cubic[ind]=all_f_cubic[f_index_procs_others_offset[proc]+i];
  //          query_values[ind + dof * N_pts] =
  //              f_cubic_unordered[f_index_procs_self_offset[proc]
  //                  + i + dof * N_pts];
  //        }
  //    }
  //  }
  //
  //  return;
  //}
  //
  //
  //
  //
  void InterpPlan::scatter(int* N_reg, int * isize, int* istart,
      const int N_pts, const int g_size, Real* query_points_in, int* c_dims,
      MPI_Comm c_comm, double * timings) {
  #ifdef FAST_INTERP
    return fast_scatter(N_reg, isize, istart, N_pts,
        g_size, query_points_in, c_dims, c_comm, timings);
  #else
    std::cout << "SCATTER CALLED!\n" << std::endl;
    int nprocs, procid;
    MPI_Comm_rank(c_comm, &procid);
    MPI_Comm_size(c_comm, &nprocs);

    if (this->allocate_baked == false) {
      std::cout
          << "ERROR InterpPlan Scatter called before calling allocate.\n";
      return;
    }
    if (this->scatter_baked == true) {
      for (int proc = 0; proc < nprocs; ++proc) {
        std::vector<int>().swap(f_index[proc]);
        std::vector<Real>().swap(query_outside[proc]);
      }
    }
    all_query_points_allocation = 0;

    {

      //int N_reg_g[3], isize_g[3];
      N_reg_g[0] = N_reg[0] + 2 * g_size;
      N_reg_g[1] = N_reg[1] + 2 * g_size;
      N_reg_g[2] = N_reg[2] + 2 * g_size;

      isize_g[0] = isize[0] + 2 * g_size;
      isize_g[1] = isize[1] + 2 * g_size;
      isize_g[2] = isize[2] + 2 * g_size;

      Real h[3]; // original grid size along each axis
      h[0] = 1. / N_reg[0];
      h[1] = 1. / N_reg[1];
      h[2] = 1. / N_reg[2];

      // We copy query_points_in to query_points to aviod overwriting the input coordinates
      memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
      // Enforce periodicity
      for (int i = 0; i < N_pts; i++) {
        while (query_points[i * COORD_DIM + 0] <= -h[0]) {
          query_points[i * COORD_DIM + 0] =
              query_points[i * COORD_DIM + 0] + 1;
        }
        while (query_points[i * COORD_DIM + 1] <= -h[1]) {
          query_points[i * COORD_DIM + 1] =
              query_points[i * COORD_DIM + 1] + 1;
        }
        while (query_points[i * COORD_DIM + 2] <= -h[2]) {
          query_points[i * COORD_DIM + 2] =
              query_points[i * COORD_DIM + 2] + 1;
        }

        while (query_points[i * COORD_DIM + 0] >= 1) {
          query_points[i * COORD_DIM + 0] =
              query_points[i * COORD_DIM + 0] - 1;
        }
        while (query_points[i * COORD_DIM + 1] >= 1) {
          query_points[i * COORD_DIM + 1] =
              query_points[i * COORD_DIM + 1] - 1;
        }
        while (query_points[i * COORD_DIM + 2] >= 1) {
          query_points[i * COORD_DIM + 2] =
              query_points[i * COORD_DIM + 2] - 1;
        }
      }

      // Compute the start and end coordinates that this processor owns
      Real iX0[3], iX1[3];
      for (int j = 0; j < 3; j++) {
        iX0[j] = istart[j] * h[j];
        iX1[j] = iX0[j] + (isize[j] - 1) * h[j];
      }

      // Now march through the query points and split them into nprocs parts.
      // These are stored in query_outside which is an array of vectors of size nprocs.
      // That is query_outside[i] is a vector that contains the query points that need to
      // be sent to process i. Obviously for the case of query_outside[procid], we do not
      // need to send it to any other processor, as we own the necessary information locally,
      // and interpolation can be done locally.
      int Q_local = 0, Q_outside = 0;

      // This is needed for one-to-one correspondence with output f. This is becaues we are reshuffling
      // the data according to which processor it land onto, and we need to somehow keep the original
      // index to write the interpolation data back to the right location in the output.

      // This is necessary because when we want to compute dproc0 and dproc1 we have to divide by
      // the max isize. If the proc grid is unbalanced, the last proc's isize will be different
      // than others. With this approach we always use the right isize0 for all procs.
      int isize0 = std::ceil(N_reg[0] * 1. / c_dims[0]);
      int isize1 = std::ceil(N_reg[1] * 1. / c_dims[1]);
      for (int i = 0; i < N_pts; i++) {
        // The if condition checks whether the query points fall into the locally owned domain or not
        if (iX0[0] - h[0] <= query_points[i * COORD_DIM + 0]
            && query_points[i * COORD_DIM + 0] <= iX1[0] + h[0]
            && iX0[1] - h[1] <= query_points[i * COORD_DIM + 1]
            && query_points[i * COORD_DIM + 1] <= iX1[1] + h[1]
            && iX0[2] - h[2] <= query_points[i * COORD_DIM + 2]
            && query_points[i * COORD_DIM + 2] <= iX1[2] + h[2]) {
          query_outside[procid].push_back(
              query_points[i * COORD_DIM + 0]);
          query_outside[procid].push_back(
              query_points[i * COORD_DIM + 1]);
          query_outside[procid].push_back(
              query_points[i * COORD_DIM + 2]);
          f_index[procid].push_back(i);
          ++Q_local;
          //PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
          continue;
        } else {
          // If the point does not reside in the processor's domain then we have to
          // first compute which processor owns the point. After computing that
          // we add the query point to the corresponding vector.
          int dproc0 = (int) (query_points[i * COORD_DIM + 0] / h[0])
              / isize0;
          int dproc1 = (int) (query_points[i * COORD_DIM + 1] / h[1])
              / isize1;
          int proc = dproc0 * c_dims[1] + dproc1; // Compute which proc has to do the interpolation
          //PCOUT<<"proc="<<proc<<std::endl;
          query_outside[proc].push_back(query_points[i * COORD_DIM + 0]);
          query_outside[proc].push_back(query_points[i * COORD_DIM + 1]);
          query_outside[proc].push_back(query_points[i * COORD_DIM + 2]);
          f_index[proc].push_back(i);
          ++Q_outside;
          //PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
          continue;
        }

      }

      // Now sort the query points in zyx order
  #ifdef SORT_QUERIES
      timings[3]+=-MPI_Wtime();
      sort_queries(query_outside,f_index,N_reg,h,c_comm);
      timings[3]+=+MPI_Wtime();
      //if(procid==0) std::cout<<"Sorting time="<<s_time<<std::endl;;
      //if(procid==0) std::cout<<"Sorting Queries\n";
  #endif

      // Now we need to send the query_points that land onto other processor's domain.
      // This done using a sparse alltoallv.
      // Right now each process knows how much data to send to others, but does not know
      // how much data it should receive. This is a necessary information both for the MPI
      // command as well as memory allocation for received data.
      // So we first do an alltoall to get the f_index[proc].size from all processes.

      for (int proc = 0; proc < nprocs; proc++) {
        if (!f_index[proc].empty())
          f_index_procs_self_sizes[proc] = f_index[proc].size();
        else
          f_index_procs_self_sizes[proc] = 0;
      }
      timings[0] += -MPI_Wtime();
      MPI_Alltoall(f_index_procs_self_sizes, 1, MPI_INT,
          f_index_procs_others_sizes, 1, MPI_INT, c_comm);
      timings[0] += +MPI_Wtime();

      // Now we need to allocate memory for the receiving buffer of all query
      // points including ours. This is simply done by looping through
      // f_index_procs_others_sizes and adding up all the sizes.
      // Note that we would also need to know the offsets.
      f_index_procs_others_offset[0] = 0;
      f_index_procs_self_offset[0] = 0;
      for (int proc = 0; proc < nprocs; ++proc) {
        // The reason we multiply by COORD_DIM is that we have three coordinates per interpolation request
        all_query_points_allocation += f_index_procs_others_sizes[proc]
            * COORD_DIM;
        if (proc > 0) {
          f_index_procs_others_offset[proc] =
              f_index_procs_others_offset[proc - 1]
                  + f_index_procs_others_sizes[proc - 1];
          f_index_procs_self_offset[proc] = f_index_procs_self_offset[proc
              - 1] + f_index_procs_self_sizes[proc - 1];
        }
      }
      total_query_points = all_query_points_allocation / COORD_DIM;

      // This if condition is to allow multiple calls to scatter fucntion with different query points
      // without having to create a new plan
      if (this->scatter_baked == true) {
        pvfmm::aligned_delete<Real>(this->all_query_points);
        pvfmm::aligned_delete<Real>(this->all_f_cubic);
        all_query_points = pvfmm::aligned_new<Real>(
            all_query_points_allocation);
        all_f_cubic = pvfmm::aligned_new<Real>(
            total_query_points * data_dof_max);
      } else {
        all_query_points = pvfmm::aligned_new<Real>(
            all_query_points_allocation);
        all_f_cubic = pvfmm::aligned_new<Real>(
            total_query_points * data_dof_max);
      }

      // Now perform the allotall to send/recv query_points
      timings[0] += -MPI_Wtime();
      {
        int dst_r, dst_s;
        for (int i = 0; i < nprocs; ++i) {
          dst_r = i;    //(procid+i)%nprocs;
          dst_s = i;    //(procid-i+nprocs)%nprocs;
          s_request[dst_s] = MPI_REQUEST_NULL;
          request[dst_r] = MPI_REQUEST_NULL;
          int roffset = f_index_procs_others_offset[dst_r] * COORD_DIM; // notice that COORD_DIM is needed because query_points are 3 times f
          int soffset = f_index_procs_self_offset[dst_s] * COORD_DIM;
          if (f_index_procs_others_sizes[dst_r] != 0)
            MPI_Irecv(&all_query_points[roffset],
                f_index_procs_others_sizes[dst_r] * COORD_DIM,
                MPI_T, dst_r, 0, c_comm, &request[dst_r]);
          if (!query_outside[dst_s].empty())
            MPI_Isend(&query_outside[dst_s][0],
                f_index_procs_self_sizes[dst_s] * COORD_DIM, MPI_T,
                dst_s, 0, c_comm, &s_request[dst_s]);
        }
        // Wait for all the communication to finish
        MPI_Status ierr;
        for (int proc = 0; proc < nprocs; ++proc) {
          if (request[proc] != MPI_REQUEST_NULL)
            MPI_Wait(&request[proc], &ierr);
          if (s_request[proc] != MPI_REQUEST_NULL)
            MPI_Wait(&s_request[proc], &ierr);
        }
      }
      timings[0] += +MPI_Wtime();
    }

    for(int ver = 0; ver < nplans_; ++ver){
    for (int i = 0; i < nprocs; ++i) {
      MPI_Type_vector(data_dofs_[ver], f_index_procs_self_sizes[i], N_pts, MPI_T,
          &rtypes[i+ver*nprocs]);
      MPI_Type_vector(data_dofs_[ver], f_index_procs_others_sizes[i],
          total_query_points, MPI_T, &stypes[i+ver*nprocs]);
      MPI_Type_commit(&stypes[i+ver*nprocs]);
      MPI_Type_commit(&rtypes[i+ver*nprocs]);
    }
    }

    rescale_xyz(g_size, N_reg, N_reg_g, istart, isize, isize_g, total_query_points,
        all_query_points);
    //rescale_xyz(g_size, N_reg, N_reg_g, istart, isize, total_query_points,
    //    all_query_points);
      if(!procs_i_recv_from_.empty()) procs_i_recv_from_.clear();
      if(!procs_i_send_to_.empty()) procs_i_send_to_.clear();
    this->scatter_baked = true;
    return;
  #endif
  } // end scatter


  /*
   * multiply and shift query points based on each proc's domain
   */
  void rescale_xyz(const int g_size, int* N_reg, int* N_reg_g, int* istart,
  		int* isize, int* isize_g, const int N_pts, Real* Q_) {

  	if (g_size == 0)
  		return;
  	Real h[3];

  	h[0] = 1. / (N_reg[0]);
  	h[1] = 1. / (N_reg[1]);
  	h[2] = 1. / (N_reg[2]);

    const Real iX0 = istart[0]*h[0];
    const Real iX1 = istart[1]*h[1];
    const Real iX2 = istart[2]*h[2];
    const Real N0 = N_reg[0];
    const Real N1 = N_reg[1];
    const Real N2 = N_reg[2];

  #pragma omp parallel for
  	for (int i = 0; i < N_pts; i++) {
      Real* Q_ptr = &Q_[COORD_DIM * i];
      Q_ptr[0] = (Q_ptr[0]-iX0)*N0+g_size;
      Q_ptr[1] = (Q_ptr[1]-iX1)*N1+g_size;
      Q_ptr[2] = (Q_ptr[2]-iX2)*N2+g_size;
  	}

    //std::cout <<std::floor(N_pts/16.0)*16<< '\t' <<std::ceil(N_pts/16.0)*16 << std::endl;
    // set Q to be one so that for the peeled loop we only access grid indxx =0
    if(N_pts%16 != 0)
  	for (int i = std::floor(N_pts/16.0)*16+N_pts%16; i < std::ceil(N_pts/16.0)*16; i++) {
      Real* Q_ptr = &Q_[COORD_DIM * i];
      Q_ptr[0] = 1;
      Q_ptr[1] = 1;
      Q_ptr[2] = 1;
  	}
  	return;
  } // end of rescale_xyz

  void rescale_xyzgrid(const int g_size, int* N_reg, int* N_reg_g, int* istart,
  		int* isize, int* isize_g, const int N_pts, pvfmm::Iterator<Real> Q_) {

  	if (g_size == 0)
  		return;
  	Real h[3];

  	h[0] = 1. / (N_reg[0]);
  	h[1] = 1. / (N_reg[1]);
  	h[2] = 1. / (N_reg[2]);

    const Real iX0 = istart[0]*h[0];
    const Real iX1 = istart[1]*h[1];
    const Real iX2 = istart[2]*h[2];
    const Real Nx = N_reg[0];
    const Real Ny = N_reg[1];
    const Real Nz = N_reg[2];
    const int isize_g2 = isize_g[2];
    const int isize_g1g2 = isize_g2 * isize_g[1];
    pvfmm::Iterator<Real> tmp = pvfmm::aligned_new<Real>(N_pts*3);
    pvfmm::memcopy(tmp, Q_, N_pts*3);

  #pragma omp parallel for
  	for (int i = 0; i < N_pts; i++) {
      Real* Q_ptr = &Q_[4 * i];
      Real* tmp_ptr = &tmp[3 * i];
      // std::cout << tmp_ptr[0] << '\t' << tmp_ptr[1] << '\t' << tmp_ptr[2] << std::endl;
      Q_ptr[0] = (tmp_ptr[0]-iX0)*Nx+g_size;
      Q_ptr[1] = (tmp_ptr[1]-iX1)*Ny+g_size;
      Q_ptr[2] = (tmp_ptr[2]-iX2)*Nz+g_size;

  		const int grid_indx0 = ((int)(Q_ptr[0])) - 1;
  		Q_ptr[0] -= grid_indx0;
  		const int grid_indx1 = ((int)(Q_ptr[1])) - 1;
  		Q_ptr[1] -= grid_indx1;
  		const int grid_indx2 = ((int)(Q_ptr[2])) - 1;
  		Q_ptr[2] -= grid_indx2;
  		const int indxx = isize_g1g2 * grid_indx0 + grid_indx2 + isize_g2 * grid_indx1 ;
      Q_ptr[3] = (Real)indxx;
      // std::cout << grid_indx0 << '\t'
      // << grid_indx1 << '\t'
      // << grid_indx2 << std::endl;
  	}
    pvfmm::aligned_delete(tmp);

    //std::cout <<std::floor(N_pts/16.0)*16<< '\t' <<std::ceil(N_pts/16.0)*16 << std::endl;
    // set Q to be one so that for the peeled loop we only access grid indxx =0
    if(N_pts%16 != 0)
  	for (int i = std::floor(N_pts/16.0)*16+N_pts%16; i < std::ceil(N_pts/16.0)*16; i++) {
      Real* Q_ptr = &Q_[4 * i];
      Q_ptr[0] = 1;
      Q_ptr[1] = 1;
      Q_ptr[2] = 1;
      Q_ptr[3] = 0;
  	}
  	return;
  } // end of rescale_xyz
  // acknowledgemet to http://stackoverflow.com/questions/13219146/how-to-sum-m256-horizontally
  // x = ( x7, x6, x5, x4, x3, x2, x1, x0 )
  float sum8(__m256 x) {
      // hiQuad = ( x7, x6, x5, x4 )
      const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
      // loQuad = ( x3, x2, x1, x0 )
      const __m128 loQuad = _mm256_castps256_ps128(x);
      // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
      const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
      // loDual = ( -, -, x1 + x5, x0 + x4 )
      const __m128 loDual = sumQuad;
      // hiDual = ( -, -, x3 + x7, x2 + x6 )
      const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
      // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
      const __m128 sumDual = _mm_add_ps(loDual, hiDual);
      // lo = ( -, -, -, x0 + x2 + x4 + x6 )
      const __m128 lo = sumDual;
      // hi = ( -, -, -, x1 + x3 + x5 + x7 )
      const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
      // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
      const __m128 sum = _mm_add_ss(lo, hi);
      return _mm_cvtss_f32(sum);
  }
  void print128(__m128 x, const char* name) {
    Real* ptr = (Real*)&x;
    std::cout << name
              << " [0] = " << ptr[0]
              << " [1] = " << ptr[1]
              << " [2] = " << ptr[2]
              << " [3] = " << ptr[3] << std::endl;
  }
  void print256(__m256 x, const char* name) {
    Real* ptr = (Real*)&x;
    std::cout << name
              << "\n [0] = " << ptr[0]
              << "\n [1] = " << ptr[1]
              << "\n [2] = " << ptr[2]
              << "\n [3] = " << ptr[3]
              << "\n [4] = " << ptr[4]
              << "\n [5] = " << ptr[5]
              << "\n [6] = " << ptr[6]
              << "\n [7] = " << ptr[7] << std::endl;
  }
  void print512(__m512 &x, const char* name) {
    Real* ptr = (Real*)&x;
    std::cout << name
              << "\n [0] = " << ptr[0]
              << "\n [1] = " << ptr[1]
              << "\n [2] = " << ptr[2]
              << "\n [3] = " << ptr[3]
              << "\n [4] = " << ptr[4]
              << "\n [5] = " << ptr[5]
              << "\n [6] = " << ptr[6]
              << "\n [7] = " << ptr[7]
              << "\n [8] = " << ptr[8]
              << "\n [9] = " << ptr[9]
              << "\n [10] = " << ptr[10]
              << "\n [11] = " << ptr[11]
              << "\n [12] = " << ptr[12]
              << "\n [13] = " << ptr[13]
              << "\n [14] = " << ptr[14]
              << "\n [15] = " << ptr[15] << std::endl;
  }





  #ifdef FAST_INTERPV
  #if defined(KNL)
  void vectorized_interp3_ghost_xyz_p(__restrict Real* reg_grid_vals, int data_dof, const int* __restrict N_reg,
  		const int* __restrict N_reg_g, const int * __restrict isize_g, const int* __restrict istart, const int N_pts,
  		const int g_size, Real* __restrict query_points, Real* __restrict query_values,
  		bool query_values_already_scaled) {

  #ifdef INTERP_DEBUG
  	int nprocs, procid;
  	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    PCOUT << "In KNL kernel\n";
  #endif
    const __m512  c1000_512 = _mm512_broadcast_f32x4(_mm_setr_ps(-1.0,-0.0,-0.0,-0.0));
    const __m512  c2211_512 = _mm512_broadcast_f32x4(_mm_setr_ps(-2.0,-2.0,-1.0,-1.0));
    const __m512  c3332_512 = _mm512_broadcast_f32x4(_mm_setr_ps(-3.0,-3.0,-3.0,-2.0));

    const __m512  c1000_512_ = _mm512_set_ps(
        -1.0,-1.0,-1.0,-1.0,
        -0.0,-0.0,-0.0,-0.0,
        -0.0,-0.0,-0.0,-0.0,
        -0.0,-0.0,-0.0,-0.0
        );
    const __m512  c2211_512_ = _mm512_set_ps(
        -2.0,-2.0,-2.0,-2.0,
        -2.0,-2.0,-2.0,-2.0,
        -1.0,-1.0,-1.0,-1.0,
        -1.0,-1.0,-1.0,-1.0
        );
    const __m512  c3332_512_ = _mm512_set_ps(
        -3.0,-3.0,-3.0,-3.0,
        -3.0,-3.0,-3.0,-3.0,
        -3.0,-3.0,-3.0,-3.0,
        -2.0,-2.0,-2.0,-2.0
        );
    const __m512 vlagr_0000_1111_2222_3333_512  = _mm512_set_ps(
        -0.1666666667,-0.1666666667,-0.1666666667,-0.1666666667,
         0.5,0.5,0.5,0.5,
        -0.5,-0.5,-0.5,-0.5,
        +0.1666666667,0.1666666667,0.1666666667,0.1666666667
        );
    //print512(c1000_512,"512");
    //print512(c3332_512,"");
    //do{}while(1);

    const __m512 vlagr_512 = _mm512_set_ps(
        -0.1666666667,0.5,-0.5, 0.1666666667,-0.1666666667,0.5,-0.5, 0.1666666667,
        -0.1666666667,0.5,-0.5, 0.1666666667,-0.1666666667,0.5,-0.5, 0.1666666667
        );
    const int isize_g2 = isize_g[2];
    const int two_isize_g2 = 2*isize_g2;
    const int three_isize_g2 = 3*isize_g2;
    const int reg_plus = isize_g[1]*isize_g2;
    const int NzNy = isize_g2 * isize_g[1];
    Real* Q_ptr = query_points;


    // std::cout << "KNL" << std::endl;
    //_mm_prefetch( (char*)Q_ptr,_MM_HINT_NTA);



    int CHUNK=16;

  //#pragma omp parallel for
  //	for (int ii = 0; ii < (int)std::ceil(N_pts/(float)CHUNK); ii++) {
  #pragma omp parallel for
  	for (int i = 0; i < N_pts; i++) {
  #ifdef INTERP_USE_MORE_MEM_L1
  		Real point[COORD_DIM];
  		point[0] = Q_ptr[i*4+0];
  		point[1] = Q_ptr[i*4+1];
  		point[2] = Q_ptr[i*4+2];
      const int indxx = (int) Q_ptr[4*i + 3];
  #else
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		point[0] = Q_ptr[i*3+0];
  		grid_indx[0] = ((int)(point[0])) - 1;
  		point[0] -= grid_indx[0];

  		point[1] = Q_ptr[i*3+1];
  		grid_indx[1] = ((int)(point[1])) - 1;
  		point[1] -= grid_indx[1];

  		point[2] = Q_ptr[i*3+2];
  		grid_indx[2] = ((int)(point[2])) - 1;
  		point[2] -= grid_indx[2];
      // Q_ptr += 3;
  		const int indxx = NzNy * grid_indx[0] + grid_indx[2] + isize_g2 * grid_indx[1] ;
  #endif
      //_mm_prefetch( (char*)Q_ptr,_MM_HINT_T2);

      __m512 vM1_0000_1111_2222_3333(vlagr_0000_1111_2222_3333_512);
      __m512 vM2_512(vlagr_512);
      //__m512 vM2_512(vlagr_3333_2222_0000_1111_512);
      __m512 vM0_512(vlagr_512);

      {
      const __m512 vx0_512 =  _mm512_set1_ps(point[0]);
      vM0_512  = _mm512_mul_ps(vM0_512 , _mm512_add_ps(vx0_512,c1000_512));
      vM0_512  = _mm512_mul_ps(vM0_512 , _mm512_add_ps(vx0_512,c2211_512));
      vM0_512  = _mm512_mul_ps(vM0_512 , _mm512_add_ps(vx0_512,c3332_512));

      const __m512 vx1_512 =  _mm512_set1_ps(point[1]);
      vM1_0000_1111_2222_3333  = _mm512_mul_ps(vM1_0000_1111_2222_3333 , _mm512_add_ps(vx1_512,c1000_512_));
      vM1_0000_1111_2222_3333  = _mm512_mul_ps(vM1_0000_1111_2222_3333 , _mm512_add_ps(vx1_512,c2211_512_));
      vM1_0000_1111_2222_3333  = _mm512_mul_ps(vM1_0000_1111_2222_3333 , _mm512_add_ps(vx1_512,c3332_512_));

      //const __m512 vx2_512 =  _mm512_set1_ps(point[2]);
      //vM2_512  = _mm512_mul_ps(vM2_512, _mm512_add_ps(vx2_512,c0010_512));
      //vM2_512  = _mm512_mul_ps(vM2_512, _mm512_add_ps(vx2_512,c1122_512));
      //vM2_512  = _mm512_mul_ps(vM2_512, _mm512_add_ps(vx2_512,c3233_512));
      //vM0_512  = _mm512_mul_ps(vM0_512 , vM2_512);
      const __m512 vx2_512 =  _mm512_set1_ps(point[2]);
      vM2_512  = _mm512_mul_ps(vM2_512 , _mm512_add_ps(vx2_512,c1000_512));
      vM2_512  = _mm512_mul_ps(vM2_512 , _mm512_add_ps(vx2_512,c2211_512));
      vM2_512  = _mm512_mul_ps(vM2_512 , _mm512_add_ps(vx2_512,c3332_512));
      //vM0_512  = _mm512_mul_ps(vM0_512 , vM2_512);
      //print512(vM0_512, "vM0");
      //print512(vM2_512, "vM2");
      //print512(vM2_512, "vM2");
      //print512(vM1_0000_1111_2222_3333  , "vM1");
      //do{}while(1);
      }

      int indx = 0;
      Real* reg_ptr = reg_grid_vals + indxx;//&reg_grid_vals[indxx];
      //_mm_prefetch( (char*)reg_ptr,_MM_HINT_T0);

      // load all vfij

            __m512 vf_i0_j0123 = _mm512_setzero_ps();
            vf_i0_j0123 = _mm512_mask_expandloadu_ps(vf_i0_j0123, 0b1111000000000000, reg_ptr);
            vf_i0_j0123 = _mm512_mask_expandloadu_ps(vf_i0_j0123, 0b0000111100000000, reg_ptr+isize_g2);
            vf_i0_j0123 = _mm512_mask_expandloadu_ps(vf_i0_j0123, 0b0000000011110000, reg_ptr+two_isize_g2);
            vf_i0_j0123 = _mm512_mask_expandloadu_ps(vf_i0_j0123, 0b0000000000001111, reg_ptr+three_isize_g2);
            reg_ptr +=  reg_plus;

            __m512 vf_i1_j0123 = _mm512_setzero_ps();
            vf_i1_j0123 = _mm512_mask_expandloadu_ps(vf_i1_j0123, 0b1111000000000000, reg_ptr);
            vf_i1_j0123 = _mm512_mask_expandloadu_ps(vf_i1_j0123, 0b0000111100000000, reg_ptr+isize_g2);
            vf_i1_j0123 = _mm512_mask_expandloadu_ps(vf_i1_j0123, 0b0000000011110000, reg_ptr+two_isize_g2);
            vf_i1_j0123 = _mm512_mask_expandloadu_ps(vf_i1_j0123, 0b0000000000001111, reg_ptr+three_isize_g2);
            reg_ptr +=  reg_plus;

            __m512 vf_i2_j0123 = _mm512_setzero_ps();
            vf_i2_j0123 = _mm512_mask_expandloadu_ps(vf_i2_j0123, 0b1111000000000000, reg_ptr);
            vf_i2_j0123 = _mm512_mask_expandloadu_ps(vf_i2_j0123, 0b0000111100000000, reg_ptr+isize_g2);
            vf_i2_j0123 = _mm512_mask_expandloadu_ps(vf_i2_j0123, 0b0000000011110000, reg_ptr+two_isize_g2);
            vf_i2_j0123 = _mm512_mask_expandloadu_ps(vf_i2_j0123, 0b0000000000001111, reg_ptr+three_isize_g2);
            reg_ptr +=  reg_plus;

            __m512 vf_i3_j0123 = _mm512_setzero_ps();
            vf_i3_j0123 = _mm512_mask_expandloadu_ps(vf_i3_j0123, 0b1111000000000000, reg_ptr);
            vf_i3_j0123 = _mm512_mask_expandloadu_ps(vf_i3_j0123, 0b0000111100000000, reg_ptr+isize_g2);
            vf_i3_j0123 = _mm512_mask_expandloadu_ps(vf_i3_j0123, 0b0000000011110000, reg_ptr+two_isize_g2);
            vf_i3_j0123 = _mm512_mask_expandloadu_ps(vf_i3_j0123, 0b0000000000001111, reg_ptr+three_isize_g2);
            reg_ptr +=  reg_plus;

            const __m512 vt_i0_512 = _mm512_mul_ps(vM1_0000_1111_2222_3333,vf_i0_j0123);
            const __m512 vt_i1_512 = _mm512_mul_ps(vM1_0000_1111_2222_3333,vf_i1_j0123);
            const __m512 vt_i2_512 = _mm512_mul_ps(vM1_0000_1111_2222_3333,vf_i2_j0123);
            const __m512 vt_i3_512 = _mm512_mul_ps(vM1_0000_1111_2222_3333,vf_i3_j0123);

            __m512 vt0_512 = _mm512_mul_ps(_mm512_permute_ps(vM0_512,0b00000000), vt_i0_512);
            __m512 vt1_512 = _mm512_mul_ps(_mm512_permute_ps(vM0_512,0b01010101), vt_i1_512);
            __m512 vt2_512 = _mm512_mul_ps(_mm512_permute_ps(vM0_512,0b10101010), vt_i2_512);
            __m512 vt3_512 = _mm512_mul_ps(_mm512_permute_ps(vM0_512,0b11111111), vt_i3_512);



             //__m512 vt_512 = vt0_512;
             //vt0_512 = _mm512_add_ps(vt1_512, vt1_512);
             //vt2_512 = _mm512_add_ps(vt2_512, vt3_512);
             //vt_512 = _mm512_add_ps(vt0_512, vt2_512);
             //vt_512 = _mm512_mul_ps(vt_512, vM2_512);

              //__m512 vt_512 = vt0_512;
              //vt_512 = _mm512_add_ps(vt_512, vt1_512);
              //vt_512 = _mm512_add_ps(vt_512, vt2_512);
              //vt_512 = _mm512_add_ps(vt_512, vt3_512);
              //vt_512 = _mm512_mul_ps(vt_512, vM2_512);

             __m512 vt_512;
             vt0_512 = _mm512_add_ps(vt0_512, vt1_512);
             vt2_512 = _mm512_add_ps(vt2_512, vt3_512);
             vt_512 = _mm512_add_ps(vt0_512, vt2_512);
              vt_512 = _mm512_mul_ps(vt_512, vM2_512);

             //val[jj] = _mm512_reduce_add_ps (vt_512);
             query_values[i] = _mm512_reduce_add_ps (vt_512);
  	  } //end jj loop
    //__m512 tmp = _mm512_loadu_ps(val);
    //_mm512_stream_ps (&query_values[ii*CHUNK],tmp);
  	//} //end ii loop

  	return;

  }  // end of interp3_ghost_xyz_p

  void ectorized_interp3_ghost_xyz_p(__restrict Real* reg_grid_vals, int data_dof, const int* __restrict N_reg,
  		const int* __restrict N_reg_g, const int * __restrict isize_g, const int* __restrict istart, const int N_pts,
  		const int g_size, Real* __restrict query_points, Real* __restrict query_values,
  		bool query_values_already_scaled) {

    const __m256  c1000 = _mm256_set_ps(-1.0,-0.0,-0.0,-0.0,-1.0,-0.0,-0.0,-0.0);
    const __m256  c2211 = _mm256_set_ps(-2.0,-2.0,-1.0,-1.0,-2.0,-2.0,-1.0,-1.0);
    const __m256  c3332 = _mm256_set_ps(-3.0,-3.0,-3.0,-2.0,-3.0,-3.0,-3.0,-2.0);
    const __m512  c1000_512 = _mm512_broadcast_f32x4(_mm_setr_ps(-1.0,-0.0,-0.0,-0.0));
    const __m512  c2211_512 = _mm512_broadcast_f32x4(_mm_setr_ps(-2.0,-2.0,-1.0,-1.0));
    const __m512  c3332_512 = _mm512_broadcast_f32x4(_mm_setr_ps(-3.0,-3.0,-3.0,-2.0));

    const __m512  c0010_512 = _mm512_set_ps(
        -0.0,-0.0,-0.0,-0.0,
        -0.0,-0.0,-0.0,-0.0,
        -1.0,-1.0,-1.0,-1.0,
        -0.0,-0.0,-0.0,-0.0
        );
    const __m512  c1122_512 = _mm512_set_ps(
        -1.0,-1.0,-1.0,-1.0,
        -1.0,-1.0,-1.0,-1.0,
        -2.0,-2.0,-2.0,-2.0,
        -2.0,-2.0,-2.0,-2.0
        );
    const __m512  c3233_512 = _mm512_set_ps(
        -3.0,-3.0,-3.0,-3.0,
        -2.0,-2.0,-2.0,-2.0,
        -3.0,-3.0,-3.0,-3.0,
        -3.0,-3.0,-3.0,-3.0
        );
    const __m512 vlagr_3333_2222_0000_1111_512  = _mm512_set_ps(
        -0.5,-0.5,-0.5,-0.5,
        +0.1666666667,0.1666666667,0.1666666667,0.1666666667,
        -0.1666666667,-0.1666666667,-0.1666666667,-0.1666666667,
         0.5,0.5,0.5,0.5
        );
    //print512(c1000_512,"512");
    //print512(c3332_512,"");
    //do{}while(1);

    const __m256 vlagr = _mm256_set_ps(-0.1666666667,0.5,-0.5, 0.1666666667,-0.1666666667,0.5,-0.5, 0.1666666667);
    const __m512 vlagr_512 = _mm512_set_ps(
        -0.1666666667,0.5,-0.5, 0.1666666667,-0.1666666667,0.5,-0.5, 0.1666666667,
        -0.1666666667,0.5,-0.5, 0.1666666667,-0.1666666667,0.5,-0.5, 0.1666666667
        );
    const __m256  c33332222 = _mm256_set_ps(-3.0,-3.0,-3.0,-3.0,-2.0,-2.0,-2.0,-2.0);
    const __m256  c22223333 = _mm256_setr_ps(-3.0,-3.0,-3.0,-3.0,-2.0,-2.0,-2.0,-2.0);
    const __m256  c11110000 = _mm256_set_ps(-1.0,-1.0,-1.0,-1.0,0,0,0,0);
    const __m256  c00001111 = _mm256_setr_ps(-1.0,-1.0,-1.0,-1.0,0,0,0,0);
    const __m256  l0l1 = _mm256_set_ps (-0.1666666667,-0.1666666667,-0.1666666667,-0.1666666667,+0.5,+0.5,+0.5,+0.5);
    const __m256  l2l3 = _mm256_setr_ps(+0.1666666667,+0.1666666667,+0.1666666667,+0.1666666667,-0.5,-0.5,-0.5,-0.5);
    const int isize_g2 = isize_g[2];
    const int two_isize_g2 = 2*isize_g2;
    const int reg_plus = isize_g[1]*isize_g2 - two_isize_g2;
    const int NzNy = isize_g2 * isize_g[1];
    Real* Q_ptr = query_points;


    // std::cout << "KNL" << std::endl;
    //_mm_prefetch( (char*)Q_ptr,_MM_HINT_NTA);
  #pragma omp parallel for
  	for (int i = 0; i < N_pts; i++) {

  #ifdef INTERP_USE_MORE_MEM_L1
  		Real point[COORD_DIM];
  		point[0] = Q_ptr[i*4+0];
  		point[1] = Q_ptr[i*4+1];
  		point[2] = Q_ptr[i*4+2];
      const int indxx = (int) Q_ptr[4*i + 3];
  #else
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		point[0] = Q_ptr[i*3+0];
  		grid_indx[0] = ((int)(point[0])) - 1;
  		point[0] -= grid_indx[0];

  		point[1] = Q_ptr[i*3+1];
  		grid_indx[1] = ((int)(point[1])) - 1;
  		point[1] -= grid_indx[1];

  		point[2] = Q_ptr[i*3+2];
  		grid_indx[2] = ((int)(point[2])) - 1;
  		point[2] -= grid_indx[2];
      // Q_ptr += 3;
  		const int indxx = NzNy * grid_indx[0] + grid_indx[2] + isize_g2 * grid_indx[1] ;
  #endif
      //_mm_prefetch( (char*)Q_ptr,_MM_HINT_T2);

      //__m256 vM0(vlagr), vM1(vlagr), vM2(vlagr);
      __m256 vM0(vlagr), vM2(vlagr);
      // __m256 vM0_tttt[4]; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM1_0000_1111; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM1_2222_3333; // elements will be M2[2] for the first 4 reg and M2[3] for the rest

      __m512 vM1_2222_3333_0000_1111(vlagr_3333_2222_0000_1111_512);
      __m512 vM2_512(vlagr_512);
      __m512 vM0_512(vlagr_512);

      {
      const __m256 vx0 = _mm256_set1_ps(point[0]);
      vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c1000));
      vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c2211));
      vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c3332));

      Real* dum, *dum2;
      const __m512 vx1_512 =  _mm512_set1_ps(point[1]);
      vM1_2222_3333_0000_1111  = _mm512_mul_ps(vM1_2222_3333_0000_1111 , _mm512_add_ps(vx1_512,c0010_512));
      vM1_2222_3333_0000_1111  = _mm512_mul_ps(vM1_2222_3333_0000_1111 , _mm512_add_ps(vx1_512,c1122_512));
      vM1_2222_3333_0000_1111  = _mm512_mul_ps(vM1_2222_3333_0000_1111 , _mm512_add_ps(vx1_512,c3233_512));

      const __m512 vx0_512 =  _mm512_set1_ps(point[0]);
      vM0_512  = _mm512_mul_ps(vM0_512 , _mm512_add_ps(vx0_512,c1000_512));
      vM0_512  = _mm512_mul_ps(vM0_512 , _mm512_add_ps(vx0_512,c2211_512));
      vM0_512  = _mm512_mul_ps(vM0_512 , _mm512_add_ps(vx0_512,c3332_512));

      const __m512 vx2_512 =  _mm512_set1_ps(point[2]);
      vM2_512  = _mm512_mul_ps(vM2_512 , _mm512_add_ps(vx2_512,c1000_512));
      vM2_512  = _mm512_mul_ps(vM2_512 , _mm512_add_ps(vx2_512,c2211_512));
      vM2_512  = _mm512_mul_ps(vM2_512 , _mm512_add_ps(vx2_512,c3332_512));

      const __m256 vx1 = _mm256_set1_ps(point[1]);
      __m256 tmp = _mm256_add_ps(vx1,c33332222); // x-3,...;x-2,...
      tmp = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c11110000));
      vM1_0000_1111 = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c22223333));
      vM1_0000_1111 = _mm256_mul_ps(vM1_0000_1111, l0l1);
      vM1_2222_3333 = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c00001111));
      vM1_2222_3333  = _mm256_mul_ps(vM1_2222_3333, l2l3);

      const __m256 vx2 = _mm256_set1_ps(point[2]);
      vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c1000));
      vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c2211));
      vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c3332));
      // todo remove permute completely by using different c's in the beginning
      vM2 = _mm256_permute_ps(vM2,0b00011011);
      ///print256(vM2,"256");
      ///print512(vM2_512,"256");
      ///print256(vM0,"256");
      ///print512(vM0_512,"256");
      ///do{}while(1);
      }

      int indx = 0;
      Real* reg_ptr = reg_grid_vals + indxx;//&reg_grid_vals[indxx];
      //_mm_prefetch( (char*)reg_ptr,_MM_HINT_T0);
  		Real val = 0;

      // load all vfij
            __m256 vt;
            vt = _mm256_setzero_ps();

            __m512 vf_i0_j0123 = _mm512_setzero_ps();
            vf_i0_j0123 = _mm512_mask_expandloadu_ps(vf_i0_j0123, 0b0000000011110000, reg_ptr);
            vf_i0_j0123 = _mm512_mask_expandloadu_ps(vf_i0_j0123, 0b0000000000001111, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vf_i0_j0123 = _mm512_mask_expandloadu_ps(vf_i0_j0123, 0b1111000000000000, reg_ptr);
            vf_i0_j0123 = _mm512_mask_expandloadu_ps(vf_i0_j0123, 0b0000111100000000, reg_ptr+isize_g2);
             reg_ptr +=  reg_plus;

            __m512 vf_i1_j0123 = _mm512_setzero_ps();
            vf_i1_j0123 = _mm512_mask_expandloadu_ps(vf_i1_j0123, 0b0000000011110000, reg_ptr);
            vf_i1_j0123 = _mm512_mask_expandloadu_ps(vf_i1_j0123, 0b0000000000001111, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vf_i1_j0123 = _mm512_mask_expandloadu_ps(vf_i1_j0123, 0b1111000000000000, reg_ptr);
            vf_i1_j0123 = _mm512_mask_expandloadu_ps(vf_i1_j0123, 0b0000111100000000, reg_ptr+isize_g2);
            reg_ptr +=  reg_plus;

            __m512 vf_i2_j0123 = _mm512_setzero_ps();
            vf_i2_j0123 = _mm512_mask_expandloadu_ps(vf_i2_j0123, 0b0000000011110000, reg_ptr);
            vf_i2_j0123 = _mm512_mask_expandloadu_ps(vf_i2_j0123, 0b0000000000001111, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vf_i2_j0123 = _mm512_mask_expandloadu_ps(vf_i2_j0123, 0b1111000000000000, reg_ptr);
            vf_i2_j0123 = _mm512_mask_expandloadu_ps(vf_i2_j0123, 0b0000111100000000, reg_ptr+isize_g2);
            reg_ptr +=  reg_plus;

            __m512 vf_i3_j0123 = _mm512_setzero_ps();
            vf_i3_j0123 = _mm512_mask_expandloadu_ps(vf_i3_j0123, 0b0000000011110000, reg_ptr);
            vf_i3_j0123 = _mm512_mask_expandloadu_ps(vf_i3_j0123, 0b0000000000001111, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vf_i3_j0123 = _mm512_mask_expandloadu_ps(vf_i3_j0123, 0b1111000000000000, reg_ptr);
            vf_i3_j0123 = _mm512_mask_expandloadu_ps(vf_i3_j0123, 0b0000111100000000, reg_ptr+isize_g2);
            reg_ptr +=  reg_plus;

            reg_ptr = reg_grid_vals + indxx;//&reg_grid_vals[indxx];
            __m256 vf_i0_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            const __m256 vf_i0_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr +=  reg_plus;
            const __m256 vf_i1_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            const __m256 vf_i1_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr +=  reg_plus;

            const __m256 vf_i2_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            const __m256 vf_i2_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr +=  reg_plus;

            const __m256 vf_i3_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            const __m256 vf_i3_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);


            const __m256 vt_i0_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i0_j01);
            const __m256 vt_i0_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i0_j23);
            const __m256 vt_i0 = _mm256_add_ps(vt_i0_j01, vt_i0_j23);
            const __m512 vt_i0_512 = _mm512_mul_ps(vM1_2222_3333_0000_1111,vf_i0_j0123);

            const __m256 vt_i1_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i1_j01);
            const __m256 vt_i1_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i1_j23);
            const __m256 vt_i1 = _mm256_add_ps(vt_i1_j01, vt_i1_j23);
            const __m512 vt_i1_512 = _mm512_mul_ps(vM1_2222_3333_0000_1111,vf_i1_j0123);

            const __m256 vt_i2_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i2_j01);
            const __m256 vt_i2_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i2_j23);
            const __m256 vt_i2 = _mm256_add_ps(vt_i2_j01, vt_i2_j23);
            const __m512 vt_i2_512 = _mm512_mul_ps(vM1_2222_3333_0000_1111,vf_i2_j0123);

            const __m256 vt_i3_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i3_j01);
            const __m256 vt_i3_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i3_j23);
            const __m256 vt_i3 = _mm256_add_ps(vt_i3_j01, vt_i3_j23);
            const __m512 vt_i3_512 = _mm512_mul_ps(vM1_2222_3333_0000_1111,vf_i3_j0123);

            const __m256 vt0 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b11111111), vt_i0);
            const __m256 vt1 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b10101010), vt_i1);
            const __m256 vt2 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b01010101), vt_i2);
            const __m256 vt3 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b00000000), vt_i3);

            const __m512 vt0_512 = _mm512_mul_ps(_mm512_permute_ps(vM0_512,0b00000000), vt_i0_512);
            const __m512 vt1_512 = _mm512_mul_ps(_mm512_permute_ps(vM0_512,0b01010101), vt_i1_512);
            const __m512 vt2_512 = _mm512_mul_ps(_mm512_permute_ps(vM0_512,0b10101010), vt_i2_512);
            const __m512 vt3_512 = _mm512_mul_ps(_mm512_permute_ps(vM0_512,0b11111111), vt_i3_512);


             __m512 vt_512 = vt0_512;
             vt_512 = _mm512_add_ps(vt_512, vt1_512);
             vt_512 = _mm512_add_ps(vt_512, vt2_512);
             vt_512 = _mm512_add_ps(vt_512, vt3_512);
             vt_512 = _mm512_mul_ps(vt_512, vM2_512);


             vt = _mm256_add_ps(vt0, vt1);
             vt = _mm256_add_ps(vt, vt2);
             vt = _mm256_add_ps(vt, vt3);

             vt = _mm256_mul_ps(vM2, vt);
              val = sum8(vt);
  		        query_values[i] = val;

             //print256(vt,"");
             //print512(vt_512,"");
             //do{}while(1);
             val = _mm512_reduce_add_ps (vt_512);
  		       query_values[i] = val;
  	}

  	return;

  }  // end of interp3_ghost_xyz_p

  #elif defined(__AVX2__) || defined(HASWELL)
  void vectorized_interp3_ghost_xyz_p(__restrict Real* reg_grid_vals, int data_dof, const int* __restrict N_reg,
  		const int* __restrict N_reg_g, const int * __restrict isize_g, const int* __restrict istart, const int N_pts,
  		const int g_size, Real* __restrict query_points, Real* __restrict query_values,
  		bool query_values_already_scaled) {

  #ifdef INTERP_DEBUG
  	int nprocs, procid;
  	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    PCOUT << "In Haswel kernel\n";
  #endif
    const __m256  c1000 = _mm256_set_ps(-1.0,-0.0,-0.0,-0.0,-1.0,-0.0,-0.0,-0.0);
    const __m256  c2211 = _mm256_set_ps(-2.0,-2.0,-1.0,-1.0,-2.0,-2.0,-1.0,-1.0);
    const __m256  c3332 = _mm256_set_ps(-3.0,-3.0,-3.0,-2.0,-3.0,-3.0,-3.0,-2.0);

    const __m256 vlagr = _mm256_set_ps(-0.1666666667,0.5,-0.5, 0.1666666667,-0.1666666667,0.5,-0.5, 0.1666666667);
    const __m256  c33332222 = _mm256_set_ps(-3.0,-3.0,-3.0,-3.0,-2.0,-2.0,-2.0,-2.0);
    const __m256  c22223333 = _mm256_setr_ps(-3.0,-3.0,-3.0,-3.0,-2.0,-2.0,-2.0,-2.0);
    const __m256  c11110000 = _mm256_set_ps(-1.0,-1.0,-1.0,-1.0,0,0,0,0);
    const __m256  c00001111 = _mm256_setr_ps(-1.0,-1.0,-1.0,-1.0,0,0,0,0);
    const __m256  l0l1 = _mm256_set_ps (-0.1666666667,-0.1666666667,-0.1666666667,-0.1666666667,+0.5,+0.5,+0.5,+0.5);
    const __m256  l2l3 = _mm256_setr_ps(+0.1666666667,+0.1666666667,+0.1666666667,+0.1666666667,-0.5,-0.5,-0.5,-0.5);
    const int isize_g2 = isize_g[2];
    const int two_isize_g2 = 2*isize_g2;
    const int three_isize_g2 = 3*isize_g2;
    const int reg_plus = isize_g[1]*isize_g2;
    const int NzNy = isize_g2 * isize_g[1];
    Real* Q_ptr = query_points;
    //std::cout << "AVX2" << std::endl;
    //_mm_prefetch( (char*)Q_ptr,_MM_HINT_NTA);

   #pragma omp parallel for
  	 for (int i = 0; i < N_pts; i++) {
  //  int CHUNK=8;
  //#pragma omp parallel for
  //  for (int ii = 0; ii < (int)std::ceil(N_pts/(float)CHUNK); ii++) {
  //		Real val[CHUNK];
  //	for (int jj = 0; jj < CHUNK; jj++) {
  //    int i = ii*CHUNK + jj;
  #ifdef INTERP_USE_MORE_MEM_L1
  		Real point[COORD_DIM];
  		point[0] = Q_ptr[i*4+0];
  		point[1] = Q_ptr[i*4+1];
  		point[2] = Q_ptr[i*4+2];
      const int indxx = (int) Q_ptr[4*i + 3];
  #else
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		point[0] = Q_ptr[i*3+0];
  		grid_indx[0] = ((int)(point[0])) - 1;
  		point[0] -= grid_indx[0];

  		point[1] = Q_ptr[i*3+1];
  		grid_indx[1] = ((int)(point[1])) - 1;
  		point[1] -= grid_indx[1];

  		point[2] = Q_ptr[i*3+2];
  		grid_indx[2] = ((int)(point[2])) - 1;
  		point[2] -= grid_indx[2];
      // Q_ptr += 3;
  		const int indxx = NzNy * grid_indx[0] + grid_indx[2] + isize_g2 * grid_indx[1] ;
  #endif

      ////_mm_prefetch( (char*)Q_ptr,_MM_HINT_T2);
  //    int indx = 0;
      Real* reg_ptr = reg_grid_vals + indxx;//&reg_grid_vals[indxx];
      //_mm_prefetch( (char*)reg_ptr,_MM_HINT_T0);



      //__m256 vM0(vlagr), vM1(vlagr), vM2(vlagr);
      __m256 vM0(vlagr), vM2(vlagr);
      // __m256 vM0_tttt[4]; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM1_0000_1111; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM1_2222_3333; // elements will be M2[2] for the first 4 reg and M2[3] for the rest


      {
      const __m256 vx0 = _mm256_set1_ps(point[0]);
      vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c1000));
      vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c2211));
      vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c3332));

      const __m256 vx1 = _mm256_set1_ps(point[1]);
      __m256 tmp = _mm256_add_ps(vx1,c33332222); // x-3,...;x-2,...
      tmp = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c11110000));
      vM1_0000_1111 = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c22223333));
      vM1_0000_1111 = _mm256_mul_ps(vM1_0000_1111, l0l1);
      vM1_2222_3333 = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c00001111));
      vM1_2222_3333  = _mm256_mul_ps(vM1_2222_3333, l2l3);
      //vM1 = _mm256_mul_ps(vM1, _mm256_add_ps(vx1,c1000));
      //vM1 = _mm256_mul_ps(vM1, _mm256_add_ps(vx1,c2211));
      //vM1 = _mm256_mul_ps(vM1, _mm256_add_ps(vx1,c3332));

      const __m256 vx2 = _mm256_set1_ps(point[2]);
      vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c1000));
      vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c2211));
      vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c3332));
      // todo remove permute completely by using different c's in the beginning
      vM2 = _mm256_permute_ps(vM2,0b00011011);
      //vM2 = _mm256_shuffle_ps(vM2,vM2,_MM_SHUFFLE(0, 1, 2, 3));

      //const Real* M1 = (Real*)&vM1;
      //vM1_0000_1111 = _mm256_set_ps(M1[7],M1[7],M1[7],M1[7],M1[6],M1[6],M1[6],M1[6]);
      //vM1_2222_3333 = _mm256_set_ps(M1[5],M1[5],M1[5],M1[5],M1[4],M1[4],M1[4],M1[4]);
      // vM1_0000_1111 = _mm256_set_ps(M1[3],M1[3],M1[3],M1[3],M1[2],M1[2],M1[2],M1[2]);
      // vM1_2222_3333 = _mm256_set_ps(M1[1],M1[1],M1[1],M1[1],M1[0],M1[0],M1[0],M1[0]);
      //vM0_tttt[0] = _mm256_permute_ps(vM0,0b11111111); // last element
      //vM0_tttt[1] = _mm256_permute_ps(vM0,0b10101010);
      //vM0_tttt[2] = _mm256_permute_ps(vM0,0b01010101);
      //vM0_tttt[3] = _mm256_permute_ps(vM0,0b00000000);
      }


      // load all vfij
            __m256 vt;
            vt = _mm256_setzero_ps();
            const __m256 vf_i0_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            const __m256 vf_i0_j23 = _mm256_loadu2_m128(reg_ptr+two_isize_g2, reg_ptr+three_isize_g2);
            reg_ptr +=  reg_plus;

            const __m256 vf_i1_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            const __m256 vf_i1_j23 = _mm256_loadu2_m128(reg_ptr+two_isize_g2, reg_ptr+three_isize_g2);
            reg_ptr +=  reg_plus;

            const __m256 vf_i2_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            const __m256 vf_i2_j23 = _mm256_loadu2_m128(reg_ptr+two_isize_g2, reg_ptr+three_isize_g2);
            reg_ptr +=  reg_plus;

            const __m256 vf_i3_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            const __m256 vf_i3_j23 = _mm256_loadu2_m128(reg_ptr+two_isize_g2, reg_ptr+three_isize_g2);

            //__m256 vt0, vt1, vt2, vt3;
            //vt0 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b11111111),_mm256_fmadd_ps(vM1_0000_1111, vf_i0_j01, _mm256_mul_ps(vM1_2222_3333, vf_i0_j23)));
            //vt1 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b10101010),_mm256_fmadd_ps(vM1_0000_1111, vf_i1_j01, _mm256_mul_ps(vM1_2222_3333, vf_i1_j23)));
            //vt2 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b01010101),_mm256_fmadd_ps(vM1_0000_1111, vf_i2_j01, _mm256_mul_ps(vM1_2222_3333, vf_i2_j23)));
            //vt3 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b00000000),_mm256_fmadd_ps(vM1_0000_1111, vf_i3_j01, _mm256_mul_ps(vM1_2222_3333, vf_i3_j23)));

            vt = _mm256_fmadd_ps(_mm256_permute_ps(vM0,0b11111111),_mm256_fmadd_ps(vM1_0000_1111, vf_i0_j01, _mm256_mul_ps(vM1_2222_3333, vf_i0_j23)) , vt);
            vt = _mm256_fmadd_ps(_mm256_permute_ps(vM0,0b10101010),_mm256_fmadd_ps(vM1_0000_1111, vf_i1_j01, _mm256_mul_ps(vM1_2222_3333, vf_i1_j23)) , vt);
            vt = _mm256_fmadd_ps(_mm256_permute_ps(vM0,0b01010101),_mm256_fmadd_ps(vM1_0000_1111, vf_i2_j01, _mm256_mul_ps(vM1_2222_3333, vf_i2_j23)) , vt);
            vt = _mm256_fmadd_ps(_mm256_permute_ps(vM0,0b00000000),_mm256_fmadd_ps(vM1_0000_1111, vf_i3_j01, _mm256_mul_ps(vM1_2222_3333, vf_i3_j23)) , vt);

            vt = _mm256_mul_ps(vM2, vt);
            //val[jj] = sum8(vt);
            query_values[i] = sum8(vt);
  	        }

  //  __m256 tmp = _mm256_loadu_ps(val);
  //  _mm256_stream_ps (&query_values[ii*CHUNK],tmp);
  //	}

  	return;

  }  // end of interp3_ghost_xyz_p

  #else
  void vectorized_interp3_ghost_xyz_p(__restrict Real* reg_grid_vals, int data_dof, const int* __restrict N_reg,
  		const int* __restrict N_reg_g, const int * __restrict isize_g, const int* __restrict istart, const int N_pts,
  		const int g_size, Real* __restrict query_points, Real* __restrict query_values,
  		bool query_values_already_scaled) {

  #ifdef INTERP_DEBUG
  	int nprocs, procid;
  	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    PCOUT << "In ivybridge  kernel\n";
  #endif
    const __m256  c1000 = _mm256_set_ps(-1.0,-0.0,-0.0,-0.0,-1.0,-0.0,-0.0,-0.0);
    const __m256  c2211 = _mm256_set_ps(-2.0,-2.0,-1.0,-1.0,-2.0,-2.0,-1.0,-1.0);
    const __m256  c3332 = _mm256_set_ps(-3.0,-3.0,-3.0,-2.0,-3.0,-3.0,-3.0,-2.0);

    const __m256 vlagr = _mm256_set_ps(-0.1666666667,0.5,-0.5, 0.1666666667,-0.1666666667,0.5,-0.5, 0.1666666667);
    const __m256  c33332222 = _mm256_set_ps(-3.0,-3.0,-3.0,-3.0,-2.0,-2.0,-2.0,-2.0);
    const __m256  c22223333 = _mm256_setr_ps(-3.0,-3.0,-3.0,-3.0,-2.0,-2.0,-2.0,-2.0);
    const __m256  c11110000 = _mm256_set_ps(-1.0,-1.0,-1.0,-1.0,0,0,0,0);
    const __m256  c00001111 = _mm256_setr_ps(-1.0,-1.0,-1.0,-1.0,0,0,0,0);
    const __m256  l0l1 = _mm256_set_ps (-0.1666666667,-0.1666666667,-0.1666666667,-0.1666666667,+0.5,+0.5,+0.5,+0.5);
    const __m256  l2l3 = _mm256_setr_ps(+0.1666666667,+0.1666666667,+0.1666666667,+0.1666666667,-0.5,-0.5,-0.5,-0.5);
    const int isize_g2 = isize_g[2];
    const int two_isize_g2 = 2*isize_g2;
    const int reg_plus = isize_g[1]*isize_g2 - two_isize_g2;
    const int NzNy = isize_g2 * isize_g[1];
    Real* Q_ptr = query_points;
    //_mm_prefetch( (char*)Q_ptr,_MM_HINT_NTA);
  	for (int i = 0; i < N_pts; i++) {
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		point[0] = Q_ptr[0];
  		grid_indx[0] = ((int)(point[0])) - 1;
  		point[0] -= grid_indx[0];

  		point[1] = Q_ptr[1];
  		grid_indx[1] = ((int)(point[1])) - 1;
  		point[1] -= grid_indx[1];

  		point[2] = Q_ptr[2];
  		grid_indx[2] = ((int)(point[2])) - 1;
  		point[2] -= grid_indx[2];
      Q_ptr += 3;

  		const int indxx = NzNy * grid_indx[0] + grid_indx[2] + isize_g2 * grid_indx[1] ;
      //_mm_prefetch( (char*)Q_ptr,_MM_HINT_T2);

  //    int indx = 0;
      Real* reg_ptr = reg_grid_vals + indxx;//&reg_grid_vals[indxx];
      //_mm_prefetch( (char*)reg_ptr,_MM_HINT_T0);
  		Real val = 0;



  //  _mm_prefetch( (char*)reg_ptr,_MM_HINT_T1 );
  //  _mm_prefetch( (char*)reg_ptr+isize_g2,_MM_HINT_T1 );
  //  reg_ptr += two_isize_g2;
  //  _mm_prefetch( (char*)reg_ptr,_MM_HINT_T0 );
  //  _mm_prefetch( (char*)reg_ptr+isize_g2,_MM_HINT_T0 );
  //  reg_ptr += reg_plus;
  //           const __m256 vf_i0_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr += two_isize_g2;
  //           const __m256 vf_i0_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr +=  reg_plus;
  //
  //           const __m256 vf_i1_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr += two_isize_g2;
  //           const __m256 vf_i1_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr +=  reg_plus;
  //
  //           const __m256 vf_i2_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr += two_isize_g2;
  //           const __m256 vf_i2_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr +=  reg_plus;
  //
  //           const __m256 vf_i3_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr += two_isize_g2;
  //           const __m256 vf_i3_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           // reg_ptr +=  reg_plus;




      //__m256 vM0(vlagr), vM1(vlagr), vM2(vlagr);
      __m256 vM0(vlagr), vM2(vlagr);
      // __m256 vM0_tttt[4]; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM1_0000_1111; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM1_2222_3333; // elements will be M2[2] for the first 4 reg and M2[3] for the rest
      __m256 vM0_tttt[4];


      {
      const __m256 vx0 = _mm256_set1_ps(point[0]);
      vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c1000));
      vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c2211));
      vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c3332));

      const __m256 vx1 = _mm256_set1_ps(point[1]);
      __m256 tmp = _mm256_add_ps(vx1,c33332222); // x-3,...;x-2,...
      tmp = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c11110000));
      vM1_0000_1111 = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c22223333));
      vM1_0000_1111 = _mm256_mul_ps(vM1_0000_1111, l0l1);
      vM1_2222_3333 = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c00001111));
      vM1_2222_3333  = _mm256_mul_ps(vM1_2222_3333, l2l3);
      //vM1 = _mm256_mul_ps(vM1, _mm256_add_ps(vx1,c1000));
      //vM1 = _mm256_mul_ps(vM1, _mm256_add_ps(vx1,c2211));
      //vM1 = _mm256_mul_ps(vM1, _mm256_add_ps(vx1,c3332));

      const __m256 vx2 = _mm256_set1_ps(point[2]);
      vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c1000));
      vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c2211));
      vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c3332));
      // todo remove permute completely by using different c's in the beginning
      vM2 = _mm256_permute_ps(vM2,0b00011011);
      //vM2 = _mm256_shuffle_ps(vM2,vM2,_MM_SHUFFLE(0, 1, 2, 3));

      //const Real* M1 = (Real*)&vM1;
      //vM1_0000_1111 = _mm256_set_ps(M1[7],M1[7],M1[7],M1[7],M1[6],M1[6],M1[6],M1[6]);
      //vM1_2222_3333 = _mm256_set_ps(M1[5],M1[5],M1[5],M1[5],M1[4],M1[4],M1[4],M1[4]);
      // vM1_0000_1111 = _mm256_set_ps(M1[3],M1[3],M1[3],M1[3],M1[2],M1[2],M1[2],M1[2]);
      // vM1_2222_3333 = _mm256_set_ps(M1[1],M1[1],M1[1],M1[1],M1[0],M1[0],M1[0],M1[0]);
      vM0_tttt[0] = _mm256_permute_ps(vM0,0b11111111); // last element
      vM0_tttt[1] = _mm256_permute_ps(vM0,0b10101010);
      vM0_tttt[2] = _mm256_permute_ps(vM0,0b01010101);
      vM0_tttt[3] = _mm256_permute_ps(vM0,0b00000000);
      }


      // load all vfij




            //
            const __m256 vf_i0_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;

            const __m256 vt_i0_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i0_j01); //8

            const __m256 vf_i0_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr +=  reg_plus;
            const __m256 vt_i0_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i0_j23);//8

            const __m256 vt_i0 = _mm256_add_ps(vt_i0_j01, vt_i0_j23);//8

            //
            const __m256 vf_i1_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;

            const __m256 vt_i1_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i1_j01);//8

            const __m256 vf_i1_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr +=  reg_plus;
            const __m256 vt_i1_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i1_j23);//8

            const __m256 vt_i1 = _mm256_add_ps(vt_i1_j01, vt_i1_j23);//8

            //
            const __m256 vf_i2_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;

            const __m256 vt_i2_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i2_j01);//8

            const __m256 vf_i2_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr +=  reg_plus;

            const __m256 vt_i2_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i2_j23);//8
            const __m256 vt_i2 = _mm256_add_ps(vt_i2_j01, vt_i2_j23);//8

            //
            const __m256 vf_i3_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            // reg_ptr +=  reg_plus;

            const __m256 vt_i3_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i3_j01);//8
            const __m256 vf_i3_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            const __m256 vt_i3_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i3_j23);//8
            const __m256 vt_i3 = _mm256_add_ps(vt_i3_j01, vt_i3_j23);//8

            const __m256 vt0 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b11111111), vt_i0);//8
            const __m256 vt1 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b10101010), vt_i1);//8
            const __m256 vt2 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b01010101), vt_i2);//8
            const __m256 vt3 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b00000000), vt_i3);//8
            //const __m256 vt0 = _mm256_mul_ps(vM0_tttt[0], vt_i0);
            //const __m256 vt1 = _mm256_mul_ps(vM0_tttt[1], vt_i1);
            //const __m256 vt2 = _mm256_mul_ps(vM0_tttt[2], vt_i2);
            //const __m256 vt3 = _mm256_mul_ps(vM0_tttt[3], vt_i3);

            __m256 vt = _mm256_add_ps(vt0, vt1);//8
            vt = _mm256_add_ps(vt, vt2);//8
            vt = _mm256_add_ps(vt, vt3);//8

            vt = _mm256_mul_ps(vM2, vt);//8
            val = sum8(vt);//7
  		      query_values[i] = val;
  	}

  	return;

  }  // end of interp3_ghost_xyz_p
  #endif

  #endif



  // void vectorized_interp3_ghost_xyz_p(__restrict Real* reg_grid_vals, int data_dof, const int* __restrict N_reg,
  // 		const int* __restrict N_reg_g, const int * __restrict isize_g, const int* __restrict istart, const int N_pts,
  // 		const int g_size, Real* __restrict query_points, Real* __restrict query_values,
  // 		bool query_values_already_scaled) {
  //
  //   const __m256  c1000 = _mm256_set_ps(-1.0,-0.0,-0.0,-0.0,-1.0,-0.0,-0.0,-0.0);
  //   const __m256  c2211 = _mm256_set_ps(-2.0,-2.0,-1.0,-1.0,-2.0,-2.0,-1.0,-1.0);
  //   const __m256  c3332 = _mm256_set_ps(-3.0,-3.0,-3.0,-2.0,-3.0,-3.0,-3.0,-2.0);
  //
  //   const __m256 vlagr = _mm256_set_ps(-0.1666666667,0.5,-0.5, 0.1666666667,-0.1666666667,0.5,-0.5, 0.1666666667);
  //   const __m256  c33332222 = _mm256_set_ps(-3.0,-3.0,-3.0,-3.0,-2.0,-2.0,-2.0,-2.0);
  //   const __m256  c22223333 = _mm256_setr_ps(-3.0,-3.0,-3.0,-3.0,-2.0,-2.0,-2.0,-2.0);
  //   const __m256  c11110000 = _mm256_set_ps(-1.0,-1.0,-1.0,-1.0,0,0,0,0);
  //   const __m256  c00001111 = _mm256_setr_ps(-1.0,-1.0,-1.0,-1.0,0,0,0,0);
  //   const __m256  l0l1 = _mm256_set_ps (-0.1666666667,-0.1666666667,-0.1666666667,-0.1666666667,+0.5,+0.5,+0.5,+0.5);
  //   const __m256  l2l3 = _mm256_setr_ps(+0.1666666667,+0.1666666667,+0.1666666667,+0.1666666667,-0.5,-0.5,-0.5,-0.5);
  // 	for (int i = 0; i < N_pts; i++) {
  // 		Real point[COORD_DIM];
  // 		int grid_indx[COORD_DIM];
  //
  // 		point[0] = query_points[COORD_DIM * i + 0] * N_reg_g[0];
  // 		grid_indx[0] = ((int)(point[0])) - 1;
  // 		point[0] -= grid_indx[0];
  //
  // 		point[1] = query_points[COORD_DIM * i + 1] * N_reg_g[1];
  // 		grid_indx[1] = ((int)(point[1])) - 1;
  // 		point[1] -= grid_indx[1];
  //
  // 		point[2] = query_points[COORD_DIM * i + 2] * N_reg_g[2];
  // 		grid_indx[2] = ((int)(point[2])) - 1;
  // 		point[2] -= grid_indx[2];
  //
  // 		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
  //     Real* reg_ptr = &reg_grid_vals[indxx];
  // 		Real val = 0;
  //     int indx = 0;
  //     const int isize_g2 = isize_g[2];
  //     const int two_isize_g2 = 2*isize_g[2];
  //     const int reg_plus = isize_g[1]*isize_g2 - two_isize_g2;
  //
  //
  //     __m256 vM0(vlagr), vM1(vlagr), vM2(vlagr);
  //     // __m256 vM0_tttt[4]; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
  //     __m256 vM1_0000_1111; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
  //     __m256 vM1_2222_3333; // elements will be M2[2] for the first 4 reg and M2[3] for the rest
  //     __m256 vM0_tttt[4];
  //
  //
  //     {
  //     const __m256 vx0 = _mm256_set1_ps(point[0]);
  //     const __m256 vx1 = _mm256_set1_ps(point[1]);
  //     const __m256 vx2 = _mm256_set1_ps(point[2]);
  //     vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c1000));
  //     vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c2211));
  //     vM0 = _mm256_mul_ps(vM0, _mm256_add_ps(vx0,c3332));
  //
  //     __m256 tmp = _mm256_add_ps(vx1,c33332222); // x-3,...;x-2,...
  //     tmp = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c11110000));
  //     vM1_0000_1111 = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c22223333));
  //     vM1_0000_1111 = _mm256_mul_ps(vM1_0000_1111, l0l1);
  //     vM1_2222_3333 = _mm256_mul_ps(tmp, _mm256_add_ps(vx1,c00001111));
  //     vM1_2222_3333  = _mm256_mul_ps(vM1_2222_3333, l2l3);
  //     //vM1 = _mm256_mul_ps(vM1, _mm256_add_ps(vx1,c1000));
  //     //vM1 = _mm256_mul_ps(vM1, _mm256_add_ps(vx1,c2211));
  //     //vM1 = _mm256_mul_ps(vM1, _mm256_add_ps(vx1,c3332));
  //
  //     vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c1000));
  //     vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c2211));
  //     vM2 = _mm256_mul_ps(vM2, _mm256_add_ps(vx2,c3332));
  //     // todo remove permute completely by using different c's in the beginning
  //     vM2 = _mm256_permute_ps(vM2,0b00011011);
  //     //vM2 = _mm256_shuffle_ps(vM2,vM2,_MM_SHUFFLE(0, 1, 2, 3));
  //
  //     //const Real* M1 = (Real*)&vM1;
  //     //vM1_0000_1111 = _mm256_set_ps(M1[7],M1[7],M1[7],M1[7],M1[6],M1[6],M1[6],M1[6]);
  //     //vM1_2222_3333 = _mm256_set_ps(M1[5],M1[5],M1[5],M1[5],M1[4],M1[4],M1[4],M1[4]);
  //     // vM1_0000_1111 = _mm256_set_ps(M1[3],M1[3],M1[3],M1[3],M1[2],M1[2],M1[2],M1[2]);
  //     // vM1_2222_3333 = _mm256_set_ps(M1[1],M1[1],M1[1],M1[1],M1[0],M1[0],M1[0],M1[0]);
  //     vM0_tttt[0] = _mm256_permute_ps(vM0,0b11111111); // last element
  //     vM0_tttt[1] = _mm256_permute_ps(vM0,0b10101010);
  //     vM0_tttt[2] = _mm256_permute_ps(vM0,0b01010101);
  //     vM0_tttt[3] = _mm256_permute_ps(vM0,0b00000000);
  //     }
  //
  //
  //     // load all vfij
  //           const __m256 vf_i0_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr += two_isize_g2;
  //           const __m256 vf_i0_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr +=  reg_plus;
  //
  //           const __m256 vf_i1_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr += two_isize_g2;
  //           const __m256 vf_i1_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr +=  reg_plus;
  //
  //           const __m256 vf_i2_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr += two_isize_g2;
  //           const __m256 vf_i2_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr +=  reg_plus;
  //
  //           const __m256 vf_i3_j01 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           reg_ptr += two_isize_g2;
  //           const __m256 vf_i3_j23 = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
  //           // reg_ptr +=  reg_plus;
  //
  //           const __m256 vt_i0_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i0_j01);
  //           const __m256 vt_i0_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i0_j23);
  //           const __m256 vt_i0 = _mm256_add_ps(vt_i0_j01, vt_i0_j23);
  //
  //           const __m256 vt_i1_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i1_j01);
  //           const __m256 vt_i1_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i1_j23);
  //           const __m256 vt_i1 = _mm256_add_ps(vt_i1_j01, vt_i1_j23);
  //
  //           const __m256 vt_i2_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i2_j01);
  //           const __m256 vt_i2_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i2_j23);
  //           const __m256 vt_i2 = _mm256_add_ps(vt_i2_j01, vt_i2_j23);
  //
  //           const __m256 vt_i3_j01 = _mm256_mul_ps(vM1_0000_1111, vf_i3_j01);
  //           const __m256 vt_i3_j23 = _mm256_mul_ps(vM1_2222_3333, vf_i3_j23);
  //           const __m256 vt_i3 = _mm256_add_ps(vt_i3_j01, vt_i3_j23);
  //
  //           const __m256 vt0 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b11111111), vt_i0);
  //           const __m256 vt1 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b10101010), vt_i1);
  //           const __m256 vt2 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b01010101), vt_i2);
  //           const __m256 vt3 = _mm256_mul_ps(_mm256_permute_ps(vM0,0b00000000), vt_i3);
  //           //const __m256 vt0 = _mm256_mul_ps(vM0_tttt[0], vt_i0);
  //           //const __m256 vt1 = _mm256_mul_ps(vM0_tttt[1], vt_i1);
  //           //const __m256 vt2 = _mm256_mul_ps(vM0_tttt[2], vt_i2);
  //           //const __m256 vt3 = _mm256_mul_ps(vM0_tttt[3], vt_i3);
  //
  //           __m256 vt = _mm256_add_ps(vt0, vt1);
  //           vt = _mm256_add_ps(vt, vt2);
  //           vt = _mm256_add_ps(vt, vt3);
  //
  //           vt = _mm256_mul_ps(vM2, vt);
  //           val = sum8(vt);
  // 		      query_values[i] = val;
  // 	}
  //
  // 	return;
  //
  // }  // end of interp3_ghost_xyz_p


  void rescale_xyz(const int g_size, int* N_reg, int* N_reg_g, int* istart,
  		const int N_pts, Real* Q_) {

  	if (g_size == 0)
  		return;
  	Real hp[3];
  	Real h[3];
  	hp[0] = 1. / N_reg_g[0]; // New mesh size
  	hp[1] = 1. / N_reg_g[1]; // New mesh size
  	hp[2] = 1. / N_reg_g[2]; // New mesh size

  	h[0] = 1. / (N_reg[0]); // old mesh size
  	h[1] = 1. / (N_reg[1]); // old mesh size
  	h[2] = 1. / (N_reg[2]); // old mesh size

  	const Real factor0 = (1. - (2. * g_size + 1.) * hp[0]) / (1. - h[0]);
  	const Real factor1 = (1. - (2. * g_size + 1.) * hp[1]) / (1. - h[1]);
  	const Real factor2 = (1. - (2. * g_size + 1.) * hp[2]) / (1. - h[2]);
    const Real iX0 = istart[0]*h[0];
    const Real iX1 = istart[1]*h[1];
    const Real iX2 = istart[2]*h[2];

  	for (int i = 0; i < N_pts; i++) {
  		Q_[0 + COORD_DIM * i] = (Q_[0 + COORD_DIM * i]
  				- iX0) * factor0 + g_size * hp[0];
  		Q_[1 + COORD_DIM * i] = (Q_[1 + COORD_DIM * i]
  				- iX1) * factor1 + g_size * hp[1];
  		Q_[2 + COORD_DIM * i] = (Q_[2 + COORD_DIM * i]
  				- iX2) * factor2 + g_size * hp[2];
  	}
  	return;
  } // end of rescale_xyz


  #ifdef FAST_INTERPV

  //#include "v1.cpp" // corresponding optimized version
  void vec_torized_interp3_ghost_xyz_p(__restrict Real* reg_grid_vals, int data_dof, const int* N_reg,
  		const int* N_reg_g, const int * isize_g, const int* istart, const int N_pts,
  		const int g_size, __restrict Real* query_points, __restrict Real* query_values,
  		bool query_values_already_scaled) {

    const __m128  c1000 = _mm_set_ps(-1.0,-0.0,-0.0,-0.0);
    const __m128  c2211 = _mm_set_ps(-2.0,-2.0,-1.0,-1.0);
    const __m128  c3332 = _mm_set_ps(-3.0,-3.0,-3.0,-2.0);
    const __m128 vlagr = _mm_set_ps(-0.1666666667,0.5,-0.5, 0.1666666667);

  	for (int i = 0; i < N_pts; i++) {
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		point[0] = query_points[COORD_DIM * i + 0];
  		grid_indx[0] = ((int)(point[0])) - 1;
  		point[0] -= grid_indx[0];

  		point[1] = query_points[COORD_DIM * i + 1];
  		grid_indx[1] = ((int)(point[1])) - 1;
  		point[1] -= grid_indx[1];

  		point[2] = query_points[COORD_DIM * i + 2];
  		grid_indx[2] = ((int)(point[2])) - 1;
  		point[2] -= grid_indx[2];

  		// for (int j = 0; j < COORD_DIM; j++) {
  		//	Real x = point[j];
  		//	for (int k = 0; k < 4; k++) {
  		//		M[j][k] = lagr_denom[k];
  		//		for (int l = 0; l < 4; l++) {
  		//			if (k != l)
  		//				M[j][k] *= (x - l);
  		//		}
  		//	}
  		// }
      //M[0][0] = lagr_denom[0];
      //M[0][1] = lagr_denom[1];
      //M[0][2] = lagr_denom[2];
      //M[0][3] = lagr_denom[3];

      //M[0][0] *= (x-1);
      //M[0][1] *= (x-0);
      //M[0][2] *= (x-0);
      //M[0][3] *= (x-0);

      //M[0][0] *= (x-2);
      //M[0][1] *= (x-2);
      //M[0][2] *= (x-1);
      //M[0][3] *= (x-1);

      //M[0][0] *= (x-3);
      //M[0][1] *= (x-3);
      //M[0][2] *= (x-3);
      //M[0][3] *= (x-2);

      __m128 vx;

      __m128 vM0(vlagr), vM1(vlagr), vM2(vlagr);
      __m256 vM0_tttt[4]; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM1_0000_1111; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM1_2222_3333; // elements will be M2[2] for the first 4 reg and M2[3] for the rest
            __m256 vVal_, vPtr_;
            __m256 vVal2_;//= _mm256_set1_ps(point[0]);


      vx = _mm_set1_ps(point[0]);
      vM0 = _mm_mul_ps(vM0, _mm_add_ps(vx,c1000));
      vM0 = _mm_mul_ps(vM0, _mm_add_ps(vx,c2211));
      vM0 = _mm_mul_ps(vM0, _mm_add_ps(vx,c3332));

      vx = _mm_set1_ps(point[1]);
      vM1 = _mm_mul_ps(vM1, _mm_add_ps(vx,c1000));
      vM1 = _mm_mul_ps(vM1, _mm_add_ps(vx,c2211));
      vM1 = _mm_mul_ps(vM1, _mm_add_ps(vx,c3332));

      vx = _mm_set1_ps(point[2]);
      vM2 = _mm_mul_ps(vM2, _mm_add_ps(vx,c1000));
      vM2 = _mm_mul_ps(vM2, _mm_add_ps(vx,c2211));
      vM2 = _mm_mul_ps(vM2, _mm_add_ps(vx,c3332));
      vM2 = _mm_shuffle_ps(vM2,vM2,_MM_SHUFFLE(0, 1, 2, 3));


      vM1_0000_1111 = _mm256_set_m128(
                _mm_shuffle_ps(vM1,vM1,_MM_SHUFFLE(3, 3, 3, 3)), // M[1][0]
                _mm_shuffle_ps(vM1,vM1,_MM_SHUFFLE(2, 2, 2, 2)));// M[1][1]
      vM1_2222_3333 = _mm256_set_m128(
                _mm_shuffle_ps(vM1,vM1,_MM_SHUFFLE(1, 1, 1, 1)), // M[1][2]
                _mm_shuffle_ps(vM1,vM1,_MM_SHUFFLE(0, 0, 0, 0)));// M[1][3]


      Real* M0 = (Real*)&vM0;
      vM0_tttt[3] = _mm256_set1_ps(M0[0]);
      vM0_tttt[2] = _mm256_set1_ps(M0[1]);
      vM0_tttt[1] = _mm256_set1_ps(M0[2]);
      vM0_tttt[0] = _mm256_set1_ps(M0[3]);

      //vM0_tttt[0] = _mm256_set_m128(
      //          _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(3, 3, 3, 3)), // M[0][0]
      //          _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(3, 3, 3, 3)));// M[0][0]
      //vM0_tttt[1] = _mm256_set_m128(
      //          _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(2, 2, 2, 2)), // M[0][0]
      //          _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(2, 2, 2, 2)));// M[0][0]
      //vM0_tttt[2] = _mm256_set_m128(
      //          _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(1, 1, 1, 1)), // M[0][0]
      //          _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(1, 1, 1, 1)));// M[0][0]
      //vM0_tttt[3] = _mm256_set_m128(
      //          _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(0, 0, 0, 0)), // M[0][0]
      //          _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(0, 0, 0, 0)));// M[0][0]
      //Real* dum1 = (Real*)&vM0;
      //Real* dum2 = (Real*)&vM1;
      //Real* dum3 = (Real*)&vM2;
      //Real* dum4 = (Real*)&vM1_0000_1111;
      //Real* dum5 = (Real*)&vM1_2222_3333;
      //Real* dum6 = (Real*)&vVal_;
      //Real* dum7 = (Real*)&vM0_tttt[0];
      //Real* dum8 = (Real*)&vM0_tttt[1];
      //Real* dum9 = (Real*)&vM0_tttt[2];
      //Real* dum10 = (Real*)&vM0_tttt[3];
      //query_values[i] = point[0]*point[1]*point[2];
      //continue;
  		//query_values[i] = dum1[0]*dum2[0]*dum3[2]*dum4[5]*dum5[5]*dum5[0]*dum5[2]
      //  *dum7[0]*dum8[5]*dum9[0]*dum10[2];//*dum5[3];
      //continue;


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
      Real* reg_ptr = &reg_grid_vals[indxx];
  		Real val = 0;
      //int indx = 0;
      const int isize_g2 = isize_g[2];
      const int two_isize_g2 = 2*isize_g[2];
      const int reg_plus = isize_g[1]*isize_g2 - two_isize_g2;
  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            vVal_ = _mm256_setzero_ps();


            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vVal2_ = _mm256_mul_ps(vM1_0000_1111, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_tttt[0]);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            vVal2_ = _mm256_mul_ps(vM1_2222_3333, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_tttt[0]);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);


            reg_ptr +=  reg_plus;

        // ------------------------------------ //
            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vVal2_ = _mm256_mul_ps(vM1_0000_1111, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_tttt[1]);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);


            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            vVal2_ = _mm256_mul_ps(vM1_2222_3333, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_tttt[1]);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);
            reg_ptr +=  reg_plus;

        // ------------------------------------ //
            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vVal2_ = _mm256_mul_ps(vM1_0000_1111, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_tttt[2]);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            vVal2_ = _mm256_mul_ps(vM1_2222_3333, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_tttt[2]);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            reg_ptr +=  reg_plus;
        // ------------------------------------ //
            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vVal2_ = _mm256_mul_ps(vM1_0000_1111, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_tttt[3]);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            vVal2_ = _mm256_mul_ps(vM1_2222_3333, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_tttt[3]);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            // set vm_inv = M[2][0], [1] [2] [3] in reverse order
            __m256 vM2_256 = _mm256_set_m128(vM2, vM2);
            vVal_ = _mm256_mul_ps(vVal_, vM2_256);
            val = sum8(vVal_);
  		query_values[i] = val;
  	}

  	return;

  }  // end of interp3_ghost_xyz_p


  void _vectorized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
      {
  #ifdef VERBOSE2
  		std::cout<<"q[0]="<<query_points[i*3+0]<<std::endl;
  		std::cout<<"q[1]="<<query_points[i*3+1]<<std::endl;
  		std::cout<<"q[2]="<<query_points[i*3+2]<<std::endl;
  #endif
    }
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			//while (grid_indx[j] < 0)
  			//	grid_indx[j] += N_reg_g[j];
  		}
  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
      //__m256 vM_ = _mm256_set_ps(M[2][0], M[2][1], M[2][2], M[2][3], 0, 0, 0, 0);
      register __m128 vM_ = _mm_set_ps(M[2][0], M[2][1], M[2][2], M[2][3]);
      //__m128 vM_ = _mm_loadu_ps(&M[2][0]);
      register __m128 vVal =  _mm_setzero_ps();
      register __m128 vVal_;
      // std::cout << "indxx = " << indxx << std::endl;
  //		Real val = 0;
      int indx = 0;
  		for (int j0 = 0; j0 < 4; j0++) {
  			for (int j1 = 0; j1 < 4; j1++) {
            const register __m128 M0M1 = _mm_set1_ps(M[0][j0]*M[1][j1]);

            __m128 vPtr_ = _mm_loadu_ps(&reg_grid_vals[indx + indxx]);
            vVal_ = _mm_mul_ps(vM_, vPtr_);
            vVal_ = _mm_mul_ps(vVal_, M0M1);

            //__m256 vVal_;
            //__m256 vPtr_ = _mm256_set_ps(ptr[0], ptr[1], ptr[2], ptr[3], 0, 0, 0, 0);
            //vVal_ = _mm256_mul_ps(vM_, vPtr_);
            //Real val_ = sum8(vVal_);
            //val += (val_[4] + val_[5] + val_[6] + val_[7]) * M0M1;
            vVal = _mm_add_ps(vVal, vVal_);
            indx += isize_g[2];
  			}
        indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];
  		}
      vVal = _mm_hadd_ps(vVal, vVal);
      vVal = _mm_hadd_ps(vVal, vVal);
      Real* val_ = (Real*)&vVal;
  		query_values[i] = val_[0];
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p

  void __vectorized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
      {
  #ifdef VERBOSE2
  		std::cout<<"q[0]="<<query_points[i*3+0]<<std::endl;
  		std::cout<<"q[1]="<<query_points[i*3+1]<<std::endl;
  		std::cout<<"q[2]="<<query_points[i*3+2]<<std::endl;
  #endif
    }
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			//while (grid_indx[j] < 0)
  			//	grid_indx[j] += N_reg_g[j];
  		}
  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
  //		Real val = 0;
      int indx = 0;


  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            Real M0M1, M0M1_2;
            __m256 vVal_, vM0M1_, vM2_, vM_, vPtr_;
            Real* ptr, *ptr2;

  					// val_ = M[2][0] * ptr[0];
  					// val_ += M[2][1] * ptr[1];
  					// val_ += M[2][2] * ptr[2];
  					// val_ += M[2][3] * ptr[3];
            // val += val_ * M0M1;
            // indx += isize_g[2];
            // M0M1 = M[0][0]*M[1][1];
            // ptr = &reg_grid_vals[indx + indxx];
  					// val_ = M[2][0] * ptr[0];
  					// val_ += M[2][1] * ptr[1];
  					// val_ += M[2][2] * ptr[2];
  					// val_ += M[2][3] * ptr[3];
            // val += val_ * M0M1;
            //indx += isize_g[2];

            M0M1 = M[0][0]*M[1][0];
            M0M1_2 = M[0][0]*M[1][1];
            vM2_ = _mm256_set_ps(M[2][0], M[2][1], M[2][2], M[2][3], M[2][0], M[2][1], M[2][2], M[2][3]);
            vM0M1_ = _mm256_set_ps(M0M1,M0M1,M0M1,M0M1,M0M1_2,M0M1_2,M0M1_2,M0M1_2);
            vM_ = _mm256_mul_ps(vM0M1_, vM2_);

            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            vPtr_ = _mm256_set_ps(ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3]);
            vVal_ = _mm256_mul_ps(vM_, vPtr_);
            //----//

            M0M1 = M[0][0]*M[1][2];
            M0M1_2 = M[0][0]*M[1][3];
            vM0M1_ = _mm256_set_ps(M0M1,M0M1,M0M1,M0M1,M0M1_2,M0M1_2,M0M1_2,M0M1_2);
            vM_ = _mm256_mul_ps(vM0M1_, vM2_);

            indx += isize_g[2];
            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            vPtr_ = _mm256_set_ps(ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3]);

            vVal_ = _mm256_mul_ps(vM_, vPtr_);
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];


            // M0M1 = M[0][0]*M[1][2];
            // ptr = &reg_grid_vals[indx + indxx];
  					// val_ = M[2][0] * ptr[0];
  					// val_ += M[2][1] * ptr[1];
  					// val_ += M[2][2] * ptr[2];
  					// val_ += M[2][3] * ptr[3];
            // val += val_ * M0M1;
            // indx += isize_g[2];

            // M0M1 = M[0][0]*M[1][3];
            // ptr = &reg_grid_vals[indx + indxx];
  					// val_ = M[2][0] * ptr[0];
  					// val_ += M[2][1] * ptr[1];
  					// val_ += M[2][2] * ptr[2];
  					// val_ += M[2][3] * ptr[3];
            // val += val_ * M0M1;
            // indx += isize_g[2];
            // indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][1]*M[1][0];
            M0M1_2 = M[0][1]*M[1][1];
            vM2_ = _mm256_set_ps(M[2][0], M[2][1], M[2][2], M[2][3], M[2][0], M[2][1], M[2][2], M[2][3]);
            vM0M1_ = _mm256_set_ps(M0M1,M0M1,M0M1,M0M1,M0M1_2,M0M1_2,M0M1_2,M0M1_2);
            vM_ = _mm256_mul_ps(vM0M1_, vM2_);

            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            vPtr_ = _mm256_set_ps(ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3]);
            vVal_ = _mm256_mul_ps(vM_, vPtr_);
            //----//

            M0M1 = M[0][1]*M[1][2];
            M0M1_2 = M[0][1]*M[1][3];
            vM0M1_ = _mm256_set_ps(M0M1,M0M1,M0M1,M0M1,M0M1_2,M0M1_2,M0M1_2,M0M1_2);
            vM_ = _mm256_mul_ps(vM0M1_, vM2_);

            indx += isize_g[2];
            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            vPtr_ = _mm256_set_ps(ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3]);

            vVal_ = _mm256_mul_ps(vM_, vPtr_);
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

            //M0M1 = M[0][1]*M[1][0];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];


            //M0M1 = M[0][1]*M[1][1];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];


            //M0M1 = M[0][1]*M[1][2];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];

            //M0M1 = M[0][1]*M[1][3];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];
            //indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // -//----------------------------------- //
            M0M1 = M[0][2]*M[1][0];
            M0M1_2 = M[0][2]*M[1][1];
            vM2_ = _mm256_set_ps(M[2][0], M[2][1], M[2][2], M[2][3], M[2][0], M[2][1], M[2][2], M[2][3]);
            vM0M1_ = _mm256_set_ps(M0M1,M0M1,M0M1,M0M1,M0M1_2,M0M1_2,M0M1_2,M0M1_2);
            vM_ = _mm256_mul_ps(vM0M1_, vM2_);

            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            vPtr_ = _mm256_set_ps(ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3]);
            vVal_ = _mm256_mul_ps(vM_, vPtr_);
            //----//

            M0M1 = M[0][2]*M[1][2];
            M0M1_2 = M[0][2]*M[1][3];
            vM0M1_ = _mm256_set_ps(M0M1,M0M1,M0M1,M0M1,M0M1_2,M0M1_2,M0M1_2,M0M1_2);
            vM_ = _mm256_mul_ps(vM0M1_, vM2_);

            indx += isize_g[2];
            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            vPtr_ = _mm256_set_ps(ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3]);

            vVal_ = _mm256_mul_ps(vM_, vPtr_);
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

            //M0M1 = M[0][2]*M[1][0];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];


            //M0M1 = M[0][2]*M[1][1];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];


            //M0M1 = M[0][2]*M[1][2];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];

            //M0M1 = M[0][2]*M[1][3];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];
            //indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // -//----------------------------------- //
            M0M1 = M[0][3]*M[1][0];
            M0M1_2 = M[0][3]*M[1][1];
            vM2_ = _mm256_set_ps(M[2][0], M[2][1], M[2][2], M[2][3], M[2][0], M[2][1], M[2][2], M[2][3]);
            vM0M1_ = _mm256_set_ps(M0M1,M0M1,M0M1,M0M1,M0M1_2,M0M1_2,M0M1_2,M0M1_2);
            vM_ = _mm256_mul_ps(vM0M1_, vM2_);

            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            vPtr_ = _mm256_set_ps(ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3]);
            vVal_ = _mm256_mul_ps(vM_, vPtr_);
            //----//

            M0M1 = M[0][3]*M[1][2];
            M0M1_2 = M[0][3]*M[1][3];
            vM0M1_ = _mm256_set_ps(M0M1,M0M1,M0M1,M0M1,M0M1_2,M0M1_2,M0M1_2,M0M1_2);
            vM_ = _mm256_mul_ps(vM0M1_, vM2_);

            indx += isize_g[2];
            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            vPtr_ = _mm256_set_ps(ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3]);

            vVal_ = _mm256_mul_ps(vM_, vPtr_);
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

            //M0M1 = M[0][3]*M[1][0];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];


            //M0M1 = M[0][3]*M[1][1];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];


            //M0M1 = M[0][3]*M[1][2];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
            //indx += isize_g[2];

            //M0M1 = M[0][3]*M[1][3];
            //ptr = &reg_grid_vals[indx + indxx];
  					//val_ = M[2][0] * ptr[0];
  					//val_ += M[2][1] * ptr[1];
  					//val_ += M[2][2] * ptr[2];
  					//val_ += M[2][3] * ptr[3];
            //val += val_ * M0M1;
  		//}
            //Real val_ = sum8(vVal_);
  		query_values[i] =sum8(vVal_);
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p

  void ____vectorized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
      {
  #ifdef VERBOSE2
  		std::cout<<"q[0]="<<query_points[i*3+0]<<std::endl;
  		std::cout<<"q[1]="<<query_points[i*3+1]<<std::endl;
  		std::cout<<"q[2]="<<query_points[i*3+2]<<std::endl;
  #endif
    }
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			//while (grid_indx[j] < 0)
  			//	grid_indx[j] += N_reg_g[j];
  		}
  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
  		Real val = 0;
      int indx = 0;
  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            Real M0M1_[2];
            Real vM[8];

            M0M1_[0] = M[0][0]*M[1][0];
            M0M1_[1] = M[0][0]*M[1][1];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            //register Real val_;
            Real vVal[8]={0};
            Real vVal2[8]={0};
            Real* ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];


            M0M1_[0] = M[0][0]*M[1][2];
            M0M1_[1] = M[0][0]*M[1][3];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1_[0] = M[0][1]*M[1][0];
            M0M1_[1] = M[0][1]*M[1][1];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];

            M0M1_[0] = M[0][1]*M[1][2];
            M0M1_[1] = M[0][1]*M[1][3];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1_[0] = M[0][1]*M[1][0];
            M0M1_[1] = M[0][1]*M[1][1];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];

            M0M1_[0] = M[0][1]*M[1][2];
            M0M1_[1] = M[0][1]*M[1][3];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];
        // ------------------------------------ //
            M0M1_[0] = M[0][3]*M[1][0];
            M0M1_[1] = M[0][3]*M[1][1];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];

            M0M1_[0] = M[0][3]*M[1][2];
            M0M1_[1] = M[0][3]*M[1][3];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            val += (vVal[0]+vVal[1]+vVal[2]+vVal[3]); // * M0M1_[0];
            val += (vVal[4]+vVal[5]+vVal[6]+vVal[7]); // * M0M1_[1];
  		//}
  		query_values[i] = val;
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p



  void _v2_ectorized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
    const Real lagr_denom0 = -1.0/6.0;
    const Real lagr_denom1 = 0.5;
    const Real lagr_denom2 = -0.5;
    const Real lagr_denom3 = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		point[0] = query_points[COORD_DIM * i + 0];
  		grid_indx[0] = ((int)(point[0])) - 1;
  		point[0] -= grid_indx[0];

  		point[1] = query_points[COORD_DIM * i + 1];
  		grid_indx[1] = ((int)(point[1])) - 1;
  		point[1] -= grid_indx[1];

  		point[2] = query_points[COORD_DIM * i + 2];
  		grid_indx[2] = ((int)(point[2])) - 1;
  		point[2] -= grid_indx[2];

      __m128 vx;
      vx = _mm_set1_ps(point[0]);
      __m128 c1000, c2211, c3332, vlagr;
      vlagr = _mm_set_ps(lagr_denom0,lagr_denom1,lagr_denom2,lagr_denom3);

      __m128 vM0(vlagr), vM1(vlagr), vM2(vlagr);
      __m256 vM0_0000; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM0_1111; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM0_2222; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM0_3333; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM2_0000_1111; // elements will be M2[0] for the first 4 reg and M2[1] for the rest
      __m256 vM2_2222_3333; // elements will be M2[2] for the first 4 reg and M2[3] for the rest

      c1000 = _mm_set_ps(-1.0,-0.0,-0.0,-0.0);
      c2211 = _mm_set_ps(-2.0,-2.0,-1.0,-1.0);
      c3332 = _mm_set_ps(-3.0,-3.0,-3.0,-2.0);

      vM0 = _mm_mul_ps(vM0, _mm_add_ps(vx,c1000));
      vM0 = _mm_mul_ps(vM0, _mm_add_ps(vx,c2211));
      vM0 = _mm_mul_ps(vM0, _mm_add_ps(vx,c3332));

      vx = _mm_set1_ps(point[1]);
      vM1 = _mm_mul_ps(vM1, _mm_add_ps(vx,c1000));
      vM1 = _mm_mul_ps(vM1, _mm_add_ps(vx,c2211));
      vM1 = _mm_mul_ps(vM1, _mm_add_ps(vx,c3332));

      vx = _mm_set1_ps(point[2]);
      vM2 = _mm_mul_ps(vM2, _mm_add_ps(vx,c1000));
      vM2 = _mm_mul_ps(vM2, _mm_add_ps(vx,c2211));
      vM2 = _mm_mul_ps(vM2, _mm_add_ps(vx,c3332));
      vM2 = _mm_shuffle_ps(vM2,vM2,_MM_SHUFFLE(0, 1, 2, 3));

      vM2_0000_1111 = _mm256_set_m128(
                _mm_shuffle_ps(vM1,vM1,_MM_SHUFFLE(3, 3, 3, 3)), // M[1][0]
                _mm_shuffle_ps(vM1,vM1,_MM_SHUFFLE(2, 2, 2, 2)));// M[1][1]
      vM2_2222_3333 = _mm256_set_m128(
                _mm_shuffle_ps(vM1,vM1,_MM_SHUFFLE(1, 1, 1, 1)), // M[1][2]
                _mm_shuffle_ps(vM1,vM1,_MM_SHUFFLE(0, 0, 0, 0)));// M[1][3]


      vM0_0000 = _mm256_set_m128(
                _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(3, 3, 3, 3)), // M[0][0]
                _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(3, 3, 3, 3)));// M[0][0]
      vM0_1111 = _mm256_set_m128(
                _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(2, 2, 2, 2)), // M[0][0]
                _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(2, 2, 2, 2)));// M[0][0]
      vM0_2222 = _mm256_set_m128(
                _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(1, 1, 1, 1)), // M[0][0]
                _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(1, 1, 1, 1)));// M[0][0]
      vM0_3333 = _mm256_set_m128(
                _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(0, 0, 0, 0)), // M[0][0]
                _mm_shuffle_ps(vM0,vM0,_MM_SHUFFLE(0, 0, 0, 0)));// M[0][0]


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
      Real* reg_ptr = &reg_grid_vals[indxx];
  		Real val = 0;
      //int indx = 0;
      const int isize_g2 = isize_g[2];
      const int two_isize_g2 = 2*isize_g[2];
      const int reg_plus = isize_g[1]*isize_g2 - two_isize_g2;
  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            __m256 vVal_, vPtr_;
            __m256 vVal2_;
            vVal_ = _mm256_setzero_ps();


            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vVal2_ = _mm256_mul_ps(vM2_0000_1111, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_0000);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            vVal2_ = _mm256_mul_ps(vM2_2222_3333, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_0000);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);


            reg_ptr +=  reg_plus;

        // ------------------------------------ //
            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vVal2_ = _mm256_mul_ps(vM2_0000_1111, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_1111);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);


            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            vVal2_ = _mm256_mul_ps(vM2_2222_3333, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_1111);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);
            reg_ptr +=  reg_plus;

        // ------------------------------------ //
            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vVal2_ = _mm256_mul_ps(vM2_0000_1111, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_2222);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            vVal2_ = _mm256_mul_ps(vM2_2222_3333, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_2222);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            reg_ptr +=  reg_plus;
        // ------------------------------------ //
            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            reg_ptr += two_isize_g2;
            vVal2_ = _mm256_mul_ps(vM2_0000_1111, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_3333);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            // set vPtr_ = {ptr[0], ptr[1], ptr[2], ptr[3], ptr2[0], ptr2[1], ptr2[2], ptr2[3])}
            vPtr_ = _mm256_loadu2_m128(reg_ptr, reg_ptr+isize_g2);
            vVal2_ = _mm256_mul_ps(vM2_2222_3333, vPtr_);
            vVal2_ = _mm256_mul_ps(vVal2_, vM0_3333);
            vVal_ = _mm256_add_ps(vVal_, vVal2_);

            // set vm_inv = M[2][0], [1] [2] [3] in reverse order
            __m256 vM2_256 = _mm256_set_m128(vM2, vM2);
            vVal_ = _mm256_mul_ps(vVal_, vM2_256);
            val = sum8(vVal_);
  		query_values[i] = val;
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p

  void _v1_ectorized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
      {
  #ifdef VERBOSE2
  		std::cout<<"q[0]="<<query_points[i*3+0]<<std::endl;
  		std::cout<<"q[1]="<<query_points[i*3+1]<<std::endl;
  		std::cout<<"q[2]="<<query_points[i*3+2]<<std::endl;
  #endif
    }
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			//while (grid_indx[j] < 0)
  			//	grid_indx[j] += N_reg_g[j];
  		}
  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
  		Real val = 0;
      int indx = 0;
  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            Real M0M1 = M[0][0]*M[1][0];
            register Real val_;
            Real* ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][0]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][0]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][0]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][1]*M[1][0];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][1]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][1]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][1]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][2]*M[1][0];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][2]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][2]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][2]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][3]*M[1][0];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][3]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][3]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][3]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
  		//}
  		query_values[i] = val;
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p
  #endif

  void optimized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			//while (grid_indx[j] < 0)
  			//	grid_indx[j] += N_reg_g[j];
  		}
  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
  		Real val = 0;
      Real vVal2_[8] = {0};
      Real vVal1_[8] = {0};
      int indx = 0;
      Real* ptr, *ptr2;
  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            Real M0M1;
            //register Real val_;

            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            M0M1 = M[1][0];
            vVal2_[0] = M0M1 * ptr[0];
            vVal2_[1] = M0M1 * ptr[1];
            vVal2_[2] = M0M1 * ptr[2];
            vVal2_[3] = M0M1 * ptr[3];
            M0M1 = M[1][1];
            vVal2_[4] = M0M1 * ptr2[0];
            vVal2_[5] = M0M1 * ptr2[1];
            vVal2_[6] = M0M1 * ptr2[2];
            vVal2_[7] = M0M1 * ptr2[3];
            for(int j = 0; j < 8; ++j)
              vVal1_[j] += M[0][0]*vVal2_[j];


            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            M0M1 = M[1][2];
            vVal2_[0] = M0M1 * ptr[0];
            vVal2_[1] = M0M1 * ptr[1];
            vVal2_[2] = M0M1 * ptr[2];
            vVal2_[3] = M0M1 * ptr[3];
            M0M1 = M[1][3];
            vVal2_[4] = M0M1 * ptr2[0];
            vVal2_[5] = M0M1 * ptr2[1];
            vVal2_[6] = M0M1 * ptr2[2];
            vVal2_[7] = M0M1 * ptr2[3];
            for(int j = 0; j < 8; ++j)
              vVal1_[j] += M[0][0]*vVal2_[j];

            //for(int j = 0; j < 8; ++j)
            //  std::cout << "[" << j << "] = " << vVal1_[j] << std::endl;
            //do{}while(1);

            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            M0M1 = M[1][0];
            vVal2_[0] = M0M1 * ptr[0];
            vVal2_[1] = M0M1 * ptr[1];
            vVal2_[2] = M0M1 * ptr[2];
            vVal2_[3] = M0M1 * ptr[3];
            M0M1 = M[1][1];
            vVal2_[4] = M0M1 * ptr2[0];
            vVal2_[5] = M0M1 * ptr2[1];
            vVal2_[6] = M0M1 * ptr2[2];
            vVal2_[7] = M0M1 * ptr2[3];
            for(int j = 0; j < 8; ++j)
              vVal1_[j] += M[0][1]*vVal2_[j];


            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            M0M1 = M[1][2];
            vVal2_[0] = M0M1 * ptr[0];
            vVal2_[1] = M0M1 * ptr[1];
            vVal2_[2] = M0M1 * ptr[2];
            vVal2_[3] = M0M1 * ptr[3];
            M0M1 = M[1][3];
            vVal2_[4] = M0M1 * ptr2[0];
            vVal2_[5] = M0M1 * ptr2[1];
            vVal2_[6] = M0M1 * ptr2[2];
            vVal2_[7] = M0M1 * ptr2[3];
            for(int j = 0; j < 8; ++j)
              vVal1_[j] += M[0][1]*vVal2_[j];

            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            M0M1 = M[1][0];
            vVal2_[0] = M0M1 * ptr[0];
            vVal2_[1] = M0M1 * ptr[1];
            vVal2_[2] = M0M1 * ptr[2];
            vVal2_[3] = M0M1 * ptr[3];
            M0M1 = M[1][1];
            vVal2_[4] = M0M1 * ptr2[0];
            vVal2_[5] = M0M1 * ptr2[1];
            vVal2_[6] = M0M1 * ptr2[2];
            vVal2_[7] = M0M1 * ptr2[3];
            for(int j = 0; j < 8; ++j)
              vVal1_[j] += M[0][2]*vVal2_[j];

            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            M0M1 = M[1][2];
            vVal2_[0] = M0M1 * ptr[0];
            vVal2_[1] = M0M1 * ptr[1];
            vVal2_[2] = M0M1 * ptr[2];
            vVal2_[3] = M0M1 * ptr[3];
            M0M1 = M[1][3];
            vVal2_[4] = M0M1 * ptr2[0];
            vVal2_[5] = M0M1 * ptr2[1];
            vVal2_[6] = M0M1 * ptr2[2];
            vVal2_[7] = M0M1 * ptr2[3];
            for(int j = 0; j < 8; ++j)
              vVal1_[j] += M[0][2]*vVal2_[j];

            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            M0M1 = M[1][0];
            vVal2_[0] = M0M1 * ptr[0];
            vVal2_[1] = M0M1 * ptr[1];
            vVal2_[2] = M0M1 * ptr[2];
            vVal2_[3] = M0M1 * ptr[3];
            M0M1 = M[1][1];
            vVal2_[4] = M0M1 * ptr2[0];
            vVal2_[5] = M0M1 * ptr2[1];
            vVal2_[6] = M0M1 * ptr2[2];
            vVal2_[7] = M0M1 * ptr2[3];
            for(int j = 0; j < 8; ++j)
              vVal1_[j] += M[0][3]*vVal2_[j];

            ptr = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            ptr2 = &reg_grid_vals[indx + indxx];
            indx += isize_g[2];
            M0M1 = M[1][2];
            vVal2_[0] = M0M1 * ptr[0];
            vVal2_[1] = M0M1 * ptr[1];
            vVal2_[2] = M0M1 * ptr[2];
            vVal2_[3] = M0M1 * ptr[3];
            M0M1 = M[1][3];
            vVal2_[4] = M0M1 * ptr2[0];
            vVal2_[5] = M0M1 * ptr2[1];
            vVal2_[6] = M0M1 * ptr2[2];
            vVal2_[7] = M0M1 * ptr2[3];
            for(int j = 0; j < 8; ++j)
              vVal1_[j] += M[0][3]*vVal2_[j];

            val = 0;
            for(int j = 0; j < 4; ++j)
              val+=vVal1_[j]*M[2][j];
            for(int j = 0; j < 4; ++j)
              val+=vVal1_[j+4]*M[2][j];
            //for(int j = 0; j < 4; ++j)
            //  std::cout << "[" << j << "] = " << vVal1_[j] * M[2][j] << std::endl;
            //for(int j = 0; j < 4; ++j)
            //  std::cout << "[" << j << "] = " << vVal1_[j+4] * M[2][j] << std::endl;

  		//}
  		query_values[i] = val;
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p

  void gold_optimized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			//while (grid_indx[j] < 0)
  			//	grid_indx[j] += N_reg_g[j];
  		}
  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
  		Real val = 0;
      int indx = 0;
  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            Real M0M1 = M[0][0]*M[1][0];
            register Real val_;
            Real* ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][0]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][0]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][0]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][1]*M[1][0];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][1]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][1]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][1]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][2]*M[1][0];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][2]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][2]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][2]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][3]*M[1][0];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][3]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][3]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][3]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
  		//}
  		query_values[i] = val;
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p

  void ___optimized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			//while (grid_indx[j] < 0)
  			//	grid_indx[j] += N_reg_g[j];
  		}
  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
  		Real val = 0;
      int indx = 0;
  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            Real M0M1_[2];
            Real vM[8];

            M0M1_[0] = M[0][0]*M[1][0];
            M0M1_[1] = M[0][0]*M[1][1];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            //register Real val_;
            Real vVal[8]={0};
            Real vVal2[8]={0};
            Real* ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];


            M0M1_[0] = M[0][0]*M[1][2];
            M0M1_[1] = M[0][0]*M[1][3];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1_[0] = M[0][1]*M[1][0];
            M0M1_[1] = M[0][1]*M[1][1];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];

            M0M1_[0] = M[0][1]*M[1][2];
            M0M1_[1] = M[0][1]*M[1][3];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1_[0] = M[0][1]*M[1][0];
            M0M1_[1] = M[0][1]*M[1][1];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];

            M0M1_[0] = M[0][1]*M[1][2];
            M0M1_[1] = M[0][1]*M[1][3];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];
        // ------------------------------------ //
            M0M1_[0] = M[0][3]*M[1][0];
            M0M1_[1] = M[0][3]*M[1][1];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            indx += isize_g[2];

            M0M1_[0] = M[0][3]*M[1][2];
            M0M1_[1] = M[0][3]*M[1][3];
            vM[0]=M0M1_[0];vM[1]=M0M1_[0];vM[2]=M0M1_[0];vM[3]=M0M1_[0];vM[4]=M0M1_[1];vM[5]=M0M1_[1];vM[6]=M0M1_[1];vM[7]=M0M1_[1];
            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[0] = M[2][0] * ptr[0];
  					vVal2[1] = M[2][1] * ptr[1];
  					vVal2[2] = M[2][2] * ptr[2];
  					vVal2[3] = M[2][3] * ptr[3];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					vVal2[4] = M[2][0] * ptr[0];
  					vVal2[5] = M[2][1] * ptr[1];
  					vVal2[6] = M[2][2] * ptr[2];
  					vVal2[7] = M[2][3] * ptr[3];
            for(int k = 0; k < 8; ++k)
              vVal2[k] *= vM[k];
            for(int k = 0; k < 8; ++k)
              vVal[k] += vVal2[k];
            val += (vVal[0]+vVal[1]+vVal[2]+vVal[3]); // * M0M1_[0];
            val += (vVal[4]+vVal[5]+vVal[6]+vVal[7]); // * M0M1_[1];
  		//}
  		query_values[i] = val;
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p

  void _v4_optimized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
      {
  #ifdef VERBOSE2
  		std::cout<<"q[0]="<<query_points[i*3+0]<<std::endl;
  		std::cout<<"q[1]="<<query_points[i*3+1]<<std::endl;
  		std::cout<<"q[2]="<<query_points[i*3+2]<<std::endl;
  #endif
    }
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			//while (grid_indx[j] < 0)
  			//	grid_indx[j] += N_reg_g[j];
  		}
  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
  		Real val = 0;
      int indx = 0;
  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            Real M0M1 = M[0][0]*M[1][0];
            register Real val_;
            Real* ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][0]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][0]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][0]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][1]*M[1][0];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][1]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][1]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][1]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][2]*M[1][0];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][2]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][2]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][2]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];
            indx += isize_g[1]*isize_g[2] - 4 * isize_g[2];

        // ------------------------------------ //
            M0M1 = M[0][3]*M[1][0];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][3]*M[1][1];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];


            M0M1 = M[0][3]*M[1][2];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
            indx += isize_g[2];

            M0M1 = M[0][3]*M[1][3];
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val += val_ * M0M1;
  		//}
  		query_values[i] = val;
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p

  void _optimized_interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;

  	for (int i = 0; i < N_pts; i++) {
      {
  #ifdef VERBOSE2
  		std::cout<<"q[0]="<<query_points[i*3+0]<<std::endl;
  		std::cout<<"q[1]="<<query_points[i*3+1]<<std::endl;
  		std::cout<<"q[2]="<<query_points[i*3+2]<<std::endl;
  #endif
    }
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			//while (grid_indx[j] < 0)
  			//	grid_indx[j] += N_reg_g[j];
  		}
  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}


  		const int indxx = isize_g[2] * isize_g[1] * grid_indx[0] + grid_indx[2] + isize_g[2] * grid_indx[1] ;
      register Real val_j0[4] = {0};
      int indx = 0;
  		//for (int j0 = 0; j0 < 4; j0++) {
        // ------------------------------------ //
            register Real val_ = 0;
            Real* ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[0] += val_ * M[1][0];
            indx += isize_g[2];


            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[0] += val_ * M[1][1];
            indx += isize_g[2];


            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[0] += val_ * M[1][2];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[0] += val_ * M[1][3];
            indx += isize_g[1]*isize_g[2] - 3 * isize_g[2];

        // ------------------------------------ //
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[1] += val_ * M[1][0];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[1] += val_ * M[1][1];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[1] += val_ * M[1][2];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[1] += val_ * M[1][3];
            indx += isize_g[1]*isize_g[2] - 3 * isize_g[2];

        // ------------------------------------ //
            val_ = 0;
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[2] += val_ * M[1][0];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[2] += val_ * M[1][1];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[2] += val_ * M[1][2];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[2] += val_ * M[1][3];
            indx += isize_g[1]*isize_g[2] - 3 * isize_g[2];

        // ------------------------------------ //
            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[3] += val_ * M[1][0];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[3] += val_ * M[1][1];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[3] += val_ * M[1][2];
            indx += isize_g[2];

            ptr = &reg_grid_vals[indx + indxx];
  					val_ = M[2][0] * ptr[0];
  					val_ += M[2][1] * ptr[1];
  					val_ += M[2][2] * ptr[2];
  					val_ += M[2][3] * ptr[3];
            val_j0[3] += val_ * M[1][3];
            query_values[i]  = M[0][0]*val_j0[0] + M[0][1]*val_j0[1] + M[0][2]*val_j0[2] + M[0][3]*val_j0[3];
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p


  /*
   * Performs a parallel 3D cubic interpolation for a row major periodic input (x \in [0,1) )
   * This function assumes that the input grid values have been padded on all sides
   * by g_size grids.
   * @param[in] reg_grid_vals The function value at the regular grid
   * @param[in] data_dof The degrees of freedom of the input function. In general
   * you can input a vector to be interpolated. In that case, each dimension of the
   * vector should be stored in a linearized contiguous order. That is the first dimension
   * should be stored in reg_grid_vals and then the second dimension, ...
   *
   * @param[in] N_reg: The size of the original grid in each dimension.
   * @param[in] isize: The locally owned sizes that each process owns
   * @param[in] istart: The start index of each process in the global array
   * @param[in] N_pts The number of query points
   * @param[in] g_size The number of ghost points padded around the input array
   *
   * @param[in] query_points_in The coordinates of the query points where the interpolated values are sought
   * One must store the coordinate values back to back. That is each 3 consecutive values in its array
   * determine the x,y, and z coordinate of 1 query point in 3D.
   *
   * @param[out] query_values The interpolated values
   *
   * @param[in] c_dims: Size of the cartesian MPI communicator
   * @param[in] c_comm: MPI Communicator
   *
   */

  void par_interp3_ghost_xyz_p(Real* ghost_reg_grid_vals, int data_dof,
  		int* N_reg, int * isize, int* istart, const int N_pts, const int g_size,
  		Real* query_points_in, Real* query_values, int* c_dims,
  		MPI_Comm c_comm) {
  	int nprocs, procid;
  	MPI_Comm_rank(c_comm, &procid);
  	MPI_Comm_size(c_comm, &nprocs);

  	int N_reg_g[3], isize_g[3];
  	N_reg_g[0] = N_reg[0] + 2 * g_size;
  	N_reg_g[1] = N_reg[1] + 2 * g_size;
  	N_reg_g[2] = N_reg[2] + 2 * g_size;

  	isize_g[0] = isize[0] + 2 * g_size;
  	isize_g[1] = isize[1] + 2 * g_size;
  	isize_g[2] = isize[2] + 2 * g_size;

  	Real h[3]; // original grid size along each axis
  	h[0] = 1. / N_reg[0];
  	h[1] = 1. / N_reg[1];
  	h[2] = 1. / N_reg[2];

  	// We copy query_points_in to query_points to aviod overwriting the input coordinates
  	Real* query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  	memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  	// Enforce periodicity
  	for (int i = 0; i < N_pts; i++) {
  		while (query_points[i * COORD_DIM + 0] <= -h[0]) {
  			query_points[i * COORD_DIM + 0] = query_points[i * COORD_DIM + 0]
  					+ 1;
  		}
  		while (query_points[i * COORD_DIM + 1] <= -h[1]) {
  			query_points[i * COORD_DIM + 1] = query_points[i * COORD_DIM + 1]
  					+ 1;
  		}
  		while (query_points[i * COORD_DIM + 2] <= -h[2]) {
  			query_points[i * COORD_DIM + 2] = query_points[i * COORD_DIM + 2]
  					+ 1;
  		}

  		while (query_points[i * COORD_DIM + 0] >= 1) {
  			query_points[i * COORD_DIM + 0] = query_points[i * COORD_DIM + 0]
  					- 1;
  		}
  		while (query_points[i * COORD_DIM + 1] >= 1) {
  			query_points[i * COORD_DIM + 1] = query_points[i * COORD_DIM + 1]
  					- 1;
  		}
  		while (query_points[i * COORD_DIM + 2] >= 1) {
  			query_points[i * COORD_DIM + 2] = query_points[i * COORD_DIM + 2]
  					- 1;
  		}
  	}

  	// Compute the start and end coordinates that this processor owns
  	Real iX0[3], iX1[3];
  	for (int j = 0; j < 3; j++) {
  		iX0[j] = istart[j] * h[j];
  		iX1[j] = iX0[j] + (isize[j] - 1) * h[j];
  	}

  	// Now march through the query points and split them into nprocs parts.
  	// These are stored in query_outside which is an array of vectors of size nprocs.
  	// That is query_outside[i] is a vector that contains the query points that need to
  	// be sent to process i. Obviously for the case of query_outside[procid], we do not
  	// need to send it to any other processor, as we own the necessary information locally,
  	// and interpolation can be done locally.
  	int Q_local = 0, Q_outside = 0;

  	// This is needed for one-to-one correspondence with output f. This is becaues we are reshuffling
  	// the data according to which processor it land onto, and we need to somehow keep the original
  	// index to write the interpolation data back to the right location in the output.
  	std::vector<int> f_index[nprocs];
  	std::vector<Real> query_outside[nprocs];
  	for (int i = 0; i < N_pts; i++) {
  		// The if condition check whether the query points fall into the locally owned domain or not
  		if (iX0[0] - h[0] <= query_points[i * COORD_DIM + 0]
  				&& query_points[i * COORD_DIM + 0] <= iX1[0] + h[0]
  				&& iX0[1] - h[1] <= query_points[i * COORD_DIM + 1]
  				&& query_points[i * COORD_DIM + 1] <= iX1[1] + h[1]
  				&& iX0[2] - h[2] <= query_points[i * COORD_DIM + 2]
  				&& query_points[i * COORD_DIM + 2] <= iX1[2] + h[2]) {
  			query_outside[procid].push_back(query_points[i * COORD_DIM + 0]);
  			query_outside[procid].push_back(query_points[i * COORD_DIM + 1]);
  			query_outside[procid].push_back(query_points[i * COORD_DIM + 2]);
  			f_index[procid].push_back(i);
  			Q_local++;
  			//PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
  			continue;
  		} else {
  			// If the point does not reside in the processor's domain then we have to
  			// first compute which processor owns the point. After computing that
  			// we add the query point to the corresponding vector.
  			int dproc0 = (int) (query_points[i * COORD_DIM + 0] / h[0])
  					/ isize[0];
  			int dproc1 = (int) (query_points[i * COORD_DIM + 1] / h[1])
  					/ isize[1];
  			int proc = dproc0 * c_dims[1] + dproc1; // Compute which proc has to do the interpolation
  			//PCOUT<<"proc="<<proc<<std::endl;
  			query_outside[proc].push_back(query_points[i * COORD_DIM + 0]);
  			query_outside[proc].push_back(query_points[i * COORD_DIM + 1]);
  			query_outside[proc].push_back(query_points[i * COORD_DIM + 2]);
  			f_index[proc].push_back(i);
  			Q_outside++;
  			//PCOUT<<"j=0 else ---------- i="<<i<<std::endl;
  			continue;
  		}

  	}

  	// Now we need to send the query_points that land onto other processor's domain.
  	// This done using a sparse alltoallv.
  	// Right now each process knows how much data to send to others, but does not know
  	// how much data it should receive. This is a necessary information both for the MPI
  	// command as well as memory allocation for received data.
  	// So we first do an alltoall to get the f_index[proc].size from all processes.
  	int f_index_procs_self_sizes[nprocs]; // sizes of the number of interpolations that need to be sent to procs
  	int f_index_procs_others_sizes[nprocs]; // sizes of the number of interpolations that need to be received from procs

  	for (int proc = 0; proc < nprocs; proc++) {
  		if (!f_index[proc].empty())
  			f_index_procs_self_sizes[proc] = f_index[proc].size();
  		else
  			f_index_procs_self_sizes[proc] = 0;
  	}
  	MPI_Alltoall(f_index_procs_self_sizes, 1, MPI_INT,
  			f_index_procs_others_sizes, 1, MPI_INT, c_comm);

  #ifdef VERBOSE2
  	sleep(1);
  	if(procid==0) {
  		std::cout<<"procid="<<procid<<std::endl;
  		std::cout<<"f_index_procs_self[0]="<<f_index_procs_self_sizes[0]<<" [1]= "<<f_index_procs_self_sizes[1]<<std::endl;
  		std::cout<<"f_index_procs_others[0]="<<f_index_procs_others_sizes[0]<<" [1]= "<<f_index_procs_others_sizes[1]<<std::endl;
  	}
  	sleep(1);
  	if(procid==1) {
  		std::cout<<"procid="<<procid<<std::endl;
  		std::cout<<"f_index_procs_self[0]="<<f_index_procs_self_sizes[0]<<" [1]= "<<f_index_procs_self_sizes[1]<<std::endl;
  		std::cout<<"f_index_procs_others[0]="<<f_index_procs_others_sizes[0]<<" [1]= "<<f_index_procs_others_sizes[1]<<std::endl;
  	}
  #endif

  	// Now we need to allocate memory for the receiving buffer of all query
  	// points including ours. This is simply done by looping through
  	// f_index_procs_others_sizes and adding up all the sizes.
  	// Note that we would also need to know the offsets.
  	size_t all_query_points_allocation = 0;
  	int f_index_procs_others_offset[nprocs]; // offset in the all_query_points array
  	int f_index_procs_self_offset[nprocs]; // offset in the query_outside array
  	f_index_procs_others_offset[0] = 0;
  	f_index_procs_self_offset[0] = 0;
  	for (int proc = 0; proc < nprocs; ++proc) {
  		// The reason we multiply by COORD_DIM is that we have three coordinates per interpolation request
  		all_query_points_allocation += f_index_procs_others_sizes[proc]
  				* COORD_DIM;
  		if (proc > 0) {
  			f_index_procs_others_offset[proc] = f_index_procs_others_offset[proc
  					- 1] + f_index_procs_others_sizes[proc - 1];
  			f_index_procs_self_offset[proc] =
  					f_index_procs_self_offset[proc - 1]
  							+ f_index_procs_self_sizes[proc - 1];
  		}
  	}
  	int total_query_points = all_query_points_allocation / COORD_DIM;
  	Real * all_query_points = (Real*) malloc(
  			all_query_points_allocation * sizeof(Real));
  #ifdef VERBOSE2
  	if(procid==0) {
  		std::cout<<"procid="<<procid<<std::endl;
  		for (int proc=0;proc<nprocs;++proc)
  		std::cout<<"proc= "<<proc<<" others_offset= "<<f_index_procs_others_offset[proc]<<" others_sizes= "<<f_index_procs_others_sizes[proc]<<std::endl;
  		for (int proc=0;proc<nprocs;++proc)
  		std::cout<<"proc= "<<proc<<" self_offset= "<<f_index_procs_self_offset[proc]<<" self_sizes= "<<f_index_procs_self_sizes[proc]<<std::endl;
  	}
  #endif

  	MPI_Request * s_request = new MPI_Request[nprocs];
  	MPI_Request * request = new MPI_Request[nprocs];

  	// Now perform the allotall to send/recv query_points
  	{
  		int dst_r, dst_s;
  		for (int i = 0; i < nprocs; ++i) {
  			dst_r = i; //(procid+i)%nprocs;
  			dst_s = i; //(procid-i+nprocs)%nprocs;
  			s_request[dst_s] = MPI_REQUEST_NULL;
  			request[dst_r] = MPI_REQUEST_NULL;
  			int roffset = f_index_procs_others_offset[dst_r] * COORD_DIM; // notice that COORD_DIM is needed because query_points are 3 times f
  			//int soffset = f_index_procs_self_offset[dst_s] * COORD_DIM;
  			if (f_index_procs_others_sizes[dst_r] != 0)
  				MPI_Irecv(&all_query_points[roffset],
  						f_index_procs_others_sizes[dst_r] * COORD_DIM, MPI_T,
  						dst_r, 0, c_comm, &request[dst_r]);
  			if (!query_outside[dst_s].empty())
  				MPI_Isend(&query_outside[dst_s][0],
  						f_index_procs_self_sizes[dst_s] * COORD_DIM, MPI_T,
  						dst_s, 0, c_comm, &s_request[dst_s]);
  			//if(procid==1){
  			//std::cout<<"soffset="<<soffset<<" roffset="<<roffset<<std::endl;
  			//std::cout<<"f_index_procs_self_sizes[0]="<<f_index_procs_self_sizes[0]<<std::endl;
  			//std::cout<<"f_index_procs_others_sizes[0]="<<f_index_procs_others_sizes[0]<<std::endl;
  			//std::cout<<"q_outside["<<dst_s<<"]="<<query_outside[dst_s][0]<<std::endl;
  			//}
  		}
  		// Wait for all the communication to finish
  		MPI_Status ierr;
  		for (int proc = 0; proc < nprocs; ++proc) {
  			if (request[proc] != MPI_REQUEST_NULL)
  				MPI_Wait(&request[proc], &ierr);
  			if (s_request[proc] != MPI_REQUEST_NULL)
  				MPI_Wait(&s_request[proc], &ierr);
  		}
  	}

  	//if(procid==1){
  	//  std::cout<<"total_query_points="<<total_query_points<<std::endl;
  	//  std::cout<<"----- procid="<<procid<<" Q="<<all_query_points[0]<<" "<<all_query_points[1]<<" "<<all_query_points[2]<<std::endl;
  	//  std::cout<<"----- procid="<<procid<<" Q="<<all_query_points[3]<<" "<<all_query_points[4]<<" "<<all_query_points[5]<<std::endl;
  	//}
  	//PCOUT<<"**** Q_local="<<Q_local<<" f_index_procs_self_sizes[procid]="<<f_index_procs_self_sizes[procid]<<std::endl;
  	//int dum=0;
  	//for (int i=0;i<Q_local;i++){
  	//  dum+=query_local[i*COORD_DIM+0]-all_query_points[f_index_procs_others_offset[procid]*3+i*COORD_DIM+0];
  	//  dum+=query_local[i*COORD_DIM+1]-all_query_points[f_index_procs_others_offset[procid]*3+i*COORD_DIM+1];
  	//  dum+=query_local[i*COORD_DIM+2]-all_query_points[f_index_procs_others_offset[procid]*3+i*COORD_DIM+2];
  	//}

  	// Now perform the interpolation on all query points including those that need to
  	// be sent to other processors and store them into all_f_cubic
  	Real* all_f_cubic = (Real*) malloc(
  			total_query_points * sizeof(Real) * data_dof);
  	interp3_ghost_xyz_p(ghost_reg_grid_vals, data_dof, N_reg, N_reg_g, isize_g,
  			istart, total_query_points, g_size, all_query_points, all_f_cubic);

  	//if(procid==0){
  	//  std::cout<<"total_query_points="<<total_query_points<<std::endl;
  	//  std::cout<<"procid="<<procid<<" Q="<<all_query_points[0]<<" "<<all_query_points[1]<<" "<<all_query_points[2]<<" f= "<<all_f_cubic[0]<<std::endl;
  	//  std::cout<<"procid="<<procid<<" Q="<<all_query_points[3]<<" "<<all_query_points[4]<<" "<<all_query_points[5]<<" f= "<<all_f_cubic[1]<<std::endl;
  	//}

  	// Now we have to do an alltoall to distribute the interpolated data from all_f_cubic to
  	// f_cubic_unordered.
  	Real * f_cubic_unordered = (Real*) malloc(N_pts * sizeof(Real) * data_dof); // The reshuffled semi-final interpolated values are stored here
  	{
  		//PCOUT<<"total_query_points="<<total_query_points<<" N_pts="<<N_pts<<std::endl;
  		int dst_r, dst_s;
  		MPI_Datatype stype[nprocs], rtype[nprocs];
  		for (int i = 0; i < nprocs; ++i) {
  			MPI_Type_vector(data_dof, f_index_procs_self_sizes[i], N_pts, MPI_T,
  					&rtype[i]);
  			MPI_Type_vector(data_dof, f_index_procs_others_sizes[i],
  					total_query_points, MPI_T, &stype[i]);
  			MPI_Type_commit(&stype[i]);
  			MPI_Type_commit(&rtype[i]);
  		}
  		for (int i = 0; i < nprocs; ++i) {
  			dst_r = i; //(procid+i)%nprocs;
  			dst_s = i; //(procid-i+nprocs)%nprocs;
  			s_request[dst_s] = MPI_REQUEST_NULL;
  			request[dst_r] = MPI_REQUEST_NULL;
  			// Notice that this is the adjoint of the first comm part
  			// because now you are sending others f and receiving your part of f
  			int soffset = f_index_procs_others_offset[dst_r];
  			int roffset = f_index_procs_self_offset[dst_s];
  			//if(procid==0)
  			//  std::cout<<"procid="<<procid<<" dst_s= "<<dst_s<<" soffset= "<<soffset<<" s_size="<<f_index_procs_others_sizes[dst_s]<<" dst_r= "<<dst_r<<" roffset="<<roffset<<" r_size="<<f_index_procs_self_sizes[dst_r]<<std::endl;
  			//if(f_index_procs_self_sizes[dst_r]!=0)
  			//  MPI_Irecv(&f_cubic_unordered[roffset],f_index_procs_self_sizes[dst_r],rtype, dst_r,
  			//      0, c_comm, &request[dst_r]);
  			//if(f_index_procs_others_sizes[dst_s]!=0)
  			//  MPI_Isend(&all_f_cubic[soffset],f_index_procs_others_sizes[dst_s],stype,dst_s,
  			//      0, c_comm, &s_request[dst_s]);
  			//
  			if (f_index_procs_self_sizes[dst_r] != 0)
  				MPI_Irecv(&f_cubic_unordered[roffset], 1, rtype[i], dst_r, 0,
  						c_comm, &request[dst_r]);
  			if (f_index_procs_others_sizes[dst_s] != 0)
  				MPI_Isend(&all_f_cubic[soffset], 1, stype[i], dst_s, 0, c_comm,
  						&s_request[dst_s]);
  		}
  		MPI_Status ierr;
  		for (int proc = 0; proc < nprocs; ++proc) {
  			if (request[proc] != MPI_REQUEST_NULL)
  				MPI_Wait(&request[proc], &ierr);
  			if (s_request[proc] != MPI_REQUEST_NULL)
  				MPI_Wait(&s_request[proc], &ierr);
  		}
  		for (int i = 0; i < nprocs; ++i) {
  			MPI_Type_free(&stype[i]);
  			MPI_Type_free(&rtype[i]);
  		}
  	}

  	// Now copy back f_cubic_unordered to f_cubic in the correct f_index
  	for (int dof = 0; dof < data_dof; ++dof) {
  		for (int proc = 0; proc < nprocs; ++proc) {
  			if (!f_index[proc].empty())
  				for (int i = 0; i < (int)f_index[proc].size(); ++i) {
  					int ind = f_index[proc][i];
  					//f_cubic[ind]=all_f_cubic[f_index_procs_others_offset[proc]+i];
  					query_values[ind + dof * N_pts] =
  							f_cubic_unordered[f_index_procs_self_offset[proc]
  									+ i + dof * N_pts];
  				}
  		}
  	}

  	free(query_points);
  	free(all_query_points);
  	free(all_f_cubic);
  	free(f_cubic_unordered);
  	delete[] s_request;
  	delete[] request;
  	//vector
  	for (int proc = 0; proc < nprocs; ++proc) {
  		std::vector<int>().swap(f_index[proc]);
  		std::vector<Real>().swap(query_outside[proc]);
  	}
  	return;
  } // end of par_interp3_ghost_xyz_p

  // the factor is computed by the following transform:
  // X=0 -> Y = ghp
  // X=1-h -> Y = 1-hp - ghp
  void rescale(const int g_size, int* N_reg, int* N_reg_g, int* istart,
  		const int N_pts, Real* query_points) {

  	if (g_size == 0)
  		return;
  	Real hp[3];
  	Real h[3];
  	hp[0] = 1. / N_reg_g[0]; // New mesh size
  	hp[1] = 1. / N_reg_g[1]; // New mesh size
  	hp[2] = 1. / N_reg_g[2]; // New mesh size

  	h[0] = 1. / (N_reg[0]); // old mesh size
  	h[1] = 1. / (N_reg[1]); // old mesh size
  	h[2] = 1. / (N_reg[2]); // old mesh size

  	Real factor[3];
  	factor[0] = (1. - (2. * g_size + 1.) * hp[0]) / (1. - h[0]);
  	factor[1] = (1. - (2. * g_size + 1.) * hp[1]) / (1. - h[1]);
  	factor[2] = (1. - (2. * g_size + 1.) * hp[2]) / (1. - h[2]);
  	for (int i = 0; i < N_pts; i++) {
  		query_points[0 + COORD_DIM * i] = (query_points[0 + COORD_DIM * i]
  				- istart[0] * h[0]) * factor[0] + g_size * hp[0];
  		query_points[1 + COORD_DIM * i] = (query_points[1 + COORD_DIM * i]
  				- istart[1] * h[1]) * factor[1] + g_size * hp[1];
  		//query_points[2+COORD_DIM*i]=(query_points[2+COORD_DIM*i]-istart[2]*h[2])*factor[2]+g_size*hp[2];
  	}
  	return;
  } // end of rescale

  /*
   * Performs a 3D cubic interpolation for a row major periodic input (x \in [0,1) )
   * This function assumes that the input grid values have been padded on all sides
   * by g_size grids.
   * @param[in] reg_grid_vals The function value at the regular grid
   * @param[in] data_dof The degrees of freedom of the input function. In general
   * you can input a vector to be interpolated. In that case, each dimension of the
   * vector should be stored in a linearized contiguous order. That is the first dimension
   * should be stored in reg_grid_vals and then the second dimension, ...
   *
   * @param[in] N_reg An integer pointer that specifies the size of the grid in each dimension.
   *
   * @param[in] N_pts The number of query points
   *
   * @param[in] g_size The number of ghost points padded around the input array
   *
   * @param[in] query_points The coordinates of the query points where the interpolated values are sought.
   * One must store the coordinate values back to back. That is each 3 consecutive values in its array
   * determine the x,y, and z coordinate of 1 query point in 3D.
   *
   * @param[out] query_values The interpolated values
   * snafu
   *
   */

  void interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[4];
    lagr_denom[0] = -1.0/6.0;
    lagr_denom[1] = 0.5;
    lagr_denom[2] = -0.5;
    lagr_denom[3] = 1.0/6.0;
  	int N_reg3 = isize_g[0] * isize_g[1] * isize_g[2];

  	for (int i = 0; i < N_pts; i++) {
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j];// * N_reg_g[j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			while (grid_indx[j] < 0)
  				grid_indx[j] += N_reg_g[j];
  		}

  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}

  		for (int k = 0; k < data_dof; k++) {
  			Real val = 0;
  			for (int j2 = 0; j2 < 4; j2++) {
  				for (int j1 = 0; j1 < 4; j1++) {
  					for (int j0 = 0; j0 < 4; j0++) {
  						int indx = ((grid_indx[2] + j2) % isize_g[2])
  								+ isize_g[2]
  										* ((grid_indx[1] + j1) % isize_g[1])
  								+ isize_g[2] * isize_g[1]
  										* ((grid_indx[0] + j0) % isize_g[0]);
  						//val += M[0][j0] * M[1][j1] * M[2][j2]
  						//		* reg_grid_vals[0];
  						val += M[0][j0] * M[1][j1] * M[2][j2]
  								* reg_grid_vals[indx + k * N_reg3];
  					}
  				}
  			}
  			//query_values[0] = val;
  			query_values[i + k * N_pts] = val;
  		}
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p

  /*
   * Performs a 3D cubic interpolation for a row major periodic input (x \in [0,1) )
   * This function assumes that the input grid values have been padded on all sides
   * by g_size grids.
   * @param[in] reg_grid_vals The function value at the regular grid
   * @param[in] data_dof The degrees of freedom of the input function. In general
   * you can input a vector to be interpolated. In that case, each dimension of the
   * vector should be stored in a linearized contiguous order. That is the first dimension
   * should be stored in reg_grid_vals and then the second dimension, ...
   *
   * @param[in] N_reg An integer pointer that specifies the size of the grid in each dimension.
   *
   * @param[in] N_pts The number of query points
   *
   * @param[in] g_size The number of ghost points padded around the input array
   *
   * @param[in] query_points The coordinates of the query points where the interpolated values are sought.
   * One must store the coordinate values back to back. That is each 3 consecutive values in its array
   * determine the x,y, and z coordinate of 1 query point in 3D.
   *
   * @param[in] interp_order The order of interpolation (e.g. 3 for cubic)
   * @param[out] query_values The interpolated values
   *
   */

  void interp3_ghost_xyz_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values,
      int interp_order,
  		bool query_values_already_scaled) {
  	Real* query_points;

  	if (query_values_already_scaled == false) {
  		// First we need to rescale the query points to the new padded dimensions
  		// To avoid changing the user's input we first copy the query points to a
  		// new array
  		query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  		memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  		rescale_xyz(g_size, N_reg, N_reg_g, istart, N_pts, query_points);
  	} else {
  		query_points = query_points_in;
  	}
  	Real lagr_denom[interp_order + 1];
  	for (int i = 0; i < interp_order + 1; i++) {
  		lagr_denom[i] = 1;
  		for (int j = 0; j < interp_order + 1; j++) {
  			if (i != j)
  				lagr_denom[i] /= (Real) (i - j);
  		}
  	}

  	int N_reg3 = isize_g[0] * isize_g[1] * isize_g[2];

  	for (int i = 0; i < N_pts; i++) {
  #ifdef VERBOSE2
  		std::cout<<"q[0]="<<query_points[i*3+0]<<std::endl;
  		std::cout<<"q[1]="<<query_points[i*3+1]<<std::endl;
  		std::cout<<"q[2]="<<query_points[i*3+2]<<std::endl;
  #endif
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];

  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j] * N_reg_g[j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			while (grid_indx[j] < 0)
  				grid_indx[j] += N_reg_g[j];
  		}

  #ifdef VERBOSE2
  		std::cout<<"***** grid_index="<<grid_indx[0]<<" "<<grid_indx[1]<<" "<<grid_indx[2]<<std::endl;
  		std::cout<<"***** point="<<point[0]<<" "<<point[1]<<" "<<point[2]<<std::endl;
  		std::cout<<"f @grid_index="<<reg_grid_vals[grid_indx[0]*isize_g[1]*isize_g[2]+grid_indx[1]*isize_g[2]+grid_indx[2]]<<std::endl;
  		std::cout<<"hp= "<<1./N_reg_g[0]<<std::endl;
  		std::cout<<"N_reg_g= "<<N_reg_g[0]<<" "<<N_reg_g[1]<<" "<<N_reg_g[2]<<std::endl;
  #endif

  		Real M[3][interp_order + 1];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < interp_order + 1; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < interp_order + 1; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}

  		for (int k = 0; k < data_dof; k++) {
  			Real val = 0;
  			for (int j2 = 0; j2 < interp_order + 1; j2++) {
  				for (int j1 = 0; j1 < interp_order + 1; j1++) {
  					for (int j0 = 0; j0 < interp_order + 1; j0++) {
  						int indx = ((grid_indx[2] + j2) % isize_g[2])
  								+ isize_g[2]
  										* ((grid_indx[1] + j1) % isize_g[1])
  								+ isize_g[2] * isize_g[1]
  										* ((grid_indx[0] + j0) % isize_g[0]);
  						val += M[0][j0] * M[1][j1] * M[2][j2]
  								* reg_grid_vals[indx + k * N_reg3];
  					}
  				}
  			}
  			query_values[i + k * N_pts] = val;
  		}
  	}

  	if (query_values_already_scaled == false) {
  		free(query_points);
  	}
  	return;

  }  // end of interp3_ghost_xyz_p

  /*
   * Performs a 3D cubic interpolation for a row major periodic input (x \in [0,1) )
   * This function assumes that the input grid values have been padded on x, and y directions
   * by g_size grids.
   * @param[in] reg_grid_vals The function value at the regular grid
   * @param[in] data_dof The degrees of freedom of the input function. In general
   * you can input a vector to be interpolated. In that case, each dimension of the
   * vector should be stored in a linearized contiguous order. That is the first dimension
   * should be stored in reg_grid_vals and then the second dimension, ...
   *
   * @param[in] N_reg An integer pointer that specifies the size of the grid in each dimension.
   *
   * @param[in] N_pts The number of query points
   *
   * @param[in] g_size The number of ghost points padded around the input array
   *
   * @param[in] query_points The coordinates of the query points where the interpolated values are sought.
   * One must store the coordinate values back to back. That is each 3 consecutive values in its array
   * determine the x,y, and z coordinate of 1 query point in 3D.
   *
   * @param[out] query_values The interpolated values
   *
   */

  void interp3_ghost_p(Real* reg_grid_vals, int data_dof, int* N_reg,
  		int* N_reg_g, int * isize_g, int* istart, const int N_pts,
  		const int g_size, Real* query_points_in, Real* query_values) {

  	// First we need to rescale the query points to the new padded dimensions
  	// To avoid changing the user's input we first copy the query points to a
  	// new array
  	Real* query_points = (Real*) malloc(N_pts * COORD_DIM * sizeof(Real));
  	memcpy(query_points, query_points_in, N_pts * COORD_DIM * sizeof(Real));
  	rescale(g_size, N_reg, N_reg_g, istart, N_pts, query_points);

  	//std::cout<<"N_reg[0]="<<N_reg[0]<<" N_reg[1]="<<N_reg[1]<<" N_reg[2]="<<N_reg[2]<<std::endl;
  	//std::cout<<"N_reg_g[0]="<<N_reg_g[0]<<" N_reg_g[1]="<<N_reg_g[1]<<" N_reg_g[2]="<<N_reg_g[2]<<std::endl;

  	Real lagr_denom[4];
  	for (int i = 0; i < 4; i++) {
  		lagr_denom[i] = 1;
  		for (int j = 0; j < 4; j++) {
  			if (i != j)
  				lagr_denom[i] /= (Real) (i - j);
  		}
  	}

  	int N_reg3 = isize_g[0] * isize_g[1] * isize_g[2];
  	//int N_pts=query_points.size()/COORD_DIM;

  	for (int i = 0; i < N_pts; i++) {
  #ifdef VERBOSE2
  		std::cout<<"q[0]="<<query_points[i*3+0]<<std::endl;
  		std::cout<<"q[1]="<<query_points[i*3+1]<<std::endl;
  		std::cout<<"q[2]="<<query_points[i*3+2]<<std::endl;
  #endif
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];
  		//grid_indx[0]=15;
  		//grid_indx[1]=15;
  		//grid_indx[2]=14;
  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j] * N_reg_g[j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			while (grid_indx[j] < 0)
  				grid_indx[j] += N_reg_g[j];
  		}
  #ifdef VERBOSE2
  		std::cout<<"***** grid_index="<<grid_indx[0]<<" "<<grid_indx[1]<<" "<<grid_indx[2]<<std::endl;
  		std::cout<<"***** point="<<point[0]<<" "<<point[1]<<" "<<point[2]<<std::endl;
  		std::cout<<"f @grid_index="<<reg_grid_vals[grid_indx[0]*isize_g[1]*isize_g[2]+grid_indx[1]*isize_g[2]+grid_indx[2]]<<std::endl;
  		std::cout<<"hp= "<<1./N_reg_g[0]<<std::endl;
  		std::cout<<"N_reg_g= "<<N_reg_g[0]<<" "<<N_reg_g[1]<<" "<<N_reg_g[2]<<std::endl;
  #endif

  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}

  		for (int k = 0; k < data_dof; k++) {
  			Real val = 0;
  			for (int j2 = 0; j2 < 4; j2++) {
  				for (int j1 = 0; j1 < 4; j1++) {
  					for (int j0 = 0; j0 < 4; j0++) {
  						//int indx = ((grid_indx[2]+j2)%N_reg) + N_reg*((grid_indx[1]+j1)%N_reg) + N_reg*N_reg*((grid_indx[0]+j0)%N_reg);
  						//int indx = ((grid_indx[2]+j2)%N_reg_g[2]) + N_reg_g[2]*((grid_indx[1]+j1)%N_reg_g[1]) + N_reg_g[2]*N_reg_g[1]*((grid_indx[0]+j0)%N_reg_g[0]);
  						int indx = ((grid_indx[2] + j2) % isize_g[2])
  								+ isize_g[2]
  										* ((grid_indx[1] + j1) % isize_g[1])
  								+ isize_g[2] * isize_g[1]
  										* ((grid_indx[0] + j0) % isize_g[0]);
  						val += M[0][j0] * M[1][j1] * M[2][j2]
  								* reg_grid_vals[indx + k * N_reg3];
  					}
  				}
  			}
  			query_values[i + k * N_pts] = val;
  		}
  	}
  	free(query_points);
  }            //end of interp3_ghost_p

  /*
   * Performs a 3D cubic interpolation for a row major periodic input (x \in [0,1) )
   * @param[in] reg_grid_vals The function value at the regular grid
   * @param[in] data_dof The degrees of freedom of the input function. In general
   * you can input a vector to be interpolated. In that case, each dimension of the
   * vector should be stored in a linearized contiguous order. That is the first dimension
   * should be stored in reg_grid_vals and then the second dimension, ...
   *
   * @param[in] N_reg An integer pointer that specifies the size of the grid in each dimension.
   *
   * @param[in] N_pts The number of query points
   *
   * @param[in] query_points The coordinates of the query points where the interpolated values are sought.
   * One must store the coordinate values back to back. That is each 3 consecutive values in its array
   * determine the x,y, and z coordinate of 1 query point in 3D.
   *
   * @param[out] query_values The interpolated values
   *
   */

  void interp3_p(Real* reg_grid_vals, int data_dof, int* N_reg, const int N_pts,
  		Real* query_points, Real* query_values) {

  	Real lagr_denom[4];
  	for (int i = 0; i < 4; i++) {
  		lagr_denom[i] = 1;
  		for (int j = 0; j < 4; j++) {
  			if (i != j)
  				lagr_denom[i] /= (Real) (i - j);
  		}
  	}

  	int N_reg3 = N_reg[0] * N_reg[1] * N_reg[2];
  	//int N_pts=query_points.size()/COORD_DIM;
  	//query_values.resize(N_pts*data_dof);

  	for (int i = 0; i < N_pts; i++) {
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];
  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j] * N_reg[j];
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			while (grid_indx[j] < 0)
  				grid_indx[j] += N_reg[j];
  		}
  		//std::cout<<"grid_index="<<grid_indx[0]<<" "<<grid_indx[1]<<" "<<grid_indx[2]<<std::endl;

  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}

  		for (int k = 0; k < data_dof; k++) {
  			Real val = 0;
  			for (int j2 = 0; j2 < 4; j2++) {
  				for (int j1 = 0; j1 < 4; j1++) {
  					for (int j0 = 0; j0 < 4; j0++) {
  						//int indx = ((grid_indx[2]+j2)%N_reg) + N_reg*((grid_indx[1]+j1)%N_reg) + N_reg*N_reg*((grid_indx[0]+j0)%N_reg);
  						int indx = ((grid_indx[2] + j2) % N_reg[2])
  								+ N_reg[2] * ((grid_indx[1] + j1) % N_reg[1])
  								+ N_reg[2] * N_reg[1]
  										* ((grid_indx[0] + j0) % N_reg[0]);
  						val += M[0][j0] * M[1][j1] * M[2][j2]
  								* reg_grid_vals[indx + k * N_reg3];
  					}
  				}
  			}
  			query_values[i + k * N_pts] = val;
  		}
  	}
  } // end of interp3_p

  /*
   * Performs a 3D cubic interpolation for a row major periodic input (x \in [0,1) ).
   * Limitation: The grid must be cubic, i.e. the number of grid points must be the same
   * in all dimensions.
   * @param[in] reg_grid_vals The function value at the regular grid
   * @param[in] data_dof The degrees of freedom of the input function. In general
   * you can input a vector to be interpolated. In that case, each dimension of the
   * vector should be stored in a linearized contiguous order. That is the first dimension
   * should be stored in reg_grid_vals and then the second dimension, ...
   *
   * @param[in] N_reg The size of the regular grid (The grid must have the same size in all dimensions)
   *
   * @param[in] N_pts The number of query points
   *
   * @param[in] query_points The coordinates of the query points where the interpolated values are sought.
   * One must store the coordinate values back to back. That is each 3 consecutive values in its array
   * determine the x,y, and z coordinate of 1 query point in 3D.
   *
   * @param[out] query_values The interpolated values
   *
   */

  void interp3_p(Real* reg_grid_vals, int data_dof, int N_reg, const int N_pts,
  		Real* query_points, Real* query_values) {

  	Real lagr_denom[4];
  	for (int i = 0; i < 4; i++) {
  		lagr_denom[i] = 1;
  		for (int j = 0; j < 4; j++) {
  			if (i != j)
  				lagr_denom[i] /= (Real) (i - j);
  		}
  	}

  	int N_reg3 = N_reg * N_reg * N_reg;
  	//int N_pts=query_points.size()/COORD_DIM;
  	//query_values.resize(N_pts*data_dof);

  	for (int i = 0; i < N_pts; i++) {
  		Real point[COORD_DIM];
  		int grid_indx[COORD_DIM];
  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j] * N_reg;
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			while (grid_indx[j] < 0)
  				grid_indx[j] += N_reg;
  		}
  		//std::cout<<"grid_index="<<grid_indx[0]<<" "<<grid_indx[1]<<" "<<grid_indx[2]<<std::endl;

  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}

  		for (int k = 0; k < data_dof; k++) {
  			Real val = 0;
  			for (int j2 = 0; j2 < 4; j2++) {
  				for (int j1 = 0; j1 < 4; j1++) {
  					for (int j0 = 0; j0 < 4; j0++) {
  						//int indx = ((grid_indx[0]+j0)%N_reg) + N_reg*((grid_indx[1]+j1)%N_reg) + N_reg*N_reg*((grid_indx[2]+j2)%N_reg);
  						int indx = ((grid_indx[2] + j2) % N_reg)
  								+ N_reg * ((grid_indx[1] + j1) % N_reg)
  								+ N_reg * N_reg * ((grid_indx[0] + j0) % N_reg);
  						val += M[0][j0] * M[1][j1] * M[2][j2]
  								* reg_grid_vals[indx + k * N_reg3];
  					}
  				}
  			}
  			query_values[i + k * N_pts] = val;
  		}
  	}
  } // end of interp3_p

  /*
   * Performs a 3D cubic interpolation for a column major periodic input (x \in [0,1) )
   * Limitation: The grid must be cubic, i.e. the number of grid points must be the same
   * in all dimensions.
   * @param[in] reg_grid_vals The function value at the regular grid
   * @param[in] data_dof The degrees of freedom of the input function. In general
   * you can input a vector to be interpolated. In that case, each dimension of the
   * vector should be stored in a linearized contiguous order. That is the first dimension
   * should be stored in reg_grid_vals and then the second dimension, ...
   *
   * @param[in] N_reg The size of the regular grid (The grid must have the same size in all dimensions)
   *
   * @param[in] N_pts The number of query points
   *
   * @param[in] query_points The coordinates of the query points where the interpolated values are sought.
   * One must store the coordinate values back to back. That is each 3 consecutive values in its array
   * determine the x,y, and z coordinate of 1 query point in 3D.
   *
   * @param[out] query_values The interpolated values
   *
   */
  void interp3_p_col(Real* reg_grid_vals, int data_dof, int N_reg,
  		const int N_pts, Real* query_points, Real* query_values) {

  	Real lagr_denom[4];
  	for (int i = 0; i < 4; i++) {
  		lagr_denom[i] = 1;
  		for (int j = 0; j < 4; j++) {
  			if (i != j)
  				lagr_denom[i] /= (Real) (i - j);
  		}
  	}

  	int N_reg3 = N_reg * N_reg * N_reg;

  	Real point[COORD_DIM];
  	int grid_indx[COORD_DIM];
  	for (int i = 0; i < N_pts; i++) {
  		for (int j = 0; j < COORD_DIM; j++) {
  			point[j] = query_points[COORD_DIM * i + j] * N_reg;
  			grid_indx[j] = (floor(point[j])) - 1;
  			point[j] -= grid_indx[j];
  			while (grid_indx[j] < 0)
  				grid_indx[j] += N_reg;
  		}

  		Real M[3][4];
  		for (int j = 0; j < COORD_DIM; j++) {
  			Real x = point[j];
  			for (int k = 0; k < 4; k++) {
  				M[j][k] = lagr_denom[k];
  				for (int l = 0; l < 4; l++) {
  					if (k != l)
  						M[j][k] *= (x - l);
  				}
  			}
  		}

  		for (int k = 0; k < data_dof; k++) {
  			Real val = 0;
  			for (int j2 = 0; j2 < 4; j2++) {
  				for (int j1 = 0; j1 < 4; j1++) {
  					for (int j0 = 0; j0 < 4; j0++) {
  						int indx = ((grid_indx[0] + j0) % N_reg)
  								+ N_reg * ((grid_indx[1] + j1) % N_reg)
  								+ N_reg * N_reg * ((grid_indx[2] + j2) % N_reg);
  						val += M[0][j0] * M[1][j1] * M[2][j2]
  								* reg_grid_vals[indx + k * N_reg3];
  					}
  				}
  			}
  			query_values[i + k * N_pts] = val;
  		}
  	}
  } // end of interp3_p_col


  /*
   * Get the left right ghost cells.
   *
   * @param[out] padded_data: The output of the function which pads the input array data, with the ghost
   * cells from left and right neighboring processors.
   * @param[in] data: Input data to be padded
   * @param[in] g_size: The size of the ghost cell padding. Note that it cannot exceed the neighboring processor's
   * local data size
   * @param[in] plan: AccFFT R2C plan
   */
  void ghost_left_right(pvfmm::Iterator<Real> padded_data, Real* data, int g_size,
      accfft_plan_t<Real, TC, PL> * plan) {
    int nprocs, procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  //  MPI_Comm c_comm = plan->c_comm;

    /* Get the local pencil size and the allocation size */
    //int isize[3],osize[3],istart[3],ostart[3];
    int * isize = plan->isize;
  //  int * osize = plan->isize;
  //  int * istart = plan->istart;
  //  int * ostart = plan->ostart;
  //  int alloc_max = plan->alloc_max;

    MPI_Comm row_comm = plan->row_comm;
    int nprocs_r, procid_r;
    MPI_Comm_rank(row_comm, &procid_r);
    MPI_Comm_size(row_comm, &nprocs_r);
    /* Halo Exchange along y axis
     * Phase 1: Write local data to be sent to the right process to RS
     */
  #ifdef VERBOSE2
    PCOUT<<"\nGL Row Communication\n";
  #endif

    int rs_buf_size = g_size * isize[2] * isize[0];
    Real *RS = (Real*) accfft_alloc(rs_buf_size * sizeof(Real)); // Stores local right ghost data to be sent
    Real *GL = (Real*) accfft_alloc(rs_buf_size * sizeof(Real)); // Left Ghost cells to be received

    for (int x = 0; x < isize[0]; ++x)
      memcpy(&RS[x * g_size * isize[2]],
          &data[x * isize[2] * isize[1] + (isize[1] - g_size) * isize[2]],
          g_size * isize[2] * sizeof(Real));

    /* Phase 2: Send your data to your right process
     * First question is who is your right process?
     */
    int dst_s = (procid_r + 1) % nprocs_r;
    int dst_r = (procid_r - 1) % nprocs_r;
    if (procid_r == 0)
      dst_r = nprocs_r - 1;
    MPI_Request rs_s_request, rs_r_request;
    MPI_Status ierr;
    MPI_Isend(RS, rs_buf_size, MPI_T, dst_s, 0, row_comm, &rs_s_request);
    MPI_Irecv(GL, rs_buf_size, MPI_T, dst_r, 0, row_comm, &rs_r_request);
    MPI_Wait(&rs_s_request, &ierr);
    MPI_Wait(&rs_r_request, &ierr);

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"procid="<<procid<<" data\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1];++j)
        std::cout<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<g_size;++j)
        std::cout<<RS[(i*g_size+j)*isize[2]]<<" ";
        //PCOUT<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
    sleep(1);
    if(procid==1) {
      std::cout<<"procid="<<procid<<" GL data\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<g_size;++j)
        std::cout<<GL[(i*g_size+j)*isize[2]]<<" ";
        //PCOUT<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
    PCOUT<<"\nGR Row Communication\n";
  #endif

    /* Phase 3: Now do the exact same thing for the right ghost side */
    int ls_buf_size = g_size * isize[2] * isize[0];
    Real *LS = (Real*) accfft_alloc(ls_buf_size * sizeof(Real)); // Stores local right ghost data to be sent
    Real *GR = (Real*) accfft_alloc(ls_buf_size * sizeof(Real)); // Left Ghost cells to be received
    for (int x = 0; x < isize[0]; ++x)
      memcpy(&LS[x * g_size * isize[2]], &data[x * isize[2] * isize[1]],
          g_size * isize[2] * sizeof(Real));

    /* Phase 4: Send your data to your right process
     * First question is who is your right process?
     */
    dst_s = (procid_r - 1) % nprocs_r;
    dst_r = (procid_r + 1) % nprocs_r;
    if (procid_r == 0)
      dst_s = nprocs_r - 1;
    MPI_Isend(LS, ls_buf_size, MPI_T, dst_s, 0, row_comm, &rs_s_request);
    MPI_Irecv(GR, ls_buf_size, MPI_T, dst_r, 0, row_comm, &rs_r_request);
    MPI_Wait(&rs_s_request, &ierr);
    MPI_Wait(&rs_r_request, &ierr);

  #ifdef VERBOSE2
    if(procid==1) {
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1];++j)
        std::cout<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
      std::cout<<"1 sending to dst_s="<<dst_s<<std::endl;
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<g_size;++j)
        std::cout<<LS[(i*g_size+j)*isize[2]]<<" ";
        //PCOUT<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
      std::cout<<"\n";
    }
    sleep(1);
    if(procid==0) {
      std::cout<<"0 receiving from dst_r="<<dst_r<<std::endl;
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<g_size;++j)
        std::cout<<GR[(i*g_size+j)*isize[2]]<<" ";
        //PCOUT<<data[(i*isize[1]+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
  #endif

    // Phase 5: Pack the data GL+ data + GR
    for (int i = 0; i < isize[0]; ++i) {
      memcpy(&padded_data[i * isize[2] * (isize[1] + 2 * g_size)],
          &GL[i * g_size * isize[2]], g_size * isize[2] * sizeof(Real));
      memcpy(
          &padded_data[i * isize[2] * (isize[1] + 2 * g_size)
              + g_size * isize[2]], &data[i * isize[2] * isize[1]],
          isize[1] * isize[2] * sizeof(Real));
      memcpy(
          &padded_data[i * isize[2] * (isize[1] + 2 * g_size)
              + g_size * isize[2] + isize[2] * isize[1]],
          &GR[i * g_size * isize[2]], g_size * isize[2] * sizeof(Real));
    }

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"procid= "<<procid<<" padded_array=\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<padded_data[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
  #endif

    accfft_free(LS);
    accfft_free(GR);
    accfft_free(RS);
    accfft_free(GL);

  }

  /*
   * Get the top bottom ghost cells AFTER getting the left right ones.
   *
   * @param[out] ghost_data: The output of the function which pads the input array data, with the ghost
   * cells from neighboring processors.
   * @param[in] padded_data: Input data that is already padded with ghost cells from left and right.
   * @param[in] g_size: The size of the ghost cell padding. Note that it cannot exceed the neighboring processor's
   * local data size
   * @param[in] plan: AccFFT R2C plan
   */
  void ghost_top_bottom(pvfmm::Iterator<Real> ghost_data, pvfmm::Iterator<Real> padded_data, int g_size,
      accfft_plan_t<Real, TC, PL> * plan) {
    int nprocs, procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  //  MPI_Comm c_comm = plan->c_comm;

    /* Get the local pencil size and the allocation size */
    //int isize[3],osize[3],istart[3],ostart[3];
    int * isize = plan->isize;
  //  int * osize = plan->isize;
  //  int * istart = plan->istart;
  //  int * ostart = plan->ostart;
  //  int alloc_max = plan->alloc_max;

    MPI_Comm col_comm = plan->col_comm;
    int nprocs_c, procid_c;
    MPI_Comm_rank(col_comm, &procid_c);
    MPI_Comm_size(col_comm, &nprocs_c);

    /* Halo Exchange along x axis
     * Phase 1: Write local data to be sent to the bottom process
     */
  #ifdef VERBOSE2
    PCOUT<<"\nGB Col Communication\n";
  #endif
    int bs_buf_size = g_size * isize[2] * (isize[1] + 2 * g_size); // isize[1] now includes two side ghost cells
    //Real *BS=(Real*)accfft_alloc(bs_buf_size*sizeof(Real)); // Stores local right ghost data to be sent
    pvfmm::Iterator<Real> GT = pvfmm::aligned_new<Real>(bs_buf_size); // Left Ghost cells to be received
    // snafu: not really necessary to do memcpy, you can simply use padded_data directly
    //memcpy(BS,&padded_data[(isize[0]-g_size)*isize[2]*(isize[1]+2*g_size)],bs_buf_size*sizeof(Real));
    Real* BS = &padded_data[(isize[0] - g_size) * isize[2]
                              * (isize[1] + 2 * g_size)];
    /* Phase 2: Send your data to your bottom process
     * First question is who is your bottom process?
     */
    int dst_s = (procid_c + 1) % nprocs_c;
    int dst_r = (procid_c - 1) % nprocs_c;
    if (procid_c == 0)
      dst_r = nprocs_c - 1;
    MPI_Request bs_s_request, bs_r_request;
    MPI_Status ierr;
    MPI_Isend(&BS[0], bs_buf_size, MPI_T, dst_s, 0, col_comm, &bs_s_request);
    MPI_Irecv(&GT[0], bs_buf_size, MPI_T, dst_r, 0, col_comm, &bs_r_request);
    MPI_Wait(&bs_s_request, &ierr);
    MPI_Wait(&bs_r_request, &ierr);

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"procid= "<<procid<<" padded_array=\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<padded_data[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }

      std::cout<<"procid= "<<procid<<" dst_s="<<dst_s<<" BS_array=\n";
      for (int i=0;i<g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<BS[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
    sleep(1);
    if(procid==2) {
      std::cout<<"procid= "<<procid<<" dst_r="<<dst_r<<" GT=\n";
      for (int i=0;i<g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<GT[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }

    PCOUT<<"\nGB Col Communication\n";
  #endif

    /* Phase 3: Now do the exact same thing for the right ghost side */
    int ts_buf_size = g_size * isize[2] * (isize[1] + 2 * g_size); // isize[1] now includes two side ghost cells
    //Real *TS=(Real*)accfft_alloc(ts_buf_size*sizeof(Real)); // Stores local right ghost data to be sent
    pvfmm::Iterator<Real> GB = pvfmm::aligned_new<Real>(ts_buf_size); // Left Ghost cells to be received
    // snafu: not really necessary to do memcpy, you can simply use padded_data directly
    //memcpy(TS,padded_data,ts_buf_size*sizeof(Real));
    Real *TS = &padded_data[0];

    /* Phase 4: Send your data to your right process
     * First question is who is your right process?
     */
    MPI_Request ts_s_request, ts_r_request;
    dst_s = (procid_c - 1) % nprocs_c;
    dst_r = (procid_c + 1) % nprocs_c;
    if (procid_c == 0)
      dst_s = nprocs_c - 1;
    MPI_Isend(&TS[0], ts_buf_size, MPI_T, dst_s, 0, col_comm, &ts_s_request);
    MPI_Irecv(&GB[0], ts_buf_size, MPI_T, dst_r, 0, col_comm, &ts_r_request);
    MPI_Wait(&ts_s_request, &ierr);
    MPI_Wait(&ts_r_request, &ierr);

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"procid= "<<procid<<" padded_array=\n";
      for (int i=0;i<isize[0];++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<padded_data[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }

      std::cout<<"procid= "<<procid<<" dst_s="<<dst_s<<" BS_array=\n";
      for (int i=0;i<g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<TS[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
    sleep(1);
    if(procid==2) {
      std::cout<<"procid= "<<procid<<" dst_r="<<dst_r<<" GB=\n";
      for (int i=0;i<g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<GB[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
  #endif

    // Phase 5: Pack the data GT+ padded_data + GB
    memcpy(&ghost_data[0], &GT[0],
        g_size * isize[2] * (isize[1] + 2 * g_size) * sizeof(Real));
    memcpy(&ghost_data[g_size * isize[2] * (isize[1] + 2 * g_size)],
        &padded_data[0],
        isize[0] * isize[2] * (isize[1] + 2 * g_size) * sizeof(Real));
    memcpy(
        &ghost_data[g_size * isize[2] * (isize[1] + 2 * g_size)
            + isize[0] * isize[2] * (isize[1] + 2 * g_size)], &GB[0],
        g_size * isize[2] * (isize[1] + 2 * g_size) * sizeof(Real));

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"\n final ghost data\n";
      for (int i=0;i<isize[0]+2*g_size;++i) {
        for (int j=0;j<isize[1]+2*g_size;++j)
        std::cout<<ghost_data[(i*(isize[1]+2*g_size)+j)*isize[2]]<<" ";
        std::cout<<"\n";
      }
    }
  #endif

    //accfft_free(TS);
    pvfmm::aligned_delete<Real>(GB);
    //accfft_free(BS);
    //accfft_free(GT);
    pvfmm::aligned_delete<Real>(GT);
  }

  /*
   * Perform a periodic z padding of size g_size. This function must be called after the ghost_top_bottom.
   * The idea is to have a symmetric periodic padding in all directions not just x, and y.
   *
   * @param[out] ghost_data_z: The output of the function which pads the input array data, with the ghost
   * cells from neighboring processors.
   * @param[in] ghost_data: Input data that is already padded in x, and y direction with ghost cells
   * @param[in] g_size: The size of the ghost cell padding. Note that it cannot exceed the neighboring processor's
   * local data size
   * @param[in] isize_g: An integer array specifying ghost cell padded local sizes.
   * @param[in] plan: AccFFT R2C plan
   */
  void ghost_z(Real *ghost_data_z, pvfmm::Iterator<Real> ghost_data, int g_size, int* isize_g,
      accfft_plan_t<Real, TC, PL>* plan) {

    int * isize = plan->isize;
    for (int i = 0; i < isize_g[0]; ++i)
      for (int j = 0; j < isize_g[1]; ++j) {
        memcpy(&ghost_data_z[(i * isize_g[1] + j) * isize_g[2]],
            &ghost_data[(i * isize_g[1] + j) * isize[2] + isize[2]
                - g_size], g_size * sizeof(Real));
        memcpy(&ghost_data_z[(i * isize_g[1] + j) * isize_g[2] + g_size],
            &ghost_data[(i * isize_g[1] + j) * isize[2]],
            isize[2] * sizeof(Real));
        memcpy(
            &ghost_data_z[(i * isize_g[1] + j) * isize_g[2] + g_size
                + isize[2]],
            &ghost_data[(i * isize_g[1] + j) * isize[2]],
            g_size * sizeof(Real));
      }
    return;
  }

  /*
   * Returns the necessary memory allocation in Bytes for the ghost data, as well
   * the local ghost sizes when ghost cell padding is desired only in x and y directions (and not z direction
   * which is locally owned).
   * @param[in] plan: AccFFT plan
   * @param[in] g_size: The number of ghost cells desired. Note that g_size cannot be bigger than
   * the minimum isize in each dimension among all processors. This means that you cannot get ghost
   * cells from a processor that is not a neighbor of the calling processor.
   * @param[out] isize_g: The new local sizes after getting the ghost cells.
   * @param[out] istart_g: Returns the new istart after getting the ghost cells. Note that this is the global
   * istart of the ghost cells. So for example, if processor zero gets 3 ghost cells, the left ghost cells will
   * come from the last process because of the periodicity. Then the istart_g would be the index of those elements
   * (that originally resided in the last processor).
   */

  size_t accfft_ghost_local_size_dft_r2c(accfft_plan_t<Real, TC, PL>* plan, int g_size,
      int * isize_g, int* istart_g) {

    size_t alloc_max = plan->alloc_max;
    int *isize = plan->isize;
    int *istart = plan->istart;
    int* n = plan->N;
    istart_g[2] = istart[2];
    isize_g[2] = isize[2];

    istart_g[0] = istart[0] - g_size;
    istart_g[1] = istart[1] - g_size;

    if (istart_g[0] < 0)
      istart_g[0] += n[0];
    if (istart_g[1] < 0)
      istart_g[1] += n[1];

    isize_g[0] = isize[0] + 2 * g_size;
    isize_g[1] = isize[1] + 2 * g_size;
    return (alloc_max + 2 * g_size * isize[2] * isize[0] * sizeof(Real)
        + 2 * g_size * isize[2] * (isize[1] + 2 * g_size) * sizeof(Real));
  }

  size_t accfft_ghost_local_size_dft_r2c(accfft_plan* plan, int g_size,
      int * isize_g, int* istart_g) {
    return accfft_ghost_local_size_dft_r2c((accfft_plan_t<Real, TC, PL>*)plan, g_size, isize_g, istart_g);
  }
  /*
   * Gather the ghost cells for a real input when ghost cell padding is desired only in x and y
   * directions (and not z direction which is locally owned). This function currently has the following limitations:
   *  - The AccFFT plan has to be outplace. Note that for inplace R2C plans, the input array has to be
   *  padded, which would slightly change the pattern of communicating ghost cells.
   *  - The number of ghost cells needed has to be the same in both x and y directions (note that each
   *  processor owns the whole z direction locally, so typically you would not need ghost cells. You can
   *  call accfft_get_ghost_xyz which would actually get ghost cells in z direction as well, but that is
   *  like a periodic padding in z direction).
   *  - The number of ghost cells requested cannot be bigger than the minimum isize of all processors. This
   *  means that a process cannot get a ghost element that does not belong to its immediate neighbors. This
   *  is a limitation that should barely matter, as typically ghost_size is a small integer while the global
   *  array sizes are very large. In mathematical terms g_size< min(isize[0],isize[1]) among isize of all
   *  processors.
   *
   * @param[in] plan: AccFFT plan
   * @param[in] g_size: The number of ghost cells desired. Note that g_size cannot be bigger than
   * the minimum isize in each dimension among all processors. This means that you cannot get ghost
   * cells from a processor that is not a neighbor of the calling processor.
   * @param[in] isize_g: An integer array specifying ghost cell padded local sizes.
   * @param[in] data: The local data whose ghost cells from other processors are sought.
   * @param[out] ghost_data: An array that is the ghost cell padded version of the input data.
   */
  void accfft_get_ghost(accfft_plan_t<Real, TC, PL>* plan, int g_size, int* isize_g, Real* data,
      Real* ghost_data) {
    int nprocs, procid;
    MPI_Comm_rank(plan->c_comm, &procid);
    MPI_Comm_size(plan->c_comm, &nprocs);

    if (plan->inplace == true) {
      PCOUT << "accfft_get_ghost_r2c does not support inplace transforms."
          << std::endl;
      return;
    }

    if (g_size == 0) {
      memcpy(ghost_data, data, plan->alloc_max);
      return;
    }

    int *isize = plan->isize;
    //int *istart = plan->istart;
    //int *n = plan->N;
    if (g_size > isize[0] || g_size > isize[1]) {
      std::cout
          << "accfft_get_ghost_r2c does not support g_size greater than isize."
          << std::endl;
      return;
    }

    pvfmm::Iterator<Real> padded_data = pvfmm::aligned_new<Real>
      (plan->alloc_max + 2 * g_size * isize[2] * isize[0]);
    ghost_left_right(padded_data, data, g_size, plan);
    ghost_top_bottom(ghost_data, padded_data, g_size, plan);
    pvfmm::aligned_delete<Real>(padded_data);
    return;

  }

  void accfft_get_ghost(accfft_plan* plan, int g_size, int* isize_g, Real* data,
      Real* ghost_data) {
    accfft_get_ghost((accfft_plan_t<Real, TC, PL>*)plan, g_size, isize_g, data, ghost_data);
  }
  /*
   * Returns the necessary memory allocation in Bytes for the ghost data, as well
   * the local ghost sizes when padding in all directions (including z direction that is locally owned by each process).
   * @param[in] plan: AccFFT plan
   * @param[in] g_size: The number of ghost cells desired. Note that g_size cannot be bigger than
   * the minimum isize in each dimension among all processors. This means that you cannot get ghost
   * cells from a processor that is not a neighbor of the calling processor.
   * @param[out] isize_g: The new local sizes after getting the ghost cells.
   * @param[out] istart_g: Returns the new istart after getting the ghost cells. Note that this is the global
   * istart of the ghost cells. So for example, if processor zero gets 3 ghost cells, the left ghost cells will
   * come from the last process because of the periodicity. Then the istart_g would be the index of those elements
   * (that originally resided in the last processor).
   */

  size_t accfft_ghost_xyz_local_size_dft_r2c(accfft_plan_t<Real, TC, PL>* plan, int g_size,
      int * isize_g, int* istart_g) {

    size_t alloc_max = plan->alloc_max;
    int *isize = plan->isize;
    int *istart = plan->istart;
    int* n = plan->N;

    istart_g[0] = istart[0] - g_size;
    istart_g[1] = istart[1] - g_size;
    istart_g[2] = istart[2] - g_size;

    if (istart_g[0] < 0)
      istart_g[0] += n[0];
    if (istart_g[1] < 0)
      istart_g[1] += n[1];
    if (istart_g[2] < 0)
      istart_g[2] += n[2];

    isize_g[0] = isize[0] + 2 * g_size;
    isize_g[1] = isize[1] + 2 * g_size;
    isize_g[2] = isize[2] + 2 * g_size;
    size_t alloc_max_g = alloc_max + 2 * g_size * isize[2] * isize[0] * sizeof(Real)
        + 2 * g_size * isize[2] * isize_g[1] * sizeof(Real)
        + 2 * g_size * isize_g[0] * isize_g[1] * sizeof(Real);
    alloc_max_g += (16*isize_g[2]*isize_g[1]+16*isize_g[1]+16)*sizeof(Real); // to account for padding required for peeled loop in interp
    return alloc_max_g;
  }

  size_t accfft_ghost_xyz_local_size_dft_r2c(accfft_planf* plan, int g_size,
      int * isize_g, int* istart_g) {
    return accfft_ghost_xyz_local_size_dft_r2c((accfft_plan_t<Real, TC, PL>*)plan, g_size, isize_g, istart_g);
  }

  size_t accfft_ghost_xyz_local_size_dft_r2c(accfft_plan* plan, int g_size,
      int * isize_g, int* istart_g) {
    return accfft_ghost_xyz_local_size_dft_r2c((accfft_plan_t<Real, TC, PL>*)plan, g_size, isize_g, istart_g);
  }
  /*
   * Gather the ghost cells for a real input when ghost cell padding is desired in all directions including z direction
   * (which is locally owned). This function currently has the following limitations:
   *  - The AccFFT plan has to be outplace. Note that for inplace R2C plans, the input array has to be
   *  padded, which would slightly change the pattern of communicating ghost cells.
   *  - The number of ghost cells needed has to be the same in both x and y directions (note that each
   *  processor owns the whole z direction locally, so typically you would not need ghost cells. You can
   *  call accfft_get_ghost_xyz which would actually get ghost cells in z direction as well, but that is
   *  like a periodic padding in z direction).
   *  - The number of ghost cells requested cannot be bigger than the minimum isize of all processors. This
   *  means that a process cannot get a ghost element that does not belong to its immediate neighbors. This
   *  is a limitation that should barely matter, as typically ghost_size is a small integer while the global
   *  array sizes are very large. In mathematical terms g_size< min(isize[0],isize[1],isize[2]) among isize of all
   *  processors.
   *
   * @param[in] plan: AccFFT plan
   * @param[in] g_size: The number of ghost cells desired. Note that g_size cannot be bigger than
   * the minimum isize in each dimension among all processors. This means that you cannot get ghost
   * cells from a processor that is not a neighbor of the calling processor.
   * @param[in] isize_g: An integer array specifying ghost cell padded local sizes.
   * @param[in] data: The local data whose ghost cells from other processors are sought.
   * @param[out] ghost_data: An array that is the ghost cell padded version of the input data.
   */
  void accfft_get_ghost_xyz(accfft_plan_t<Real, TC, PL>* plan, int g_size, int* isize_g,
      Real* data, Real* ghost_data) {
    int nprocs, procid;
    MPI_Comm_rank(plan->c_comm, &procid);
    MPI_Comm_size(plan->c_comm, &nprocs);

    if (plan->inplace == true) {
      PCOUT << "accfft_get_ghost_r2c does not support inplace transforms."
          << std::endl;
      return;
    }

    if (g_size == 0) {
      memcpy(ghost_data, data, plan->alloc_max);
      return;
    }

    int *isize = plan->isize;
  //  int *istart = plan->istart;
  //  int *n = plan->N;
    if (g_size > isize[0] || g_size > isize[1]) {
      std::cout
          << "accfft_get_ghost_r2c does not support g_size greater than isize."
          << std::endl;
      return;
    }

    pvfmm::Iterator<Real> padded_data = pvfmm::aligned_new<Real>
      (plan->alloc_max + 2 * g_size * isize[2] * isize[0]);
    pvfmm::Iterator<Real> ghost_data_xy = pvfmm::aligned_new<Real>(
        plan->alloc_max + 2 * g_size * isize[2] * isize[0]
            + 2 * g_size * isize[2] * isize_g[1]);

    ghost_left_right(padded_data, data, g_size, plan);
    ghost_top_bottom(ghost_data_xy, padded_data, g_size, plan);
    ghost_z(&ghost_data[0], ghost_data_xy, g_size, isize_g, plan);

  #ifdef VERBOSE2
    if(procid==0) {
      std::cout<<"\n final ghost data\n";
      for (int i=0;i<isize_g[0];++i) {
        for (int j=0;j<isize_g[1];++j)
        std::cout<<ghost_data[(i*isize_g[1]+j)*isize_g[2]]<<" ";
        std::cout<<"\n";
      }

      std::cout<<"\n a random z\n";
      int i=3+0*isize_g[0]/2;
      int j=3+0*isize_g[1]/2;
      for(int k=0;k<isize_g[2];++k)
      std::cout<<ghost_data[(i*isize_g[1]+j)*isize_g[2]+k]<<" ";
      std::cout<<"\n";
    }
  #endif

    pvfmm::aligned_delete<Real>(padded_data);
    pvfmm::aligned_delete<Real>(ghost_data_xy);
    return;
  }

  void accfft_get_ghost_xyz(accfft_plan* plan, int g_size, int* isize_g,
      Real* data, Real* ghost_data) {
    accfft_get_ghost_xyz((accfft_plan_t<Real, TC, PL>*)plan, g_size, isize_g, data, ghost_data);
  }

#endif
