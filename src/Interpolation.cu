#include <stdio.h>
#include <cuda.h>
#include <Interp.h>





/*
 * Rescales the query points to [0,1) range for the parallel case. Note that the input query_points are initially
 * in the global range, however, each parallel process needs to rescale it to [0,1) for its local interpolation.
 * Since interpolation is scale invariant, this would not affect the interpolation result.
 * This function assumes that the ghost padding was done in x, y, and z directions.
 *
 * @param[in] g_size: The ghost padding size
 * @param[in] N_reg: The original (unpadded) global size
 * @param[in] N_reg_g: The padded global size
 * @param[in] istart: The original isize for each process
 * @param[in] N_pts: The number of query points
 * @param[in,out]: query_points: The query points coordinates
 *
 */
__device__ void gpu_rescale_xyz_kernel(const int g_size,  int* N_reg, int* N_reg_g, int* istart, const int N_pts, Real* query_points, int thid){

  if(g_size==0)
    return;
  Real hp[3];
  Real h[3];
  hp[0]=1./N_reg_g[0]; // New mesh size
  hp[1]=1./N_reg_g[1]; // New mesh size
  hp[2]=1./N_reg_g[2]; // New mesh size

  h[0]=1./(N_reg[0]); // old mesh size
  h[1]=1./(N_reg[1]); // old mesh size
  h[2]=1./(N_reg[2]); // old mesh size

  Real factor[3];
  factor[0]=(1.-(2.*g_size+1.)*hp[0])/(1.-h[0]);
  factor[1]=(1.-(2.*g_size+1.)*hp[1])/(1.-h[1]);
  factor[2]=(1.-(2.*g_size+1.)*hp[2])/(1.-h[2]);
  query_points[0+COORD_DIM*thid]=(query_points[0+COORD_DIM*thid]-istart[0]*h[0])*factor[0]+g_size*hp[0];
  query_points[1+COORD_DIM*thid]=(query_points[1+COORD_DIM*thid]-istart[1]*h[1])*factor[1]+g_size*hp[1];
  query_points[2+COORD_DIM*thid]=(query_points[2+COORD_DIM*thid]-istart[2]*h[2])*factor[2]+g_size*hp[2];
  return;
}



/*
 * GPU kernel for performing a 3D cubic interpolation. For input/output details please see
 * the comments of gpu_interp3_ghost_xyz_p function below.
 */
__global__ void gpu_interp3_ghost_xyz_p_kernel( Real* reg_grid_vals, int data_dof,
    int* N_reg, int* isize,int* istart, const int N_pts, const int g_size, Real* query_points,
    Real* query_values,bool query_values_already_scaled)
{

  int thid=blockIdx.x * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x + threadIdx.x;;
  //printf("Hello reg_grid_vals[%d]= %f\n",thid, reg_grid_vals[thid]);
  if(thid>=N_pts)
    return;

  int isize_g[3],N_reg_g[3];
  isize_g[0]=isize[0]+2*g_size;
  isize_g[1]=isize[1]+2*g_size;
  isize_g[2]=isize[2]+2*g_size;

  N_reg_g[0]=N_reg[0]+2*g_size;
  N_reg_g[1]=N_reg[1]+2*g_size;
  N_reg_g[2]=N_reg[2]+2*g_size;

  // First we need to rescale the query points to the new padded dimensions
  // To avoid changing the user's input we first copy the query points to a
  // new array
  //Real* query_points=(Real*) malloc(N_pts*COORD_DIM*sizeof(Real));
  //memcpy(query_points,query_points_in,N_pts*COORD_DIM*sizeof(Real));
  //printf("\n q[0]=%f  q[1]=%f q[2]=%f \n\n\n",query_points[0],query_points[1],query_points[2]);
  if(query_values_already_scaled==false)
    gpu_rescale_xyz_kernel(g_size,N_reg,N_reg_g,istart,N_pts,query_points,thid);
  //printf("\n q[0]=%f  q[1]=%f q[2]=%f \n\n\n",query_points[0],query_points[1],query_points[2]);

  //printf("isize[0]=%d [1]=%d [2]=%d \n",isize[0],isize[1],isize[2]);
  //printf("istart[0]=%d [1]=%d [2]=%d \n",istart[0],istart[1],istart[2]);

  Real lagr_denom[4];
  for(int i=0;i<4;i++){
    lagr_denom[i]=1;
    for(int j=0;j<4;j++){
      if(i!=j) lagr_denom[i]/=(Real)(i-j);
    }
  }

  int N_reg3=isize_g[0]*isize_g[1]*isize_g[2];

  Real point[COORD_DIM];
  int grid_indx[COORD_DIM];

  for(int j=0;j<COORD_DIM;j++){
    point[j]=query_points[COORD_DIM*thid+j]*N_reg_g[j];
    grid_indx[j]=(floor(point[j]))-1;
    point[j]-=grid_indx[j];
    while(grid_indx[j]<0) grid_indx[j]+=N_reg_g[j];
  }

  Real M[3][4];
  for(int j=0;j<COORD_DIM;j++){
    Real x=point[j];
    for(int k=0;k<4;k++){
      M[j][k]=lagr_denom[k];
      for(int l=0;l<4;l++){
        if(k!=l) M[j][k]*=(x-l);
      }
    }
  }

  for(int k=0;k<data_dof;k++){
    Real val=0;
    for(int j2=0;j2<4;j2++){
      for(int j1=0;j1<4;j1++){
        for(int j0=0;j0<4;j0++){
          int indx = ((grid_indx[2]+j2)%isize_g[2]) + isize_g[2]*((grid_indx[1]+j1)%isize_g[1]) + isize_g[2]*isize_g[1]*((grid_indx[0]+j0)%isize_g[0]);
          val += M[0][j0]*M[1][j1]*M[2][j2] * reg_grid_vals[indx+k*N_reg3];
        }
      }
    }
    query_values[thid+k*N_pts]=val;
  }
  return;
}

__global__ void gpu_interp3_ghost_xyz_p_kernel_scaled( __restrict__ Real* reg_grid_vals, int data_dof,
//__global__ void gpu_interp3_ghost_xyz_p_kernel_scaled( Real* reg_grid_vals, int data_dof,
    int* N_reg_g, int* isize_g,const int N_pts, Real* query_points,
    Real* query_values,size_t offset,const int N_PTS,__restrict__ Real* lagr_denom)
{


  int thid=blockIdx.x * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x + threadIdx.x;;
  if(thid>=N_pts)
    return;

  //reg_grid_vals+=offset;
  //query_points+=offset*COORD_DIM;
  //query_values+=offset;
  thid+=offset;

  const int N_reg3=isize_g[0]*isize_g[1]*isize_g[2];

  Real point[COORD_DIM];
  int grid_indx[COORD_DIM];

  for(int j=0;j<COORD_DIM;j++){
    point[j]=query_points[COORD_DIM*thid+j]*N_reg_g[j];
    grid_indx[j]=(floor(point[j]))-1;
    point[j]-=grid_indx[j];
    while(grid_indx[j]<0) grid_indx[j]+=N_reg_g[j];
  }

  Real M[3][4];
  for(int j=0;j<COORD_DIM;j++){
    Real x=point[j];
    for(int k=0;k<4;k++){
      M[j][k]=lagr_denom[k];
      for(int l=0;l<4;l++){
        if(k!=l) M[j][k]*=(x-l);
      }
    }
  }


  for(int k=0;k<data_dof;k++){
    Real val=0;
    for(int j2=0;j2<4;j2++){
      for(int j1=0;j1<4;j1++){
        for(int j0=0;j0<4;j0++){
          int indx = ((grid_indx[2]+j2)%isize_g[2]) + isize_g[2]*((grid_indx[1]+j1)%isize_g[1]) + isize_g[2]*isize_g[1]*((grid_indx[0]+j0)%isize_g[0]);
          val += M[0][j0]*M[1][j1]*M[2][j2] * reg_grid_vals[indx+k*N_reg3];
        }
      }
    }
    query_values[thid+k*N_PTS]=val;
  }
  return;
}



/*
 * GPU kernel for performing a 3D cubic interpolation. For input/output details please see
 * the comments of gpu_interp3_p function below.
 */
__global__ void gpu_interp3_p_kernel( Real* reg_grid_vals, int data_dof,
    int* N_reg, const int N_pts, Real* query_points,
    Real* query_values){

  int thid=blockIdx.x * blockDim.x * blockDim.y+ threadIdx.y * blockDim.x + threadIdx.x;;
  //printf("Hello reg_grid_vals[%d]= %f\n",thid, reg_grid_vals[thid]);
  if(thid>=N_pts)
    return;


  Real lagr_denom[4];
  for(int i=0;i<4;i++){
    lagr_denom[i]=1;
    for(int j=0;j<4;j++){
      if(i!=j) lagr_denom[i]/=(Real)(i-j);
    }
  }

  int N_reg3=N_reg[0]*N_reg[1]*N_reg[2];
  //printf("Hello from block %d, thread %d thid %d N_Pts %d N_reg3 %d dat_dof %d\n", blockIdx.x, threadIdx.x, thid,N_pts,N_reg3,data_dof);

  Real point[COORD_DIM];
  int grid_indx[COORD_DIM];

  for(int j=0;j<COORD_DIM;j++){
    point[j]=query_points[COORD_DIM*thid+j]*N_reg[j];
    grid_indx[j]=(floor(point[j]))-1;
    point[j]-=grid_indx[j];
    while(grid_indx[j]<0) grid_indx[j]+=N_reg[j];
  }
  //std::cout<<"grid_index="<<grid_indx[0]<<" "<<grid_indx[1]<<" "<<grid_indx[2]<<std::endl;

  Real M[3][4];
  for(int j=0;j<COORD_DIM;j++){
    Real x=point[j];
    for(int k=0;k<4;k++){
      M[j][k]=lagr_denom[k];
      for(int l=0;l<4;l++){
        if(k!=l) M[j][k]*=(x-l);
      }
    }
  }

  register int indx;
  register Real factor;
  for(int k=0;k<data_dof;k++){
    Real val=0;
    for(int j0=0;j0<4;j0++){
      indx =N_reg[2]*N_reg[1]*((grid_indx[0]+j0)%N_reg[0]);
      factor=M[0][j0];
      for(int j1=0;j1<4;j1++){
        indx +=N_reg[2]*((grid_indx[1]+j1)%N_reg[1]);
        factor*=M[1][j1];
        for(int j2=0;j2<4;j2++){
          indx += ((grid_indx[2]+j2)%N_reg[2]);
          val += factor*M[2][j2] * reg_grid_vals[indx+k*N_reg3];
        }
      }
    }
    query_values[thid+k*N_pts]=val;
  }

  //printf("Hello reg_grid_vals[0] %f\n", reg_grid_vals[0]);

  return;
}


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
void gpu_interp3_ghost_xyz_p( Real* reg_grid_vals_d, int data_dof,
    int* N_reg, int * isize, int* istart,const int N_pts, const int g_size, Real* query_points_d,
    Real* query_values_d,bool query_values_already_scaled){

  int * N_reg_d,*isize_d,*istart_d;

  //printf("isize[0]=%d [1]=%d [2]=%d \n",isize[0],isize[1],isize[2]);
  //printf("istart[0]=%d [1]=%d [2]=%d \n",istart[0],istart[1],istart[2]);
  cudaMalloc((void**)&N_reg_d, sizeof(int)*3);
  cudaMalloc((void**)&isize_d, sizeof(int)*3);
  cudaMalloc((void**)&istart_d, sizeof(int)*3);


  int isize_g[3],N_reg_g[3];
  isize_g[0]=isize[0]+2*g_size;
  isize_g[1]=isize[1]+2*g_size;
  isize_g[2]=isize[2]+2*g_size;

  N_reg_g[0]=N_reg[0]+2*g_size;
  N_reg_g[1]=N_reg[1]+2*g_size;
  N_reg_g[2]=N_reg[2]+2*g_size;
  cudaMemcpy(N_reg_d,N_reg_g,sizeof(int)*3,cudaMemcpyHostToDevice);
  cudaMemcpy(isize_d,isize_g,sizeof(int)*3,cudaMemcpyHostToDevice);

  //cudaMemcpy(N_reg_d,N_reg,sizeof(int)*3,cudaMemcpyHostToDevice);
  //cudaMemcpy(isize_d,isize,sizeof(int)*3,cudaMemcpyHostToDevice);
  //cudaMemcpy(istart_d,istart,sizeof(int)*3,cudaMemcpyHostToDevice);

  Real lagr_denom[4];
  for(int i=0;i<4;i++){
    lagr_denom[i]=1;
    for(int j=0;j<4;j++){
      if(i!=j) lagr_denom[i]/=(Real)(i-j);
    }
  }

  Real* lagr_denom_d;
  cudaMalloc((void**)&lagr_denom_d, sizeof(Real)*4);
  cudaMemcpy(lagr_denom_d,lagr_denom,sizeof(Real)*4,cudaMemcpyHostToDevice);

  dim3 blocksize(16,32);
  int gridsize=(int)ceil(double(N_pts)/(blocksize.x*blocksize.y));
  float time=0,dummy_time=0;
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  //int nstream=1;
  //cudaStream_t stream[nstream];
  //cudaError_t result;
  //for(int i=0;i<nstream;++i)
  //result = cudaStreamCreate(&stream[i]);

  cudaEventRecord(startEvent,0);
  //gpu_interp3_ghost_xyz_p_kernel<<<gridsize,blocksize>>>(reg_grid_vals_d,data_dof,N_reg_d,isize_d,istart_d,N_pts,g_size,query_points_d,query_values_d, query_values_already_scaled);
  //for(int i=0;i<nstream;++i)
  //gpu_interp3_ghost_xyz_p_kernel_scaled<<<gridsize,blocksize,0,stream[i]>>>(reg_grid_vals_d      ,data_dof,N_reg_d,isize_d,N_pts/nstream,query_points_d        ,query_values_d      ,i*N_pts/nstream,N_pts);
  gpu_interp3_ghost_xyz_p_kernel_scaled<<<gridsize,blocksize>>>(reg_grid_vals_d,data_dof,N_reg_d,isize_d,N_pts,query_points_d,query_values_d,0,N_pts,lagr_denom_d);
  cudaEventRecord(stopEvent,0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&dummy_time, startEvent, stopEvent);
  time+=dummy_time/1000;
  cudaDeviceSynchronize();

  //printf("\n3D interpolation of Q=%d query point on a grid size of N=%dx%dx%d  took %f\n\n\n",N_pts,N_reg[0],N_reg[1],N_reg[2],time);
  //for(int i=0;i<nstream;++i)
  //result = cudaStreamDestroy(stream[i]);

  cudaFree(N_reg_d);
  cudaFree(isize_d);
  cudaFree(istart_d);
  cudaFree(lagr_denom_d);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

}




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


void gpu_interp3_p( Real* reg_grid_vals_d, int data_dof,
    int *N_reg, const int N_pts, Real*  query_points_d,
    Real* query_values_d){



  int * N_reg_d;
  cudaMalloc((void**)&N_reg_d, sizeof(int)*3);
  cudaMemcpy(N_reg_d,N_reg,sizeof(int)*3,cudaMemcpyHostToDevice);

  dim3 blocksize(16,32);
  int gridsize=(int)ceil(double(N_pts)/(blocksize.x*blocksize.y));
  float time=0,dummy_time=0;
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  cudaEventRecord(startEvent,0);
  gpu_interp3_p_kernel<<<gridsize,blocksize>>>(reg_grid_vals_d,data_dof,N_reg_d,N_pts,query_points_d,query_values_d);
  cudaEventRecord(stopEvent,0);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&dummy_time, startEvent, stopEvent);
  time+=dummy_time/1000;
  cudaDeviceSynchronize();
  //printf("\n3D interpolation of Q=%d query point on a grid size of N=%dx%dx%d  took %f\n\n\n",N_pts,N_reg[0],N_reg[1],N_reg[2],time);
  cudaFree(N_reg_d);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

}


