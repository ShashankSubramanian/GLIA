#!/bin/bash

### RUN SCRIPT FOR INVERSE TUMOR SOLVER  ###
############################################################################################################################

### Inversion tumor parameters  -- Tumor is inverted with these parameters: Use k_inv=0 if diffusivity is being inverted
rho_inv=20
k_inv=0.1
nt_inv=15
dt_inv=0.02

### tumor regularization type -- L1, L2, L2b  (Use L1 or L2b for all tests)
reg_type=L1
### generic name for results folders, job outputs etc
TESTCASE=check
### Synthetic data parameters  -- Tumor is grown with these parameters
rho_data=20
k_data=0.1
nt_data=15
dt_data=0.02

### Interpolation flag   -- Flag to solve an interpolation problem (find parameterization of the data) only
interp_flag=0
### Diffusivity inversion flag  -- Flag to invert for diffusivity/diffusion coefficient
diffusivity_flag=1
### Radial basis flag: 1 - data driven, 0 - grid-based (bounding box)  (Use data-driven for all tests)
basis_type=1
### Lambda continuation flag -- Flag for parameter continuation in L1 optimization (Keep turned on)
lam_cont=1
### Tumor L2 regularization
beta=0e-4
### No of discretization points (Assumed uniform)
N=256
### No of radial basis functions (Only used if basis_type is grid-based)
np=27
### Factor (integer only) which controls the variance of the basis function for synthetic data (\sigma = fac * 2 * pi / 256)
fac=2
### Spacing factor between radial basis functions (Keep as 2 to have a well-conditioned matrix for the radial basis functions)
space=2
### Gaussian volume fraction -- Fraction of Gaussian that has to be tumorous to switch on the basis function at any grid point
gvf=0.99
### Threshold of data tumor concentration above which Gaussians are switched on
data_thres=0.1
### Target sparsity we expect for our initial tumor condition
target_spars=0.99
### Factor (integer only) which controls the variance of the basis function for tumor inversion (\sigma = fac * 2 * pi / 256)
dd_fac=2
### Solver type: QN - Quasi newton, GN - Gauss newton
solvertype=QN
### Newton max iterations
max_iter=35
### GIST max iterations (for L1 solver)
max_gist_iter=50
### Krylov max iterations
max_krylov_iter=30
### Path to all output results (Directories are created automatically)
RESULTS_DIR=./results/${reg_type}/${TESTCASE}/
mkdir -p ${RESULTS_DIR}
### Input data
### Flag to create synthetic data
create_synthetic=1
### Path to data (used if create_synthetic = 1)
data_path=./brain_data/${N}/cpl/c1p.nc
### Atlas
### Path to gm
gm_path=./brain_data/${N}/gray_matter.nc
### Path to csf
csf_path=./brain_data/${N}/csf.nc
### Path to wm
wm_path=./brain_data/${N}/white_matter.nc
### Path to gl,
glm_path=./brain_data/${N}/glial_matter.nc

############################################################################################################################

### Create job submission file -- Change this according to the system
cat <<EOF > submit
#!/bin/bash

#SBATCH -J ITP
#SBATCH -o itp_${TESTCASE}_${reg_type}
#SBATCH -p rebels
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 10:00:00

source ~/.bashrc
export OMP_NUM_THREADS=1

mpirun build/last/inverse -nx ${N} -ny ${N} -nz ${N} -beta ${beta} \
-rho_inversion ${rho_inv} -k_inversion ${k_inv} -nt_inversion ${nt_inv} -dt_inversion ${dt_inv} \
-rho_data ${rho_data} -k_data ${k_data} -nt_data ${nt_data} -dt_data ${dt_data} \
-regularization ${reg_type} -interpolation ${interp_flag} -diffusivity_inversion ${diffusivity_flag} \
-basis_type ${basis_type} -number_gaussians ${np} -sigma_factor ${fac} -sigma_spacing ${space} \
-gaussian_volume_fraction ${gvf} \
-lambda_continuation ${lam_cont} \
-target_sparsity ${target_spars} \
-threshold_data_driven ${data_thres} \
-sigma_data_driven ${dd_fac} \
-output_dir ${RESULTS_DIR} \
-newton_solver ${solvertype} \
-newton_maxit ${max_iter} \
-gist_maxit ${max_gist_iter} \
-krylov_maxit ${max_krylov_iter} \
-syn_flag ${create_synthetic} \
-data_path ${data_path} \
-gm_path ${gm_path} \
-wm_path ${wm_path} \
-csf_path ${csf_path} \
-glm_path ${glm_path}
EOF
sbatch submit