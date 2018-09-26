#!/bin/bash

### name of results sub-directory
REGTYPE=L1
### reg type
reg_type=L1
### name of results sub-directory
TESTCASE=pc_lm

### Synthetic data parameters
rho_data=20
k_data=0
nt_data=15
dt_data=0.02

### Inversion tumor parameters
rho_inv=20
k_inv=0
nt_inv=15
dt_inv=0.02

### Interpolation flag
interp_flag=0
### Diffusivity inversion flag
diffusivity_flag=0
### Radial basis flag: 1 - data driven, 0 - grid-based (bounding box)
basis_type=1
lam_cont=1

beta=1e-4
N=256
np=27
fac=2
space=2
gvf=0.99
data_thres=0.1
target_spars=0.99
dd_fac=2

# space=0.09817477042

#declare -a rhoarray=(0)
#for rho_inv in ${rhoarray[@]}; do
# declare -a facarray=(0.1 0.3 0.5 0.7 1)
# for fac in ${facarray[@]}; do
# declare -a karray=(0e+00 1e-07 5e-07 1e-06 5e-05 1e-05 5e-05 1e-04 5e-04 1e-03 5e-03 1e-02 5e-02 1e-01 5e-01 1e+00)
# for k_inv in ${karray[@]}; do

# RESULTS_DIR=./results/${REGTYPE}/${TESTCASE}/
RESULTS_DIR=./results/${REGTYPE}/${TESTCASE}/

mkdir -p ${RESULTS_DIR}

cat <<EOF > submit
#!/bin/bash

#SBATCH -J ITP
#SBATCH -o itp_${TESTCASE}_${REGTYPE}
#SBATCH -p rebels
#SBATCH -N 2
#SBATCH -n 40
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
-output_dir ${RESULTS_DIR}
EOF

sbatch submit
# done
# -sigma_data_driven ${space}
