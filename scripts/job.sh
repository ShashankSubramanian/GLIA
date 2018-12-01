#!/bin/bash
#SBATCH -J ITP
#SBATCH -o /workspace/shashank/pglistr_tumor/scripts/..//results/checkL1vsmall//itp_check
#SBATCH -p rebels
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 48:00:00
source ~/.bashrc
export OMP_NUM_THREADS=1
mpirun /workspace/shashank/pglistr_tumor/scripts/..//build/last/inverse -nx 128 -ny 128 -nz 128 -beta 0.0001 -rho_inversion 15 -k_inversion 0.01 -nt_inversion 10 -dt_inversion 0.02 -rho_data 15 -k_data 0.01 -nt_data 10 -dt_data 0.02 -regularization L1c -interpolation 0 -diffusivity_inversion 1 -basis_type 0 -number_gaussians 125 -sigma_factor 2 -sigma_spacing 2 -gaussian_volume_fraction 0.99 -lambda_continuation 1 -target_sparsity 0.99 -sparsity_level 5 -threshold_data_driven 0.1 -sigma_data_driven 2 -output_dir /workspace/shashank/pglistr_tumor/scripts/..//results/checkL1vsmall/ -newton_solver QN -newton_maxit 50 -gist_maxit 50 -krylov_maxit 30 -syn_flag 1 -data_path /workspace/shashank/pglistr_tumor/scripts/..//results/check/data.nc -gm_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/128/gray_matter.nc -wm_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/128/white_matter.nc -csf_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/128/csf.nc -model 1 -tao_lmm_vectors 50 -tao_lmm_scale_type broyden -tao_lmm_scalar_history 5 -tao_lmm_rescale_type scalar -tao_lmm_rescale_history 5