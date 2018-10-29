#!/bin/bash
#SBATCH -J ITP
#SBATCH -o /workspace/shashank/pglistr_tumor/scripts/..//results/check2//itp_check
#SBATCH -p rebels
#SBATCH -N 1
#SBATCH -n 20
#SBATCH -t 10:00:00
source ~/.bashrc
export OMP_NUM_THREADS=1
mpirun /workspace/shashank/pglistr_tumor/scripts/..//build/last/inverse -nx 256 -ny 256 -nz 256 -beta 0.0001 -rho_inversion 20 -k_inversion 0.0 -nt_inversion 15 -dt_inversion 0.02 -rho_data 20 -k_data 0.1 -nt_data 15 -dt_data 0.02 -regularization L1 -interpolation 0 -diffusivity_inversion 1 -basis_type 1 -number_gaussians 27 -sigma_factor 2 -sigma_spacing 2 -gaussian_volume_fraction 0.99 -lambda_continuation 1 -target_sparsity 0.99 -threshold_data_driven 0.1 -sigma_data_driven 2 -output_dir /workspace/shashank/pglistr_tumor/scripts/..//results/check2/ -newton_solver QN -newton_maxit 35 -gist_maxit 50 -krylov_maxit 30 -syn_flag 0 -data_path /workspace/shashank/pglistr_tumor/scripts/..//results/check/data.nc -gm_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/256/gray_matter.nc -wm_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/256/white_matter.nc -csf_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/256/csf.nc -model 1