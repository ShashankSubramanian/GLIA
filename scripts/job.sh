#!/bin/bash
#SBATCH -J ITP
#SBATCH -o /workspace/shashank/pglistr_tumor/scripts/..//results/check//log
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -t 48:00:00
source ~/.bashrc
export OMP_NUM_THREADS=1
mpirun /workspace/shashank/pglistr_tumor/scripts/..//build/last/inverse -nx 128 -ny 128 -nz 128 -beta 0.0001 -rho_inversion 15 -k_inversion 0.0 -nt_inversion 20 -dt_inversion 0.05 -rho_data 12 -k_data 0.05 -nt_data 100 -dt_data 0.01 -regularization L1c -interpolation 0 -diffusivity_inversion 1 -reaction_inversion 1 -basis_type 1 -number_gaussians 64 -sigma_factor 2 -sigma_spacing 2 -testcase 0 -gaussian_volume_fraction 0.99 -lambda_continuation 1 -target_sparsity 0.99 -sparsity_level 10 -threshold_data_driven 0.1 -sigma_data_driven 2 -output_dir /workspace/shashank/pglistr_tumor/scripts/..//results/check/ -newton_solver QN -newton_maxit 50 -gist_maxit 2 -krylov_maxit 30 -syn_flag 1 -data_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/128/cpl/c1p.nc -gm_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/128/gray_matter.nc -wm_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/128/white_matter.nc -csf_path /workspace/shashank/pglistr_tumor/scripts/..//brain_data/128/csf.nc -model 1 -smooth 1.5 -observation_threshold 0.0 -k_gm_wm 0.0 -r_gm_wm 0.0 -low_freq_noise 0.0 -prediction 1 -tao_lmm_vectors 50 -tao_lmm_scale_type broyden -tao_lmm_scalar_history 5 -tao_lmm_rescale_type scalar -tao_lmm_rescale_history 5