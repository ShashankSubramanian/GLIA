#!/bin/bash
#SBATCH -J ITP
#SBATCH -o /scratch1/04678/scheufks/alzh/syn_test/test04/tc/t_series//log
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
source ~/.bashrc
export OMP_NUM_THREADS=1

module load cuda
module load cudnn
module load nccl
module load petsc/3.11-rtx
export ACCFFT_DIR=/work/04678/scheufks/frontera/libs/accfft/build_gpu/
export ACCFFT_LIB=${ACCFFT_DIR}/lib/
export ACCFFT_INC=${ACCFFT_DIR}/include/
export CUDA_DIR=${TACC_CUDA_DIR}/
ibrun /work/04678/scheufks/frontera/code/tumor-tools/3rdparty/pglistr_tumor/scripts/..//build/last/inverse_gpu_scale  -nx 256 -ny 256 -nz 256 -beta 0 -multilevel 0 -inject_solution 0 -pre_reacdiff_solve 0 -rho_inversion 10 -k_inversion 0.0 -nt_inversion 20 -dt_inversion 0.05 -rho_data 8 -k_data 0.18 -nt_data 100 -dt_data 0.01 -regularization L1c -interpolation 0 -diffusivity_inversion 1 -reaction_inversion 1 -basis_type 1 -number_gaussians 64 -sigma_factor 2 -sigma_spacing 2 -testcase 0 -solve_rho_k 0 -gaussian_volume_fraction 0.99 -lambda_continuation 1 -target_sparsity 0.99 -sparsity_level 10 -threshold_data_driven 0.1 -sigma_data_driven 2 -output_dir /scratch1/04678/scheufks/alzh/syn_test/test04/tc/t_series/ -newton_solver QN -line_search mt -newton_maxit 50 -gist_maxit 2 -krylov_maxit 1 -rel_grad_tol 1e-05 -syn_flag 1 -data_path_t1 /work/04678/scheufks/frontera/code/tumor-tools/3rdparty/pglistr_tumor/scripts/..//brain_data/256/cpl/c1p.nc -data_path_t0  -two_snapshot 0 -low_res_data 0 -gm_path /scratch1/04678/scheufks/alzh/syn_test/test04/data/0368Y02_seg_gm.nc -wm_path /scratch1/04678/scheufks/alzh/syn_test/test04/data/0368Y02_seg_wm.nc -csf_path /scratch1/04678/scheufks/alzh/syn_test/test04/data/0368Y02_seg_csf.nc -glm_path  -z_cm1 97 -y_cm1 138 -x_cm1 119 -cm1_s 1 -z_cm2 97 -y_cm2 138 -x_cm2 146 -cm2_s 1 -z_cm3 101 -y_cm3 213 -x_cm3 115 -cm3_s 1 -z_cm4 101 -y_cm4 200 -x_cm4 115 -cm4_s 1 -obs_mask_path  -support_data_path  -gaussian_cm_path  -pvec_path  -data_comp_path  -data_comp_dat_path  -init_tumor_path  -model 1 -smooth 1.5 -observation_threshold -100 -k_gm_wm 0.0 -r_gm_wm 0.0 -low_freq_noise 0.0 -prediction 0 -forward 1 -order 2 -verbosity 3 -forcing_factor 150000.0 -kappa_lb 0.0 -kappa_ub 0.1 -rho_lb 3 -rho_ub 15 -v_x1  -v_x2  -v_x3  -tao_lmm_vectors 3 -tao_lmm_scale_type broyden -tao_lmm_scalar_history 5 -tao_lmm_rescale_type scalar -tao_lmm_rescale_history 5  -tumor_tao_ls_max_funcs 10 