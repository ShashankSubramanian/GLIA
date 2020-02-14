#!/bin/bash
#SBATCH -J claire
#SBATCH -p normal
#SBATCH -o stdout.o
#SBATCH -e stderr.e
#SBATCH -N 2
#SBATCH -n 96
#SBATCH -t 0:30:00
#SBATCH -A FTA-Biros



#symlink the normal_p in 256x256x124 resolution
ln -sf ../../../698_templates_nifti_256x256x124/0110Y02_segmented_aff2jakob_256x256x124.nii.gz normal_p_256x256x124.nii.gz


#Resample the BraTS_2019_subject
ln -sf ../../../training/HGG/BraTS19_TCIA02_471_1/BraTS19_TCIA02_471_1_seg_tu_dl_256x256x124.nii.gz BraTS19_TCIA02_471_1_seg_tu_dl_256x256x124.nii.gz


python3 /home1/04716/naveen15/pglistr_tumor/scripts/data-gen/fix_tumor_init_condition.py --atlas /work/04716/naveen15/frontera/698_normal_brains_segmented/mri_images_nifti_256x256x256_aff2jakob/0110Y02_segmented_256x256x256_aff2jakob.nii.gz --centers /work/04716/naveen15/frontera/brats18/results/brats19-gridcont-concomp-cosamp/Brats18_TCIA02_471_1/tumor_inversion/nx256/obs-1.0/phi-mesh-scaled.txt --activations /work/04716/naveen15/frontera/brats18/results/brats19-gridcont-concomp-cosamp/Brats18_TCIA02_471_1/tumor_inversion/nx256/obs-1.0/p-rec-scaled.txt --output /scratch1/04716/naveen15/brats19_syn_tumor/HGG/BraTS19_TCIA02_471_1/tuME-rho-12.87-kappa-2.54E-02-gamma-90000.0

#Check if c0.nc exists in the output_path
if [ ! -f /scratch1/04716/naveen15/brats19_syn_tumor/HGG/BraTS19_TCIA02_471_1/tuME-rho-12.87-kappa-2.54E-02-gamma-90000.0/c0.nc ]; then
echo /scratch1/04716/naveen15/brats19_syn_tumor/HGG/BraTS19_TCIA02_471_1/tuME-rho-12.87-kappa-2.54E-02-gamma-90000.0/c0.nc does not exist
exit;
fi

#Tumor forward command
ibrun /home1/04716/naveen15/pglistr_tumor/build/last/inverse -nx 256 -ny 256 -nz 256 -beta 0 -multilevel 0 -inject_solution 0 -pre_reacdiff_solve 0 -rho_inversion 15 -k_inversion 0.0 -nt_inversion 20 -dt_inversion 0.05 -rho_data 12.87 -k_data 0.0254 -nt_data 100 -dt_data 0.01 -regularization L1c -interpolation 0 -diffusivity_inversion 1 -reaction_inversion 1 -basis_type 1 -number_gaussians 64 -sigma_factor 2 -sigma_spacing 2 -testcase 4 -solve_rho_k 0 -gaussian_volume_fraction 0.99 -lambda_continuation 1 -target_sparsity 0.99 -sparsity_level 10 -threshold_data_driven 0.1 -sigma_data_driven 2 -output_dir /scratch1/04716/naveen15/brats19_syn_tumor/HGG/BraTS19_TCIA02_471_1/tuME-rho-12.87-kappa-2.54E-02-gamma-90000.0/ -newton_solver QN -line_search mt -newton_maxit 50 -gist_maxit 2 -krylov_maxit 1 -rel_grad_tol 1e-05 -syn_flag 1 -data_path /home1/04716/naveen15/pglistr_tumor/brain_data/128/cpl/c1p.nc -gm_path /scratch1/04716/naveen15/brats19_syn_tumor/698_templates_netcdf_256x256x256/0110Y02_seg_gm.nc -wm_path /scratch1/04716/naveen15/brats19_syn_tumor/698_templates_netcdf_256x256x256/0110Y02_seg_wm.nc -csf_path /scratch1/04716/naveen15/brats19_syn_tumor/698_templates_netcdf_256x256x256/0110Y02_seg_ve.nc -glm_path /scratch1/04716/naveen15/brats19_syn_tumor/698_templates_netcdf_256x256x256/0110Y02_seg_csf.nc -z_cm 112 -y_cm 136 -x_cm 144 -obs_mask_path  -support_data_path  -gaussian_cm_path  -pvec_path  -data_comp_path  -data_comp_dat_path  -init_tumor_path /scratch1/04716/naveen15/brats19_syn_tumor/HGG/BraTS19_TCIA02_471_1/tuME-rho-12.87-kappa-2.54E-02-gamma-90000.0/c0.nc -model 4 -smooth 1.5 -observation_threshold -100 -k_gm_wm 0.0 -r_gm_wm 0.0 -low_freq_noise 0.0 -prediction 0 -forward 1 -order 2 -verbosity 1 -forcing_factor 90000.0 -kappa_lb 0.0 -kappa_ub 0.1 -tao_lmm_vectors 50 -tao_lmm_scale_type broyden -tao_lmm_scalar_history 5 -tao_lmm_rescale_type scalar -tao_lmm_rescale_history 5  -tumor_tao_ls_max_funcs 10 

#Convert final segmentation to nifti
python3 /home1/04716/naveen15/pglistr_tumor/scripts/grid-cont/utils.py -convert_netcdf_to_nii --name_old /scratch1/04716/naveen15/brats19_syn_tumor/HGG/BraTS19_TCIA02_471_1/tuME-rho-12.87-kappa-2.54E-02-gamma-90000.0/seg_t[50].nc --name_new /scratch1/04716/naveen15/brats19_syn_tumor/HGG/BraTS19_TCIA02_471_1/tuME-rho-12.87-kappa-2.54E-02-gamma-90000.0/abnormal_p_256x256x256.nii.gz --reference_image /scratch1/04716/naveen15/brats19_syn_tumor/HGG/BraTS19_TCIA02_471_1/tuME-rho-12.87-kappa-2.54E-02-gamma-90000.0/c0.nii.gz