BRATS=/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia
DSET=obs-train
OFF=${BRATS}/results/HGG-grid-cont/concomp-cm-DATA-cm_phi-grad-ls10/
OBS_REF='obs-1.0'
SCRIPT_DIR=/zhome/academic/HLRS/ipv/ipvscheu/software/tumor-tools/scripts
CODE_DIR=/zhome/academic/HLRS/ipv/ipvscheu/software/tumor-tools/3rdparty/pglistr_tumor/build/last


RHO_INIT=8
K_INIT=0

for ID in Brats18_CBICA_AXO_1 Brats18_CBICA_AQJ_1 Brats18_TCIA03_257_1 Brats18_TCIA04_328_1
do

  DIR=${OFF}/${ID}/tumor_inversion/nx256
  DIR_PREV=${OFF}/${ID}/tumor_inversion/nx128
  LAMBDA_OBS=1
  CMDATA_DIR=${DIR}/cm-data-obs-${LAMBDA_OBS}.0
  mkdir -p ${CMDATA_DIR}
  cd ${CMDATA_DIR}
  PVEC=${DIR}/${OBS_REF}/p-cm-data.txt
  GCM=${DIR}/${OBS_REF}/phi-cm-data.txt
cat <<EOF > job-submission.sh
#!/bin/bash
#PBS -N cm-data-cmp
#PBS -l nodes=4:ppn=24
#PBS -l walltime=10:00:00
#PBS -m abe
#PBS -M kscheufele@austin.utexas.edu

source /zhome/academic/HLRS/ipv/ipvscheu/env_intel.sh
cd ${CMDATA_DIR}
export OMP_NUM_THREADS=1
umask 002


# extract reconstructed rho, k
# python3 ${SCRIPT_DIR}/utils.py -extract -output_path ${DIR_PREV}/${OBS_REF}/
# source ${DIR_PREV}/${OBS_REF}/env_rhok.sh
export LAMBDA_OBS=1

aprun -n 96 -N 24  ${CODE_DIR}/inverse -nx 256 -ny 256 -nz 256 -beta 0.0001 \
   -multilevel 1 \
   -rho_inversion ${RHO_INIT} -k_inversion ${K_INIT}                                         \
   -nt_inversion 20 -dt_inversion 0.05 -rho_data 12 -k_data 0.05 -nt_data 100 -dt_data 0.01  \
   -regularization L1c -lambda_continuation 1 -target_sparsity 0.99 -sparsity_level 10       \
   -interpolation 0                                                                          \
   -diffusivity_inversion 1 -reaction_inversion 1                                            \
   -basis_type 1 -number_gaussians 64 -sigma_factor 2 -sigma_spacing 2  -gaussian_volume_fraction 0.99 -threshold_data_driven 0.1 -sigma_data_driven 1 \
   -testcase 0    \
   -solve_rho_k 1 \
   -output_dir    ${CMDATA_DIR}/                    \
   -data_path     ${DIR}/init/patient_seg_tc.nc     \
   -gm_path       ${DIR}/init/patient_seg_gm.nc     \
   -wm_path       ${DIR}/init/patient_seg_wm_wt.nc  \
   -csf_path      ${DIR}/init/patient_seg_csf.nc    \
   -obs_mask_path ${DIR}/init/obs_mask_lbd-${LAMBDA_OBS}.nc \
   -support_data_path    \
   -data_comp_path       \
   -data_comp_dat_path   \
   -gaussian_cm_path ${GCM}  -pvec_path ${PVEC}     \
   -model 1 -smooth 1.5 -observation_threshold 0.0 -k_gm_wm 0.0 -r_gm_wm 0.0 -low_freq_noise 0.0 -prediction 1 -forward 0 -order 2 -verbosity 3                  \
   -newton_solver QN -line_search armijo  -newton_maxit 50 -gist_maxit 2 -krylov_maxit 30 -rel_grad_tol 1e-04 -syn_flag 0                                        \
   -tao_lmm_vectors 50 -tao_lmm_scale_type broyden -tao_lmm_scalar_history 5 -tao_lmm_rescale_type scalar -tao_lmm_rescale_history 5  -tumor_tao_ls_max_funcs 10 \
   &> ${CMDATA_DIR}/tumor_solver_log_nx256-cm-data-comparison-obs-${LAMBDA_OBS}.txt

# NOT USED: 
# -support_data_path  ${DIR}/init/support_data.nc  \
# -data_comp_path     ${DIR}/init/data_comps.nc    \
# -data_comp_dat_path ${DIR}/${OBS_REF}/dcomp.dat  \

# postproc, compute dice
#python ${SCRIPT_DIR}/postprocess.py -input_path ${OFF}/${ID}/ -reference_image_path /lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/training/HGG/${DSET}/${ID}/${ID}_seg_tu.nii.gz -patient_labels 0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm -convert_images  -compute_tumor_stats -tu_path tumor_inversion/nx256/obs-${LAMBDA_OBS}/
EOF
   qsub job-submission.sh
done  

