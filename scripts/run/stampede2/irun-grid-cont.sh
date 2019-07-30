BRATS=/scratch/04678/scheufks/brats18/
#BRATS=/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/

DSET=ALL

#for ID in  Brats18_CBICA_AQD_1 Brats18_CBICA_AQR_1 Brats18_CBICA_AQA_1  # Brats18_CBICA_AQO_1 Brats18_CBICA_AQP_1 Brats18_TCIA03_257_1 Brats18_TCIA08_406_1  # Brats18_CBICA_AQJ_1 #  Brats18_TCIA04_328_1
#for ID in  Brats18_CBICA_AQR_1  Brats18_TCIA08_406_1  Brats18_CBICA_AQJ_1 Brats18_TCIA03_257_1 Brats18_CBICA_AQA_1 Brats18_CBICA_AQD_1 Brats18_CBICA_AQO_1 Brats18_CBICA_AQP_1
#for ID in Brats18_CBICA_AMH_1 Brats18_CBICA_ANG_1 Brats18_CBICA_ANI_1 Brats18_CBICA_ANP_1 Brats18_CBICA_ANZ_1 Brats18_CBICA_AZH_1
#for ID in Brats18_CBICA_AXO_1 Brats18_CBICA_AQJ_1 Brats18_TCIA03_257_1 Brats18_TCIA04_328_1
#for ID in Brats18_CBICA_ABM_1 Brats18_CBICA_ABY_1 Brats18_CBICA_ASN_1 Brats18_CBICA_AAG_1 Brats18_CBICA_ASA_1 Brats18_CBICA_ASH_1 Brats18_CBICA_ASO_1 Brats18_CBICA_ASW_1 Brats18_CBICA_AYI_1 Brats18_CBICA_ABO_1 Brats18_CBICA_APY_1 Brats18_CBICA_ASE_1 Brats18_CBICA_ASK_1 Brats18_CBICA_ASU_1

for ID in Brats18_2013_11_1 # Brats18_TCIA03_257_1
do
  RES=${BRATS}/results/brats19-cc-supphi-kbound-s5-gradls10-beta-1e-4/${ID}
  mkdir -p ${RES}
  python3  run_gridcont.py -patient_path   ${BRATS}/training/HGG/${DSET}/${ID}/${ID}_seg_tu.nii.gz                   \
                           -atlas_path     ${BRATS}/atlas/jakob_segmented_with_cere_lps_240240155_bratslabels.nii.gz \
                           -cluster stampede2                                          \
                           --use_patient_segmentation                                  \
                           --use_atlas_segmentation                                    \
                           -patient_labels 0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm   \
                           -atlas_labels   0=bg,8=csf,7=vt,5=gm,6=wm                   \
                           -x              ${RES}                                      \
                           -nx 256                                                     \
                           --obs_lambda 1                                              \
                           --vary_obs_lambda
done
