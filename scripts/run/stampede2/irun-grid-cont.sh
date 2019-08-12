BRATS=/scratch/04678/scheufks/brats18/
#BRATS=/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/

DSET=ALL


for ID in Brats18_CBICA_ASA_1 # Brats18_2013_20_1 Brats18_CBICA_ASE_1 Brats18_CBICA_ASO_1 Brats18_TCIA03_419_1 Brats18_TCIA06_165_1 Brats18_CBICA_ASY_1 Brats18_2013_11_1 Brats18_CBICA_AQG_1 Brats18_CBICA_AQG_1 Brats18_TCIA05_444_1 Brats18_TCIA02_368_1 Brats18_TCIA06_603_1 Brats18_TCIA03_265_1 Brats18_TCIA08_469_1 Brats18_TCIA08_469_1 Brats18_CBICA_BHM_1 Brats18_CBICA_AQU_1 Brats18_CBICA_AQQ_1 Brats18_TCIA03_296_1 Brats18_TCIA03_296_1 Brats18_CBICA_AMH_1 Brats18_CBICA_AMH_1 Brats18_2013_22_1 Brats18_TCIA03_474_1 Brats18_TCIA08_162_1 Brats18_2013_19_1 Brats18_2013_19_1 Brats18_2013_19_1 Brats18_TCIA02_290_1 Brats18_CBICA_ABO_1 Brats18_CBICA_ABO_1 Brats18_TCIA05_277_1 Brats18_TCIA05_277_1 Brats18_2013_21_1 Brats18_CBICA_AME_1 Brats18_TCIA06_211_1 Brats18_TCIA06_211_1 Brats18_CBICA_AVJ_1 Brats18_TCIA02_605_1 Brats18_CBICA_ASV_1 Brats18_2013_14_1 Brats18_TCIA02_374_1 Brats18_CBICA_AYU_1 Brats18_TCIA02_274_1 Brats18_CBICA_AQY_1
 
do
  RES=${BRATS}/results/brats19-cc-supphi-kbound-s5mod-gradls10-beta-1e-4/${ID}
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
