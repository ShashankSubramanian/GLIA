BRATS=/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/

#for ID in Brats18_CBICA_AME_1 Brats18_TCIA08_162_1 Brats18_TCIA06_211_1

#for ID in Brats18_TCIA02_471_1 Brats18_TCIA02_321_1 Brats18_CBICA_AQJ_1 Brats18_TCIA02_171_1 Brats18_CBICA_AQR_1 Brats18_TCIA02_135_1 Brats18_TCIA08_242_1 Brats18_TCIA08_406_1 Brats18_CBICA_AOZ_1 Brats18_CBICA_AOZ_1 Brats18_2013_4_1 Brats18_CBICA_AWG_1 Brats18_CBICA_ASG_1 Brats18_CBICA_APY_1 Brats18_2013_25_1 Brats18_CBICA_AXQ_1 Brats18_2013_10_1


#for ID in Brats18_2013_19_1  Brats18_TCIA02_368_1  Brats18_TCIA03_296_1
#for ID in Brats18_2013_19_1  Brats18_TCIA03_257_1
#for ID in Brats18_CBICA_AXO_1 # Brats18_CBICA_ANG_1 Brats18_TCIA03_257_1
#for ID in Brats18_CBICA_AXO_1 Brats18_CBICA_AQJ_1 Brats18_TCIA03_257_1 Brats18_TCIA04_328_1 Brats18_TCIA03_296_1
#for ID in Brats18_TCIA04_328_1  Brats18_CBICA_AXO_1  Brats18_TCIA03_257_1 Brats18_CBICA_AQJ_1 
for ID in  Brats18_CBICA_AXO_1 Brats18_TCIA03_257_1
do
  #RES=${BRATS}/results/brats19-grad-sol-out/${ID}
  #RES=${BRATS}/results/brats19-GNK-hessian/${ID}
  #RES=${BRATS}/results/new_brats19-sigma-2h-inject-sol/${ID}
  #RES=${BRATS}/results/new_brats19-sigma-h-inject-sol-pre-rho-kappa/${ID}
  RES=${BRATS}/results/new_regularization_brats19-sigma-h-inject-sol-pre-rho-kappa-double-tstep-grad-rs-beta-0/${ID}
  #RES=${BRATS}/results/brats19-cm-data-phi-bump/${ID}

  mkdir -p ${RES}
  python  run_gridcont.py -patient_path   ${BRATS}/training/HGG/ALL/${ID}/${ID}_seg_tu.nii.gz               \
                      -atlas_path     ${BRATS}/atlas/jakob_segmented_with_cere_lps_240240155_bratslabels.nii.gz \
                      -cluster hazelhen                                           \
                      --use_patient_segmentation                                  \
                      --use_atlas_segmentation                                    \
                      -patient_labels 0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm   \
                      -atlas_labels   0=bg,8=csf,7=vt,5=gm,6=wm                   \
                      -x              ${RES}                                      \
                      -nx 256                                                     \
                      --obs_lambda 1                                              \
                      --vary_obs_lambda
done
