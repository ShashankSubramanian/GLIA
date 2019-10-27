for ID in  patient1
do
  RES=${HOME}/Desktop/$ID
  mkdir -p ${RES}
  python3  run_gridcont.py -patient_path   /path/to/patient_segmentation.nii.gz        \
                           -atlas_path     /path/to/atlas_segmentation.nii.gz          \
                           -cluster cbica                                              \
                           --use_patient_segmentation                                  \
                           --use_atlas_segmentation                                    \
                           -patient_labels 0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm   \
                           -atlas_labels   0=bg,8=csf,7=vt,5=gm,6=wm                   \
                           -x              ${RES}                                      \
                           -nx 256                                                     \
                           --obs_lambda 1                                              \
                           --run_registration                                          \
                           --reg_code_dir /path/to/claire/repo                         \
                           --tumor_code_dir ${HOME}/Desktop/code/pglistr_tumor
done
