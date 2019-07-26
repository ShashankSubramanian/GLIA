BRATS=/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/
#DSET=b-2013     # 
#DSET=b-tcia08   # 
#DSET=b-tcia06   # 
#DSET=b-tcia05   # 
#DSET=b-tcia04   # hazel (r)
#DSET=b-tcia03   # hazel (r)
#DSET=b-tcia02   # 
#DSET=b-cbica    # hazel (r)
#DSET=b-cbica-1  # hazel (r)
#DSET=b-cbica-2  # hazel (r)
DSET=b-cbica-3  # hazel
#DSET=b-cbica-4  # 
#DSET=b-cbica-5  # hazel (r) 

#DSET=b-obs-train

TOL=1E-4
RES=${BRATS}/results/HGG-grid-cont/brats19-cc-supphi-kbound-s5-gradls10/
mkdir -p ${RES}
python  run_gridcont.py -patient_path   ${BRATS}/training/HGG/${DSET}             \
                        -atlas_path     ${BRATS}/atlas/jakob_segmented_with_cere_lps_240240155_bratslabels.nii.gz \
                        -cluster hazelhen                                           \
                        --use_patient_segmentation                                  \
                        --use_atlas_segmentation                                    \
                        -patient_labels 0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm   \
                        -atlas_labels   0=bg,8=csf,7=vt,5=gm,6=wm                   \
                        -x              ${RES}                                      \
                        -nx 256                                                     \
                        --obs_lambda 1                                              \
                        --opttol ${TOL}                                             \
                        --vary_obs_lambda                                           \
                        --multiple_patients

