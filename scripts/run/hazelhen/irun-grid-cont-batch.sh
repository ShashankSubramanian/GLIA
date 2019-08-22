BRATS=/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/
#DSET=b-2013     # stamp (r) 
#DSET=b-tcia08   # hazel (r) 
#DSET=b-tcia06   # hazel (r) 
#DSET=b-tcia05   # hazel (r)
#DSET=b-tcia04   # hazel (r)
#DSET=b-tcia03   #
#DSET=b-tcia02   # hazel (r)
#DSET=b-tcia021  # 
#DSET=b-cbica    # hazel (r)
#DSET=b-cbica-1  # hazel (r)
#DSET=b-cbica-2  # hazel (r)
#DSET=b-cbica-3  # hazel (r)
#DSET=b-cbica-4  # hazel (r)
#DSET=b-cbica-5  # hazel (r) 

DSET=BCHUNK-0

RES=${BRATS}/results/BRATS19_NEW
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
                        --multiple_patients

