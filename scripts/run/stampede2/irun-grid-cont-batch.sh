BRATS=/scratch/04678/scheufks/brats18/
#BRATS=/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/

#### DATA SETS ###
#DSET=b-2013-1     # stamp
DSET=b-2013-2     # stamp
#DSET=b-2013-3     # stamp
#DSET=b-tcia08   # 
#DSET=b-tcia06   # 
#DSET=b-tcia05   # 
#DSET=b-tcia04   # 
#DSET=b-tcia03   # 
#DSET=b-tcia02   # 
#DSET=b-cbica    # 
#DSET=b-cbica-1  # 
#DSET=b-cbica-2  #
#DSET=b-cbica-3  # 
#DSET=b-cbica-4  # 
#DSET=b-cbica-5  #

#DSET=b-obs-train

RES=${BRATS}/results/brats19-cc-supphi-kbound-s5-gradls10-beta-1e-4/
mkdir -p ${RES}
python3  run_gridcont.py -patient_path   ${BRATS}/training/HGG/${DSET}               \
                         -atlas_path     ${BRATS}/atlas/jakob_segmented_with_cere_lps_240240155_bratslabels.nii.gz \
                         -cluster stampede2                                          \
                         --use_patient_segmentation                                  \
                         --use_atlas_segmentation                                    \
                         -patient_labels 0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm   \
                         -atlas_labels   0=bg,8=csf,7=vt,5=gm,6=wm                   \
                         -x              ${RES}                                      \
                         -nx 256                                                     \
                         --obs_lambda 1                                              \
                         --vary_obs_lambda                                           \
                         --multiple_patients

