#!/bin/bash

CODEDIR=/scratch/05027/shas1693/pglistr_tumor/
REGCODEDIR=/scratch/05027/shas1693/claire-dev/
RES=$CODEDIR/results/rd-ce-/
mkdir -p ${RES}
python3  run_gridcont.py -patient_path $CODEDIR/brain_data/ce_test/Brats18_TCIA03_257_1_seg_tu_aff2jakob.nii.gz            \
                      -atlas_path     $CODEDIR/brain_data/jakob/jakob.nii.gz \
                      -cluster maverick2 \
                      --use_patient_segmentation                                  \
                      --use_atlas_segmentation                                    \
                      -patient_labels 0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm   \
                      -atlas_labels   0=bg,8=csf,7=vt,5=gm,6=wm                   \
                      -x              ${RES}                                      \
                      -nx 256                                                     \
                      --obs_lambda 1                                              \
                      --tumor_code_dir      ${CODEDIR}                            \
                      --reg_code_dir ${REGCODEDIR}
