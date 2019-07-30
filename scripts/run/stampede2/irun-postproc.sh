#!/bin/bash
set -x
#set -e

#CONVERT_IMG=' - convert_images' 
CONVERT_IMG=' -convert_images '
BASE=/scratch/04678/scheufks/brats18/results
TRAIN=/scratch/04678/scheufks/brats18/training/HGG
SCRIPT=/work/04678/scheufks/stampede2/code/tumor-tools/scripts


cd $1
for DIR in Brats*;
do
  #BID="$(cut -d'-' -f1 <<<"$DIR")"
  BID=$DIR
  echo "$BID"
  rm -rf ${BASE}/$1/$DIR/vis*
  python3 ${SCRIPT}/postprocess.py -input_path "${BASE}/$1/$DIR" -reference_image_path "${TRAIN}/ALL/${BID}/${BID}_seg_tu.nii.gz" -patient_labels "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"  ${CONVERT_IMG}  -gridcont  -compute_tumor_stats -analyze_concomps -generate_slices
done
cd ..
