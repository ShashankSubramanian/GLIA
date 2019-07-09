#!/bin/bash
set -x
#set -e
  
cd $1
for DIR in Brats*;
do
  BID="$(cut -d'-' -f1 <<<"$DIR")"
  echo "$BID"
  python /zhome/academic/HLRS/ipv/ipvscheu/software/tumor-tools/scripts/postprocess.py -input_path "/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/results/HGG-grid-cont/$1/$DIR" -reference_image_path "/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/training/HGG/obs-train/${BID}/${BID}_seg_tu.nii.gz" -patient_labels "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm" -tu_path   "tumor_inversion/nx256/obs-1.0/"   -compute_tumor_stats -analyze_concomps -generate_slices
done
cd ..
