#!/bin/bash
#PBS -N postproc
#PBS -l nodes=1:ppn=24
#PBS -l walltime=05:00:00
#PBS -m abe
#PBS -M kscheufele@austin.utexas.edu

source /zhome/academic/HLRS/ipv/ipvscheu/env_intel.sh
cd /lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/results/HGG-grid-cont/brats19-cc-supphi-kbound-s5-gradls10
export OMP_NUM_THREADS=1
umask 002

for DIR in Brats*;
do
  #BID="$(cut -d'-' -f1 <<<"$DIR")"
  BID=$DIR
  echo "$BID"
  rm -rf /lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/results/HGG-grid-cont/brats19-cc-supphi-kbound-s5-gradls10/$DIR/vis*
  python /zhome/academic/HLRS/ipv/ipvscheu/software/tumor-tools/scripts/postprocess.py -input_path "/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/results/HGG-grid-cont/brats19-cc-supphi-kbound-s5-gradls10/$DIR" -reference_image_path "/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/training/HGG/ALL/${BID}/${BID}_seg_tu.nii.gz" -patient_labels "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"  -convert_images  -gridcont  -compute_tumor_stats -analyze_concomps -generate_slices 

  #python /zhome/academic/HLRS/ipv/ipvscheu/software/tumor-tools/scripts/pp.py -input_path "/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/results/HGG-grid-cont/$1/$DIR" -reference_image_path "/lustre/cray/ws9/4/ws/ipvscheu-ws-sibia/training/HGG/ALL/${BID}/${BID}_seg_tu.nii.gz" -patient_labels "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm" -rdir "cm-data" ${CONVERT_IMG} -compute_tumor_stats -analyze_concomps -generate_slices
done
cd ..

