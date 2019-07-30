#!/bin/bash

#SBATCH -J postproc
#SBATCH -n 24
#SBATCH -p skx-normal
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH --mail-user=kscheufele@austin.utexas.edu
#SBATCH --mail-type=fail
#SBATCH -A PADAS
#SBATCH -o /scratch/04678/scheufks/brats18//results/brats19-cc-supphi-kbound-s5-gradls10-beta-1e-4//postproc.out 


source ~/.bashrc
#### define paths
DATA_DIR=/scratch/04678/scheufks/brats18//results/brats19-cc-supphi-kbound-s5-gradls10-beta-1e-4/
OUTPUT_DIR=/scratch/04678/scheufks/brats18//results/brats19-cc-supphi-kbound-s5-gradls10-beta-1e-4/
cd /scratch/04678/scheufks/brats18//results/brats19-cc-supphi-kbound-s5-gradls10-beta-1e-4/
export OMP_NUM_THREADS=1
umask 002

for DIR in Brats*;
do
  #BID="$(cut -d'-' -f1 <<<"$DIR")"
  BID=$DIR
  echo "$BID"
  rm -rf /scratch/04678/scheufks/brats18//results/brats19-cc-supphi-kbound-s5-gradls10-beta-1e-4/$DIR/vis*
  python /work/04678/scheufks/stampede2/code/tumor-tools/scripts/postprocess.py -input_path "/scratch/04678/scheufks/brats18//results/brats19-cc-supphi-kbound-s5-gradls10-beta-1e-4/$DIR" -reference_image_path "/scratch/04678/scheufks/brats18//training/HGG/ALL/${BID}/${BID}_seg_tu.nii.gz" -patient_labels "0=bg,1=nec,4=en,2=ed,8=csf,7=vt,5=gm,6=wm"  -convert_images  -gridcont  -compute_tumor_stats -analyze_concomps -generate_slices 
done
cd ..

