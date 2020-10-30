#/bin/bash


#CODE_DIR=/workspace/shashank/pglistr_tumor/
#RESDIR=$CODE_DIR/scripts/masseffect/
#SURVDIR=/scratch/shashank/brats18/survival_data.csv
#PATDIR=/scratch/data/tmi-results/
#SCRATCHDIR=/scratch/shashank/
SCRATCHDIR=/scratch/05027/shas1693/
CODE_DIR=$SCRATCHDIR/pglistr_tumor/
#MEDIR=/localscratch/shashank/me_inversion/brats/
#MEDIR=$SCRATCHDIR/miccai-results/full_4_pat/masseffect/
#DATADIR=$MEDIR
RESDIR=$CODE_DIR/scripts/masseffect/
#SURVDIR=/scratch/data/penn_gbm_survival20/info.csv
PATDIR=$CODE_DIR/results/syn/
#SCRATCHDIR=/scratch/shashank/penn/
#MEDIR=$SCRATCHDIR/penn_masseffect/
#NMEDIR=$SCRATCHDIR/penn_nomasseffect/
#DATADIR=$SCRATCHDIR/penn_masseffect/

MEDIR=$CODE_DIR/results/syn_rec_ctil_256/
DATADIR=$MEDIR
MNI=$SCRATCHDIR/MNI/aff2jakob/MNI_seg_aff2jakob.nii.gz
#MNI=/scratch/data/Atlases/MNI/aff2jakob/MNI_seg_aff2jakob.nii.gz

#python3 analysis.py -p ${MEDIR} -x ${RESDIR} -s ${SURVDIR} -d ${DATADIR} -i ${PATDIR} -pnm ${NMEDIR} -mni $MNI
#python3 analysis.py -p ${MEDIR} -x ${RESDIR} -s ${SURVDIR} -d ${DATADIR} -i ${PATDIR} -pnm ${NMEDIR}
python3 analysis.py -p ${MEDIR} -x ${RESDIR} -d ${DATADIR} -i ${PATDIR} -mni $MNI -n 256
