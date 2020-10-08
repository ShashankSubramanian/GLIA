#/bin/bash

CODE_DIR=/workspace/shashank/pglistr_tumor/
RESDIR=$CODE_DIR/scripts/masseffect/
SURVDIR=/scratch/shashank/brats18/survival_data.csv
PATDIR=/scratch/data/tmi-results/
SCRATCHDIR=/scratch/shashank/
MEDIR=$SCRATCHDIR/miccai-results/full_4_pat/masseffect/
DATADIR=$MEDIR
#RESDIR=$CODE_DIR/scripts/masseffect/
#SURVDIR=/scratch/data/penn_gbm_survival20/info.csv
#PATDIR=/scratch/data/penn_gbm_survival20/Data/
#SCRATCHDIR=/scratch/shashank/penn/
#MEDIR=$SCRATCHDIR/penn_masseffect/
#NMEDIR=$SCRATCHDIR/penn_nomasseffect/
#DATADIR=$SCRATCHDIR/penn_masseffect/

#python3 analysis.py -p ${MEDIR} -x ${RESDIR} -s ${SURVDIR} -d ${DATADIR} -i ${PATDIR} -pnm ${NMEDIR}
python3 analysis.py -p ${MEDIR} -x ${RESDIR} -s ${SURVDIR} -d ${DATADIR} -i ${PATDIR}
