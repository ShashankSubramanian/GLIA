#/bin/bash

CODE_DIR=/workspace/shashank/pglistr_tumor/
RESDIR=$CODE_DIR/scripts/masseffect/
SURVDIR=/scratch/data/penn_gbm_survival20/info.csv
SCRATCHDIR=/scratch/shashank/penn/
MEDIR=$SCRATCHDIR/penn_nomasseffect/
DATADIR=$SCRATCHDIR/penn_masseffect/

python3 analysis.py -p ${MEDIR} -x ${RESDIR} -s ${SURVDIR} -d ${DATADIR} -pnm ${NMEDIR}
