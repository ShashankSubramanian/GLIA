#/bin/bash

CODE_DIR=/scratch/05027/shas1693/pglistr_tumor/
RESDIR=$CODE_DIR/scripts/masseffect/
SURVDIR=/scratch/05027/shas1693/penn_gbm_survival20/info.csv
MEDIR=$CODE_DIR/results/penn_nomasseffect/
DATADIR=$CODE_DIR/results/penn_masseffect/

python3 analysis.py -p ${MEDIR} -x ${RESDIR} -s ${SURVDIR} -d ${DATADIR}
