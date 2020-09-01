#/bin/bash

RESDIR=/workspace/shashank/pglistr_tumor/scripts/masseffect/
SURVDIR=/scratch/data/penn_gbm_survival20/info.csv
MEDIR=/scratch/shashank/penn/penn_masseffect/

python3 analysis.py -p ${MEDIR} -x ${RESDIR} -s ${SURVDIR}
