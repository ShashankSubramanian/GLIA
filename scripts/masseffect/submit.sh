#!/bin/bash
#### default script to run mass effect

#PAT_DIR=/scratch/05027/shas1693/penn_gbm_survival20/Data/
AT_DIR=/scratch/05027/shas1693/adni-nc/
CODE_DIR=/scratch/05027/shas1693/pglistr_tumor/
CLAIRE_DIR=/scratch/05027/shas1693/claire-dev/bingpu/
RES_DIR=$CODE_DIR/results/syn_rec_ctil_256/
JOB_DIR=$CODE_DIR/results/syn_rec_jobs/
COMP_SYS=longhorn
N=256
reg=1
PAT_DIR=$CODE_DIR/results/syn/

#python3 run_masseffect.py -p $PAT_DIR -a $AT_DIR -csys $COMP_SYS -c $CODE_DIR -x $RES_DIR -n $N -r $reg -rc $CLAIRE_DIR -syn
#python3 run_masseffect.py -p $PAT_DIR -a $AT_DIR -csys $COMP_SYS -c $CODE_DIR -x $RES_DIR -n $N -r $reg -rc $CLAIRE_DIR
python3 run_multiple_patients.py -p $PAT_DIR -csys $COMP_SYS -c $CODE_DIR -x $RES_DIR -j $JOB_DIR
#python3 run_masseffect.py -p $PAT_DIR -a $AT_DIR -csys $COMP_SYS -c $CODE_DIR -x $RES_DIR -n $N -r $reg -rc $CLAIRE_DIR -submit

