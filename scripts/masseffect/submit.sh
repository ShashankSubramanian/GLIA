#!/bin/bash
#### default script to run mass effect
PAT_DIR=/scratch/05027/shas1693/tmi-results/
AT_DIR=/scratch/05027/shas1693/adni-nc/
CODE_DIR=/scratch/05027/shas1693/pglistr_tumor/
CLAIRE_DIR=/scratch/05027/shas1693/claire-dev/bingpu/
RES_DIR=$CODE_DIR/results/noreg/
COMP_SYS=longhorn
N=160
reg=0

#python3 run_masseffect_gridcont.py -p $PAT_DIR -a $AT_DIR -csys $COMP_SYS -c $CODE_DIR -x $RES_DIR
python3 run_masseffect.py -p $PAT_DIR -a $AT_DIR -csys $COMP_SYS -c $CODE_DIR -x $RES_DIR -n $N -r $reg -rc $CLAIRE_DIR
##python3 run_masseffect.py -p $PAT_DIR -a $AT_DIR -csys $COMP_SYS -c $CODE_DIR -x $RES_DIR -n $N -r $reg -rc $CLAIRE_DIR -submit

