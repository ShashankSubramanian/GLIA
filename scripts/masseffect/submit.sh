#!/bin/bash
#### default script to run mass effect
PAT_DIR=/scratch1/05027/shas1693/tmi-results/
AT_DIR=/scratch1/05027/shas1693/adni-nc/
CODE_DIR=/work/05027/shas1693/frontera/pglistr_tumor/
RES_DIR=$CODE_DIR/results/
COMP_SYS=frontera
N=160

python3 run_masseffect_gridcont.py -p $PAT_DIR -a $AT_DIR -csys $COMP_SYS -c $CODE_DIR -x $RES_DIR
#python3 run_masseffect.py -p $PAT_DIR -a $AT_DIR -csys $COMP_SYS -c $CODE_DIR -x $RES_DIR -n $N

