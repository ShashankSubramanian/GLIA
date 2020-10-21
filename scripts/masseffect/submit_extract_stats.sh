#!/bin/bash
SCRATCHDIR=/scratch/05027/shas1693
#SCRATCHDIR=/scratch/
PAT_DIR=$SCRATCHDIR/penn_gbm_survival20/Data/
AT_DIR=$SCRATCHDIR/adni-nc/
#CODEDIR=/workspace/shashank/pglistr_tumor/
CODEDIR=$SCRATCHDIR/pglistr_tumor/
scripts_path=$CODEDIR/scripts/masseffect
res_path=$CODEDIR/results/syn_rec_ctil_256/
#res_path=$SCRATCHDIR/shashank/miccai-results/full_4_pat/masseffect/
sz=256
at_stat_path=$AT_DIR/adni-atlas-stats.csv
pat_stat_path=$CODEDIR/results/syn/pat_stats.csv
#pat_stat_path=$SCRATCHDIR/data/penn_gbm_survival20/Data/pat_stats.csv

python3 $scripts_path/extract_stats.py -n $sz -results_path $res_path -atlas_stat_path $at_stat_path -patient_stat_path $pat_stat_path -patient_dir $PAT_DIR
