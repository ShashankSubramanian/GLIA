#!/bin/bash
SCRATCHDIR=/scratch/05027/shas1693
PAT_DIR=$SCRATCHDIR/pglistr_tumor/checkreal/
AT_DIR=$SCRATCHDIR/adni-nc/
CODEDIR=$SCRATCHDIR/pglistr_tumor/
scripts_path=$CODEDIR/scripts/masseffect
res_path=$CODEDIR/results/test_me_gpu/
sz=160
at_stat_path=$AT_DIR/adni-atlas-stats.csv
pat_stat_path=$CODEDIR/checkreal/pat_stats.csv

python3 $scripts_path/extract_stats.py -n $sz -results_path $res_path -atlas_stat_path $at_stat_path -patient_stat_path $pat_stat_path -patient_dir $PAT_DIR
