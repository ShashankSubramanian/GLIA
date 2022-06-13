#!/bin/bash
SCRATCHDIR=/scratch/07544/ghafouri
PAT_DIR=${SCRATCHDIR}/results/syndata/brats_dir/
AT_DIR=$SCRATCHDIR/data/adni-nc/
CODEDIR=/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/
scripts_path=$CODEDIR/scripts/masseffect
res_path=/scratch1/07544/ghafouri/results/syn_results/inv_p_temp_m/
sz=160
at_stat_path=$AT_DIR/adni-atlas-stats.csv
pat_stat_path=${PAT_DIR}/pat_stats.csv

python $scripts_path/extract_stats.py -n $sz -results_path $res_path -atlas_stat_path $at_stat_path -patient_stat_path $pat_stat_path -patient_dir $PAT_DIR
