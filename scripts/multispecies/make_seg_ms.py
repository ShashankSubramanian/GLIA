

import numpy as np
import os,subprocess

import sys
scripts_path = os.path.dirname(os.path.realpath(__file__))
code_dir = scripts_path + '/../../'
sys.path.append(code_dir)
from scripts.utils.file_io import readNetCDF, createNetCDF 

#res_path = os.path.join(code_dir, 'syndata','160', 'ed_inv')

prefix = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/scripts/test'

 

ed = readNetCDF(os.path.join(prefix, 'ed_rec_final.nc'))
infl = readNetCDF(os.path.join(prefix, 'i_rec_final.nc'))
nec = readNetCDF(os.path.join(prefix, 'nec_rec_final.nc'))
en = readNetCDF(os.path.join(prefix, 'en_rec_final.nc'))
seg = readNetCDF(os.path.join(prefix, 'seg_rec_final.nc'))


seg_all = seg.copy()


total = np.maximum(infl, nec)
total = np.maximum(total, en)

seg_tmp = np.zeros(seg_all.shape)
seg_tmp[total == nec] = 1
seg_tmp[total == en] = 4
seg_tmp[seg_all != 1] = 0
#seg_tmp[total == ed] = 2
tmp1 = (total == infl)
tmp2 = (seg_all != 7)
tmp3 = (seg_all != 8)
tmp4 = (ed >= 0.5)
#tmp4 = (infl >= 0.09856694224220497)
tmp = tmp1 * tmp2 * tmp3 * tmp4
#seg_tmp[seg_all >= 5] = 0
#seg_tmp[seg == 0] = 0
#seg_all[total == ed] = 0
seg_all[seg_all == 1] = 0
seg_all += seg_tmp
seg_all[tmp > 0] = 2


createNetCDF(os.path.join(prefix, 'seg_all_rec_final.nc'), seg_all.shape, seg_all)


        











