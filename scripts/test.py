from utils.file_io import readNetCDF,createNetCDF
import numpy as np
import os






path = '/work2/07544/ghafouri/frontera/gits/GLIA_CMA_Py/syndata/'

syn = 'case4'

en = readNetCDF(os.path.join(path, syn, '160', 'en_t1.nc'))
nec = readNetCDF(os.path.join(path, syn, '160', 'nec_t1.nc'))
tc = en+nec

createNetCDF(os.path.join(path, syn, '160', 'tc_t1.nc'), tc.shape, tc)








vt = readNetCDF(os.path.join(path, 'seg_t1.nc'))
vt[vt != 7] = 0
vt[vt == 7] = 1
w = np.sum((vt >0))
print(w)
print(1/w)



ed = readNetCDF(os.path.join(path, 'ed_t1.nc'))
w = np.sum((ed >0))
print(w)
print(1/w)


norm = np.linalg.norm(ed)

print(norm**2)
