from tools import *
import argparse

code_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
data_path = code_dir + 'syn_data1/data/'
results_dir = code_dir + 'd_nc-80/'

DIRS = os.listdir(results_dir + '/.')
TAB = []
meth = 'RD'
NOISE = {
      '0.0' : {'d1' : 0,  'd0' : 0}, 
      '0.1' : {'d1' : 8,  'd0' : 45},
      '0.5' : {'d1' : 14, 'd0' : 45},
      '1.0'   : {'d1' : 24, 'd0' : 45},
      '1.5' : {'d1' : 35, 'd0' : 45},
      '2.0' : {'d1' : 46, 'd0' : 46}
      }
for dir in DIRS:
    print(dir) 
    # Reading log file
    n_it = 0
    rho_inv = 0
    k_inv = 0
    eps_k = 0
    eps_rho = 0
    miss = 0
    time = 0
    descr = ''
    init_r = 6
    init_k = 1e-1
    tpoint = 0
    success = False
    d0 = str(NOISE[dir]['d0']) + '\%' 
    d1 = str(NOISE[dir]['d1']) + '\%' 
    d_path = data_path + '../tc/d_nc-0/'
    d1_true = readNetCDF(results_dir +dir +  '/d1.nc')
    d_path = data_path + '../tc/t=1.2/'
    d12_true = readNetCDF(d_path + 'dataBeforeObservation.nc')
    d_path = data_path + '../tc/t=1.5/'
    d15_true = readNetCDF(d_path + 'dataBeforeObservation.nc')
    c1_rec = readNetCDF(results_dir + dir + '/c1_rec.nc')
    c12_rec = readNetCDF(results_dir + dir + '/c_pred_at_[t=0.3].nc')
    c15_rec = readNetCDF(results_dir + dir + '/c_pred_at_[t=0.6].nc')
    res = np.linalg.norm(c1_rec - d1_true)
    print(res)
    miss_t1   = res / np.linalg.norm(d1_true)
    print(np.linalg.norm(d1_true))
    print(miss_t1)
    miss_t12  = np.linalg.norm(c12_rec - d12_true) / np.linalg.norm(d12_true)
    miss_t15  = np.linalg.norm(c15_rec - d15_true) / np.linalg.norm(d15_true)
    descr = 'RD'     
    
    with open(os.path.join(results_dir+dir, 'log')) as f:
        for line in f.readlines():
            if "optimization done:" in line:
                n_it = int(line.split("#N-it:")[-1].split(",")[0])
                time = float(line.split("exec time:")[-1].split()[0])
                success=True
            if "r1: " in line:
                rho_inv = float(line.split("r1: ")[-1].split(",")[0])
            if "k1: " in line:
                k_inv = float(line.split("k1: ")[-1].split(",")[0])
            if "rel. l2-error (at observation points):" in line:
                miss = float(line.split("(at observation points): ")[-1].split()[0])
            eps_k = np.abs(k_inv - 0.18)/0.18
            eps_rho = np.abs(rho_inv - 8)/8
    if success:
       TAB.append("\textit{{{0:22s}}} & \textit{{{1:4d}}}  & {2:6s} & {3:6s}  & \textit{{{4:14s}}} & \num{{{5:1.0f}}}  & \num{{{6:e}}}  & {7:4.2f} & \num{{{8:e}}}& \num{{{9:e}}} & \num{{{10:e}}} & \num{{{11:e}}} & \num{{{12:e}}} & \num{{{13:e}}} & \num{{{14:e}}}  & {15:2d} & \num{{{16:e}}}  ".format(descr,tpoint, d1, d0, meth, init_r, init_k,  rho_inv, k_inv, eps_rho, eps_k, miss, miss_t1, miss_t12, miss_t15,  n_it, time))

TAB.sort()        
for t in TAB:
    print(t, " \\ ")



