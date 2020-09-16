from tools import *
import argparse

code_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
data_path = code_dir + 'real_data/data/'
results_dir = code_dir + 'inverse_RD_real_l2/'

case_dir = 'CASE_035_S_4114'

DIRS = os.listdir(results_dir + '/.')

TAB = []

meth = 'RD'

TF = {
        'CASE_035_S_4114' :  {'t1' : 0.42,  't2' : 0.63, 't02' : 1.63},
     }

th = 0.6
tol = ''

for dir in DIRS:
    
    # Reading log file
    n_it = 0
    rho_inv = 0
    k_inv = 0
    eps_k = 0
    eps_rho = 0
    miss = 0
    time = 0
    descr = dir
    init_r = 4
    init_k = 1e-2
    tpoint = 0
    success = False

    
    print('[] processing {}'.format(dir))

    d1_true = readNetCDF(results_dir + dir + '/d1.nc')
    d0_true = readNetCDF(results_dir + dir + '/d0.nc')
    d1_obs = np.where(d1_true > th, d1_true, 0) 
    d0_obs = np.where(d0_true > th, d0_true, 0)
    c0 = readNetCDF(results_dir + dir + '/c0_rec.nc')
    c1 = readNetCDF(results_dir + dir + '/c1_rec.nc')
    res_t1 = np.linalg.norm(c1-d1_true)
    res_t1_obs = np.linalg.norm(c1-d1_obs)
    miss_t1   = res_t1 / np.linalg.norm(d1_true)
    miss_t1_obs  = res_t1_obs / np.linalg.norm(d1_obs)
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
            if "l2-error in reconstruction:" in line:
                miss = float(line.split("l2-error in reconstruction:")[-1].split()[0])
            if "rel. l2-error (everywhere) :" in line:
                mu_1 = float(line.split('rel. l2-error (everywhere) (T=1.0) :')[-1].split()[0])

    if success:
        TAB.append("\\textit{{{0:22s}}}  & \\num{{{1:4.2f}}} &  \\num{{{2:e}}} & \\num{{{3:e}}}  & \\num{{{4:e}}} ".format(descr,  rho_inv, k_inv, miss_t1, miss_t1_obs))

        
for t in TAB:
    print(t, " \\\ ")



