import os
from os import listdir
import numpy as np
import statistics as st
import ntpath
import argparse
import shutil
import csv
import pandas as pd
import re
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.spatial import distance
from tabulate import tabulate
import pdfkit as pdf
import sqlite3
import math
import time
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

anthrazit = (0.2431, 0.2666, 0.2980)
mittelblau = (0., 0.31764, 0.61960)
hellblau = (0., 0.7529411, 1.)
signalred = (1., 0., 0.12549)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def getSurvivalClass(x):
    m = x/30.
    if m < 10:
        return "short"
    elif m < 15:
        return "mid"
    else:
        return "long"

def getSurvivalSigma(x, std, mean):
    return (x - mean) / std;



###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    pd.options.display.float_format = '{1.2e}%'.format
    # parse arguments
    basedir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='process BRATS results')
    parser.add_argument ('-x',  '--dir', type = str, help = 'path to the results folder');
    args = parser.parse_args();
    survival_data = pd.read_csv(os.path.join(basedir,"survival_data.csv"), header = 0, error_bad_lines=True, skipinitialspace=True)
    col_names = ["BraTS18ID", "level", "rho-inv", "k-inv", "l2Oc1", "l2c1(TC,s)", "l2c1(TC,#1,..,#n)", "l2c1", "sparsity", "np", "#comp", "xcm-dist", "exec-time",  "N#it", "|g|_r/k",  "age", "srgy", "srvl[]", "srvl", "dice_tc8", "#wt/#b", "#ed/#b", "#tc/#b", "#c0/#b", "I_EDc1", "I_TCc1", "I_B\WTc1", "Ic0/Ic1","comment"]
    df = pd.DataFrame(columns = col_names)

    BIDs = []

    ##
    ## These brains failed due to aliasing. For all brains rho is around 10 and kappa reaches regime of 1E-1.
    ## If not already failed in L1 phase, rescaling further amplifies spectral errors in c(0).
    FILTER     = ['Brats18_CBICA_ANG_1','Brats18_CBICA_AUR_1','Brats18_TCIA02_370_1', 'Brats18_TCIA02_118_1', 'Brats18_TCIA05_478_1', 'Brats18_TCIA06_372_1','Brats18_TCIA02_430_1', 'Brats18_TCIA02_606_1']
    FILTER_MOD = ['Brats18_TCIA02_605_1','Brats18_TCIA01_378_1']
    FILTER     += FILTER_MOD
    FAILEDTOADD = {}
    REDO_SPARSITY = {}
    levels = [64,128,256]
    RUNS = os.listdir(args.dir);
    print(bcolors.BOLD + "   ### DIR = %s ###" % args.dir + bcolors.ENDC);

    survival_mean = np.mean(survival_data["Survival"])
    survival_var  = np.var(survival_data["Survival"])
    survival_std  = np.std(survival_data["Survival"])
    # survival_data.hist() #(column="Survival")
    # plt.show()
    print("Survival Data Statistics: mean %1.3f, variance: %1.3f, std: %1.3f" % (survival_mean, survival_var, survival_std));

    ncases  = {};
    nfailed = {};
    fltr    = 0;
    for l in levels:
        ncases[l]  = 0;
        nfailed[l] = 0;
        FAILEDTOADD[l  ] = []
        REDO_SPARSITY[l] = []
    for run in RUNS:
        if not run.startswith('Brats'):
            continue;
        args.bid = str(run.split('-')[0])
        # print("BID", args.bid)
        if args.bid not in BIDs:
            BIDs.append(args.bid);
        if args.bid in FILTER:
            print(bcolors.FAIL + "   --> filtering ", args.bid , " due to errors in simulation." + bcolors.ENDC);
            fltr += 1;
            continue;
        print(bcolors.OKBLUE + "   ### processing", run , "###" + bcolors.ENDC)

        run_path = os.path.join(args.dir, run);
        tumor_output_path = os.path.join(run_path, "tumor_inversion");

        for level in levels:
            sum_p = 0;
            p_dict = {}
            p_dict['BraTS18ID'] = args.bid;
            p_dict['comment']   = "";#str(run.split(args.bid + "-")[-1].split("4-")[-1]);
            p_dict['level']  = level;
            p_dict['np'] = float('nan');
            p_dict['N#it']  = "";

            survival_row = survival_data.loc[survival_data['BraTS18ID'] == args.bid]
            level_path = os.path.join(os.path.join(tumor_output_path, 'nx'+str(level)), 'obs-1.0');
            f_exist = False;
            info_exist = True;
            f_empty = True;
            t_exec_time = 0;


            ###### COMPONENTS.TXT #####
            p_dict["#comp"]    = -1;
            p_dict["xcm-dist"] = -1;
            if os.path.exists(os.path.join(run_path,'components_obs-1.0.txt')):
                with open(os.path.join(run_path,'components_obs-1.0.txt'), 'r') as f:
                    lines  = f.readlines();
                    curr_level = False;
                    for line, ln in zip(lines,range(len(lines))):
                        if "## level" in line:
                            if int(line.split("## level ")[-1].split("##")[0]) == level:
                                curr_level = True;
                                p_dict["#comp"] = int(lines[ln+1].split("DATA:   #comp:")[-1].split(",")[0]); # number of data components
                            else:
                                curr_level = False;
                        if curr_level and ("SOL(L): #comp:" in line):
                            diststr = line.split("=")[-1]
                            distsep = diststr.split(",");
                            if len(distsep) > 3:
                                p_dict["xcm-dist"] = ', '.join(['%s']*3) % tuple([str(distsep[i]) for i in range(3) ])
                                p_dict["xcm-dist"] += ', ...]px';
                            else:
                                p_dict["xcm-dist"] = diststr;

            if p_dict['#comp'] > 3:
                REDO_SPARSITY[l].append(args.bid)
            ###### INFO.DAT ######
            if os.path.exists(os.path.join(level_path,'info.dat')):
                with open(os.path.join(level_path,'info.dat'), 'r') as f:
                    f_exist = True;
                    lines  = f.readlines();
                    if len(lines) > 1:
                        f_empty = False;
                        param      = lines[0].split(" ");
                        values     = lines[1].split(" ");
                        p_dict['rho-inv'] = float(values[0]);                   # rho-inv
                        p_dict['k-inv']   = float(values[1]);                   # k-inv
                        # p_dict['l2Oc1']   = float(values[2]);                 # l2error_c1
                    else:
                        f_empty    = True;
                        info_exist = False;
                        print( "  WARNING: output file info.dat is empty for tumor inversion of patient " + level_path);
                    f.close()
            else:
                f_exist    = False;
                info_exist = False;
                print("  WARNING: no output file info.dat for tumor inversion of patient " + level_path );
            if f_empty or not f_exist or not info_exist:
                p_dict['rho-inv'] = float('nan');
                p_dict['k-inv']   = float('nan');
                # p_dict['l2Oc1']   = float('nan');

            ###### SURVIVAL ######
            p_dict['srvl[]'] = 'N/A';
            p_dict['srvl'] = -1;
            # p_dict['srvl(s)'] = -1;
            p_dict['age']    = -1;
            p_dict['srgy']   = "no"
            if not survival_row.empty:
                p_dict['age']    = float(survival_row.iloc[0]['Age']);                          # age
                p_dict['srvl[]']  = getSurvivalClass(float(survival_row.iloc[0]['Survival']));  # survival class
                p_dict['srvl']    = float(survival_row.iloc[0]['Survival']);                    # survival
                p_dict['srgy']    = str(survival_row.iloc[0]['ResectionStatus']) if (str(survival_row.iloc[0]['ResectionStatus']) != 'nan' and str(survival_row.iloc[0]['ResectionStatus']) != "NA") else "no";
                # p_dict['srvl(s)'] = getSurvivalSigma(float(survival_row.iloc[0]['Survival']), survival_std, survival_mean);

            sparsity_str = "";
            rel_sparsity = 1;
            f_exist = False;
            f_empty = True;

            ###### RECONP ######
            if os.path.exists(os.path.join(level_path,'reconP.dat')):
                with open(os.path.join(level_path,'reconP.dat'), 'r') as f:
                    f_exist = True;
                    lines = f.readlines()
                    np_ = len(lines)
                    if np_ > 0:
                        recon_p = np.loadtxt(os.path.join(level_path,'reconP.dat'), converters = {0: lambda s: float(s.decode('utf8').replace('\x00', ''))});
                        f_empty    = False;
                        if np_ > 1:
                            recon_p = np.sort(recon_p)[::-1]; # ascending order, reverse for descending order
                        else:
                            recon_p = np.array([recon_p])
                        p_dict['np'] = recon_p.shape[0];
                        sum = np.sum(recon_p);
                        sum_p = sum;
                        sum_frac = [0., 0., 0., 0.];
                        frac     = [.9, .7, .5, .3];
                        num_frac = [-1, -1, -1, -1];
                        sparsity_str = "("
                        for j in range(len(frac)):
                            k = 1;
                            for i in range(recon_p.size):
                                if sum_frac[j] + recon_p[i] < frac[j] * sum:
                                    sum_frac[j] += recon_p[i]
                                    if recon_p[i] > 0:
                                        k += 1
                                else:
                                    num_frac[j] = k;
                                    if j > 0:
                                        sparsity_str += ";"
                                    sparsity_str += str(num_frac[j])
                                    break;
                        sparsity_str += ")"
                    else:
                        f_empty    = True;
                        sparsity_str = ""
                        print(bcolors.FAIL + "  Error: output file reconP.dat is empty for tumor inversion of patient " + level_path + bcolors.ENDC);
                    f.close()
            else:
                sparsity_str = ""
                f_exist    = False;
                print(bcolors.FAIL + "  Error: no output file reconP.dat for tumor inversion of patient " + level_path + bcolors.ENDC);
            p_dict['sparsity'] =  sparsity_str;

            ###### LOGFILE ######
            logfile = 'tumor_solver_log_nx' + str(level) + '.txt'
            log_exist = True;
            if os.path.exists(os.path.join(level_path, logfile)):
                with open(os.path.join(level_path, logfile), 'r') as f:
                    f_exist   = True;
                    err       = True;
                    lines = f.readlines();
                    no = 0;
                    reac_estimation = False;
                    l2gradstr = "nan,"
                    l2gradreacstr = "nan,"
                    for line in lines:
                        f_empty = False;
                        if "Reaction and diffusivity inversion with scaled L2 solution guess" in line:
                            reac_estimation = True;
                        if " optimization done:" in line:
                            back = 0;
                            while not "[  0" in lines[no-back]:
                                back = back + 1;
                            try:
                                p_dict['N#it']    +=  str(int(lines[no-back].split()[1])) + '/'
                                if reac_estimation:
                                    p_dict['|g|_r/k']  = float(lines[no-back].split()[3])
                                    # print(lines[no-back])
                            except (RuntimeError, TypeError, NameError, ValueError):
                                print(lines[no-back].split())
                                print("Error reading log file of patient ", args.bid)
                                p_dict['N#it']    += ",err"
                                p_dict['|g|_r/k']  = -1;
                        if " ----- NP:" in line:
                            np_ = int(line.split(" ----- NP:")[-1].split("------")[0])
                            # if not p_dict['np'] == np_:
                                # print("Error: length of reconP.dat != np", p_dict['np'], np_)
                            p_dict['np'] = int(np_)
                        if "solve-tumor-inverse-tao" in line:
                            t_exec_time = float(line.split()[-1])
                            crash = False;
                        if "Estimated reaction coefficients" in line:
                            rho_ = float(lines[no+1].split("r1:")[-1])
                            found_rho = True;
                            if info_exist and not np.isclose(p_dict['rho-inv'], rho_):
                                print(bcolors.WARNING + "  WARNING: rho-inv in tumor_solver_log.txt %1.2e != %1.2e rho-inv in info.dat." % (rho_, p_dict['rho-inv']) + bcolors.ENDC);
                            p_dict['rho-inv'] = rho_;                           # rho-inv
                        if "Estimated diffusion coefficients" in line:
                            kf_ = float(lines[no+1].split("k1:")[-1]);
                            if not info_exist:
                                print("  Warning: kf-inv =", kf_, "from tumor_solver_log.txt, info.dat empty");
                            elif not np.isclose(kf_, p_dict['k-inv']):
                                print(bcolors.FAIL + "  Error: kf-inv in tumor_solver_log.txt %1.2e != %1.2e kf-inv in info.dat." % (kf_, p_dict['k-inv']) + bcolors.ENDC);
                            p_dict['k-inv'] = kf_;                            # k-inv
                        if "Reconstructed P Norm:" in line:
                            pnorm = float(line.split("Reconstructed P Norm:")[-1])
                        if "C Reconstructed Max and Min" in line:
                            c1_max = float(line.split(":")[-1].split()[0])
                            c1_min = float(line.split(":")[-1].split()[-1])
                        if "L2 rel error at observation points:" in line:
                            p_dict['l2Oc1'] = float(line.split(":")[-1])
                        if "L2 Error in Reconstruction:" in line:
                            p_dict['l2c1'] = float(line.split(":")[-1])
                        #     l2error_c1_ = float(line.split(":")[-1])
                        #     found_l2e = True;
                        #     if not info_exist:
                        #         p_dict['l2c1'] = l2error_c1_;                    # l2error_c1
                        #         print("  Warning: l2error_c1 =", l2error_c1_, " from tumor_solver_log.txt, info.dat empty");
                        #     elif not l2error_c1_ == l2error_c1:
                        #         print("  Error: l2error_c1 in tumor_solver_log.txt != l2error_c1 in info.dat.");
                        no += 1;
                    if len(lines) == 0:
                        f_empty   = True;
                        log_exist = False;
                        print(bcolors.FAIL + " Error: tumor_solver_log.txt is empty for " + level_path + bcolors.ENDC);
            else:
                f_exist   = False;
                log_exist = False;
                error     = True;
                print(bcolors.FAIL + "  Error: tumor_solver_log.txt does not exist for " + level_path + bcolors.ENDC);

            if log_exist == True:
                p_dict['exec-time'] = float(t_exec_time/1000.);

            if sum_p == 0:
                sum_p = - p_dict['np'];
            # p_dict['rel-sparsity'] = sum_p / float(p_dict['np']);

            I_Bc1 = 1.
            dice_found = False;
            f_exist = False;
            ###### TUMOR PARAMS ######
            if os.path.exists(os.path.join(run_path,'tumor_parameters-obs-1.0.txt')):
                f_exist    = True;
                dice_found = False;
                curr_level = False;
                with open(os.path.join(run_path,'tumor_parameters-obs-1.0.txt'), 'r') as f:
                    f_empty = True;
                    for line in f.readlines():
                        if "## level" in line:
                            if int(line.split("## level")[-1].split("##")[0]) == level:
                                curr_level = True;
                            else:
                                curr_level = False;
                        f_empty = False;
                        if curr_level:
                            if "healthy tissue dice (csf,gm,wm)       =" in line:
                                dice_healthy_tissue  = line.split("(")[-1].split(")")[0].split(",")
                                # p_dict['dice_wm'] = float(dice_healthy_tissue[2])/100.;  # dice wm
                                # p_dict['dice_gm'] = float(dice_healthy_tissue[1])/100.;  # dice gm
                                # p_dict['dice_csf'] = float(dice_healthy_tissue[0])/100.; # dice csf
                                dice_found = True;
                            if "dice tumor (max): (wt,tc,nec)         =" in line:
                                tumor_dice  = line.split("(")[-1].split(")")[0].split(",")
                                # p_dict['dice_wt'] = float(tumor_dice[0])/100.;           # dice wt  (whole tumor)
                                # p_dict['dice_tc'] = float(tumor_dice[1])/100.;           # dice tc  (tumor core)
                                # p_dict['dec_nec'] = float(tumor_dice[2])/100.;           # dice nec (necrotic)
                            if "dice tumor (> x): (tc9,tc8,nec1)      =" in line:
                                tumor_dice  = line.split("(")[-1].split(")")[0].split(",")
                                # p_dict['dice_tc9'] = float(tumor_dice[0])/100.;          # dice tc(.9)  (tumor core where c(1) > 0.9)
                                p_dict['dice_tc8'] = float(tumor_dice[1])/100.;          # dice tc(.8)  (tumor core where c(1) > 0.9)
                            if "stats #tu/#brain  (wt,ed,en,nec,tc)   =" in line:
                                frac_stats  = line.split("(")[-1].split(")")[0].split(",")
                                p_dict['#wt/#b']  = float(frac_stats[0]);               # frac #wt/#brain
                                p_dict['#ed/#b']   = float(frac_stats[1]);               # frac #ed/#brain
                                # p_dict['#en/#b']  = float(frac_stats[2]);               # frac #en/#brain
                                # p_dict['#nec/#b'] = float(frac_stats[3]);               # frac #nec/#brain
                                p_dict['#tc/#b']  = float(frac_stats[4]);               # frac #tc/#brain
                            if "stats #tu/#brain  (rec_tc,pred_ct,c0) =" in line:
                                frac_stats  = line.split("(")[-1].split(")")[0].split(",")
                                # p_dict[col_names[25]] = float(frac_stats[0]);               # frac #rec_tc/#b#
                                # p_dict[col_names[26]] = float(frac_stats[1]);               # frac #pred_tc/#b#
                                p_dict['#c0/#b'] = float(frac_stats[2]);                # frac #c0/#b"
                            if "stats int_B c(1) dx                   =" in line:
                                I_Bc1 = float(line.split("=")[-1]);                     # Int_B c(1)
                            if "stats int_ED c(1) dx                  =" in line:
                                p_dict['I_EDc1'] = float(line.split("=")[-1])/I_Bc1;          # Int_ED c(1)
                            if "stats int_TC c(1) dx                  =" in line:
                                p_dict['I_TCc1'] = float(line.split("=")[-1])/I_Bc1;          # Int_TC c(1)
                            if "stats int_B/WT c(1) dx                =" in line:
                                p_dict['I_B\WTc1'] = float(line.split("=")[-1])/I_Bc1;        # Int_B\WT c(1)
                            if "stats int c(0)   / int c(1)           =" in line:
                                p_dict['Ic0/Ic1'] = float(line.split("=")[-1]);         # frac Int c(0) / Int c(1)
                            if "l2ec(1) scaled,TC (l1,l2,l3)          =" in line:
                                all_levels = line.split('l2ec(1) scaled,TC (l1,l2,l3)          = (')[-1].split(")")[0].split(",");
                                p_dict['l2c1(TC,s)'] = float(all_levels[int(math.log(level,2))-6]);
                            if level == 64 and "l2ec(1) scaled,relD (l1;#1,..,#n)     = " in line:
                                errs = line.split("l2ec(1) scaled,relD (l1;#1,..,#n)     = ")[-1].split("(")[-1].split(")")[0].split(",");
                                p_dict["l2c1(TC,#1,..,#n)"] = ', '.join(['%.2f']*len(errs)) % tuple([float(x) for x in errs])
                            if level == 128 and "l2ec(1) scaled,relD (l2;#1,..,#n)     = " in line:
                                errs = line.split("l2ec(1) scaled,relD (l2;#1,..,#n)     = ")[-1].split("(")[-1].split(")")[0].split(",");
                                if p_dict['#comp'] > 3:
                                    p_dict["l2c1(TC,#1,..,#n)"] = '[' +  ', '.join(['%.2f']*3) % tuple([float(errs[i]) for i in range(3) ])
                                    p_dict["l2c1(TC,#1,..,#n)"] += ', ...]';
                                else:
                                    p_dict["l2c1(TC,#1,..,#n)"] = '[' + ', '.join(['%.2f']*len(errs)) % tuple([float(x) for x in errs])  + ']'
                            if level == 256 and "l2ec(1) scaled,relD (l3;#1,..,#n)     = " in line:
                                errs = line.split("l2ec(1) scaled,relD (l3;#1,..,#n)     = ")[-1].split("(")[-1].split(")")[0].split(",");
                                if p_dict['#comp'] > 3:
                                    p_dict["l2c1(TC,#1,..,#n)"] = '[' + ', '.join(['%.2f']*3) % tuple([float(errs[i]) for i in range(3) ])
                                    p_dict["l2c1(TC,#1,..,#n)"] += ', ...]';
                                else:
                                    p_dict["l2c1(TC,#1,..,#n)"] = '[' + ', '.join(['%.2f']*len(errs)) % tuple([float(x) for x in errs]) + ']'
                            # if "stats int c(1.5) / int c(1)           =" in line:
                                # p_dict[col_names[29]] = float(line.split("=")[-1]);         # frac Int c(1.5) / Int c(1)
                            # if "stats int c(1.5) / int c(1.2)         =" in line:
                                # p_dict[col_names[30]] = float(line.split("=")[-1]);         # frac Int c(1.5) / Int c(1.2)
                            # if "stats int c(1.5) / int d              =" in line:
                                # p_dict[col_names[31]] = float(line.split("=")[-1]);         # frac Int c(1.5) / Int data
                            # if "l2err_c(1)         (virg,obs)         =" in line:
                                # l2_errs  = line.split("(")[-1].split(")")[0].split(",")
                                # p_dict['l2c1']  = float(l2_errs[0]);                   # l2err | c(1)-d| / |d|
                                # p_dict['l2Oc1'] = float(l2_errs[1]);                   # l2err |Oc(1)-d| / |d|

            if f_exist == False or f_empty == True:
                error   = True;
                print(bcolors.FAIL + "  Error: tumor_parameters.txt does not exist for " +run_path + bcolors.ENDC);
                # p_dict['dice_tc9'] = float('nan');
                p_dict['dice_tc8']   = float('nan');
                p_dict['#wt/#b']     = float('nan');
                p_dict['#ed/#b']     = float('nan');
                p_dict['#tc/#b']     = float('nan');
                p_dict['#c0/#b']     = float('nan');
                # p_dict['I_Bc1']    = float('nan');
                p_dict['I_EDc1']     = float('nan');
                p_dict['I_TCc1']     = float('nan');
                p_dict['I_B\WTc1']   = float('nan');
                p_dict['Ic0/Ic1']    = float('nan');
                p_dict['l2c1(TC,s)'] = float('nan');
                p_dict["l2c1(TC,#1,..,#n)"] = ""



            ###### ADD PDICT ######
            try:
                df.loc[len(df)] = p_dict;
                print(bcolors.OKGREEN, "Sucessfully added entry for", args.bid, bcolors.ENDC)
                ncases[level] += 1;
            except (ValueError):
                print(bcolors.FAIL,  "Failed adding entry for", args.bid, "\n", p_dict, bcolors.ENDC)
                FAILEDTOADD[level].append(args.bid)
                nfailed[level] += 1;

        print("")

    caption = 'Solver Analysis'
    # color palette for heatmap
    cm1 = sns.light_palette("green", as_cmap=True)
    cm2 = sns.light_palette("blue", as_cmap=True)
    cm3 = sns.light_palette("purple", as_cmap=True)
    cm1 = ListedColormap(sns.color_palette("BuGn_r").as_hex())
    cm2 = ListedColormap(sns.color_palette("Blues").as_hex())
    cm3 = ListedColormap(sns.color_palette("GnBu").as_hex())

    df['srgy'] = df['srgy'].astype('str')
    # df['srvl[]'] = df['srvl[]'].astype('category')

    # set CSS properties for th elements in dataframe
    table_props = [
      ('border-spacing', '3px'),
      ('width', '100%'),
      ('border', '1px solid #ddd')
    ]
    th_props = [
      ('font-size', '12px'),
      ('text-align', 'center'),
      ('font-weight', 'bold'),
      ('color', '#6d6d6d'),
      ('background-color', '#f7f7f9')
      ]
    # set CSS properties for td elements in dataframe
    td_props = [
      ('font-size', '12px')
      ]
    # set table styles
    styles = [
      dict(selector="table", probs=table_props),
      dict(selector="th", props=th_props),
      dict(selector="td", props=td_props),
      dict(selector="tr:hover", props=[('background-color', 'yellow'), ('font-weight', 'italic')]),
      # dict(selector="tr:nth-child(even)", props=[('background-color', '#f2f2f2')])
      ]

    df = df.sort_values(by=['BraTS18ID','level']);
    # df = df.sort_values(by=['#ed/#b','BraTS18ID','level']);

    p = str(args.dir).split("/")
    basedir = "./"
    identifier_dir = p[-1]
    if len(p) > 1:
        basedir = os.path.join(*p[:len(p)-1])
    if not os.path.exists(basedir):
        os.mkdir(basedir)
    df.to_csv(os.path.join(basedir,'grid-cont-analysis_' + str(identifier_dir) + '.csv'))
    numeric_col_mask_1 = df.dtypes.apply(lambda d: issubclass(np.dtype(d).type, np.number))

    # print("numeric columns: ", numeric_col_mask_1)
    HTML = ""
    # for bid in BIDs:
    if True:
        dftmp = df; # df[df['BraTS18ID'] == bid];
        html = dftmp.style\
            .hide_index()\
            .set_caption("BRATS ANALYSIS (GRID-CONT)")\
            .set_properties(subset=dftmp.columns[numeric_col_mask_1],  **{'width':'1*', 'text-align':'right'})\
            .set_properties(subset=dftmp.columns[~numeric_col_mask_1], **{'width':'1*', 'text-align':'right'})\
            .set_properties(subset=['comment'], **{'width':'3*', 'text-align':'left'})\
            .format(lambda x: '{:1.2e}'.format(x), subset=pd.IndexSlice[:,dftmp.columns[numeric_col_mask_1]])\
            .format(lambda x: '{:1.0f}'.format(x), subset=['np', 'age'])\
            .format(lambda x: '{:1.2f}'.format(x), subset=['rho-inv','srvl'])\
            .format(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if not math.isnan(x) else 'nan', subset=['exec-time',])\
            .set_table_styles(styles)\
            .bar(subset=['exec-time'], color='#779ecb')\
            .bar(subset=['rho-inv'], color='#FF9AA2')\
            .bar(subset=['l2Oc1'], color='#FFDAC1')\
            .bar(subset=['l2c1(TC,s)'], color='#FFDAC1')\
            .bar(subset=['k-inv'], color='#B5EAD7')\
            .render();
            #.set_caption(caption + " " + bid)\
        HTML += html;
        HTML += "\n\n"
    # write html file
    filename_html = os.path.join(basedir,'brats-grid-cont-analysis(levels)_' + str(identifier_dir) + '.html')
    html_file = open(filename_html, 'w')
    html_file.write(HTML)
    html_file.close()

    HTML = ""
    dftmp = df.loc[df['level'] == 128];
    dftmp_128 = dftmp;
    # dftmp = dftmp.loc[df['remember_ls-step'] == 'yes'];
    # dftmp = dftmp.loc[dftmp['comment'] == "phi-supp"];
    html = dftmp.style\
        .hide_index()\
        .set_caption("BRATS ANALYSIS (GRID-CONT, nx=128)")\
        .set_properties(subset=dftmp.columns[numeric_col_mask_1],  **{'width':'1*', 'text-align':'right'})\
        .set_properties(subset=dftmp.columns[~numeric_col_mask_1], **{'width':'1*', 'text-align':'right'})\
        .set_properties(subset=['comment'], **{'width':'3*', 'text-align':'left'})\
        .format(lambda x: '{:1.2e}'.format(x), subset=pd.IndexSlice[:,dftmp.columns[numeric_col_mask_1]])\
        .format(lambda x: '{:1.0f}'.format(x), subset=['np', 'age'])\
        .format(lambda x: '{:1.2f}'.format(x), subset=['rho-inv','srvl'])\
        .format(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if not math.isnan(x) else 'nan', subset=['exec-time',])\
        .set_table_styles(styles)\
        .bar(subset=['exec-time'], color='#779ecb')\
        .bar(subset=['rho-inv'], color='#FF9AA2')\
        .bar(subset=['l2Oc1'], color='#FFDAC1')\
        .bar(subset=['l2c1(TC,s)'], color='#FFDAC1')\
        .bar(subset=['I_EDc1'], color='#9199BE')\
        .bar(subset=['I_TCc1'], color='#54678F')\
        .bar(subset=['I_B\WTc1'], color='#AAAAAA')\
        .bar(subset=['Ic0/Ic1'], color='#989898')\
        .bar(subset=['k-inv'], color='#B5EAD7')\
        .render();
        #.set_caption(caption + " " + bid)\
    HTML += html;
    HTML += "\n\n"
    # write html file
    filename_html = os.path.join(basedir,'brats-grid-cont-analysis(128)_' + str(identifier_dir) + '.html')
    dftmp_128.to_csv(os.path.join(basedir,'grid-cont-analysis_' + str(identifier_dir) + '_nx128.csv'))
    html_file = open(filename_html, 'w')
    html_file.write(HTML)
    html_file.close()
    html_128 = html;

    HTML = ""
    dftmp = df.loc[df['level'] == 256];
    dftmp_256 = dftmp;
    # dftmp = dftmp.loc[df['remember_ls-step'] == 'yes'];
    # dftmp = dftmp.loc[dftmp['comment'] == "phi-supp"];
    html = dftmp.style\
        .hide_index()\
        .set_caption("BRATS ANALYSIS (GRID-CONT nx=256)")\
        .set_properties(subset=dftmp.columns[numeric_col_mask_1],  **{'width':'1*', 'text-align':'right'})\
        .set_properties(subset=dftmp.columns[~numeric_col_mask_1], **{'width':'1*', 'text-align':'right'})\
        .set_properties(subset=['comment'], **{'width':'3*', 'text-align':'left'})\
        .format(lambda x: '{:1.2e}'.format(x), subset=pd.IndexSlice[:,dftmp.columns[numeric_col_mask_1]])\
        .format(lambda x: '{:1.0f}'.format(x), subset=['np', 'age'])\
        .format(lambda x: '{:1.2e}'.format(x), subset=['rho-inv','srvl'])\
        .format(lambda x: time.strftime('%H:%M:%S', time.gmtime(x)) if not math.isnan(x) else 'nan', subset=['exec-time',])\
        .set_table_styles(styles)\
        .bar(subset=['exec-time'], color='#779ecb')\
        .bar(subset=['rho-inv'], color='#FF9AA2')\
        .bar(subset=['l2Oc1'], color='#FFDAC1')\
        .bar(subset=['l2c1(TC,s)'], color='#FFDAC1')\
        .bar(subset=['I_EDc1'], color='#9199BE')\
        .bar(subset=['I_TCc1'], color='#54678F')\
        .bar(subset=['I_B\WTc1'], color='#AAAAAA')\
        .bar(subset=['Ic0/Ic1'], color='#989898')\
        .bar(subset=['k-inv'], color='#B5EAD7')\
        .render();
        #.set_caption(caption + " " + bid)\
    HTML += html;
    HTML += "\n\n"
    HTML += html_128;
    # write html file
    filename_html = os.path.join(basedir,'brats-grid-cont-analysis(256)_' + str(identifier_dir) + '.html')
    dftmp_256.to_csv(os.path.join(basedir,'grid-cont-analysis_' + str(identifier_dir) + '_nx256.csv'))
    html_file = open(filename_html, 'w')
    html_file.write(HTML)
    html_file.close()



    #### POSTPROC/VISUALIZE ####
    plot_corr     = False;
    plot_pairgrid = False;
    plot_stats    = False;
    max_l2c1error = 1
    crop_corr     = ["rho-inv", "k-inv", "rho-over-k", "l2Oc1", "l2c1(TC,s)", "#comp", "age", "srvl[]", "Ic0/Ic1", '#tc/#b', '#ed/#b']

    # discard coarse levels
    data256 = df.loc[df['level'] == 256]
    # data256 = data256.loc[data256['comment'] == "phi-supp"];
    # make ordinal column srvl[] numeric
    data256['srvl[]'] = data256['srvl[]'].astype('category')
    cats = []
    if "N/A" in data256['srvl[]'].cat.categories:
        cats.append(-1);
    if "long" in data256['srvl[]'].cat.categories:
        cats.append(3);
    if "mid" in data256['srvl[]'].cat.categories:
        cats.append(2);
    if "short" in data256['srvl[]'].cat.categories:
        cats.append(1);
    data256['srvl[]'].cat.categories = cats; #[-1, 3,2,1]   # alpha numeric sorting: [NA, long, mid, short] -> [-1, 2,1,0]
    data256['srvl[]'] = data256['srvl[]'].astype('float')
    # filter data with patient age > 0, i.e., patient IDs where survival data exists
    data256['age'] = data256['age'].astype('float')
    dat_out = data256.loc[data256['age'] <= 0]
    data256 = data256.loc[data256['age'] >  0]
    dat_out["filter-reason"] = "no survival data"
    dat_filtered_out = dat_out;
    # filter data with too large misfit
    data256['l2c1'] = data256['l2Oc1'].astype('float')
    dat_out = data256.loc[data256['l2Oc1'] >= max_l2c1error]
    data256 = data256.loc[data256['l2Oc1'] <  max_l2c1error]
    dat_out["filter-reason"] = "l2err > "+str(max_l2c1error)
    dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)
    # add rho-over-k
    data256['rho-inv'] = data256['rho-inv'].astype('float')
    data256['k-inv']   = data256['k-inv'].astype('float')
    dat_out  = data256.loc[data256['k-inv'] <= 0]
    data256  = data256.loc[data256['k-inv'] >  0]
    dat_out["filter-reason"] = "k zero"
    dat_filtered_out = pd.concat([dat_filtered_out, dat_out], axis=0)
    data256["rho-over-k"] = data256["rho-inv"]/data256["k-inv"]

    print("\n\n### BraTS simulation data [filtered] ### ")
    print(tabulate(data256, headers='keys', tablefmt='psql'))
    print("\n\n### BraTS simulation data [filtered out] ### ")
    print(tabulate(dat_filtered_out, headers='keys', tablefmt='psql'))

    print("Entries on level 128: %d" % len(dftmp_128), " ... failed to add (%d) \n" % nfailed[128], FAILEDTOADD[128]);
    print("Entries on level 256: %d" % len(dftmp_256), " ... failed to add (%d) \n" % nfailed[256], FAILEDTOADD[256]);
    print("IDs with more than 4 components (need to be recomputed): \n", REDO_SPARSITY[256])
    print(bcolors.WARNING + "\n\n===  filtered %d cases ===" % (fltr) + bcolors.ENDC)
    print(bcolors.OKGREEN + "===  successfully added %d cases on level 64  (%d failed to add) ===" % (ncases[64],nfailed[64]) + bcolors.ENDC)
    print(bcolors.OKGREEN + "===  successfully added %d cases on level 128 (%d failed to add) ===" % (ncases[128],nfailed[128]) + bcolors.ENDC)
    print(bcolors.OKGREEN + "===  successfully added %d cases on level 256 (%d failed to add) ===" % (ncases[256],nfailed[256]) + bcolors.ENDC)



    # --------------------------
    # ### correlation matrix ###
    if plot_corr:
        data = data256.loc[:,crop_corr]
        # corr_pearson  = data.corr(method='pearson')
        corr_kendall  = data.corr(method='kendall')
        corr_spearman = data.corr(method='spearman')

        # generate mask for the upper triangle
        mask = np.zeros_like(corr_kendall, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # matplotlib figure
        # f, ax = plt.subplots(figsize=(11, 9))
        # generate custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # draw the heatmap with the mask and correct aspect ratio

        fig = plt.figure(figsize=(17,5))
        grid = plt.GridSpec(1, 2, wspace=0.2, hspace=0.7)
        ax1 = plt.subplot(grid[0,0])
        # ax1.set_title("Pearson Correlation Coeff.")
        # sns.heatmap(corr_pearson, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    # square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, ax=ax1)
        ax2 = plt.subplot(grid[0,0])
        ax2.set_title("Kendall Correlation Coeff.")
        sns.heatmap(corr_kendall, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, ax=ax2)
        ax3 = plt.subplot(grid[0,1])
        ax3.set_title("Spearman Correlation Coeff.")
        sns.heatmap(corr_spearman, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, ax=ax3)
        sns.despine(fig, top=True, bottom=True, left=True, right=True, trim=True)

        sns.despine(offset=5, trim=True)
        # sns.despine()
        plt.tight_layout()
        plt.show()

    # -------------------------------
    # ### pairgrid                ###
    if plot_pairgrid:
        data = data256.loc[:,crop_corr]
        g = sns.PairGrid(data, diag_sharey=False, size=1.8,  hue="srvl[]", palette="Set2", hue_kws={"marker": ["o", "s", "D"]})
        # g.map_lower(sns.kdeplot)
        g.map_offdiag(sns.scatterplot,  linewidths=1, edgecolor="w", s=20)
        g.map_diag(plt.hist, edgecolor='w')# histtype="step", linewidth=1)
        g.add_legend()

        sns.despine(offset=5, trim=True)
        # sns.despine()
        plt.tight_layout()
        plt.show()


    if plot_stats:
        # crop data
        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        pal = sns.color_palette('Paired')
        # dice_crop = ["BraTS18ID","srvl[]","dice_wm","dice_gm","dice_csf","dice_wt","dice_tc","dice_nec"]
        stat_crop = ["BraTS18ID","srvl[]","#tc/#b","#wt/#b","#ed/#b","Ic0/Ic1","I_TCc1", "I_EDc1", "I_WT\Bc1", "l2Oc1", "l2c1(TC,s)"]
        inv_crop  = ["BraTS18ID","srvl[]","rho-inv","k-inv","rho-over-k"]
        # dice_dat = data256.loc[:,dice_crop]
        stats_dat = data256.loc[:,stat_crop]
        inv_dat = data256.loc[:,inv_crop]
        # melt data
        # dice_dat_melted    = pd.melt(dice_dat,  id_vars=['BraTS18ID','srvl[]'], value_vars=["dice_wm","dice_gm","dice_csf","dice_wt","dice_tc","dice_nec"], value_name='dice_val', var_name=['dice_tissue'])
        stats_dat_melted    = pd.melt(stats_dat, id_vars=['BraTS18ID','srvl[]'], value_vars=["#tc/#b","#wt/#b","#ed/#b","Ic0/Ic1"],                   value_name='stat_val', var_name=['stats'])
        stats_dat_melted2   = pd.melt(stats_dat, id_vars=['BraTS18ID','srvl[]'], value_vars=["I_TCc1", "I_EDc1", "I_WT\Bc1",  "l2Oc1", "l2c1(TC,s)"], value_name='stat_val', var_name=['stats'])
        inv_dat_melted      = pd.melt(inv_dat,   id_vars=['BraTS18ID','srvl[]'], value_vars=["rho-inv","k-inv"], value_name='value', var_name=['inv_vars'])
        inv_dat_melted_rho  = pd.melt(inv_dat,   id_vars=['BraTS18ID','srvl[]'], value_vars=["rho-inv"],         value_name='value', var_name=['inv_vars'])
        inv_dat_melted_k    = pd.melt(inv_dat,   id_vars=['BraTS18ID','srvl[]'], value_vars=["k-inv"],           value_name='value', var_name=['inv_vars'])
        inv_dat_melted_rhok = pd.melt(inv_dat,   id_vars=['BraTS18ID','srvl[]'], value_vars=["rho-over-k"],      value_name='value', var_name=['inv_vars'])

        stats_dat.hist()
        inv_dat.hist()

        # figure
        mpl.style.use('seaborn')
        sns.set(style="ticks")
        f = plt.figure(figsize=(8,9))
        grid1 = plt.GridSpec(4, 1, wspace=0.2, hspace=0.4, height_ratios=[4,4,1,1])

        # boxplot settings
        boxprops = {'edgecolor': 'w', 'linewidth': 1}
        lineprops = {'color': 'k', 'linewidth': 1}
        medianprops = {'color': 'r', 'linewidth': 2}
        # kwargs
        kwargs = {'palette': pal}
        boxplot_kwargs = dict({'boxprops': boxprops, 'medianprops': medianprops,
                               'whiskerprops': lineprops, 'capprops': lineprops,
                               'width': 0.75, 'palette' : pal, 'notch' : 0},
                               **kwargs)
        stripplot_kwargs = dict({'linewidth': 0.2, 'size': 3, 'alpha': 0.5, 'color' : [0.1,0.1,0.1], 'edgecolor' : 'w'})

        # === plot ax1 ===
        # ax1 =  plt.subplot(grid1[0,0])
        # sns.boxplot(x="dice_val", y="dice_tissue", data=dice_dat_melted, hue="srvl[]", fliersize=0, ax=ax1, **boxplot_kwargs)
        # sns.stripplot(x='dice_val', y='dice_tissue', hue='srvl[]', data=dice_dat_melted, jitter=True, split=True, ax=ax1, **stripplot_kwargs)
        # # sns.swarmplot(x="dice_val", y="dice_tissue", data=dice_dat_melted, hue="srvl[]", size=3, palette=flatui, linewidth=0, ax=ax1)
        # ax1.xaxis.grid(True)
        # ax1.set(xlabel="Dice coefficient")
        # ax1.set(ylabel="")
        # ax1.set_yticklabels(["dice$_{WM}$", "dice$_{GM}$", "dice$_{CSF}$", "dice$_{WT}$", "dice$_{TC}$", "dice$_{NEC}$"])
        # # fix legend
        # handles, labels = ax1.get_legend_handles_labels()
        # labels = ["short", "mid", "long"]
        # lgd = ax1.legend(handles[0:3], labels[0:3], loc='upper left', fontsize='small', handletextpad=0.5)
        # lgd.legendHandles[0]._sizes = [40]
        # lgd.legendHandles[1]._sizes = [40]

        # === plot ax2 ===
        ax2 =  plt.subplot(grid1[0,0])
        sns.boxplot(x="stat_val", y="stats", data=stats_dat_melted, hue="srvl[]", fliersize=0, ax=ax2, **boxplot_kwargs)
        sns.stripplot(x='stat_val', y='stats', hue='srvl[]', data=stats_dat_melted, jitter=True, split=True, ax=ax2, **stripplot_kwargs)
        # fix legend
        handles, labels = ax2.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        lgd = ax2.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
        lgd.legendHandles[0]._sizes = [40]
        lgd.legendHandles[1]._sizes = [40]
        ax2.xaxis.grid(True)
        ax2.set(xlabel="Value")
        ax2.set_yticklabels(["$\\frac{\# TC}{\# B}$", "$\\frac{\# WT}{\# B}$", "$\\frac{\# ED}{\# B}$",  "$\\frac{\\int c(0)}{\\int c(1)}$"]);
        ax2.set(ylabel="")

        # === plot ax3 ===
        ax3 =  plt.subplot(grid1[1,0])
        sns.boxplot(x="stat_val", y="stats", data=stats_dat_melted2, hue="srvl[]", fliersize=0, ax=ax3, **boxplot_kwargs)
        sns.stripplot(x='stat_val', y='stats', hue='srvl[]', data=stats_dat_melted2, jitter=True, split=True, ax=ax3, **stripplot_kwargs)
        # fix legend
        handles, labels = ax3.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        lgd = ax3.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
        lgd.legendHandles[0]._sizes = [40]
        lgd.legendHandles[1]._sizes = [40]
        ax3.xaxis.grid(True)
        ax3.set(xlabel="Value")
        ax3.set_yticklabels(["$\\frac{\\int_{TC} c(1)}{\\int_B c(1)}$",  "$\\frac{\\int_{ED} c(1)}{\\int_B c(1)}$",  "$\\frac{\\int_{B\\ WT} c(1)}{\\int_B c(1)}$", "$\\|\\left.c(1)\\right|_{TC} - d\\|_{2,rel}$", "$\\|Oc(1) - d\\|_{2,rel}$"]);
        ax3.set(ylabel="")


        # === plot ax4 ===
        ax3 =  plt.subplot(grid1[2,0])
        sns.boxplot(x="value", y="inv_vars", data=inv_dat_melted_rho, hue="srvl[]", fliersize=0, ax=ax3, **boxplot_kwargs)
        sns.stripplot(x='value', y='inv_vars', hue='srvl[]', data=inv_dat_melted_rho, jitter=True, split=True, ax=ax3, **stripplot_kwargs)
        # fix legend
        handles, labels = ax3.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        lgd = ax3.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
        lgd.legendHandles[0]._sizes = [40]
        lgd.legendHandles[1]._sizes = [40]
        ax3.xaxis.grid(True)
        ax3.set_xlim([1,15])
        ax3.set(xlabel="$\\rho$")
        ax3.set_yticklabels(["$\\rho$"]);
        ax3.set(ylabel="")

        # === plot ax5 ===
        ax4 =  plt.subplot(grid1[3,0])
        sns.boxplot(x="value", y="inv_vars", data=inv_dat_melted_k, hue="srvl[]", fliersize=0, ax=ax4, **boxplot_kwargs)
        sns.stripplot(x='value', y='inv_vars', hue='srvl[]', data=inv_dat_melted_k, jitter=True, split=True, ax=ax4, **stripplot_kwargs)
        ax4.set_xscale('log')
        # fix legend
        handles, labels = ax4.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        lgd = ax4.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
        lgd.legendHandles[0]._sizes = [40]
        lgd.legendHandles[1]._sizes = [40]
        ax4.xaxis.grid(True)
        ax4.set_xlim([1e-4,1e0])
        ax4.set(xlabel="$k$")
        ax4.set_yticklabels(["$k$"]);
        ax4.set(ylabel="")

        # fig, ax = plt.subplots(figsize=[8,5])
        # sns.catplot(ax=ax, x="dice_val", y="dice_tissue", data=dice_dat_melted, hue="srvl[]", kind="boxen", palette=pal)
        # ax.xaxis.grid(True)
        # ax.set(xlabel="Dice coefficient")
        # ax.set(ylabel="")
        # ax.set_yticklabels(["dice$_{WM}$", "dice$_{GM}$", "dice$_{CSF}$", "dice$_{WT}$", "dice$_{TC}$", "dice$_{NEC}$"])
        # # fix legend
        # handles, labels = ax.get_legend_handles_labels()
        # labels = ["short", "mid", "long"]
        # lgd = ax.legend(handles[0:3], labels[0:3], loc='upper left', fontsize='small', handletextpad=0.5)
        # # lgd.legendHandles[0]._sizes = [40]

        fig, ax6 = plt.subplots(figsize=[8,5])
        sns.catplot(ax=ax6, x="stat_val", y="stats", data=stats_dat_melted, hue="srvl[]", kind="boxen", palette=pal)
        # sns.boxplot(ax=ax6, x="stat_val", y="stats", data=stats_dat_melted, hue="srvl[]", **boxplot_kwargs)
        sns.stripplot(ax=ax6, x="stat_val", y="stats", data=stats_dat_melted, hue="srvl[]", jitter=True, split=True, **stripplot_kwargs)
        ax6.xaxis.grid(True)
        ax6.set(xlabel="Fraction")
        ax6.set(ylabel="")
        ax6.set_yticklabels(["$\\frac{\# TC}{\# B}$", "$\\frac{\# WT}{\# B}$", "$\\frac{\# ED}{\# B}$",  "$\\frac{\\int c(0)}{\\int c(1)}$"]);
        # fix legend
        handles, labels = ax6.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        lgd = ax6.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)

        fig, ax3 = plt.subplots(figsize=[8,5])
        sns.boxplot(x="stat_val", y="stats", data=stats_dat_melted2, hue="srvl[]", fliersize=0, ax=ax3, **boxplot_kwargs)
        sns.stripplot(x='stat_val', y='stats', hue='srvl[]', data=stats_dat_melted2, jitter=True, split=True, ax=ax3, **stripplot_kwargs)
        ax3.xaxis.grid(True)
        ax3.set(xlabel="Fraction")
        ax3.set_yticklabels(["$\\frac{\\int_{TC} c(1)}{\\int_B c(1)}$",  "$\\frac{\\int_{ED} c(1)}{\\int_B c(1)}$",  "$\\frac{\\int_{B\\ WT} c(1)}{\\int_B c(1)}$", "$\\|\\left.c(1)\\right|_{TC} - d\\|_{2,rel}$", "$\\|Oc(1) - d\\|_{2,rel}$"]);
        ax3.set(ylabel="")
        # fix legend
        handles, labels = ax3.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        lgd.legendHandles[0]._sizes = [40]
        lgd.legendHandles[1]._sizes = [40]
        lgd = ax3.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)

        fig, ax7 = plt.subplots(figsize=[8,3])
        sns.catplot(x="value", y="inv_vars", data=inv_dat_melted_rho, hue="srvl[]", ax=ax7, kind="boxen", palette=pal)
        # sns.boxplot(x="value", y="inv_vars", data=inv_dat_melted_rho, hue="srvl[]", ax=ax7, **boxplot_kwargs)
        # fix legend
        handles, labels = ax7.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        lgd = ax7.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
        ax7.xaxis.grid(True)
        ax7.set_xlim([1,15])
        ax7.set(xlabel="$\\rho$")
        ax7.set_yticklabels(["$\\rho$"]);
        ax7.set(ylabel="")

        fig, ax8 = plt.subplots(figsize=[8,3])
        sns.catplot(x="value", y="inv_vars", data=inv_dat_melted_k, hue="srvl[]", ax=ax8, kind="boxen", palette=pal)
        # sns.boxplot(x="value", y="inv_vars", data=inv_dat_melted_k, hue="srvl[]", ax=ax8, **boxplot_kwargs)
        ax8.set_xscale('log')
        # fix legend
        handles, labels = ax8.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        lgd = ax8.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
        ax8.xaxis.grid(True)
        ax8.set_xlim([1e-4,1e0])
        ax8.set(xlabel="$k$")
        ax8.set_yticklabels(["$k$"]);
        ax8.set(ylabel="")

        fig, ax9 = plt.subplots(figsize=[8,3])
        sns.catplot(x="value", y="inv_vars", data=inv_dat_melted_rhok, hue="srvl[]", ax=ax9, kind="boxen", palette=pal)
        # sns.boxplot(x="value", y="inv_vars", data=inv_dat_melted_rhok, hue="srvl[]", ax=ax9, **boxplot_kwargs)
        ax9.set_xscale('log')
        # fix legend
        handles, labels = ax8.get_legend_handles_labels()
        labels = ["short", "mid", "long"]
        lgd = ax9.legend(handles[0:3], labels[0:3], loc='upper right', fontsize='small', handletextpad=0.5)
        ax9.xaxis.grid(True)
        # ax8.set_xlim([1e-4,1e0])
        ax9.set(xlabel="$\\frac{\\rho}{\\kappa}$")
        ax9.set_yticklabels(["$\\frac{\\rho}{\\kappa}$"]);
        ax9.set(ylabel="")

        sns.despine(offset=5, trim=True)
        # sns.despine()
        plt.tight_layout()
        plt.show()
