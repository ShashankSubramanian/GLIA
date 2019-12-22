import nibabel as nib
import scipy.ndimage as ndimage
from scipy.stats import multivariate_normal
from netCDF4 import Dataset
import numpy as np
import os, math, sys, argparse
from TumorParams import *
import pandas as pd


TARGET_DIMS = (256, 256, 256)
#LABEL_DICT = [("csf", 10), ("vt", 50), ("gm", 150), ("wm", 250)]
LABEL_DICT = {"csf":10, "vt":50, "gm":150, "wm":250}
SIGMA = 2*math.pi/256
DEBUG=False

JOB_HEADER="#!/bin/bash\n\
#SBATCH -J claire\n\
#SBATCH -p normal\n\
#SBATCH -o stdout.o\n\
#SBATCH -e stderr.e\n\
#SBATCH -N 2\n\
#SBATCH -n 96\n\
#SBATCH -t 0:30:00\n\
#SBATCH -A FTA-Biros\n\n"


def getTumorGenCmd(atlas, data_path, c0_path, output_path, rho_data=8, k_data=0.025, gamma=1.5E5):
    scripts_path = os.path.dirname(os.path.realpath(__file__))

    home = os.environ['HOME']
    work = os.environ['WORK']
    scratch=os.environ['SCRATCH']    
    params = {}
    params['N'] = TARGET_DIMS[0]
    params['code_path'] = os.path.join(home, 'pglistr_tumor')
    
    params['results_path'] = output_path + "/"
    
    params['compute_sys'] = 'frontera'
    params['forcing_factor'] = gamma
    params['k_data'] = k_data
    params['rho_data'] = rho_data
    params['model'] = 4
    params['forward_flag'] = 1

    params['gm_path'] = os.path.join(data_path, atlas+'_seg_gm.nc')
    params['wm_path'] = os.path.join(data_path, atlas+'_seg_wm.nc')
    params['glm_path'] = os.path.join(data_path, atlas+'_seg_csf.nc')
    params['csf_path'] = os.path.join(data_path, atlas+'_seg_ve.nc')
    params['init_tumor_path'] = c0_path
    
    run_str, err = getTumorRunCmd(params)  ### Use default parameters (if not, define dict with usable values)
    return run_str, err

def createJobScript(cmd, output_path):
    '''
    creates and submits the job script
    '''

     
    # if not os.path.exists(params['gm_path']):
    #     print("converting gm for {}".format(atlas))
    #     gm_path = convert_nii_to_nc(os.path.join(data_path, atlas+'_seg_gm.nii.gz'))
    # if not os.path.exists(params['wm_path']):
    #     print("converting wm for {}".format(atlas))
    #     wm_path = convert_nii_to_nc(os.path.join(data_path, atlas+'_seg_wm.nii.gz'))
    # if not os.path.exists(params['glm_path']):
    #     print("converting glm for {}".format(atlas))
    #     glm_path = convert_nii_to_nc(os.path.join(data_path, atlas+'_seg_csf.nii.gz'))
    # if not os.path.exists(params['csf_path']):
    #     print("converting csf for {}".format(atlas))
    #     csf_path = convert_nii_to_nc(os.path.join(data_path, atlas+'_seg_ve.nii.gz'))


    # if not os.path.exists(params['init_tumor_path']):
    #     print("converting init_tumor_path for {}".format(atlas))
    #     init_tumor_path = convert_nii_to_nc(os.path.join(results_path, patient+'_c0_in_'+atlas+'.nii.gz'))

    #if not err:  # No error in tumor input parameters
    #    print('No errors, submitting jobfile\n')
    #    fname = scripts_path + '/job.sh'
    #    submit_file = open(fname, 'w+')
    #    if params['compute_sys'] == 'hazelhen':
    #        submit_file.write("#!/bin/bash\n" + \
    #        "#PBS -N ITP\n" + \
    #        "#PBS -l nodes="+str(N)+":ppn=24 \n" + \
    #        "#PBS -l walltime=01:00:00 \n" + \
    #        "#PBS -m e\n" + \
    #        "#PBS -M kscheufele@austin.utexas.edu\n\n" + \
    #        "source /zhome/academic/HLRS/ipv/ipvscheu/env_intel.sh\n" + \
    #        "export OMP_NUM_THREADS=1\n")
    #    else:
    #        submit_file.write ("#!/bin/bash\n" + \
    #        "#SBATCH -J ITP\n" + \
    #        "#SBATCH -o " + params['results_path'] + "/log\n" + \
    #        "#SBATCH -p " + queue + "\n" + \
    #        "#SBATCH -N " + str(N) + "\n" + \
    #        "#SBATCH -n " + str(n) + "\n" + \
    #        "#SBATCH -t 01:00:00\n" + \
    #        "#SBATCH -A PADAS\n" + \
    #        "source ~/.bashrc\n" + \
    #        "export OMP_NUM_THREADS=1\n")

    #    submit_file.write(run_str)
    #    submit_file.close()
    #    ### submit jobfile
    #    if params['compute_sys'] == 'hazelhen':
    #        subprocess.call(['qsub', fname])
    #    else:
    #        subprocess.call(['sbatch', fname])
    #else:
    #    print('Errors, no job submitted\n')



def sampleInputParam(num_brats, atlas_list, rho_list, kappa_list, gamma_list, brats_dict):
    # set random number seed
    np.random.seed(12345)

    # sample and convert to quadruples
    n = 4
    sampled_atlas = np.random.choice(atlas_list, n*num_brats)
    sampled_atlas = breakList(sampled_atlas, n)

    sampled_gamma = np.random.choice(gamma_list, n*num_brats)
    sampled_gamma = breakList(sampled_gamma, n)
    
    n = 2
    sampled_rho = np.random.choice(rho_list, n*num_brats)
    sampled_rho = breakList(sampled_rho, n)

    sampled_kappa = np.random.uniform(kappa_list[0], kappa_list[1], n*num_brats)
    sampled_kappa = breakList(sampled_kappa, n)

    for i,bid in enumerate(brats_dict):
        brats_dict[bid]['s_atlas'] = sampled_atlas[i]
        brats_dict[bid]['s_kappa'] = sampled_kappa[i]
        brats_dict[bid]['s_rho'] = sampled_rho[i]
        brats_dict[bid]['s_gamma'] = sampled_gamma[i]
    
    return sampled_atlas, sampled_rho, sampled_kappa, sampled_gamma, brats_dict


def readBratsTumorInversionResults(tumor_inversion_feature_file, brats_name_map):
    data = pd.read_csv(tumor_inversion_feature_file)

    brats_ids = data['Unnamed: 1'][3:].tolist()
    kappa_inv = data['Unnamed: 7'][3:].tolist()
    rho_inv = data['Unnamed: 8'][3:].tolist()

    # make it a dictionary
    brats_dict = {}
    for i,bid in enumerate(brats_ids):
        # convert to brats19 naming convention
        new_bid = brats_name_map[bid][0]
        grade = brats_name_map[bid][1]
        brats_dict[new_bid] = {}
        brats_dict[new_bid]['rho_inv'] = float(rho_inv[i])
        brats_dict[new_bid]['kappa_inv'] = float(kappa_inv[i])
        brats_dict[new_bid]['grade'] = grade

    return brats_dict

def getBrats18ToBrats19Map(name_mapping_file):
    data = pd.read_csv(name_mapping_file)

    brats_name_map = {}
    brats_19_to_18 = {}
    for i in range(data.shape[0]):
        brats18 = data['BraTS_2018_subject_ID'][i]
        brats19 = data['BraTS_2019_subject_ID'][i]
        grade = data['Grade'][i]
        brats_name_map[brats18] = [brats19, grade]
        brats_19_to_18[brats19] = brats18

    return brats_name_map,brats_19_to_18

def breakList(my_list, n):
    res = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n )]
    return res


#### tumor generation part ####
#TODO
#1. For each initial condition (brats ID): 
#- sample 4 atlasses, 
#- sample 4 values of gamma from, say {0, 30k, 60k, 90k}, 
#- sample 2 values of rho from say {4,6,8,10,12,14}, (mean/std across brats18 cohort is 10.3 / 2.6)
#- sample 2 values for kappa from say [5E-3, 8E-2]  (mean/std across brats18 cohort is 1.37E-2 / 9.75E-3)

#2. Generate tumor data via ME forward solves using 2 of the above atlasses for each  rho/kappa from inversion + sampled gamma
#3. Generate tumor data via ME forward solves using other 2 of the above atlasses and sampled value for rho, kappa, gamma  
#5. For each sample, copy relevant images to a folder with structure <HGG/LGG>/<BID>/tuME-rho-<rho>-kappa-<kappa>-gamma-<gamma>/<SAMPLE_GOES_HERE>  (we need original Brats dl_tu seg, normal atlas seg, seg of abnormal fwd tumor result; all resampled to 698_templates resolution, i.e., 256x256x124)
#4. Generate as csv list, containing each sample, i.e., columns: [‘bid', ‘aid’, ‘rho’, ‘kappa’, ‘p_path’, ’normal_p’, ‘abnormal_p'] where p_path is the relative path to the patient dir, e.g., HGG//BID/, normal_p is the (relative) path to the healthy atlas segmentation, abnormal_p is the relative path to the segmented output of the tumor forward model.
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grow synthetic tumor in a healthy subject")
    parser.add_argument("-i", "--brats_results_path", type=str, help="path to base directory containing tumor inversion data for all brats id", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="path to directory where synthetic tumors need to be stored", required=True)
    parser.add_argument("-a", "--atlas_path", type=str, help="path to directory where atlas segmentation files are stored (naming convention - atlas_segmented_256x256x256_aff2jakob.nii.gz), all affine registered to jakob", required=True)
    parser.add_argument("-r", "--tumor_inversion_features", type=str, help="path to csv file containing brats tumor inversion results and features", required=True)
    parser.add_argument("-m", "--name_mapping_file", type=str, help="path to csv file mapping names brats18 to brats19", required=True)
    parser.add_argument("-j", "--submit_jobs", type=bool, default=False, help="True=submit jobs, False=Dont submit jobs")
    parser.add_argument("-c", "--tumor_code_dir", type=str, help="path to pglistr_tumor", required=True)
    parser.add_argument("-d", "--data_path", type=str, help="path to data (wm,gm,csf,ve) for tumor growth", required=True)
    args = parser.parse_args()
    
    
    tumor_code_dir = args.tumor_code_dir;
    # get a list of brats/patient ids
    brats_ids = next(os.walk(args.brats_results_path))[1]
    # number of brats patients
    num_brats = len(brats_ids)

    # get output path
    output_path = args.output_path;
    os.makedirs(output_path, exist_ok=True)

    # path to brats tumor inversion results directory
    brats_results_path = args.brats_results_path

    # relative path of tumor activation and centers
    tumor_inversion_rel_path = "tumor_inversion/nx256/obs-1.0/"
    tumor_activation_rel_path = tumor_inversion_rel_path + "p-rec-scaled.txt"
    tumor_center_rel_path = tumor_inversion_rel_path + "phi-mesh-scaled.txt"

    # get list of atlas names
    atlas_path = args.atlas_path    
    atlas_list = next(os.walk(atlas_path))[2]
    # filter atlas to only get Y02 i.e. year 2 atlases
    atlas_file_list = [x for x in atlas_list if "Y02" in x]
    atlas_list = [x.split("_")[0] for x in atlas_file_list]
    atlas_suffix = "_segmented_256x256x256_aff2jakob.nii.gz"

    # sample gamma from gamma set
    gamma_list = [0, 3E4, 6E4, 9E4]
    # sample rho values from set
    rho_list = [4,6,8,10,12,14]
    # sample kappa values from range
    kappa_list = [5E-3, 8E-2]

    # get name mapping from brats18 to brats19
    brats_name_map,brats_19_to_18 = getBrats18ToBrats19Map(args.name_mapping_file)

    # read in brats tumor inversion rho and kappa
    brats_dict = readBratsTumorInversionResults(args.tumor_inversion_features, brats_name_map)
    
    if len(brats_dict) != num_brats:
        if DEBUG:
            print("num_brats = {}, len(brats_dict) = {}".format(num_brats, len(brats_dict)))
            print("Mismatch between number of rho_inv and available brats tumor inversion results, using the lesser number of brats subjects")        
        num_brats = np.minimum(num_brats, len(brats_dict))

    # call a sampling function which samples atlas, rho, kappa, gamma
    [sampled_atlas, sampled_rho, sampled_kappa, sampled_gamma, brats_dict] = sampleInputParam(num_brats, atlas_list, rho_list, kappa_list, gamma_list, brats_dict)
    
    # csv file to write synthetic tumor param info to
    f = open(os.path.join(output_path, "syn_tumor_brats19.csv"), "w");
    f.write("bid,aid,rho,kappa,gamma,p_path,normal_p,abnormal_p\n")

    k = 0
    # iterate over the patients
    for i,bid in enumerate(brats_dict):
        bid18 = brats_19_to_18[bid]
        # specific brats id path
        bid_path = os.path.join(brats_results_path, bid18)
        #check if for this brats id, the tumor center and activation exist.
        # if not then skip this patient
        centers_path = os.path.join(bid_path, tumor_center_rel_path);
        activations_path = os.path.join(bid_path, tumor_activation_rel_path)
        if not os.path.exists(centers_path) or not os.path.exists(activations_path):
            print("no tumor activation center for this brats id, skipping")
            continue
        
        # get input parameters
        # 4 sampled atlases
        s_atlas = brats_dict[bid]['s_atlas']
        # 2 sampled rho
        s_rho = brats_dict[bid]['s_rho']
        # 2 sampled kappa
        s_kappa = brats_dict[bid]['s_kappa']
        # 4 sampled gamma
        s_gamma = brats_dict[bid]['s_gamma']
        # inverted rho and kappa from the tumor inversion result
        rho_inv = brats_dict[bid]['rho_inv']
        kappa_inv = brats_dict[bid]['kappa_inv']

        # create a 4 element list of the above params
        input_param_list = []
        input_param_list.append([s_atlas[0], rho_inv, kappa_inv, s_gamma[0]])
        input_param_list.append([s_atlas[1], rho_inv, kappa_inv, s_gamma[1]])
        input_param_list.append([s_atlas[2], s_rho[0], s_kappa[0], s_gamma[2]])
        input_param_list.append([s_atlas[3], s_rho[1], s_kappa[1], s_gamma[3]])

        # get the tumor grade to store in the csv file for future reference
        grade = brats_dict[bid]['grade']
        # loop over params to generate the tumor
        for input_param in input_param_list:
            atlas = input_param[0]
            rho = input_param[1]
            kappa = input_param[2]
            gamma = input_param[3]

            # relative path to the brats id
            p_path = "{}/{}".format(grade, bid)
            # prefix for the synthetic tumor folder tumor_grade/brats_id/tuME-rho-[]-kappa-[]-gamma-[]
            output_prefix = "{0}/{1}/tuME-rho-{2}-kappa-{3:2.2E}-gamma-{4}".format(grade,bid,rho,kappa,gamma)
            # relative path to the healthy subject (or atlas)
            normal_p = output_prefix + "/normal_p_256x256x124.nii.gz"
            # relative path to the grown synthetic tumor
            abnormal_p = output_prefix + "/abnormal_p_256x256x124.nii.gz"

            # create tumor param string to write into the syn_tumor_brats19.csv file
            s = "{0},{1},{2},{3:.2E},{4},{5},{6},{7}\n".format(bid, input_param[0], input_param[1], input_param[2], input_param[3], p_path, normal_p, abnormal_p)
            f.write(s)
            
            tumor_output_path = os.path.join(output_path, output_prefix)
            os.makedirs(tumor_output_path, exist_ok=True)
            
            jobfile = open(os.path.join(tumor_output_path, "job_file.sh"), "w")

            cmd = JOB_HEADER;
            # symlink the normal_p to the relative location
            cmd += "\n\n#symlink the normal_p in 256x256x124 resolution\n"
            cmd += "ln -sf ../../../698_templates_nifti_256x256x124/" + atlas + "_segmented_aff2jakob_256x256x124.nii.gz normal_p_256x256x124.nii.gz\n"

            cmd += "\n\n#Resample the BraTS_2019_subject\n"
            cmd += "ln -sf ../../../training/" + grade + "/" + bid + "/" + bid + "_seg_tu_dl_256x256x124.nii.gz " + bid + "_seg_tu_dl_256x256x124.nii.gz\n" 

            # create command string for fixing tumor initial condition for job submission
            cmd += "\n\npython3 " + os.path.join(tumor_code_dir,"scripts/data-gen/fix_tumor_init_condition.py") + \
                    " --atlas " + os.path.join(atlas_path, atlas+atlas_suffix) + \
                    " --centers " + centers_path + \
                    " --activations " + activations_path + \
                    " --output " + tumor_output_path


            c0_path = os.path.join(tumor_output_path, 'c0.nc');

            #command to check of the output of previous command exists
            cmd += "\n\n#Check if c0.nc exists in the output_path\n"
            cmd += "if [ ! -f " + c0_path + " ]; then\n"
            cmd += "echo " + c0_path + " does not exist\nexit;\n"            
            cmd += "fi"

            cmd += "\n\n#Tumor forward command\n"            
            tu_cmd,err = getTumorGenCmd(atlas, args.data_path, c0_path, tumor_output_path, rho, kappa, gamma)
            if not err:
                cmd += tu_cmd
            else:
                print("tumor command function returned a non zero exit code for {}".format(output_prefix))

            # convert abnormal seg to nifti
            cmd += "\n\n#Convert final segmentation to nifti\n"
            cmd += "python3 " + os.path.join(tumor_code_dir, "scripts/grid-cont/utils.py") + \
                   " -convert_netcdf_to_nii --name_old " + os.path.join(tumor_output_path, "seg_t[50].nc") + \
                   " --name_new " + os.path.join(tumor_output_path, "abnormal_p_256x256x256.nii.gz") + \
                   " --reference_image " + os.path.join(tumor_output_path, "c0.nii.gz");

            jobfile.write(cmd)
            jobfile.close()
            
            # remove this when submitting jobs for all patients.
            # keep this to test for one patient.
            sys.exit()
                        
    f.close()
