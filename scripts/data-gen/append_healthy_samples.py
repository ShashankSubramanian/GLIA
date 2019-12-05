import os, sys, warnings, argparse, subprocess, shutil
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../common/'))
from shutil import copyfile
import random 
import pandas as pd
import imageTools as imgtools
import file_io as fio
import nibabel as nib   

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Affine Register Image and transport for Brats18')
    parser.add_argument ('-atlas_dir', type = str, help = 'base directory containing all atlasses')
    parser.add_argument ('-x', '--results_directory', type = str, help = 'results directory')
    args = parser.parse_args();
    
    atlas_dir = args.atlas_dir;
    base_results_dir = args.results_directory;

    if True:
         

        col_names = ["abnormal_seg", "abnormal_t1", "normal_seg", "normal_t1", "normal_c0"]
        fname_df       = pd.DataFrame(columns=col_names)
        fname_df_train = pd.DataFrame(columns=col_names)
        fname_df_val   = pd.DataFrame(columns=col_names)
        fname_df_test  = pd.DataFrame(columns=col_names)

        atlas_list = os.listdir(os.path.join(atlas_dir));
        atlas_list = [str(os.path.basename(a).split('_')[0])  for a in atlas_list if "segmented" in a]
       
        # HGG patients
        for i in range(50):
            TEST = False;
            TRAIN = False;
            VAL = False;
            rnd2 = random.random()
            if rnd2 >= 0.0 and rnd2 < 0.7:
                TRAIN = True;
            elif rnd2 >= 0.7 and rnd2 < 0.85:
                VAL = True;
            elif rnd2 >= 0.85 and rnd2 < 1.0:
                TEST = True;

            rnd = random.random()
            idx = int(rnd*len(atlas_list))
            atlas = atlas_list[idx]

            atlas_image_path = os.path.join(atlas_dir, atlas + '_segmented.nii.gz')
            atlas_ref_path   = os.path.join(os.path.join(atlas_dir, '../256x256x124_analyze'), atlas + '_segmented.img')
            atlas_t1_image   = os.path.join(atlas_dir, atlas + '_cbq_n3.nii.gz')

            p_dict = {}
            p_dict['normal_seg']   = os.path.join(os.path.join("NORMAL", atlas), "normal_"+str(atlas)+"_seg_256x256x124.nii.gz")
            p_dict['normal_t1']    = os.path.join(os.path.join("NORMAL", atlas), "normal_"+str(atlas)+"_t1_256x256x124.nii.gz")
            p_dict['abnormal_seg'] = os.path.join(os.path.join("NORMAL", atlas), "normal_"+str(atlas)+"_seg_256x256x124.nii.gz")
            p_dict['abnormal_t1']  = os.path.join(os.path.join("NORMAL", atlas), "normal_"+str(atlas)+"_t1_256x256x124.nii.gz")
            p_dict['normal_c0']    = "";

            out_dir = os.path.join(os.path.join(base_results_dir, "NORMAL"), atlas);
            if not os.path.exists(out_dir):
                print("results folder doesn't exist, creating one!\n");
                os.makedirs(out_dir);
            
            fname_df.loc[len(fname_df)] = p_dict;
            if TRAIN:
                fname_df_train.loc[len(fname_df_train)] = p_dict;
                print("[] AID {} has been selected for TRAIN".format(atlas))
            if TEST:
                fname_df_test.loc[len(fname_df_test)] = p_dict;
                print("[] AID {} has been selected for TEST".format(atlas))
            if VAL:
                fname_df_val.loc[len(fname_df_val)] = p_dict;
                print("[] AID {} has been selected for VAL".format(atlas))

            shutil.copy2(atlas_image_path, os.path.join(out_dir, "normal_"+str(atlas)+"_seg_256x256x256.nii.gz"))
            shutil.copy2(atlas_t1_image, os.path.join(out_dir, "normal_"+str(atlas)+"_t1_256x256x256.nii.gz"))
            a_ref = nib.load(atlas_ref_path);
            # resize atlas
            a_img = nib.load(atlas_image_path);
            rz_img = imgtools.resizeNIIImage(a_img, tuple([256,256,124]), interp_order=0)
            print(" .. resmapling altas");
            fio.writeNII(rz_img.get_fdata(), os.path.join(out_dir,"normal_"+str(atlas)+"_seg_256x256x124.nii.gz"), ref_image=a_ref)
            
        fname_df.to_csv(os.path.join(base_results_dir, "dataset_healthy_atlasses_fnames.csv"), index=False);
        fname_df_train.to_csv(os.path.join(base_results_dir, "dataset_healthy_atlasses_fnames_train.csv"), index=False);
        fname_df_val.to_csv(os.path.join(base_results_dir, "dataset_healthy_atlasses_fnames_val.csv"), index=False);
        fname_df_test.to_csv(os.path.join(base_results_dir, "dataset_healthy_atlasses_fnames_test.csv"), index=False);
            
