
import os, sys, warnings, argparse, subprocess
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../common/'))
from shutil import copyfile
import nibabel as nib
import numpy as np
import nibabel as nib
import pandas as pd
import shutil
import imageTools as imgtools
import file_io as fio
import random


###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
    # repository base directory
    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
    basedir_real = os.path.dirname(os.path.realpath(__file__));
    # parse arguments
    parser = argparse.ArgumentParser(description='Process input images',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    r_args = parser.add_argument_group('required arguments')
    r_args.add_argument ('-patient_path',   '--patient_image_path',          type = str, help = 'path to patient image directory containing the T1,T2,FLAIR,T1CE and segmentation images\n (format- PatientName_{t1,t2,t1ce,flair,segmented}.ext)', required=True)
    r_args.add_argument ('-patient_labels', '--patient_segmentation_labels', type=str,   help = 'comma separated patient segmented image labels. for ex.\n  0=bg,1=nec,2=ed,4=enh,5=wm,6=gm,7=vt,8=csf\n for BRATS type segmentation. DISCLAIMER vt and every extra label mentioned will be merged with csf');
    r_args.add_argument ('-atlas_path',     '--atlas_image_path',            type = str, help = 'path to a segmented atlas image (affinely registered to given patient)', required=True)
    parser.add_argument ('-atlas_t1_image', '--atlas_t1_image',              type = str, help = 'path to t1 atlas image (affinely registered to given patient)')
    parser.add_argument ('-x', '--results_directory',                        type = str, help = 'path to destination')
    args = parser.parse_args();

    col_names = ["abnormal_seg", "abnormal_t1", "normal_seg", "normal_t1", "normal_c0"]
    fname_df       = pd.DataFrame(columns=col_names)
    fname_df_train = pd.DataFrame(columns=col_names)
    fname_df_val   = pd.DataFrame(columns=col_names)
    fname_df_test  = pd.DataFrame(columns=col_names)

    grade = "HGG"

    patient_list = os.listdir(args.patient_image_path)
    atlas_list = ['0034Y02', '0056Y02', '0101Y02', '0115Y02', '0177Y02', '0222Y02', '0241Y02', '0258Y02', '0396Y02']
    base_results_dir = args.results_directory;
    base_patient_image_path = args.patient_image_path
    base_atlas_image_path = args.atlas_image_path
    for patient in patient_list:
        if "BraTS" not in patient:
            continue;
        for atlas in atlas_list:
            print('processing patient/atlas pair [{} / {}]'.format(patient, atlas))
            patient_image_path = os.path.join(os.path.join(os.path.join(os.path.join(base_patient_image_path, patient), "data"), "affreg2-"+str(atlas)), patient + '_seg_dl_tu_256x256x256_aff2'+str(atlas)+'.nii.gz');
            if atlas == 'jacob':
                atlas_image_path = os.path.join(os.path.join(base_atlas_image_path, atlas), 'jakob_segmented_with_cere_lps_256x256x256.nii.gz');
                atlas_t1_image   = os.path.join(os.path.join(base_atlas_image_path, atlas), 'jakob_stripped_with_cere_lps_256x256x256.nii.gz');
            else:
                atlas_image_path = os.path.join(os.path.join(base_atlas_image_path, atlas), atlas + '_segmented.nii.gz');
                atlas_ref_path   = os.path.join(os.path.join(base_atlas_image_path,"../698_templates/256x256x124_analyze"), atlas + '_segmented.img');
                atlas_t1_image   = os.path.join(os.path.join(base_atlas_image_path, atlas), atlas + '_cbq_n3.nii.gz');
            # directories
            reg_out_dir = os.path.join(os.path.join(base_patient_image_path, patient), "{}_regto_{}".format(patient,atlas));
            reg_out_dir = os.path.join(reg_out_dir, "registration");
            out_dir = os.path.join(os.path.join(os.path.join(base_results_dir, grade), str(patient)), "{}_regto_{}".format(patient,atlas));
            if not os.path.exists(out_dir):
                print("results folder doesn't exist, creating one!\n");
                os.makedirs(out_dir);

            run_ok = True;
            p_dict = {}
            p_dict['norma_seg'] = os.path.join(grade, "normal_"+str(atlas)+"_seg_256x256x124.nii.gz")
            p_dict['normal_t1'] = os.path.join(grade, "normal_"+str(atlas)+"_t1_256x256x124.nii.gz")
            p_dict['normal_c0'] = os.path.join(grade, "normal_"+str(atlas)+"_with_c0_256x256x124.nii.gz")
            p_dict["abnormal_seg"] = os.path.join(grade, "abnormal_"+str(atlas)+"_from_"+str(patient)+"_seg-combined_"+"_256x256x124.nii.gz")
            p_dict["abnormal_t1"] = os.path.join(grade, "abnormal_"+str(atlas)+"_from_"+str(patient)+"_t1_"+"_256x256x124.nii.gz")

            try:
                 shutil.copy2(atlas_image_path, os.path.join(out_dir, "normal_"+str(atlas)+"_seg_256x256x256.nii.gz"))
                 shutil.copy2(atlas_t1_image, os.path.join(out_dir, "normal_"+str(atlas)+"_t1_256x256x256.nii.gz"))
                 shutil.copy2(patient_image_path, os.path.join(out_dir, str(patient)+"_seg_aff2_"+str(atlas)+"_256x256x256.nii.gz"))
                 shutil.copy2(os.path.join(reg_out_dir, "atlas_with_warped_tumor_in_Pspace.nii.gz"), os.path.join(out_dir, "abnormal_"+str(atlas)+"_from_"+str(patient)+"_seg_"+"_256x256x256.nii.gz"))
                 shutil.copy2(os.path.join(reg_out_dir, "atlas_t1_in_Pspace.nii.gz"), os.path.join(out_dir, "abnormal_"+str(atlas)+"_from_"+str(patient)+"_t1_"+"_256x256x256.nii.gz"))
                 shutil.copy2(os.path.join(reg_out_dir, "patient_seg_in_Aspace.nii.gz"), os.path.join(out_dir, str(patient)+"_seg_in_Aspace_"+"_256x256x256.nii.gz"))
            except:
                 print("Error processing [{} / {}] combination: FILES NOT AVAILABLE (maybe registration has not been run)".format(patient,atlas))
                 run_ok = False;
            if run_ok:
                 a_ref = nib.load(atlas_ref_path);
                 # resize atlas
                 a_img = nib.load(atlas_image_path);
                 rz_img = imgtools.resizeNIIImage(a_img, tuple([256,256,124]), interp_order=0)
                 print(" .. resmapling altas"); 
                 fio.writeNII(rz_img.get_fdata(), os.path.join(out_dir, "normal_"+str(atlas)+"_seg_256x256x124.nii.gz"), ref_image=a_ref)

                 # resize patient
                 p_img = nib.load(os.path.join(reg_out_dir, "atlas_with_warped_tumor_in_Pspace.nii.gz"))
                 rz_img = imgtools.resizeNIIImage(p_img, tuple([256,256,124]), interp_order=0)
                 print(" .. resmapling altas w/ tumor in p-space"); 
                 fio.writeNII(rz_img.get_fdata(), os.path.join(out_dir, "abnormal_"+str(atlas)+"_from_"+str(patient)+"_seg_"+"_256x256x124.nii.gz"), ref_image=a_ref);
                 im = rz_img.get_fdata()
                 wt = np.logical_or(np.logical_or(im == 4, im == 1), im == 2)
                 im[wt] = 3;
                 fio.writeNII(rz_img.get_fdata(), os.path.join(out_dir, "abnormal_"+str(atlas)+"_from_"+str(patient)+"_seg-combined_"+"_256x256x124.nii.gz"), ref_image=a_ref);
                 p_img = nib.load(os.path.join(reg_out_dir, "patient_seg_in_Aspace.nii.gz"))
                 rz_img = imgtools.resizeNIIImage(p_img, tuple([256,256,124]), interp_order=0)
                 print(" .. resmapling patient in a-space"); 
                 fio.writeNII(rz_img.get_fdata(), os.path.join(out_dir, str(patient)+"_seg_in_Aspace_"+"_256x256x124.nii.gz"), ref_image=a_ref);
                 p_img = nib.load(patient_image_path);
                 rz_img = imgtools.resizeNIIImage(p_img, tuple([256,256,124]), interp_order=0)
                 print(" .. resmapling patient"); 
                 fio.writeNII(rz_img.get_fdata(), os.path.join(out_dir, str(patient)+"_seg_aff2_"+str(atlas)+"_256x256x124.nii.gz"), ref_image=a_ref);

                 rnd = random.random()
                 fname_df.loc[len(fname_df)] = p_dict;
                 if rnd >= 0 and rnd < 0.7:
                     fname_df_train.loc[len(fname_df_train)] = p_dict;
                     print("Sucessfully added [{} / {}] combination to TRAIN".format(patient,atlas))
                 elif rnd >= 0.7 and rnd < 0.85:
                     fname_df_val.loc[len(fname_df_val)] = p_dict;
                     print("Sucessfully added [{} / {}] combination to VAL".format(patient,atlas))
                 elif rnd >= 0.85 and rnd < 1:
                     fname_df_test.loc[len(fname_df_test)] = p_dict;
                     print("Sucessfully added [{} / {}] combination to TEST".format(patient,atlas))
            #except:
            #     print("Error processing [{} / {}] combination".format(patient,atlas))
    fname_df.to_csv(os.path.join(out_dir, "dataset_fnames.csv"));
    fname_df_train.to_csv(os.path.join(out_dir, "dataset_fnames_train.csv"));
    fname_df_val.to_csv(os.path.join(out_dir, "dataset_fnames_val.csv"));
    fname_df_test.to_csv(os.path.join(out_dir, "dataset_fnames_test.csv"));
