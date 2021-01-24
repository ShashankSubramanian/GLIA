import os, sys, warnings, argparse, subprocess
import numpy as np
import math
import pandas as pd
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../utils/')
sys.path.append('../')


###
### ------------------------------------------------------------------------ ###
if __name__=='__main__':
  parser = argparse.ArgumentParser(description='extract/compute tumor stats')
  parser.add_argument ('-n', type=int, help = 'size', default = 160);
  parser.add_argument ('-forward_path', type=str, help = 'path to tumor forward solve results');

  args = parser.parse_args();
  fwd_path = args.forward_path
  extract_vols = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
#  extract_vols = [0.05]
  extract_time = [1]

  extracted_features = []
  pat_list = []
  for pat in os.listdir(fwd_path):
    cur_path = os.path.join(*[fwd_path, pat, "tu", str(args.n)])
    if os.path.exists(cur_path):
      pat_list.append(pat)

  for vol in extract_vols:
    pat_idx = 0
    
    # get any feature file to get columm names
    cur_path = os.path.join(*[fwd_path, pat_list[0], "tu", str(args.n)])
    for sub in os.listdir(cur_path):
      if os.path.exists(os.path.join(*[cur_path, sub, "solver_config.txt"])):
        atlas = sub
    feature_file = os.path.join(*[cur_path, atlas, "biophysical_features.csv"])
    features = pd.read_csv(feature_file, header = 0)
    features = features.iloc[:, :-1]
    # get the column names to create a new dataframe with the same list
    new_cols = ["PATIENT"] + features.columns.values.tolist()
    out_df = pd.DataFrame(columns = new_cols) # create new file/dataframe for every vol ratio 

    # populate new dataframe with features from every patient
    pat_failed = []
    for pat in pat_list:
      cur_path = os.path.join(*[fwd_path, pat, "tu", str(args.n)])
      
      for sub in os.listdir(cur_path):
        if os.path.exists(os.path.join(*[cur_path, sub, "solver_config.txt"])):
          atlas = sub

      feature_file = os.path.join(*[cur_path, atlas, "biophysical_features.csv"])
      features = pd.read_csv(feature_file, header = 0)
      features = features.iloc[:, :-1]
      vols = features.loc[:,"volc/volb"].values.tolist() #vector of vol ratios - subselect from these
      vol_diff = [np.abs(v - vol) for v in vols]
      vol_ratio = vols[vol_diff.index(min(vol_diff))]
#      if vol_ratio < 0.05 and np.abs(vol_ratio - 0.05) > 0.01:
#        pat_failed.append((pat,vol_ratio))
      out_df = out_df.append({}, ignore_index = True)
      out_df.iloc[pat_idx,0] = pat
      out_df.iloc[pat_idx,1:] = features.loc[features["volc/volb"] == vol_ratio].values
      pat_idx += 1
      print("finished patient = ", pat)
    out_df.to_csv(os.path.join(fwd_path, "temporal_stats_vol_" + str(vol) + ".csv"))
    del out_df

  for time in extract_time:
    pat_idx = 0
    
    # get any feature file to get columm names
    cur_path = os.path.join(*[fwd_path, pat_list[0], "tu", str(args.n)])
    for sub in os.listdir(cur_path):
      if os.path.exists(os.path.join(*[cur_path, sub, "solver_config.txt"])):
        atlas = sub
    feature_file = os.path.join(*[cur_path, atlas, "biophysical_features.csv"])
    features = pd.read_csv(feature_file, header = 0)
    features = features.iloc[:, :-1]
    # get the column names to create a new dataframe with the same list
    new_cols = ["PATIENT"] + features.columns.values.tolist()
    out_df = pd.DataFrame(columns = new_cols) # create new file/dataframe for every vol ratio 

    # populate new dataframe with features from every patient
    pat_failed = []
    for pat in pat_list:
      cur_path = os.path.join(*[fwd_path, pat, "tu", str(args.n)])
      
      for sub in os.listdir(cur_path):
        if os.path.exists(os.path.join(*[cur_path, sub, "solver_config.txt"])):
          atlas = sub

      feature_file = os.path.join(*[cur_path, atlas, "biophysical_features.csv"])
      features = pd.read_csv(feature_file, header = 0)
      features = features.iloc[:, :-1]
      times = features.loc[:,"t"].values.tolist() #vector of times - subselect from these
      times_diff = [np.abs(t - time) for t in times]
      time_select = times[times_diff.index(min(times_diff))]
      if time_select < 1:
        pat_failed.append((pat,time_select))
      out_df = out_df.append({}, ignore_index = True)
      out_df.iloc[pat_idx,0] = pat
      out_df.iloc[pat_idx,1:] = features.loc[features["t"] == time_select].values
      pat_idx += 1
      print("finished patient = ", pat)
    out_df.to_csv(os.path.join(fwd_path, "temporal_stats_time_" + str(time) + ".csv"))
    del out_df

  print(len(pat_failed))
  print((pat_failed))
#  with open(os.path.join(fwd_path, "failed_artificialdiffusion.txt"), "w+") as f:
#    for p in pat_failed:
#      f.write(p[0] + "\n")



