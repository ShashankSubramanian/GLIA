import os, sys
import params as par

###############
r = {}
p = {}
submit = False;
###############

### === define code path, write path, read path
code_path =
write_path =
read_path =

### === define parameters
p['output_dir'] = write_path;



### === define run configuration
r['code_path'] = code_path;

# === write config to write_path and submit job
par.submit(submit, p, r);
