# TODOS

### build/
- create a build-libs script to build all dependencies
- pnetcdf vs nifti check libs

### opt/

### pde/

### utils
- on frontera, thrust max produces NaNs on some rtx GPUs (current fix is to do max on cpu). But unsure what the problem is

### mat/
- Remove glm ratios
- can we remove the functions DiffCoef::compute_dKdm_gradc_gradp and ReacCoef::applydRdm? What are they needed for? Needed for SIBIA (unused for the time being)

### test/
- see discussion in src/test/
- most unit tests are remaining using catch

### doc/
- write documentation

### scripts/
- extract_stats and analysis (masseffect scripts) should be merged. all useless stats need to be commented out

### External functionality:
- cython functionality (most of it is done in sibia-py; needs to be refactored)

### app/


