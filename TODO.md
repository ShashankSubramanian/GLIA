# TODOS CLEANUP
---

### Solver

### gen/
- scons clean does not clean cuda files - fix.
- create a build-libs script to build all dependencies like CLAIRE

### opt/
- inconcsisten use of params->tu->nk and params->get_nk(), I think almost everywhere params->get_nk() should be used.
- dirty allocation of x_L1 in SparseTILOptimizer, revisit restrict/prolongate subspace
- test RD inverion (write test)
- test ME inverion (write test)
- test TIL inversion (write test)
- test grid cont (write test)
- cuda: performance is poor (gpu util < 30%) for full subspace solve ~ need to figure out why

### pde/

### test/
- implement integration test for ME inversion: compare rho, kappa, gamma
- implement integration test for RD inversion: compare rho, kappa
- implement integration test for TIL inversion: compare error c1, c0, kappa, p
- implement integration test for sparse TIL inversion:  compare error c1, c0, TIL, rho, kappa, gamma
- implement integration test for grid-cont integration (with coarse solution injection):  compare error c1, c0, TIL, rho, kappa, gamma
- implement unit test objective evaluation (all models)
- implement unit test gradient evaluation (all models)
- implement unit test hessian evaluation
- implement unit test adjoint solve


### doc/
- write documentation

### scripts/

### External functionality:

### inverse/Solver

### DiffCoef, ReacCoef
- Remove glm ratios?
- can we remove the functions DiffCoef::compute_dKdm_gradc_gradp and ReacCoef::applydRdm? What are they needed for? Needed for SIBIA (unused for the time being)

