# TODOS CLEANUP
---

### Solver
- set param->tu_->*_scale_ scales for FD gradients of RD only and ME inversion
- initialize a DerivativeOperator object for RDOnlyFD
- remaining alzh stuff

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

(S)
- cuda: performance is poor (gpu util < 30%) for full subspace solve ~ need to figure out why

---

### pde/
- for alzh: advection of material properties in forward solver is not yet implemented
- implement preAdvection. 

---

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


---

### doc/
- write documentation

---

### scripts/
- clean up
- preprocessing of data should be done by ANTS, re-write scripts in this regard
- clean up grid-cont, make it work with new code, and simplify.
- clean up alzh script, make it work with new code.
- script for forward run
- script for inverse run
- organize other helper scripts.



### External functionality:
 - we need to discuss how to handle this. SIBIA is not compatible with current code anyways. I think, if used, sibia has to be restructured, and fittet to new tumor code. That said, some of the legacy functinality should be dropped (which is clearly
   outdated and only supported bc of SIBIA) but I think not all of it should be deleted. This is my suggestion:
 - keep modified objective, keep functions which compute moving-atlas contributions, but maybe move them into one combined file: agreed. will be done after opt/ is complete
 ---

### inverse/Solver
 - add l1 p relative error
---


--- 
### DiffCoef, ReacCoef
- Remove glm ratios? tumor never grows or diffuses here K: yes let's remove
- can we remove the functions DiffCoef::compute_dKdm_gradc_gradp and ReacCoef::applydRdm? What are they needed for? K: needed for sibia moving atlas, we need to discuss what to do about this.

