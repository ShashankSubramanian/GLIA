# TODOS CLEANUP
---

### All files
 - (S) Format to tab-width 2, CHKERRQ(ierr) to one space after ";" and general \(\) formating
 - (K) add time point to observation operator

### inverse/Solver

**todo: (implement)**
 - in _Solver.cpp_: implement apply low frequency noise <-- add to createSynthetic

 (S):
 - add l1 p relative error
 - remove computeSegmentation ffrom _Solver.cpp_ and use it from tumor class [K done]
 - add mass effect data to Parameters, config, and to the inverse solver in _Solver.cpp_
 - Maybe bg can be dropped if lame-coefficients are initialized with its value:  modify _ElastictiySolver.cpp_

**unclear: (dropped)**
 - in _Solver.cpp_: createSynthetic: I dropped generation of MF data; this should be implemented as test in the test suite, since hard coded coordinates; you can also define coordinates in a p.txt and phi.txt file, store the min a test/ dir, and read them in (with hard coded path);
 - generate sinusoidal: do we need this? <-- move to test suite : (S) agreed
 - inverse.cpp:1603-1718: error for p (plus some weighted error) I have dropped this: (S) this is needed. the weighted error can be dropped. I will add this.
---

### IO
- change dataOut to take params_ as argument and std::string instead of char*
---

### PdeOperators
- (S) csf -> vt

**unclear**
- (S) move reaction solve to a new file ReactionSolver? I think this is more consistent even though it's an analytic solve and doesn't really need a new class, K: New class is fine. I'd say new file ReacSolver.h/cpp, then a new folder pde_solvers, and
  put all forward solvers there.
 
--- 
### DiffSolver, ElasticitySolver, AdvectionSolver
- (S) Rename DiffSolver to DiffusionSolver for consistency. ok
- move all pde solvers to new folder pde_Solvers

--- 
### SpectralOperators

--- 
### Phi

--- 
### DiffCoef, ReacCoef
- (S) csf -> vt

**unclear**
- Remove glm ratios? tumor never grows or diffuses here K: yes let's remove
- can we remove the functions DiffCoef::compute_dKdm_gradc_gradp and ReacCoef::applydRdm? What are they needed for? K: needed for sibia moving atlas, we need to discuss what to do about this.

--- 
### DerivativeOperators
- n_misc --> params changes

---
### TaoL1Solver
- Delete: make sure InvSolve does not use this

--- 
### MatProp
- (S) csf -> vt

--- 
### Tumor

---
### TumorSolverInterface
- (S) Make sure tumor::setTruP is not used

---
### Utils
- (S) Some functions depend on NMisc --> move these elsewhere

---
### TestSuite
- implement tests
- I.  SIN
  - a. forward
  - b. l2 inverse with small opttol 
- II. Forward (non-const coeff; read in sample brain; all tests 64^3, nifty)
  - a. forward solver me test
  - b. forward solver ms test
  - c. l2 inversion test
- III. Inverse:
  - a. l2 inverse, compare error c1, c0, kappa, p
  - b. l1 inverse, compare error c1, c0, TIL, rho, kappa, gamma
  - c. me inversion, compare rho, kappa, gamma
---


### Invsolver
- use params 'ls_max_func_evals', 'lbfgs_vectors_', 'lbfgs_scale_hist', 'lbfgs_scale_type'to set petsc settings
