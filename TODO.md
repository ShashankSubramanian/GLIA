# TODOS CLEANUP
---

### All files
(S) Format to tab-width 2, CHKERRQ(ierr) to one space after ";" and general \(\) formating


### inverse.cpp/Solver.cpp

**todo: (implement)**
 - in _tusolver.cpp_: config file; parse config file; populate to parameters
 - in _Solver.cpp_: implement read in mass effect data; healthy patient etc; complete ME initialize
 - in _Solver.cpp_: implement read in alzh data
 - in _Solver.cpp_: implement apply low frequency noise <-- add to createSynthetic

 (S):
 - add l1 p relative error
 - remove computeSegmentation ffrom _Solver.cpp_ and use it from tumor class [K done]
 - add mass effect data to Parameters, config, and to the inverse solver in _Solver.cpp_
 - Maybe bg can be dropped if lame-coefficients are initialized with its value:  modify _ElastictiySolver.cpp_

**unclear: (dropped)**
 - in _Solver.cpp_: createSynthetic: I dropped generation of MF data; this should be implemented as test in the test suite, since hard coded coordinates; you can also define coordinates in a p.txt and phi.txt file, store the min a test/ dir, and read them in (with hard coded path);
- generate sinusoidal: do we need this? <-- move to test suite : (S) agreed
 - inverse.cpp:840-844: no idea what this is good for, not implemented: (S) I think it was for wL2 solve.
 - inverse.cpp:877-891: not implemented; outparams not implemented: (S) agreed
 - inverse.cpp: 893-920: wL2 norm, not implemented, dropped this section: (S) agreed
 - generateSyntheticData: why is c(0) read in in generateSynthetic? Im dropping this; reading in is done in readData: (S) no reason. agreed
 - inverse.cpp:1603-1718: error for p (plus some weighted error) I have dropped this: (S) this is needed. the weighted error can be dropped. I will add this.
---

 ### Parameters.h
  - populate all parameters from utils and n_misc; structure them into groups, see examples
  - change all occurrences of n_misc parameters etc.

---

### Tumor.cpp/Tumor.h
- remove p_true_, remove setTrueP(), (already removed dependency in Solver) 

### MatProb.h
- can we get rid of bg? clarify what glm is used for; I also introduced ve;
No, bg is needed for lame parameters in the bg (which is a hard stiff material). The alternative is to drop it and compute bg everytime elasticity solver is called. I suggest we keep it there for now.
(S) remove bg later by re-initializing all lame coefficients

---

### Obs.h/Obs.cpp
- implement two time points from alzh solver; add time point to apply in all files

---

### IO.h/IO.cpp
- (S) move all IO functions from Utils to IO: [done]
- (S) change all  n_misc to params and corresponding parameters to the correct class: grid or tu: [done]
- change dataOut to take params_ as argument and std::string instead of char*

---

### PdeOperators
- change forward_flag to time_history_off_: (S): [done]
- (S) n_misc --> params changes: also change for CUDA routines: [done]
- (S) csf -> vt

**unclear**
- (S) move reaction solve to a new file ReactionSolver? I think this is more consistent even though it's an analytic solve and doesn't really need a new class

--- 
### DiffSolver, ElasticitySolver, AdvectionSOlver
- (S) n_misc --> params changes: also change for CUDA routines: [done]
- (S) Rename DiffSolver to DiffusionSolver for consistency

--- 
### DiffCoef, ReacCoef
- (S) n_misc --> params changes: [done]
- (S) Move sinusoidal setvalues to TestSuite.cpp: [done]
- (S) csf -> vt

**unclear**
- Remove glm ratios? tumor never grows or diffuses here
- can we remove the functions DiffCoef::compute_dKdm_gradc_gradp and ReacCoef::applydRdm? What are they needed for?

--- 
### MatProp
- (S) n_misc --> params changes: [done]
- (S) csf -> vt

--- 
### Tumor
- (S) n_misc --> params changes: [done]
- (S) remove setTrueP: [done]
- (S) remove weights (only used in wL2): [done]
- (S) delete p_true_ : [done]

---
### TumorSolverInterface
- (S) Make sure tumor::setTruP is not used

---
### InvSolver
- (S) Make sure tumor->weights_ is not used

---
### Utils.h
- (S) Move enums to typedefs and remove parameters (except tumor statistics): [done]

---
### TestSuite.cpp
- implement tests

---
### Invsolver
- OptimizerSettings members have a trailing underscore now.
- changed newtonsolver to newton_solver_
- moved flag_reaction_inv_ from n_misc to opt
- introduced params (in opt) 'ls_max_func_evals', 'lbfgs_vectors_', 'lbfgs_scale_hist', 'lbfgs_scale_type'; we should set those parameters directly in the code.
