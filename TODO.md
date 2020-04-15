# TODOS CLEANUP
---

### opt/ todos:
1.     Non-class methods in TaoInterfaceLandfill need to be revisited, cleaned up, and if possible combined with existing methods in TaoInterface
2.     In opt/ file includes, member function definitions (virtual vs non-virtual etc need to be fixed and reviewed. I’ll do that
[done] In Solver, I hope all occurrences of solver_interface are fixed but not sure; Solver should have it’s own .cpp file; all remaining methods from TumorSolverInterface (which are needed) should be moved to Solver.cpp;
4.     In Solver.cpp, the allocation of p_vec_ needs to be fixed; quick and dirty I’ve been re-allocating in the run() methods if required; we should find a better way of allocating p_vec_ once and for all in specialized initialize of solvers. This is necessary since optimizer now has a function setInitialGuess which assumes a specific length of the vector (for RD and ME TIL can be given as Phi(p) or c(0))
5.     In derivative operators: all Vec data has to be replaced by std::shared_ptr<Data> data, all data access by data->dt1(), all obs_->apply(..) by  bs_->apply(.., 1); the latter is for clarity, time point one is the default.
6.     Fixing many compile errors
7.     Fixing many runtime errors

---

### other general todos:
1. create directories and move files as indicated in discussion.txt, we should have directories cuda/, test/, opt/, pde/, utils/, mat/, and maybe solver/
2. derivative operators: a generic FD model should be provided, which works for mass effect and alzh, maybe we should split up derivative operators
3. Test suite
4. remaining todos in discussion.txt and TODO.md
5. Minimal documentation
6. python scripts for grid-cont, forward, etc which process the data and do everything (also have to implement functionality to read in segmentation, i.e., split up into tissues)
7. clean up in scripts/


---


### All files
 - (S) Format to tab-width 2, CHKERRQ(ierr) to one space after ";" and general \(\) formating

### External functionality:
 - we need to discuss how to handle this. SIBIA is not compatible with current code anyways. I think, if used, sibia has to be restructured, and fittet to new tumor code. That said, some of the legacy functinality should be dropped (which is clearly
   outdated and only supported bc of SIBIA) but I think not all of it should be deleted. This is my suggestion:
 - keep modified objective, keep functions which compute moving-atlas contributions, but maybe move them into one combined file: agreed. will be done after opt/ is complete
 ---

### inverse/Solver
 - add l1 p relative error
 - Maybe bg can be dropped if lame-coefficients are initialized with its value:  modify _ElastictiySolver.cpp_
---

### IO
- change dataOut to take params_ as argument and std::string instead of char*
---

### PdeOperators
- implement preAdvection
- (S) move reaction solve to a new file ReactionSolver? I think this is more consistent even though it's an analytic solve and doesn't really need a new class, K: New class is fine. I'd say new file ReacSolver.h/cpp, then a new folder pde_solvers, and
  put all forward solvers there.
 
--- 
### DiffSolver, ElasticitySolver, AdvectionSolver
- (S) Rename DiffSolver to DiffusionSolver for consistency. ok
- move all pde solvers to new folder pde/


--- 
### DiffCoef, ReacCoef
- Remove glm ratios? tumor never grows or diffuses here K: yes let's remove
- can we remove the functions DiffCoef::compute_dKdm_gradc_gradp and ReacCoef::applydRdm? What are they needed for? K: needed for sibia moving atlas, we need to discuss what to do about this.

--- 
### DerivativeOperators
- data struct
- generic FD derivative operators
- split up

---
### Utils
- GeometricCoupling() etc are opt related. Move to obj deriv ops source

---
### TestSuite
- implement apply low frequency noise
- implement generation of MF data; you can also define coordinates in a p.txt and phi.txt file, store the min a test/ dir, and read them in (with hard coded path);
- implement generate sinusoidal
- implement tests
- I.  SIN
  - a. forward
  - b. l2 inverse with small opttol 
- II. Forward (non-const coeff; read in sample brain; all tests 64^3, nifty)
  - a. forward solver me test
  - b. forward solver ms test
- III. Inverse:
  - a. l2 inverse, compare error c1, c0, kappa, p
  - b. l1 inverse, compare error c1, c0, TIL, rho, kappa, gamma
  - c. me inversion, compare rho, kappa, gamma
---

### Invsolver
 - we need to restructure, possibly difide into several files, make a folder optimizers/ and files ReactionDiffusionInversion, MassEffectInversion, SparseTILInversion, NonSparseTILInversion?, have a superclass Optimizer for shared things?
 - use params 'ls_max_func_evals', 'lbfgs_vectors_', 'lbfgs_scale_hist', 'lbfgs_scale_type'to set petsc settings
 - why does itctx have optsetting_ etC?
