#include "Solver.h"
enum Test {DEFAULTTEST, FORWARDTEST, INVERSETEST, PDEOPSTEST, DERIVOPSTEST};

class TestSuite : public Solver {
 public:
  TestSuite(Test test);

  Test testcase_;
  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  /* Unit tests */
  /* Simulator level tests */
  PetscErrorCode forwardTest();
  PetscErrorCode inverseTest();
  /* Pdeops tests */
  PetscErrorCode pdeopsTest() {}
  /* Derivops tests */
  PetscErrorCode derivopsTest() {}

  virtual ~TestSuite() {}
};

