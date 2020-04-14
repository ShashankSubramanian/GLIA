#include "Solver.h"

class TestSuite : public Solver {
 public:
  TestSuite() : Solver() {}

  virtual PetscErrorCode finalize();
  virtual PetscErrorCode initialize(std::shared_ptr<SpectralOperators> spec_ops, std::shared_ptr<Parameters> params, std::shared_ptr<ApplicationSettings> app_settings);
  virtual PetscErrorCode run();

  virtual ~TestSuite() {}
};

// /* simulator level tests */
// class ForwardTest : public TestSuite {
//  public:
//   ForwardTest() : TestSuite() {}

//   virtual PetscErrorCode finalize();
//   virtual PetscErrorCode run();

//   virtual ~ForwardTest() {}

// };

// /* simulator level tests */
// class InverseTest : public TestSuite {
//  public:
//   InverseTest() : TestSuite() {}

//   virtual PetscErrorCode finalize();
//   virtual PetscErrorCode run();

//   virtual ~InverseTest() {}

// };
