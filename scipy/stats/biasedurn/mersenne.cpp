
#include <random>
#include "stdint.h"

class CRandomMersenne {
public:

  CRandomMersenne(uint32_t seed) {
    RandomInit(seed);
  }

  void RandomInit(uint32_t seed) {
    gen.seed(seed);
  }

  uint32_t IRandom (uint32_t min, uint32_t max) {
    return gen();
  }

  double Random() {
    return udist(gen);
  };

private:
  std::mt19937 gen;
  std::uniform_real_distribution<double> udist{0.0, 1.0}; // is this the correct interval?
};
