#ifndef RANDOMC_H
#define RANDOMC_H

#include <iostream>
#include <random>

class CRandomMersenne {
public:
  CRandomMersenne(uint32_t seed);
  void RandomInit(uint32_t seed);
  uint32_t IRandom (uint32_t min, uint32_t max);
  double Random();
private:
  std::mt19937 gen;
  std::uniform_real_distribution<double> udist{0.0, 1.0}; // is this the correct interval?
};

void FatalError(const std::string& msg) {
  std::cout << "Bad things are happening: " << msg << std::endl;
}

#endif
