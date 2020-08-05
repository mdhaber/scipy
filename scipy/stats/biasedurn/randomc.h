#ifndef RANDOMC_H
#define RANDOMC_H

#include <iostream>

#include <string>
#include "stdint.h"

#include "mt19937.h"

class CRandomMersenne {
public:

  CRandomMersenne(uint32_t seed) {
    RandomInit(seed);
  }

  void RandomInit(uint32_t seed) {
    mt19937_seed(&_rng_state, seed);
  }

  uint32_t IRandom (uint32_t min, uint32_t max) {
    uint32_t word = mt19937_next32(&_rng_state);
    return (uint32_t)(((uint64_t)word * (uint64_t)(max - min)) >> 32) + min;
  }

  double Random() {
    return mt19937_next_double(&_rng_state);
  };

private:
  mt19937_state _rng_state;
};


void FatalError(const std::string& msg) {
  throw std::domain_error("Bad things are happening!");
}

#endif
