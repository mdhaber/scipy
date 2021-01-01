#ifndef _STOCR_H_
#define _STOCR_H_

#include <cstdint>
#include <stdexcept>
#include "numpy/random/bitgen.h"

class StocRBase {

  // bitgen_state is owned by the collater and its
  // lifetime will be assumed to be the same or
  // greater than this StocRBase object
  bitgen_t* bitgen_state;

public:
  StocRBase() : bitgen_state(NULL) {}
  StocRBase(std::int32_t seed) : bitgen_state(NULL) {}

  void SetBitGen(bitgen_t* that_bitgen_state) {
    bitgen_state = that_bitgen_state;
  }
  
  // Call this before first random number
  void InitRan() {
    // we can only generate random numbers if we have bitgen_t object
    if (bitgen_state == NULL) {
      throw std::runtime_error("SetBitGen(bitgen_state) has not been called!");
    }
  }

  // Call this after last random number
  void EndRan() {}

  // output random float number in the interval 0 <= x < 1
  double Random() {
    return bitgen_state->next_double(bitgen_state->state);
  }

  // normal distribution
  double Normal(double m, double s) {
    printf("I don't have a normal generator!");
    return 0;
  }

  std::int32_t Hypergeometric(std::int32_t n, std::int32_t m, std::int32_t N);
  
protected:
  std::int32_t HypInversionMod (std::int32_t n, std::int32_t M, std::int32_t N);
  std::int32_t HypRatioOfUnifoms (std::int32_t n, std::int32_t M, std::int32_t N);
  static double fc_lnpk(std::int32_t k, std::int32_t N_Mn, std::int32_t M, std::int32_t n);
};

#endif // _STOCR_H_
