
#ifndef RANDOMC_H
#define RANDOMC_H

#include <iostream>
#include <stdint.h>


/**
 * START
 * Taken from numpy/numpy/random/src/mt19937/mt19937.h
 *
 * Some functions in numpy header are not extern, so we
 * have to reproduce them here.
 */
#define RK_STATE_LEN 624
extern "C" {
  typedef struct s_mt19937_state {
    uint32_t key[RK_STATE_LEN];
    int pos;
  } mt19937_state;
  void mt19937_seed(mt19937_state *state, uint32_t seed);
  void mt19937_gen(mt19937_state *state);
}
/* Slightly optimized reference implementation of the Mersenne Twister */
static inline uint32_t mt19937_next(mt19937_state *state) {
  uint32_t y;

  if (state->pos == RK_STATE_LEN) {
    // Move to function to help inlining
    mt19937_gen(state);
  }
  y = state->key[state->pos++];

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}
static inline double mt19937_next_double(mt19937_state *state) {
  int32_t a = mt19937_next(state) >> 5, b = mt19937_next(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}
/**
 * END
 */


// implemented in fnchyppr.cpp
void FatalError(const char* msg);

class CRandomMersenne {
public:

  CRandomMersenne(int seed) {
    RandomInit(seed);
  }

  void RandomInit(int seed) {
    mt19937_seed(&_rng_state, seed);
  }

  int IRandom (int min, int max) {
    // Based on Lemire's method:
    uint32_t word = mt19937_next(&_rng_state);
    return (uint32_t)(((uint64_t)word * (uint64_t)(max - min)) >> 32) + min;
  }

  double Random() {
    return mt19937_next_double(&_rng_state);
  }
private:
  mt19937_state _rng_state;
};

#endif
