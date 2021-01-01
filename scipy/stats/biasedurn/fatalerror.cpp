#include <stdexcept>

void FatalError(const char * msg) {
  // This function outputs an error message and aborts the program.
  throw std::runtime_error(msg);
}
