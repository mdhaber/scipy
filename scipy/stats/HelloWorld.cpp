#include <iostream>
#include "HelloWorld.h"

int hello_world(double *x, int n) {
    for (int i = 0; i < n; i++){
        x[i] = float(i);
    }
    return 0;
}
