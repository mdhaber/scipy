import numpy as np
import sys
import importlib

modules = ['cluster', 'constants', 'datasets', 'differentiate', 'fft', 'fftpack',
           'integrate', 'interpolate', 'io', 'linalg', 'ndimage', 'odr', 'optimize',
           'signal', 'sparse', 'spatial', 'special', 'stats', '_lib']

# This is very weird, yes. I'm using a `npz` file to save index of the
# module we need to import next as well as the partial state of the import
# graph. `import_graph_caller` calls this repeatedly until we run out of
# indices. Obviously there are more elegant ways to do this.
try:
    res = np.load('callgraph.npz')
    A = res['A']
    k = res['k']
except:
    n = len(modules)
    A = np.zeros((n, n), dtype=int)
    k = 0

modules = ['scipy.' + module for module in modules]
module_numbers = {mod: k for k, mod in enumerate(modules)}

# Import the module and associated tests
importlib.import_module(modules[k])
importlib.import_module(modules[k] + ".tests")

# See which other modules ended up in `sys.modules`
scipy_subpackages = []
scipy_modules = [sys_module for sys_module in sys.modules
                 if 'scipy' in sys_module]
for module in modules:
    for scipy_module in scipy_modules:
        if module in scipy_modules:
            scipy_subpackages.append(module)
            break

# A simpler way of determining which modules are in `sys.modules`
# (sanity check)
scipy_subpackages2 = [sys_module for sys_module in sys.modules
                      if 'scipy' in sys_module and sys_module.count('.') == 1]
assert not (set(scipy_subpackages) - set(scipy_subpackages2))

# Encode which modules are in `sys.modules` in an adjacency matrix.
# Each row is the module we imported; each column is 1 if the corresponding
# module ended up in `sys.modules`.
for package in scipy_subpackages:
    A[k, module_numbers[package]] = 1
k += 1
np.savez('callgraph.npz', A=A, k=k)