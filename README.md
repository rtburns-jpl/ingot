# INGOT - The INteractive GPU ODE Toolbox

## Requirements

* CMake ≥ 3.14
* CUDA Toolkit ≥ 10.1 (due to Thrust ≥ 1.9.4) and C++14 support
* Eigen 3
* Googletest (if building unit tests)

If using Anaconda, you can install these from the default channel.
```bash
conda install -c defaults cmake eigen gtest
```

## Building

```bash
mkdir build
cd build
cmake ..
make
```
