# INGOT - The INteractive GPU ODE Toolbox

## Requirements

* CMake ≥ 3.14
* CUDA Toolkit ≥ 10.1 (due to Thrust ≥ 1.9.4) and C++14 support
* Eigen 3
* Googletest (if building unit tests)

If unsure, I recommend using [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

Once micromamba is installed, you can optionally create an environment for ingot dependencies:
```bash
micromamba create --name ingot
micromamba activate ingot
```

Then install ingot's dependencies:
```bash
micromamba install cmake doctest eigen python pybind11
```

## Building

```bash
mkdir build
cd build
cmake ..
make
```
