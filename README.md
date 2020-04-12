# torch-self-intersection

A "minimal" package to compute self-intersections among various faces within a triangle-mesh.


## NOTE: Under development

This package currently provides a "CPU-only" version that can be used from PyTorch. Other enhancements, including a CUDA module, are under development.

Also, we currently do NOT implement a backward method, as this module is primarily intended for benchmarking convenience (i.e., to identify the self-intersections in a mesh generated by a neural network, for example).


## Installation instructions

Assuming you have a conda/virtual environment with PyTorch (>=1.3.0) installed, run the following command from the base directory of this repo (i.e., the directory containing this readme).

```sh
python setup.py build develop
```

## Usage instructions

A minimal example that uses the CPU version of the self-intersection test is shown below
```py
# TODO: Add this
```
