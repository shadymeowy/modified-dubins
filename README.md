# modified-dubins
A modified version of Dubin's path based on altitude modifications and vector algebra

## Motivation

This project aims to provide a suitable variant and an efficient implementation of Dubin's path algorithm for aircraft navigation. The original algorithm 
solves Dubin's car problem and finds the shortest path between two positions and their directions. While the algorithm can be generalized to 3D dimensions, the
direct generalizations are unsuitable for aircraft navigation since the resulting path is generally far from ideal, requiring unfeasible maneuvers and
unbounded pitch angles. While this problem is addressed in the literature, the current state of these methods is not satisfactory. This
variant is a modification of the original algorithm that appropriately addresses the problem.

## Details of the algorithm

Firstly, a plane, including the start and the
target positions, is chosen. There are infinitely many such planes, each having a different angle from the ground plane. The
plane with zero roll is the one that is used as the reference for the calculation. After, the direction vectors of the start and the target
positions are projected onto the plane with zero roll. Then, the original algorithm is applied to the projected start and target positions and directions.
The generated path is then converted back to the original coordinate system. However, this path does not respect the original directions of the start and target 
positions. Since the zero roll plane have chosen, the z component of points on the path is a linear function of the time variable, assuming the plane is flying
at a constant speed. This linear function needs to be modified to respect the original directions of the start and target positions. While splines can
solve this problem, the resulting path is likely to contain large pitch angles. Instead, a Dubin's path-like method is used to generate the z components
of the points on the path. Unlike the original algorithm, which uses circular arcs, this method utilizes parabolic arcs. This method ensures that maximum z
acceleration is utilized to correct the pitch angles, thus avoiding large pitch angles.

## Current state of the implementation

The implementation is available as a Python package, and it does not depend on any external libraries.
It is currently written in Cython. However, it will be migrated to C soon since it is possible to convert
the code almost line by line. A more Pythonic wrapper is also planned

## Performance

While I have not published any benchmarks yet, I can say that both the algorithm and implementation are quite fast. For reference, constructing the Python dictionary storing
the desired constraints takes longer than the algorithm itself. Proper benchmarking will be published once the code is converted to C.

## Installation

Install via pip:

```
pip install git+https://github.com/shadymeowy/modified-dubins
```

or clone the repository and run the setup script:

```
git clone https://github.com/shadymeowy/modified-dubins
cd modified-dubins
python setup.py install
```