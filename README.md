This is my solution to the exercise for the `Neural Machine Translation and Sequence-to-sequence Models` [tutorial](https://arxiv.org/abs/1703.01619
). The original tutorial suggests using `python` and the `DyNet` deep learning framework. This repository will include `python3` implementation as well as [Julia](https://julialang.org) implementation to practice my Julia skill.

Setup Python Environment
------------------------
1. Download the latest [Anaconda](https://anaconda.org/)
2. Create a virtual environment by `conda env update` which will create a python virtual environment using the `environment.yml` file.

Setup Julia Environment
-----------------------
1. Download and install the latest [Julia](https://julialang.org) (0.6 as of this writing).
2. Install `Knet`

```
Pkg.update()
Pkg.add("Knet")
```
