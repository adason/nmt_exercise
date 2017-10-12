This is my solution to exercises in `Neural Machine Translation and Sequence-to-sequence Models` [tutorial](https://arxiv.org/abs/1703.01619
). The original tutorial suggests using `python` and the `DyNet` deep learning framework. Instead of `DyNet`, I choose to implementation my solution using `pytorch`. This repository will include my `python3` implementation as well as [Julia](https://julialang.org) implementation to practice my julia skill.

Setup Python Environment
------------------------
1. Download the latest [Anaconda](https://anaconda.org/)
2. To create a virtual environment, run `conda env update` which will use the `environment.yml` file.
[or] Create a virtual environment with CUDA support, run `conda env update -f environment_cuda.yml`

Setup Julia Environment
-----------------------
1. Download and install the latest [Julia](https://julialang.org) (0.6 as of this writing).
2. Install `Knet`

```
Pkg.update()
Pkg.add("Knet")
```
