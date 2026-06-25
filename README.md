# Efficient Higher-Order Sobolev Training

This repository contains the code for my M.Sc. thesis at RWTH Aachen University. For backgorund information consider reading this [summary article](https://johannes-wilhelm.com/thesis/efficient-neural-greeks-directional-sobolev-training-via-randomized-hessian-sketching/).

The project explores higher-order differential machine learning for surrogate modelling, with a focus on learning curvature and higher-order derivative information without requiring full second- and third-order supervision. The implementation uses randomized sketching methods and JAX for automatic differentiation, and applies them to option-pricing and analytic benchmark problems.

The project builds on [diff-ml](https://github.com/neilkichler/diff-ml) developed by Neil Kicher.

## Content
The package implements functionalities for Higher-Order Sobobolev Training / Differential Machine Learning.

## Installation
Clone the repo and execute the following inside the root folder.

```bash
python -m pip install -e .
```

> Requires Python 3.9+, [JAX](https://github.com/google/jax) 0.4.16+ and [Equinox](https://github.com/patrick-kidger/equinox) 0.10.5+.



