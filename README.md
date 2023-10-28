# Second-Order Differential Machine Learning

## Installation
Clone the repo and execute the following inside the root folder.

```bash
python -m pip install -e .
```

> Requires Python 3.9+, [JAX](https://github.com/google/jax) 0.4.13+ and [Equinox](https://github.com/patrick-kidger/equinox) 0.10.5+.

## Development
We use [Hatch](https://hatch.pypa.io/) as the project manager. The usual commands apply.

#### Show all available scripts
```bash
hatch env show
```
#### Run case study
```bash
hatch run python examples/bachelier/bachelier.py
```
#### Run Tests
```bash
hatch run test
```
#### Build project wheel
```bash
hatch build
```
#### Lint project
```bash
hatch run lint:fmt
```


