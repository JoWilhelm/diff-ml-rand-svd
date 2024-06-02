# Sobolev Pruning

This example shows how to use Sobolev Training and Sobolev Pruning for finding and pruning surrogate models considering sensitivity information.

We first train a sufficiently large neural network from training data or by sampling a reference model. Then, the model is pruned using Interval Adjoint Significance Analysis.
It ensures that neurons are removed that globally have the lowest significance. Finally, we fine-tune the pruned surrogate model using Sobolev Training to recover the (previously lost) sensitivity information. It results in a surrogate model that is dense, small, and can be differentiated using AD giving sensible sensitivity results.

For training we use [Equinox](https://github.com/patrick-kidger/equinox) and the surrounding [JAX](https://github.com/google/jax) ecosystem.

For pruning using Interval Adjoint Significance Analysis we currently use a C++ library (supporting AD with Interval types) which can be found [here](https://gitlab.stce.rwth-aachen.de/stce/interval_network). It accepts any valid ONNX network. Therefore, we first convert the Equinox model to ONNX.

For more information, please refer to the [paper](https://doi.org/10.1145/3659914.3659915) and its accompanying [repo](https://github.com/neilkichler/sobolev-pruning).
