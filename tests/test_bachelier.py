from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import tensorflow_datasets as tfds
from jaxtyping import Array

import diff_ml as dml
from diff_ml.model import Bachelier, generate_correlation_matrix


class TestGenerateCorrelatedSamples:
    def test_generate_correlated_samples(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        samples = generate_correlation_matrix(key, n_samples)
        assert samples.shape == (n_samples, n_samples)

    def test_generator(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        n_dims = 7
        n_batch = 128

        ds_identity = tfds.core.DatasetIdentity( # pyright: ignore
            name="bachelier",
            version="1.0.0",
            data_dir="datasets/bachelier",
            module_name="diff_ml_bachelier",
        )

        ds_info = tfds.core.DatasetInfo( # pyright: ignore
            builder=ds_identity,
            description="Bachelier Dataset with Differential Data.",
            features=tfds.features.FeaturesDict(
                {
                    "spot": tfds.features.Tensor(
                        shape=(n_dims,),
                        dtype=np.float32,
                    ),
                    "payoff": tfds.features.Scalar(
                        dtype=np.float32,
                    ),
                }
            ),
            supervised_keys=("spot", "payoff"),
        )

        writer = tfds.core.SequentialWriter(  # pyright: ignore
            ds_info=ds_info,
            max_examples_per_shard=n_samples,
            overwrite=True
        )

        writer.initialize_splits(["train"])

        key, subkey = jrandom.split(key)
        weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)
        bachelier = Bachelier(key, weights, n_dims=n_dims)
        data: dml.DifferentialData = bachelier.generator(n_samples)
        examples = [{k: v[i] for k, v in data.items()} for i in range(n_samples)]

        writer.add_examples({"train": examples})
        writer.close_all()
        assert jnp.asarray(data["spot"]).shape == (n_samples, n_dims)

        ds_builder = tfds.builder_from_directory("datasets/bachelier")
        ds = ds_builder.as_dataset(split="train", batch_size=n_batch, as_supervised=True).repeat().shuffle(1024)

        np_arrays = tfds.as_numpy(ds)
        for xs, ys in np_arrays:
            assert xs.shape == (n_batch, n_dims)
            assert ys.shape == (n_batch,)
            break

        ds_all = ds_builder.as_dataset(split="train", as_supervised=True)
        np_all = tfds.as_numpy(ds_all)

        spots = np.asarray(data["spot"])
        payoffs = np.asarray(data["payoff"])
        for i, (xs, ys) in enumerate(np_all):
            assert np.allclose(xs, spots[i])
            assert np.allclose(ys, payoffs[i])

    def test_generator_testdata(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        n_dims = 7

        ds_identity = tfds.core.DatasetIdentity( # pyright: ignore
            name="bachelier",
            version="1.0.0",
            data_dir="datasets/bachelier",
            module_name="diff_ml_bachelier",
        )

        ds_info = tfds.core.DatasetInfo( # pyright: ignore
            builder=ds_identity,
            description="Bachelier Dataset with Differential Data.",
            features=tfds.features.FeaturesDict(
                {
                    "spot": tfds.features.Tensor(
                        shape=(n_dims,),
                        dtype=np.float32,
                    ),
                    "payoff": tfds.features.Scalar(
                        dtype=np.float32,
                    ),
                }
            ),
            supervised_keys=("spot", "payoff"),
        )

        writer = tfds.core.SequentialWriter(  # pyright: ignore
            ds_info=ds_info,
            max_examples_per_shard=n_samples,
            overwrite=False
        )

        writer.initialize_splits(["test"])

        key, subkey = jrandom.split(key)
        weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)
        bachelier = Bachelier(key, weights, n_dims=n_dims)
        data = bachelier.test_generator(n_samples)
        examples = [{k: v[i] for k, v in data.items()} for i in range(n_samples)]

        writer.add_examples({"test": examples})
        writer.close_all()
        assert jnp.asarray(data["spot"]).shape == (n_samples, n_dims)

        ds_builder = tfds.builder_from_directory("datasets/bachelier")
        ds_all = ds_builder.as_dataset(split="test", as_supervised=True)
        np_all = tfds.as_numpy(ds_all)

        spots = np.asarray(data["spot"])
        payoffs = np.asarray(data["payoff"])
        for i, (xs, ys) in enumerate(np_all):
            assert np.allclose(xs, spots[i])
            assert np.allclose(ys, payoffs[i])



if __name__ == "__main__":
    TestGenerateCorrelatedSamples().test_generator()
