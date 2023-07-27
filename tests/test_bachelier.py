from typing import Any

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import tensorflow_datasets as tfds
from jaxtyping import Array

import diff_ml as dml
from diff_ml.model import Bachelier, BachelierParams, generate_correlation_matrix


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

        ds_identity = tfds.core.DatasetIdentity(
            name="bachelier",
            version="1.0.0",
            data_dir="datasets/bachelier",
            module_name="diff_ml_bachelier",
        )

        ds_info = tfds.core.DatasetInfo(
            builder=ds_identity,
            description="Bachelier Dataset with Differential Data.",
            features = tfds.features.FeaturesDict({
                    "xs": tfds.features.Tensor(
                        shape=(n_samples, n_dims),
                        dtype=np.float32,
                    ),
                    "ys": tfds.features.Tensor(
                        shape=(n_samples,),
                        dtype=np.float32,
                    ),
            })
        )

        writer = tfds.core.SequentialWriter(
            ds_info=ds_info,
            max_examples_per_shard=n_samples,
            overwrite=True
        )
        writer.initialize_splits(["train", "test"])
        bachelier = Bachelier(key, n_dims=n_dims)
        data: dml.DifferentialData = bachelier.generator(n_samples)
        example = [{"xs": np.asarray(data.xs), "ys": np.asarray(data.ys)}]
        writer.add_examples({"train": example})
        writer.close_all()
        assert jnp.asarray(data.xs).shape == (1024, 7)


if __name__ == "__main__":
    TestGenerateCorrelatedSamples().test_generator()
