import typing
from functools import partial
from itertools import islice

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import tensorflow_datasets as tfds

import diff_ml as dml
from datasets import Dataset, DatasetInfo, IterableDataset, load_from_disk
from diff_ml.model import Bachelier, generate_correlation_matrix


def bachelier_ds_info() -> DatasetInfo:
    citation = r"""@misc{dataset:bachelier,
        author = {Neil Kichler},
        title = {{Dataset of a Bachelier model for a 7-dimensional basket option},
        howpublished= {\url{https://github.com/neilkichler/diff-ml/tree/main/datasets/bachelier}}
    """

    ds_info = DatasetInfo(
        description="Example data of a Bachelier model for a 7-dimensional basket option.",
        homepage="https://github.com/neilkichler/diff-ml/tree/main/datasets/bachelier",
        license="MIT License",
        citation=citation,
        version="1.1.0",
    )

    return ds_info

class TestGenerateCorrelatedSamples:
    def test_generate_correlated_samples(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        samples = generate_correlation_matrix(key, n_samples)
        assert samples.shape == (n_samples, n_samples)

    def test_generator_to_ds_iter(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        n_dims = 7
        n_batch = 8

        key, subkey = jrandom.split(key)
        weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)
        bachelier = Bachelier(key, n_dims, weights)

        # using Dataset.from_generator
        data_gen = partial(bachelier.generator, n_precompute=n_samples)
        # data_gen_slice = islice(data_gen(), n_iter)

        # def gen():
        #     yield next(data_gen_slice)
            # yield {"key": "value", "other_key": "other_value"}

        ds_gen = Dataset.from_generator(generator=data_gen)
        ds_gen = typing.cast(Dataset, ds_gen)
        device = str(jax.devices()[0])
        ds_gen_jax = ds_gen.with_format("jax", device=device)
        ds_gen_iter = ds_gen_jax.iter(batch_size=n_batch)
        for _ in range(2):
            d = next(ds_gen_iter)
            xs_train = d["spot"]
            assert(xs_train.shape == (n_batch, n_dims))


    def test_generator_to_ds(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        n_dims = 7
        n_batch = 128

        key, subkey = jrandom.split(key)
        weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)
        bachelier = Bachelier(key, n_dims, weights)
        data = bachelier.sample(n_samples)

        ds_info = bachelier_ds_info()

        # using Dataset.from_dict
        ds = Dataset.from_dict(dict(data), info=ds_info)
        device = str(jax.devices()[0])
        ds = ds.with_format("jax", device=device)
        ds_iter = ds.iter(batch_size=n_batch)
        ds.save_to_disk("datasets/bachelier/arrow", num_shards=8)

        # load data that we just saved and compare
        ds_loaded = load_from_disk("datasets/bachelier/arrow")
        ds_loaded = typing.cast(Dataset, ds_loaded)
        ds_loaded_iter = ds_loaded.iter(batch_size=n_batch)
        for _ in range(2):
            batch = next(ds_iter)
            batch = typing.cast(dict, batch)
            batch_loaded = dict(next(ds_loaded_iter))
            batch_loaded = typing.cast(dict, batch_loaded)
            x, y = batch["spot"], batch["payoff"]
            x_loaded, y_loaded = batch_loaded["spot"], batch_loaded["payoff"]
            assert jnp.allclose(x, x_loaded)
            assert jnp.allclose(y, y_loaded)


    def tf_bachelier_info(self):
        ds_identity = tfds.core.DatasetIdentity(  # pyright: ignore
            name="bachelier",
            version="1.0.0",
            data_dir="datasets/bachelier",
            module_name="diff_ml_bachelier",
        )

        return ds_identity

    def test_generator(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        n_dims = 7
        n_batch = 128

        ds_identity = self.tf_bachelier_info()

        ds_info = tfds.core.DatasetInfo(  # pyright: ignore
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
                    "differentials": tfds.features.Tensor(
                        shape=(n_dims,),
                        dtype=np.float32
                    )
                }
            ),
            supervised_keys=("spot", ("payoff", "differentials")),
        )

        writer = tfds.core.SequentialWriter(  # pyright: ignore
            ds_info=ds_info, max_examples_per_shard=n_samples, overwrite=True
        )

        writer.initialize_splits(["train"])

        key, subkey = jrandom.split(key)
        weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)
        bachelier = Bachelier(key, n_dims, weights)
        data = bachelier.sample(n_samples=n_samples)
        examples = [{k: np.asarray(v)[i] for k, v in data.items()} for i in range(n_samples)]

        writer.add_examples({"train": examples})
        writer.close_all()
        assert jnp.asarray(data["spot"]).shape == (n_samples, n_dims)

        ds_builder = tfds.builder_from_directory("datasets/bachelier")
        ds = ds_builder.as_dataset(split="train", batch_size=n_batch, as_supervised=True)

        np_arrays = tfds.as_numpy(ds)
        np_arrays = typing.cast(typing.Iterable, np_arrays)
        for xs, (ys, zs) in np_arrays:
            assert xs.shape == (n_batch, n_dims)
            assert ys.shape == (n_batch,)
            assert zs.shape == (n_batch, n_dims)
            break

        ds_all = ds_builder.as_dataset(split="train", as_supervised=True)
        np_all = tfds.as_numpy(ds_all)
        np_all = typing.cast(typing.Iterable, np_all)

        spots = np.asarray(data["spot"])
        payoffs = np.asarray(data["payoff"])
        differentials = np.asarray(data["differentials"])
        for i, (xs, (ys, zs)) in enumerate(np_all):
            assert np.allclose(xs, spots[i])
            assert np.allclose(ys, payoffs[i])
            assert np.allclose(zs, differentials[i])

    def test_generator_testdata(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        n_dims = 7

        ds_identity = self.tf_bachelier_info()

        ds_info = tfds.core.DatasetInfo(  # pyright: ignore
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
                    "differentials": tfds.features.Scalar(
                        dtype=np.float32
                    )
                }
            ),
            supervised_keys=("spot", ("payoff", "differentials")),
        )

        writer = tfds.core.SequentialWriter(  # pyright: ignore
            ds_info=ds_info, max_examples_per_shard=n_samples, overwrite=True
        )

        writer.initialize_splits(["test"])

        key, subkey = jrandom.split(key)
        weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)
        bachelier = Bachelier(key, n_dims, weights)
        data = bachelier.test_generator(n_samples)
        examples = [{k: np.asarray(v)[i] for k, v in data.items()} for i in range(n_samples)]

        writer.add_examples({"test": examples})
        writer.close_all()
        assert jnp.asarray(data["spot"]).shape == (n_samples, n_dims)

        ds_builder = tfds.builder_from_directory("datasets/bachelier")

        ds_all = ds_builder.as_dataset(split="test", as_supervised=True)
        np_all = tfds.as_numpy(ds_all)
        np_all = typing.cast(typing.Iterable, np_all)

        spots = np.asarray(data["spot"])
        payoffs = np.asarray(data["payoff"])
        differentials = np.asarray(data["differentials"])
        for i, (xs, (ys, zs)) in enumerate(np_all):
            assert np.allclose(xs, spots[i])
            assert np.allclose(ys, payoffs[i])
            assert np.allclose(zs, differentials[i])


if __name__ == "__main__":
    TestGenerateCorrelatedSamples().test_generator_to_ds()
