from collections.abc import Generator
from typing import TypeAlias

import numpy as np
import polars as pl
from jaxtyping import Array, Float


Data: TypeAlias = dict[str, Float[Array, "n_samples ..."]]

DataGenerator: TypeAlias = Generator[Data, None, None]


# Function to add random data to the DataFrame
def add_random_data_to_dataframe(df, num_rows: int = 100):
    for _ in range(num_rows):
        new_xs = np.random.uniform(1, 100, 100)
        new_ys = np.random.uniform(1, 100, 100)
        new_dys = np.random.uniform(1, 100, 100)
        # print("test: ", new_xs)

        xs_pl = pl.Series("xs", new_xs, dtype=pl.Float64)
        ys_pl = pl.Series("ys", new_ys, dtype=pl.Float64)
        dys_pl = pl.Series("dys", new_dys, dtype=pl.Float64)

        df_b = pl.DataFrame([xs_pl, ys_pl, dys_pl])
        df.extend(df_b)
        # print(df)

        # print(new_xs_pl)
        # df["xs"] = df["xs"].append(new_xs_pl)
        # df["ys"].append(pl.Series(new_xs))
        # df["dys"].append(pl.Series(new_xs))

        # row = {
        #     "A": random.randint(1, 100),
        #     "B": random.uniform(0, 1),
        #     "C": random.choice(["foo", "bar", "baz"]),
        # }
        # df = df.push(row)
    return df


def print_df():
    # key = jrandom.PRNGKey(0)

    df = pl.DataFrame(
        [
            pl.Series("xs", [1, 2, 3], dtype=pl.Float64),
            pl.Series("ys", [1, 2, 3], dtype=pl.Float64),
            pl.Series("dys", [1, 2, 3], dtype=pl.Float64),
        ]
    )

    # key, subkey = jrandom.split(key)
    # df = add_random_data_to_dataframe(subkey, df, 5)
    df = add_random_data_to_dataframe(df, 5)
    df_sample = df.sample(5)
    # print(df_sample)
    df_sample.to_numpy()

    # print(df)
