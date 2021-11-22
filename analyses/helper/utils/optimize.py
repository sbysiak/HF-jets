import numpy as np


def convert_float64_to_float32(df):
    """returns input DataFrame with columns of type float64 downgraded to float32"""

    dtypes = df.dtypes
    dtypes[dtypes.to_numpy() == np.dtype("float64")] = np.dtype("float32")
    return df.astype(dtypes)
