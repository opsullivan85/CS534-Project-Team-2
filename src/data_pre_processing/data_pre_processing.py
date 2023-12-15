import pandas as pd
import numpy as np
from src.data_generation import BoidField
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
import src


fields_per_boid = BoidField.num_parameters
fields_per_boid -= 6  # metaparameters are not logged
num_neighbors = 3
values_per_neighbor = 1
fields_per_boid += (
    num_neighbors * values_per_neighbor
)  # add the number of neighbors to the fields per boid
is_faulty_index = BoidField.is_faulty_index - 6  # because metaparameters are not logged

y_fields_per_boid = 1
X_fields_per_boid = fields_per_boid - y_fields_per_boid

# variables to define behavior of load_data
window_size = 50
step_size = 1
x_width = window_size * X_fields_per_boid


def seperate_boids(file: str) -> list[np.ndarray]:
    """Reads in datafile, splits into a list of numpy arrays, each array
    has the data for a single boid.

    Args:
        file (str): Data path

    Returns:
        list[np.ndarray]: List of numpy arrays containing the data for each boid
    """

    df = pd.read_csv(file).astype("float32")
    data = df.to_numpy()
    boids = np.split(data, data.shape[1] / fields_per_boid, axis=1)
    return boids


def get_rolling_window(
    data: np.ndarray, window_size: int, step_size: int = 1
) -> np.ndarray:
    """Converts a 1D array into a 2D array of rolling windows

    Args:
        data (np.ndarray): Data to convert
        window_size (int): Window size, effectively the number of columns in the output
        step_size (int, optional): Size to shift over for each row. Defaults to 1.

    Returns:
        np.ndarray: rolling view of data
    """
    return sliding_window_view(data, window_shape=window_size)[::step_size]


def get_rolling_data(
    file: str, window_size: int, step_size: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Gets rolling data from a file

    Args:
        file (str): Path to data file
        window_size (int): data window size. In terms of boid iterations.
        step_size (int, optional): Ammount to step each window by. In terms of boid iterations. Defaults to 1.

    Returns:
        tuple[tuple[np.ndarray], tuple[float]]: X, y data.
            X is an np.ndarray where each row is a window of flattened data for one boid.
            y is an np.ndarray where each row is the is_faulty value for the boid
    """
    # these are scaled to be in terms of boid iterations
    window_size *= X_fields_per_boid
    step_size *= X_fields_per_boid
    boids = seperate_boids(file)
    X = []
    y = []
    data_mask = np.ones((fields_per_boid,), dtype=bool)
    data_mask[is_faulty_index] = 0  # don't include is_faulty in X
    for boid in boids:
        masked_boid_data = boid[:, data_mask]
        X.append(get_rolling_window(masked_boid_data.ravel(), window_size, step_size))
        y.append(
            np.full(shape=(X[-1].shape[0],), fill_value=boid[0, is_faulty_index])[
                :, np.newaxis
            ]
        )  # is_faulty is the same for all timesteps, so just take the first one
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y


def get_most_recent_data(file: str, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Gets rolling data from a file. only returns the most recent data for each boid

    Args:
        file (str): Path to data file
        window_size (int): data window size. In terms of boid iterations.
        step_size (int, optional): Ammount to step each window by. In terms of boid iterations. Defaults to 1.

    Returns:
        tuple[tuple[np.ndarray], tuple[float]]: X, y data.
            X is an np.ndarray where each row is a window of flattened data for one boid.
            y is an np.ndarray where each row is the is_faulty value for the boid
    """
    boids = seperate_boids(file)
    X = []
    y = []
    data_mask = np.ones((fields_per_boid,), dtype=bool)
    data_mask[is_faulty_index] = 0  # don't include is_faulty in X
    for boid in boids:
        masked_boid_data = boid[-window_size:, data_mask]
        X.append(masked_boid_data.ravel())
        y.append(
            boid[0, is_faulty_index]
        )  # is_faulty is the same for all timesteps, so just take the first one
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y


def load_most_recent_data(data_path: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Loads the most recent data

    Args:
        data_path (str, optional): Path to load data from. Defaults to `PROJECT_DIR/data/boid_log.csv`.

    Returns:
        tuple[np.ndarray, np.ndarray]: X, y
    """

    data_path = data_path or Path(src.__file__).parent.parent / "data" / "boid_log.csv"

    return get_most_recent_data(data_path, window_size=window_size)


def load_data(data_path: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Loads the data

    Args:
        data_path (str, optional): Path to load data from. Defaults to `PROJECT_DIR/data/boid_log.csv`.

    Returns:
        tuple[np.ndarray, np.ndarray]: X, y
    """

    data_path = data_path or Path(src.__file__).parent.parent / "data" / "boid_log.csv"

    return get_rolling_data(data_path, window_size=window_size, step_size=1)


def load_timeseries_data(data_path: str = None) -> tuple[np.ndarray, np.ndarray]:
    """Loads the data as a timeseries

    dim1 = number of samples
    dim2 = sequence length of the time-series
    dim3 = feature dimensions

    Args:
        data_path (str, optional): Path to load data from. Defaults to `PROJECT_DIR/data/boid_log.csv`.

    Returns:
        tuple[np.ndarray, np.ndarray]: X, y
    """
    # Get data
    X, y = load_data(data_path)

    no = -1  # number of samples
    seq_len = window_size  # sequence length of the time-series
    dim = X_fields_per_boid  # feature dimensions

    # This method wants time series data
    # so we unravel the data
    X = X.reshape((no, seq_len, dim))

    return X, y
  

def separate_good_and_bad_boids_from_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filters only data from a particular fault type from the data

    dim1 = number of samples
    dim2 = sequence length of the time-series
    dim3 = feature dimensions

    Args:
        X: Input Data
        y: Output Labels
        fault_type: 0 - Not Faulty; 1-3 - Faulty

    Returns:
        tuple[np.ndarray, np.ndarray]: Filtered input data (X) and filtered output labels (y)
    """
    NO_FAULT = 0

    # Get only good boid indices and filter labels
    good_indices = np.where(y == NO_FAULT)[0]
    bad_indices = np.where(y != NO_FAULT)[0]
    good_X = X[good_indices,:,:]
    good_y = y[good_indices]
    bad_X = X[bad_indices,:,:]
    bad_y = y[bad_indices]
    
    return good_X, good_y, bad_X, bad_y
  

def down_sample_data(X, y):
    """Down samples the number of healthy boids to match the number of faulty boids
    Specifically finds the number of non-zero values in y, and then randomly selects
    that many healthy boids to keep. Assumes that the healthy boids are at the start

    Args:
        X: X Data
        y: y Data

    Returns:
        X: X Data
        y: y Data
    """
    num_faulty = np.count_nonzero(y)
    num_healthy = y.shape[0] - num_faulty
    random_healthy_indices = np.random.choice(
        num_healthy - num_faulty, num_faulty, replace=False
    )
    X = np.concatenate((X[random_healthy_indices], X[-num_faulty:]))
    y = np.concatenate((y[random_healthy_indices], y[-num_faulty:]))
    return X, y
