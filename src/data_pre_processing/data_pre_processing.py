import pandas as pd
import numpy as np
from src.data_generation import BoidField
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
import src


fields_per_boid = BoidField.num_parameters
fields_per_boid -= 6  # metaparameters are not logged
fields_per_boid += 6  # 2 values for 3 neighbors are logged
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
    train_X, train_y = load_data(data_path)

    no = -1  # number of samples
    seq_len = window_size  # sequence length of the time-series
    dim = X_fields_per_boid  # feature dimensions

    # This method wants time series data
    # so we unravel the data
    train_X = train_X.reshape((no, seq_len, dim))

    return train_X, train_y

def filter_fault_type_boids_from_data(X: np.ndarray, y: np.ndarray, fault_type: int) -> tuple[np.ndarray, np.ndarray]:
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

    # Get only good boid indices and filter labels
    indices = np.where(y == fault_type)[0]
    filtered_y = y[indices]
    filterd_X = X[indices,:,:]

    return filterd_X, filtered_y
