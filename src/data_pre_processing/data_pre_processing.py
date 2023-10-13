import pandas as pd
import numpy as np
from src.data_generation import BoidField
from numpy.lib.stride_tricks import sliding_window_view


fields_per_boid = BoidField.num_parameters
fields_per_boid -= 6  # metaparameters are not logged
fields_per_boid += 6  # 2 values for 3 neighbors are logged
is_faulty_index = BoidField.is_faulty_index - 6  # because metaparameters are not logged


def seperate_boids(file: str) -> list[np.ndarray]:
    """Reads in datafile, splits into a list of numpy arrays, each array
    has the data for a single boid.

    Args:
        file (str): Data path

    Returns:
        list[np.ndarray]: List of numpy arrays containing the data for each boid
    """

    df = pd.read_csv(file)
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
) -> tuple[tuple[np.ndarray], tuple[float]]:
    """Gets rolling data from a file

    Args:
        file (str): Path to data file
        window_size (int): data window size. In terms of boid iterations.
        step_size (int, optional): Ammount to step each window by. In terms of boid iterations. Defaults to 1.

    Returns:
        tuple[tuple[np.ndarray], tuple[float]]: X, y data.
            X is a tuple of numpy arrays, each a rolling view of the data for a boid.
    """
    window_size *= fields_per_boid
    step_size *= fields_per_boid
    boids = seperate_boids(file)
    X = []
    y = []
    for boid in boids:
        X.append(get_rolling_window(boid.ravel(), window_size, step_size))
        y.append(
            np.full(shape=(X[-1].shape[0],), fill_value=boid[0, is_faulty_index])[
                :, np.newaxis
            ]
        )  # is_faulty is the same for all timesteps, so just take the first one
    X = np.vstack(X)
    y = np.vstack(y)
    return X, y
