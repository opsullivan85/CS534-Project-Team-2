# This file is the code that is run when you run `python -m src.sota_method2`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
import numpy as np
from src.time_gan import TimeGan
from sklearn.model_selection import train_test_split
from src.data_pre_processing import load_timeseries_data, separate_good_and_bad_boids_from_data, window_size, X_fields_per_boid

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

# Get data
NUM_FAULT_TYPES = 4
X, y = load_timeseries_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
good_X_train, good_y_train, bad_X_train, bad_y_train = separate_good_and_bad_boids_from_data(X_train, y_train)
good_X_test, good_y_test, bad_X_test, bad_y_test = separate_good_and_bad_boids_from_data(X_test, y_test)

# Adding the bad from the train split since it's not used in training.
X_test = np.concatenate((bad_X_train, bad_X_test, good_X_test), axis=0)
y_test = np.concatenate((bad_y_train, bad_y_test, good_y_test), axis=0)
print("Shape of test x: ", X_test.shape)
print("Shape of test y: ", y_test.shape)
print(y_test)

## Newtork parameters
parameters = dict()

parameters["module"] = "gru"
parameters["hidden_dim"] = 24
parameters["num_layer"] = 3
parameters["batch_size"] = 128
epoch_size = X_train.shape[0] // parameters["batch_size"]
parameters["iterations"] = epoch_size * 4
print(f"{parameters = }")

# Run TimeGAN
generated_data, discriminator_output, totals = TimeGan(X_train, parameters, X_test, y_test, NUM_FAULT_TYPES)
print("Finish Synthetic Data Generation")
