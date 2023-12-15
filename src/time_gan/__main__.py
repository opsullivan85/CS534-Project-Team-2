# This file is the code that is run when you run `python -m src.sota_method2`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
import numpy as np
from src.time_gan import TimeGan
from sklearn.model_selection import train_test_split
from src.data_pre_processing import load_timeseries_data, separate_good_and_bad_boids_from_data, window_size, X_fields_per_boid
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

# Get data
NUM_FAULT_TYPES = 4
X_train, y_train = load_timeseries_data()
X_test, y_test = load_timeseries_data("./data/boid_log_test.csv")

## Uncomment if there is no separated test set for testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

good_X_train, good_y_train, bad_X_train, bad_y_train = separate_good_and_bad_boids_from_data(X_train, y_train)

## Uncomment if you want to add the unused bad boids from the train split into the test set.
# good_X_test, good_y_test, bad_X_test, bad_y_test = separate_good_and_bad_boids_from_data(X_test, y_test)
# Adding the bad from the train split since it's not used in training.
# X_test = np.concatenate((bad_X_train, bad_X_test, good_X_test), axis=0)
# y_test = np.concatenate((bad_y_train, bad_y_test, good_y_test), axis=0)

# Mixes up test set so that good and bad boids aren't right next to each other. Helps since testing has to be done in batches for large datasets because of hardware constraints.
perm = np.random.permutation(X_test.shape[0])
X_test = X_test[perm, :, :]
y_test = y_test[perm, :]
print("Shape of test x: ", X_test.shape)
print("Shape of test y: ", y_test.shape)

## Newtork parameters
parameters = dict()

parameters["module"] = "gru"
parameters["hidden_dim"] = 24
parameters["num_layer"] = 3
parameters["batch_size"] = 128
epoch_size = X_train.shape[0] // parameters["batch_size"]
parameters["iterations"] = epoch_size * 4

# Run TimeGAN
y_pred, totals = TimeGan(X_train, parameters, X_test, y_test, NUM_FAULT_TYPES)
print("Finished Evaluating Predictions from time_GAN")

# Convert all errors into one type
fault_indices = np.where(y_test != 0)[0]
y_test[fault_indices] = 1

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(confusion)
print("Classification Report:")
print(classification_report(y_test, y_pred))

