# This file is the code that is run when you run `python -m src.sota_method2`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
from src.time_gan import TimeGan
from src.data_pre_processing import load_timeseries_data, filter_fault_type_boids_from_data, window_size, X_fields_per_boid

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")

# Get data
NOT_FAULTY = 0
train_X, train_y = load_timeseries_data()
filtered_X, filtered_y = filter_fault_type_boids_from_data(train_X, train_y, NOT_FAULTY)

## Newtork parameters
parameters = dict()

parameters["module"] = "gru"
parameters["hidden_dim"] = 24
parameters["num_layer"] = 3
parameters["batch_size"] = 128
epoch_size = filtered_X.shape[0] // parameters["batch_size"]
parameters["iterations"] = epoch_size * 4
print(f"{parameters = }")

# Run TimeGAN
generated_data = TimeGan(filtered_X, parameters)
print("Finish Synthetic Data Generation")

print(generated_data)
