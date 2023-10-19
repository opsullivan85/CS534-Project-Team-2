# This file is the code that is run when you run `python -m src.sota_method2`.
# This should contain code to test and run other code in this module, but not
# anything that is meant to be imported by other modules.

import logging
from src.time_gan import TimeGan
from src.data_pre_processing import load_data, window_size, X_fields_per_boid

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")


## Newtork parameters
parameters = dict()

parameters["module"] = "gru"
parameters["hidden_dim"] = 24
parameters["num_layer"] = 3
parameters["iterations"] = 10000
parameters["batch_size"] = 128

# Get data
train_X, train_y = load_data()

no = -1 # number of samples
seq_len = window_size # sequence length of the time-series
dim = X_fields_per_boid # feature dimensions

# This method wants time series data
# so we unravel the data
train_X = train_X.reshape((no, seq_len, dim))
print(train_X.shape)


# Run TimeGAN
generated_data = TimeGan(train_X, parameters)
print("Finish Synthetic Data Generation")

print(generated_data)
