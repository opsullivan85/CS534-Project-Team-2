from src.data_generation.boid import (
    BoidField,
    good_boid,
    low_alignment_boid,
    low_cohesion_boid,
    large_seperation_boid,
    BoidLogger,
)
import numpy as np
from src.data_pre_processing import load_most_recent_data, window_size,X_fields_per_boid
import joblib



def run_demo(
        num_good_boids,
        num_faulty_boids,
        num_iterations,
        visualize=True,
        file_name: str = r"C:\Users\Sharvi\Documents\CS534-Project-Team-2\src\data_generation\data\boid_log_test.csv"
):
    """

    @type file_name: object
    """

    def load_model(file_path):
        model = joblib.load(file_path)
        return model
    model_file_path = r"C:\Users\Sharvi\Documents\CS534-Project-Team-2\src\LSTM\lstm_model100.pkl"
    trained_model = load_model(model_file_path)


    if visualize:
        import matplotlib.pyplot as plt

    boid_params = good_boid
    faulty_boid_params = [
        large_seperation_boid,
        low_alignment_boid,
        low_cohesion_boid,
    ]
    # init boid field
    bf = BoidField.make_boid_field(
        num_good_boids, num_faulty_boids, boid_params, faulty_boid_params
    )

    # randomize initial velocities
    bf.boids[:, BoidField.vel_slice] = (
            np.random.rand(num_good_boids + num_faulty_boids, 2) * 500 - 250
    )
    # setup logger
    logger = BoidLogger(file_name, boid_field=bf)
    logger.log_header()

    print(f"Starting simulation: {file_name}")

    def pre_process_data(X_demo, y_demo):

        no = -1  # number of samples
        seq_len = window_size  # sequence length of the time-series
        dim = X_fields_per_boid  # feature dimensions

        # This method wants time series data
        # so we unravel the data
        X_demo = X_demo.reshape((no, seq_len, dim))
        y_demo = np.minimum(y_demo, 1)  # convert all values to 0 or 1

        return X_demo, y_demo

    for i in range(num_iterations):
        bf.simulate(0.01)
        logger.log()
        logger._handle.flush()

        print(f"Iteration {i:>5}/{num_iterations}", end="\r")
        if visualize and i >= window_size:
            # get predictions
            X_demo, y_demo = load_most_recent_data(file_name)
            X_demo, y_demo = pre_process_data(X_demo, y_demo)
            pred = trained_model.predict(X_demo)
            pred=np.rint(pred)
            print(type(pred))

            # plot predictions
            velocity_magnitudes = np.linalg.norm(
                bf.boids[:, BoidField.vel_slice], axis=1
            )
            colors = np.zeros_like(pred, dtype=int)
            colors = colors | (y_demo.astype(int) << 1)
            colors = colors | (pred.astype(int))
            # 0 = purple
            # 1 = blue
            # 2 = green
            # 3 = yellow

            # y, pred
            # 0, 0 = purple TN
            # 1, 0 = green FN
            # 0, 1 = blue FP
            # 1, 1 = yellow TP

            # pred correct <-> pred wrong
            # purple <-> blue # real healthy
            # yellow <-> green # real faulty

            plt.quiver(
                bf.boids[:, BoidField.x_pos_index],
                bf.boids[:, BoidField.y_pos_index],
                bf.boids[:, BoidField.x_vel_index] / velocity_magnitudes,
                bf.boids[:, BoidField.y_vel_index] / velocity_magnitudes,
                # bf.boids[:, BoidField.is_faulty_index],
                # pred * (y + 1),
                colors,
                scale=35,
            )
            # 0 purple
            # 1 yellow
            #
            # Add key
            plt.text(
                5,
                0.95,
                "Healthy",
                transform=plt.gca().transAxes,
                color="Purple",
                fontsize=12,
                ha="right",
                va="top",
            )
            plt.text(
                5,
                0.9,
                "Faulty",
                transform=plt.gca().transAxes,
                color="Yellow",
                fontsize=12,
                ha="right",
                va="top",
            )
            plt.xlim((0, bf.field_size))
            plt.ylim((0, bf.field_size))
            plt.pause(0.01)
            plt.clf()

    print()
    print()


if __name__ == "__main__":
    run_demo(
        num_good_boids=300,
        num_faulty_boids=60,
        num_iterations=2000,
        visualize=True,
    )