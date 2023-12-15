from src.data_generation.boid import (
    BoidField,
    good_boid,
    low_alignment_boid,
    low_cohesion_boid,
    large_seperation_boid,
    BoidLogger,
)
import numpy as np
from src.data_pre_processing import load_most_recent_data, window_size
from src.GDBT.GDBT import gradientB_decision_tree as generate_model
import pickle


def run_demo(
    num_good_boids,
    num_faulty_boids,
    num_iterations,
    visualize=True,
    file_name: str = "./data/demo.csv",
):
    print("loading model")
    with open('model_path', 'rb') as file:    # run __main__.py to train GDBT, get the .pkl file and replace 'model_path' with the path of the .pkl file
      loaded_model = pickle.load(file)

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

    for i in range(num_iterations):
        bf.simulate(0.01)
        logger.log()
        logger._handle.flush()

        print(f"Iteration {i:>5}/{num_iterations}", end="\r")
        if visualize and i >= window_size:
            # get predictions
            X, y = load_most_recent_data()
            y = np.minimum(y, 1)
            y = y.ravel()
            pred = loaded_model.predict(X)

            # plot predictions
            velocity_magnitudes = np.linalg.norm(
                bf.boids[:, BoidField.vel_slice], axis=1
            )
            colors = np.zeros_like(pred, dtype=int)
            colors = colors | (y.astype(int) << 1)
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
