from dataclasses import dataclass
import dataclasses
import numpy as np
from pathlib import Path
import csv
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Loading {__name__}")


@dataclass
class BoidParameters:
    """Stores the parameters of a boid."""

    separation: float
    alignment: float
    cohesion: float
    seperation_factor: float
    alignment_factor: float
    cohesion_factor: float
    name: str = "BoidParameters"


good_boid = BoidParameters(
    separation=25,
    alignment=50,
    cohesion=50,
    seperation_factor=800,
    alignment_factor=0.05,
    cohesion_factor=1,
    name="Good Boid",
)

# get copies of good boid with one parameter changed
large_seperation_boid = dataclasses.replace(good_boid)
large_seperation_boid.separation *= 2
large_seperation_boid.name = "Large Separation"

low_alignment_boid = dataclasses.replace(good_boid)
# low_alignment_boid.alignment_factor /= 5
low_alignment_boid.alignment_factor = 0
low_alignment_boid.name = "Low Alignment"

low_cohesion_boid = dataclasses.replace(good_boid)
# low_cohesion_boid.cohesion_factor /= 5
low_cohesion_boid.cohesion_factor = 0
low_cohesion_boid.name = "Low Cohesion"


class BoidField:
    """A field of boids. This class is responsible for simulating the boids.

    Boid simulation follows implementation here: https://vanhunteradams.com/Pico/Animal_Movement/Boids-algorithm.html
    """

    x_pos_index = 0
    y_pos_index = 1
    pos_slice = slice(x_pos_index, y_pos_index + 1)
    x_vel_index = 2
    y_vel_index = 3
    vel_slice = slice(x_vel_index, y_vel_index + 1)
    separation_index = 4
    alignment_index = 5
    cohesion_index = 6
    seperation_factor_index = 7
    alignment_factor_index = 8
    cohesion_factor_index = 9
    is_faulty_index = 10
    num_parameters = 11

    def __init__(
        self,
        boids: np.ndarray,
        field_size: float,
        max_velocity: float = 500,
    ) -> None:
        """Creates a new BoidField.

        Args:
            boids (np.ndarray): Boids. They are stored in a 2D array, where each row is a boid and each column is a parameter.
                Each boid has the following parameters:
                    - x position
                    - y position
                    - x velocity
                    - y velocity
                    - separation range
                    - alignment range
                    - cohesion range
                    - separation factor
                    - alignment factor
                    - cohesion factor
                    - is faulty
                The position and velocity parameters will be modified as the simulation is run.
            field_size (float): Size of the field. The field is a square with sides of length field_size. Boids wrap around field boundaries.
            max_velocity (float): Velocity to cap boids at.
        """
        assert (
            boids.shape[1] == BoidField.num_parameters
        ), f"Boids must have {BoidField.num_parameters} parameters. See docstring for details."
        self.boids = boids
        self.field_size = field_size
        self.max_velocity = max_velocity

    def apply_velocity(self, dt: float = 1) -> None:
        """Applies the velocity of each boid to its position. Boids wrap around field boundaries.

        Args:
            dt (float): Time step.
        """
        self.boids[:, BoidField.pos_slice] += self.boids[:, BoidField.vel_slice] * dt
        self.boids[:, BoidField.pos_slice] %= self.field_size

    def cap_velocity(self) -> None:
        """Caps the velocity of each boid to max_velocity.

        Args:
            max_velocity (float): Maximum velocity.
        """
        velocities = self.boids[:, BoidField.vel_slice]
        velocities = np.minimum(velocities, self.max_velocity)
        velocities = np.maximum(velocities, -self.max_velocity)
        self.boids[:, BoidField.vel_slice] = velocities

    def simulate(self, dt=1) -> None:
        new_velocity = self.boids[:, BoidField.vel_slice].copy()

        for i, boid in enumerate(self.boids):
            distances = np.linalg.norm(
                self.boids[:, BoidField.pos_slice] - boid[BoidField.pos_slice], axis=1
            )

            # fmt: off
            # get masks for boids in range of the various factors. filter out the boid itself
            seperation_range_mask = np.logical_and(distances < boid[BoidField.separation_index], distances > 0)
            alignment_range_mask = np.logical_and(distances < boid[BoidField.alignment_index], distances > 0)
            cohesion_range_mask = np.logical_and(distances < boid[BoidField.cohesion_index], distances > 0)
            # fmt: on

            if np.any(seperation_range_mask):
                # vector from boid to other boids in seperation range
                relative_position_vectors = (
                    self.boids[seperation_range_mask][:, BoidField.pos_slice]
                    - boid[BoidField.pos_slice]
                )
                seperation_factor = boid[BoidField.seperation_factor_index]
                # update velocity based on seperation factor
                # This equivalent to the (relative position unit vector) * (seperation factor) / (distance to other boid)
                neighbor_distances = np.linalg.norm(relative_position_vectors, axis=1)[
                    :, np.newaxis
                ]
                neighbor_unit_vectors = relative_position_vectors / neighbor_distances
                new_velocity[i] -= (
                    np.sum(neighbor_unit_vectors / neighbor_distances, axis=0)
                    * seperation_factor
                )

            if np.any(alignment_range_mask):
                # velocities of other boids in alignment range
                other_velocities = self.boids[alignment_range_mask][
                    :, BoidField.vel_slice
                ]
                alignment_factor = boid[BoidField.alignment_factor_index]
                current_velocity = boid[BoidField.vel_slice]
                # update velocity based on alignment factor
                new_velocity[i] += (
                    np.average(other_velocities - current_velocity, axis=0)
                ) * alignment_factor

            if np.any(cohesion_range_mask):
                # vector of positions of other boids in cohesion range
                other_positions = self.boids[cohesion_range_mask][
                    :, BoidField.pos_slice
                ]
                cohesion_factor = boid[BoidField.cohesion_factor_index]
                current_position = boid[BoidField.pos_slice]
                # update velocity based on cohesion factor
                new_velocity[i] += (
                    np.average(other_positions - current_position, axis=0)
                ) * cohesion_factor

        self.boids[:, BoidField.vel_slice] = new_velocity
        self.apply_velocity(dt)
        self.cap_velocity()

    @classmethod
    def make_boid_field(
        cls: "BoidField",
        num_boids: int,
        num_faulty_boids: int,
        boid_parameters: BoidParameters,
        faulty_boid_parameters: list[BoidParameters],
        field_size: float = 1000,
    ) -> "BoidField":
        """Makes a boid field based on parameters

        Args:
            num_boids (int): Number of functional boids
            num_faulty_boids (int): Number of faulty boids
            boid_parameters (BoidParameters): Parameters for functional boids
            faulty_boid_parameters (list[BoidParameters]): Parameters for faulty boids.
                List so that each faulty boid can have different parameters.
                The list is cycled through, so if there are more faulty boids than parameters, the parameters will be reused.
                The is_faulty field for each boid will be the index of the faulty_boid_parameters list + 1 (0 is not faulty).
            field_size (float): Size of the field. The field is a square with sides of length field_size. Boids wrap around field boundaries.

        Returns:
            BoidField: A boid field with the specified parameters
        """
        # fmt: off
        boids = np.zeros((num_boids + num_faulty_boids, cls.num_parameters))
        for i in range(num_boids):
            boids[i, cls.pos_slice] = np.random.rand(2) * field_size
            boids[i, cls.separation_index] = boid_parameters.separation
            boids[i, cls.alignment_index] = boid_parameters.alignment
            boids[i, cls.cohesion_index] = boid_parameters.cohesion
            boids[i, cls.seperation_factor_index] = boid_parameters.seperation_factor
            boids[i, cls.alignment_factor_index] = boid_parameters.alignment_factor
            boids[i, cls.cohesion_factor_index] = boid_parameters.cohesion_factor
        
        for i in range(num_faulty_boids):
            faulty_parameter_index = i % len(faulty_boid_parameters)
            faulty_boid_parameter = faulty_boid_parameters[faulty_parameter_index]
            boids[i + num_boids, cls.pos_slice] = np.random.rand(2) * field_size
            boids[i + num_boids, cls.separation_index] = faulty_boid_parameter.separation
            boids[i + num_boids, cls.alignment_index] = faulty_boid_parameter.alignment
            boids[i + num_boids, cls.cohesion_index] = faulty_boid_parameter.cohesion
            boids[i + num_boids, cls.seperation_factor_index] = faulty_boid_parameter.seperation_factor
            boids[i + num_boids, cls.alignment_factor_index] = faulty_boid_parameter.alignment_factor
            boids[i + num_boids, cls.cohesion_factor_index] = faulty_boid_parameter.cohesion_factor
            boids[i + num_boids, cls.is_faulty_index] = faulty_parameter_index + 1
        # fmt: on

        return cls(boids, field_size=field_size)


class BoidLogger:
    def __init__(
        self, file: str, boid_field: BoidField, num_neighbors: int = 3
    ) -> None:
        self.path = Path(file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _handle = open(str(self.path), "w", newline="")
        self._csv_writer = csv.writer(
            _handle, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        self.num_neighbors = num_neighbors
        self.log_mask = np.ones(BoidField.num_parameters, dtype=bool)
        self.boid_field = boid_field
        # used to scale data to be between 0 and 1 where applicable
        self.scalars = np.ones(BoidField.num_parameters)
        self.scalars[BoidField.pos_slice] = self.boid_field.field_size
        self.scalars[BoidField.vel_slice] = self.boid_field.max_velocity

    def log_header(self, log_meta_parameters: bool = False) -> None:
        labels = [
            "x_pos",
            "y_pos",
            "x_vel",
            "y_vel",
            "separation",
            "alignment",
            "cohesion",
            "seperation_factor",
            "alignment_factor",
            "cohesion_factor",
            "is_faulty",
        ]
        if len(labels) != BoidField.num_parameters:
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match number of parameters ({BoidField.num_parameters})"
            )

        if not log_meta_parameters:
            self.log_mask[
                BoidField.separation_index : BoidField.cohesion_factor_index + 1
            ] = 0
            labels = list(np.asarray(labels)[self.log_mask])

        for i in range(self.num_neighbors):
            labels.append(f"neighbor_{i}_x_distance")
            labels.append(f"neighbor_{i}_y_distance")

        all_labels = []
        for i in range(self.boid_field.boids.shape[0]):
            all_labels.extend([f"{i}-{label}" for label in labels])

        self._csv_writer.writerow(all_labels)

    def log(self) -> None:
        row_data = []
        for boid in self.boid_field.boids:
            neighbor_offsets = BoidLogger.get_neighbors(
                boid, self.boid_field, self.num_neighbors
            )
            row_data.extend((boid / self.scalars)[self.log_mask])
            row_data.extend(neighbor_offsets / self.boid_field.field_size)

        self._csv_writer.writerow(row_data)

    @staticmethod
    def get_neighbors(boid: np.ndarray, boid_field: BoidField, num_neighbors: int):
        distances = np.linalg.norm(
            boid_field.boids[:, BoidField.pos_slice] - boid[BoidField.pos_slice],
            axis=1,
        )
        neighbor_indices = np.argsort(distances)[1 : num_neighbors + 1]
        neighbors = boid_field.boids[neighbor_indices]
        neighbor_offsets = neighbors[:, BoidField.pos_slice] - boid[BoidField.pos_slice]
        neighbor_offsets = neighbor_offsets.ravel()
        return neighbor_offsets


def get_data(
    num_good_boids,
    num_faulty_boids,
    num_iterations,
    visualize=True,
    file_name: str = "data/boid_log.csv",
):
    if visualize:
        import matplotlib.pyplot as plt

    boid_params = good_boid
    faulty_boid_params = [
        large_seperation_boid,
        low_alignment_boid,
        low_cohesion_boid,
    ]
    bf = BoidField.make_boid_field(
        num_good_boids, num_faulty_boids, boid_params, faulty_boid_params
    )
    bf.max_velocity = 200
    bf.boids[:, BoidField.vel_slice] = (
        np.random.rand(num_good_boids + num_faulty_boids, 2) * 500 - 250
    )
    logger = BoidLogger(file_name, boid_field=bf)
    logger.log_header()

    print(f"Starting simulation: {file_name}")

    for i in range(num_iterations):
        bf.simulate(0.01)
        logger.log()
        print(f"Iteration {i:>5}/{num_iterations}", end="\r")
        if visualize:
            # plt.scatter(
            #     bf.boids[:, BoidField.x_pos_index],
            #     bf.boids[:, BoidField.y_pos_index],
            #     c=bf.boids[:, BoidField.is_faulty_index],
            #     s=20,
            # )
            velocity_magnitudes = np.linalg.norm(
                bf.boids[:, BoidField.vel_slice], axis=1
            )
            plt.quiver(
                bf.boids[:, BoidField.x_pos_index],
                bf.boids[:, BoidField.y_pos_index],
                bf.boids[:, BoidField.x_vel_index] / velocity_magnitudes,
                bf.boids[:, BoidField.y_vel_index] / velocity_magnitudes,
                bf.boids[:, BoidField.is_faulty_index],
                scale=35,
            )
            plt.xlim((0, bf.field_size))
            plt.ylim((0, bf.field_size))
            plt.pause(0.01)
            plt.clf()

    print()
    print()


if __name__ == "__main__":
    get_data(num_good_boids=150, num_faulty_boids=30, num_iterations=100)
