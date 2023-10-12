from dataclasses import dataclass
import numpy as np


@dataclass
class BoidParameters:
    """Stores the parameters of a boid."""

    separation: float
    alignment: float
    cohesion: float
    seperation_factor: float
    alignment_factor: float
    cohesion_factor: float


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
        max_velocity: float = 75,
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
        for boid in self.boids:
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
                boid[BoidField.vel_slice] -= (
                    np.sum(relative_position_vectors, axis=0) * seperation_factor
                )

            if np.any(alignment_range_mask):
                # velocities of other boids in alignment range
                other_velocities = self.boids[alignment_range_mask][
                    :, BoidField.vel_slice
                ]
                alignment_factor = boid[BoidField.alignment_factor_index]
                current_velocity = boid[BoidField.vel_slice]
                # update velocity based on alignment factor
                boid[BoidField.vel_slice] += (
                    np.average(other_velocities, axis=0) - current_velocity
                ) * alignment_factor

            if np.any(cohesion_range_mask):
                # vector of positions of other boids in cohesion range
                other_positions = self.boids[cohesion_range_mask][
                    :, BoidField.pos_slice
                ]
                cohesion_factor = boid[BoidField.cohesion_factor_index]
                current_position = boid[BoidField.pos_slice]
                # update velocity based on cohesion factor
                boid[BoidField.vel_slice] += (
                    np.average(other_positions, axis=0) - current_position
                ) * cohesion_factor

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
            faulty_boid_parameter = faulty_boid_parameters[i % len(faulty_boid_parameters)]
            boids[i + num_boids, cls.pos_slice] = np.random.rand(2) * field_size
            boids[i + num_boids, cls.separation_index] = faulty_boid_parameter.separation
            boids[i + num_boids, cls.alignment_index] = faulty_boid_parameter.alignment
            boids[i + num_boids, cls.cohesion_index] = faulty_boid_parameter.cohesion
            boids[i + num_boids, cls.seperation_factor_index] = faulty_boid_parameter.seperation_factor
            boids[i + num_boids, cls.alignment_factor_index] = faulty_boid_parameter.alignment_factor
            boids[i + num_boids, cls.cohesion_factor_index] = faulty_boid_parameter.cohesion_factor
            boids[i + num_boids, cls.is_faulty_index] = i + 1
        # fmt: on

        return cls(boids, field_size=field_size)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    boid_params = BoidParameters(75, 75, 75, 2.5, 1.2, 0.8)
    faulty_boid_params = [BoidParameters(200, 75, 75, 2.5, 1.2, 0.8)]
    bf = BoidField.make_boid_field(30, 1, boid_params, faulty_boid_params)

    for _ in range(1000):
        plt.scatter(
            bf.boids[:, BoidField.x_pos_index],
            bf.boids[:, BoidField.y_pos_index],
            c=bf.boids[:, BoidField.is_faulty_index],
        )
        plt.xlim((0, bf.field_size))
        plt.ylim((0, bf.field_size))
        plt.pause(0.05)
        bf.simulate(0.1)
        plt.clf()
