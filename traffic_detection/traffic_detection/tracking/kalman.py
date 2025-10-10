import numpy as np


class KalmanFilter2D:
    """Kalman Filter for 2D position and velocity tracking.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    def __init__(self, dt: float = 1.0, process_var: float = 1.0, measurement_var: float = 1.0) -> None:
        # State transition matrix (constant velocity model)
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        # Measurement matrix
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
        # Process noise covariance
        self.Q = process_var * np.eye(4, dtype=np.float32)
        # Measurement noise covariance
        self.R = measurement_var * np.eye(2, dtype=np.float32)
        # Initial state estimate
        self.x = np.zeros((4, 1), dtype=np.float32)
        # Initial covariance estimate
        self.P = np.eye(4, dtype=np.float32)

    def initiate(self, init_measurement: np.ndarray, init_velocity: float = 0.0) -> None:
        """Initialize state with first measurement [x, y] and optionally velocity."""
        self.x[:2, 0] = init_measurement
        self.x[2:, 0] = init_velocity

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Predict the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        position = self.x[:2, 0]  # Extract predicted position
        velocity = self.x[2:, 0]  # Extract predicted velocity
        return position, velocity  # Return both position and velocity

    def update(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Update with a new measurement [x, y]."""
        z = np.array(measurement, dtype=np.float32).reshape(2, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.F.shape[0], dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

        position = self.x[:2, 0]  # Extract updated position
        velocity = self.x[2:, 0]  # Extract updated velocity
        return position, velocity  # Return both position and velocity
