from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from traffic_detection.definitions import Box2D


def convert_vel_vector_ms_to_kmh(velocity: tuple[float, float]) -> float:
    """Convert velocity from m/s in xy to km/h."""
    return round((velocity[0] ** 2 + velocity[1] ** 2) ** 0.5 * 3.6, 2)


class FilterEMA:
    """Exponential Moving Average (EMA) filter for smoothing time series data."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.last_value: np.ndarray | None = None

    def reset(self) -> None:
        """Reset the filter state."""
        self.last_value = None

    def update(self, value: np.ndarray) -> np.ndarray:
        """Update the filter with a new value and return the smoothed value."""
        if self.last_value is None:
            self.last_value = value
        else:
            self.last_value = self.alpha * value + (1 - self.alpha) * self.last_value
        return self.last_value


class TrackHistoryManager:
    """Keeps per-track history of past positions."""

    def __init__(
        self,
        max_length: int = 120,
        dt: float = 1.0,
        velocity_filter_alpha: float = 0.1,
        compute_bev_velocity: bool = True,
    ) -> None:
        self.max_length = int(max_length)
        self.velocity_filter_alpha = velocity_filter_alpha
        self.compute_bev_velocity = compute_bev_velocity
        self.dt = dt
        self._hist_2d: dict[int, deque[tuple[float, float]]] = {}
        self._hist_bev: dict[int, deque[tuple[float, float]]] = {}

        # self._hist_vel_2d: dict[int, deque[tuple[float, float]]] = {} # 2d velocity is computed by kalman already
        self._hist_vel_bev: dict[int, deque[tuple[float, float]]] = {}

        # self._vel_filter: dict[int, FilterEMA] = {}
        self._vel_bev_filter: dict[int, FilterEMA] = {}

    def reset(self) -> None:
        """Reset stored history."""
        self._hist_2d.clear()
        self._hist_bev.clear()

    def update(self, boxes: Box2D) -> Box2D:
        """Attach histories to boxes. Optionally append current positions.

        Args:
            boxes: Box2D containing track_ids and bev_pos information.
        """
        assert boxes.track_ids is not None, "Track IDs must be present in boxes."
        assert boxes.bev_pos is not None, "BEV positions must be present in boxes."

        # Current positions
        pos_2d = boxes.boxes_centers
        pos_bev = boxes.bev_pos

        # Update per-track queues
        for i, tid in enumerate(boxes.track_ids.tolist()):
            # initialize a new queue if new track id
            if tid not in self._hist_2d:
                self._hist_2d[tid] = deque(maxlen=self.max_length)
                self._hist_bev[tid] = deque(maxlen=self.max_length)
                if self.compute_bev_velocity:
                    # self._hist_vel_2d[tid] = deque(maxlen=self.max_length)
                    # self._vel_filter[tid] = FilterEMA(alpha=self.velocity_filter_alpha)
                    self._hist_vel_bev[tid] = deque(maxlen=self.max_length)
                    self._vel_bev_filter[tid] = FilterEMA(alpha=self.velocity_filter_alpha)

            # append current position to the track id history
            self._hist_2d[tid].append((float(pos_2d[i, 0]), float(pos_2d[i, 1])))
            self._hist_bev[tid].append((float(pos_bev[i, 0]), float(pos_bev[i, 1])))

            if self.compute_bev_velocity:
                # compute velocity and append to velocity history
                if len(self._hist_2d[tid]) > 2:
                    # prev_pos = self._hist_2d[tid][-2]
                    prev_pos_bev = self._hist_bev[tid][-2]

                    # vel_current = (
                    #    (pos_2d[i, 0] - prev_pos[0]) / self.dt,
                    #    (pos_2d[i, 1] - prev_pos[1]) / self.dt,
                    # )
                    vel_bev_current = (
                        (pos_bev[i, 0] - prev_pos_bev[0]) / self.dt,
                        (pos_bev[i, 1] - prev_pos_bev[1]) / self.dt,
                    )
                else:
                    # vel_current = (0.0, 0.0)
                    vel_bev_current = (0.0, 0.0)

                # vel = self._vel_filter.update(np.array(vel_current))
                vel_bev = self._vel_bev_filter[tid].update(np.array(vel_bev_current))

                # self._hist_vel_2d[tid].append((float(vel[0]), float(vel[1])))
                self._hist_vel_bev[tid].append((float(vel_bev[0]), float(vel_bev[1])))

        # build lists aligned with boxes order
        out_hist_2d_list: list[list[tuple[float, float]]] = []
        out_hist_bev_list: list[list[tuple[float, float]]] = []
        # out_vel_list: list[tuple[float, float]] = []
        out_vel_bev_list: list[tuple[float, float]] = []
        for tid in boxes.track_ids.tolist():
            out_hist_2d_list.append(list(self._hist_2d.get(tid, [])))
            out_hist_bev_list.append(list(self._hist_bev.get(tid, [])))

            if self.compute_bev_velocity:
                # output last velocity
                # out_vel_list.append(list(self._hist_vel_2d.get(tid, [(0.0, 0.0)]))[-1])
                out_vel_bev_list.append(list(self._hist_vel_bev.get(tid, [(0.0, 0.0)]))[-1])

        # Return a new Box2D with histories attached
        return Box2D(
            boxes=boxes.boxes,
            scores=boxes.scores,
            labels=boxes.labels,
            colors=boxes.colors,
            bev_pos=boxes.bev_pos,
            track_ids=boxes.track_ids,
            track_ages=boxes.track_ages,
            vel=boxes.vel,  # np.array(out_vel_list, dtype=np.float32),
            vel_bev=np.array(out_vel_bev_list, dtype=np.float32),
            track_center_history=out_hist_2d_list,
            track_bev_pos_history=out_hist_bev_list,
        )

    def plot_history(self, savedir: Path) -> None:
        """Plot the history of tracks."""

        Path(savedir).mkdir(parents=True, exist_ok=True)

        for hist_name, plot_type, limits, all_histories in (
            ("pos_2d", "pos", (0, 0, 1280, 720), self._hist_2d),
            ("pos_bev", "pos", (-20, -20, 20, 200), self._hist_bev),
            # ("vel_2d", "vel",(-200,-200, 200, 200), self._hist_vel_2d),
            ("vel_bev", "vel (m/s)", (-60, -60, 60, 60), self._hist_vel_bev),
        ):
            plt.figure(figsize=(16, 16))
            plt.title(f"History ({hist_name}) for tracks")
            plt.xlabel(f"X {plot_type}")
            plt.ylabel(f"Y {plot_type}")
            plt.xlim(limits[0], limits[2])
            plt.ylim(limits[1], limits[3])
            for track_id, history in all_histories.items():
                history = np.array(history)
                plt.plot(history[:, 0], history[:, 1], marker=".", label=f"Track {track_id}")

            plt.grid()
            plt.legend()
            outpath = f"{savedir}/history_{hist_name}.png"
            plt.savefig(outpath)
            print(f"Saved history plot to {outpath}")
            plt.close()
