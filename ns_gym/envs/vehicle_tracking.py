import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
from gridworldenvutils.los import check_fov, get_visible_cells
from gridworldenvutils.path_finding import find_shortest_path
import io


class Actions(Enum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    UP_LEFT = 5
    UP_RIGHT = 6
    DOWN_LEFT = 7
    DOWN_RIGHT = 8
    ROTATE_CW = 9
    ROTATE_CCW = 10


class VehicleTrackingEnv(gym.Env):
    """A simple 2D vehicle tracking environment where a pursuer tries to catch an evader.


    KWARGS:
        obstacle_map (np.ndarray): 2D array representing the environment. 1 for obstacle, 0 for free space.
        num_pursuers (int): Number of pursuer vehicles.
        starting_pursuer_position (tuple or list of tuples): Starting position(s) of the pursuer(s).
        starting_evader_position (tuple): Starting position of the evader.
        goal_locations (list of tuples): List of possible goal locations for the evader.
        fov_distance (int): Field of view distance for the pursuer.
        fov_angle (float): Field of view angle (in radians) for the pursuer.
        game_mode (str): "capture" or "follow". In "capture" mode, pursuers get reward for capturing the evader.
                         In "follow" mode, pursuers get reward for keeping the evader in view.
        is_evader_always_observable (bool): If True, evader position is always observable in the observation.
        allow_diagonal_evader_movement (bool): If True, evader can move diagonally when computing its path.

    WARNING: Coordinates are in (col, row) format where origin is in the top left corner.

    So the map looks like:
        (0,0) (1,0) (2,0) ... (x,0) ->
        (0,1) (1,1) (2,1)
        (0,2) (1,2) (2,2)
        |
        (0,y)

    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}

    def __init__(self, **kwargs) -> None:
        super().__init__()


        self.obstacle_map = kwargs.get("obstacle_map", np.zeros((5, 5)))
        self.map_dimensions = self.obstacle_map.shape

        self.num_pursuers = kwargs.get("num_pursuers", 1)

        start_pos = kwargs.get("starting_pursuer_position", (0, 0))
        if isinstance(start_pos, tuple) or (
            isinstance(start_pos, np.ndarray) and start_pos.ndim == 1
        ):
            self.starting_pursuer_positions = [
                np.array(start_pos) for _ in range(self.num_pursuers)
            ]
        else:
            self.starting_pursuer_positions = [np.array(pos) for pos in start_pos]

        self.starting_evader_position = np.array(
            kwargs.get(
                "starting_evader_position",
                (self.map_dimensions[0] - 1, self.map_dimensions[1] - 1),
            )
        )

        if len(self.starting_pursuer_positions) != self.num_pursuers:
            raise ValueError(
                "Number of starting pursuer positions must match num_pursuers"
            )

        self.goal_locations = np.array(
            kwargs.get(
                "goal_locations",
                [(self.map_dimensions[0] // 2, self.map_dimensions[1] // 2)],
            )
        )
        self.fov_distance = kwargs.get("fov_distance", 2)
        self.fov_angle = kwargs.get("fov_angle", np.pi / 2)
        self.game_mode = kwargs.get("game_mode", "capture")
        self.is_evader_always_observable = kwargs.get(
            "is_evader_always_observable", True
        )
        self.allow_diagonal_evader_movement = kwargs.get(
            "allow_diagonal_evader_movement", False
        )

        self.flatten_obs = kwargs.get("flatten_obs", False)

        self._directions = [
            np.array([1, 0]),
            np.array([1, 1]),
            np.array([0, 1]),
            np.array([-1, 1]),
            np.array([-1, 0]),
            np.array([-1, -1]),
            np.array([0, -1]),
            np.array([1, -1]),
        ]


        self.pursuer_positions = [np.copy(p) for p in self.starting_pursuer_positions]
        self.pursuer_dir_idxs = [0] * self.num_pursuers  # All start facing East
        self.pursuer_dirs = [self._directions[i] for i in self.pursuer_dir_idxs]

        self.evader_position = self.starting_evader_position
        self.evader_path_step = 0

        num_states_per_cell = self.map_dimensions[0] * self.map_dimensions[1]


        if self.flatten_obs:
            self.observation_space = spaces.Dict( 
                {
                    "pursuer_position": spaces.MultiDiscrete(
                        [num_states_per_cell] * self.num_pursuers
                    ),
                    "evader_position": spaces.Box(
                        low=-1, high=num_states_per_cell - 1, shape=(), dtype=np.int64
                    ),
                    "goal_position": spaces.MultiDiscrete(
                        [num_states_per_cell] * len(self.goal_locations)
                    ),
                    "is_evader_in_view": spaces.Discrete(2),
                }
            )
        else:
            self.observation_space = spaces.Dict( 
                {
                    "pursuer_position": spaces.Box(
                        low=0, high=num_states_per_cell - 1, shape=(self.num_pursuers, 2), dtype=np.int64
                    ),
                    "evader_position": spaces.Box(
                        low=-1, high=num_states_per_cell - 1, shape=(2,), dtype=np.int64
                    ),
                    "goal_position": spaces.Box(
                        low=0, high=num_states_per_cell - 1, shape=(len(self.goal_locations), 2), dtype=np.int64
                    ),
                    "is_evader_in_view": spaces.Discrete(2),
                }
            )

    
        self.action_space = spaces.MultiDiscrete([len(Actions)] * self.num_pursuers)

    def step(self, actions):
        """Take a step in the environment.

        Args:
            actions (list or np.ndarray): List of actions for each pursuer.
        """
        for i, action in enumerate(actions):
            self._forward_pursuer_dynamics(i, action)

        evader_position = self._forward_evader_dynamics()

        is_captured = any(
            np.array_equal(evader_position, p_pos) for p_pos in self.pursuer_positions
        )
        is_in_view = self._evader_inview()

        if is_captured:
            reward = 1.0
        elif is_in_view:
            if self.game_mode == "follow":
                reward = 1.0
            else:
                min_dist = min(
                    np.linalg.norm(evader_position - p_pos)
                    for p_pos in self.pursuer_positions
                )
                reward = 1.0 / (min_dist + 1)
        else:
            reward = 0.0

        done = False
        if any(np.array_equal(evader_position, goal) for goal in self.goal_locations):
            done = True
        elif self.game_mode == "capture" and is_captured:
            done = True

        observation = self._get_obs(
            is_in_view=is_in_view or self.is_evader_always_observable
        )
        return observation, reward, done, False, {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # --- MODIFICATION: Reset all pursuer states ---
        self.pursuer_positions = [np.copy(p) for p in self.starting_pursuer_positions]
        self.pursuer_dir_idxs = [0] * self.num_pursuers
        self.pursuer_dirs = [self._directions[i] for i in self.pursuer_dir_idxs]

        self.evader_position = self.starting_evader_position
        self.evader_path = self._compute_evader_path()
        self.evader_path_step = 0
        is_in_view = self._evader_inview()

        return self._get_obs(
            is_in_view=is_in_view or self.is_evader_always_observable
        ), {}

    def render(self):
        outfile = io.StringIO()
        outfile.write("\033[H\033[J")  # Clear screen

        outfile.write(
            f"Vehicle Tracking Environment\nTrue Goal Location {self.true_goal_location}\n"
        )

        # --- MODIFICATION: Get union of all visible cells ---
        visible_cells = set()
        for p_pos, p_dir in zip(self.pursuer_positions, self.pursuer_dirs):
            cells = self._get_visible_cells(p_pos, p_dir)
            visible_cells.update((cell[0], cell[1]) for cell in cells)

        # Convert positions to tuples for fast lookups in the loop
        pursuer_pos_set = {tuple(p) for p in self.pursuer_positions}

        for r in range(self.map_dimensions[1]):
            for c in range(self.map_dimensions[0]):
                current_pos_tuple = (c, r)
                if current_pos_tuple in pursuer_pos_set:
                    outfile.write("\033[94mP\033[0m ")
                elif tuple(self.evader_position) == current_pos_tuple:
                    outfile.write("\033[91mE\033[0m ")
                elif any(
                    np.array_equal(current_pos_tuple, goal)
                    for goal in self.goal_locations
                ):
                    outfile.write("\033[92mG\033[0m ")
                elif self.obstacle_map[r, c]:
                    outfile.write("X ")
                elif current_pos_tuple in visible_cells:
                    outfile.write("\033[93mo\033[0m ")
                else:
                    outfile.write(". ")
            outfile.write("\n")
        return outfile.getvalue()

    def _forward_pursuer_dynamics(self, pursuer_idx, action):
        pursuer_position = self.pursuer_positions[pursuer_idx]
        new_pursuer_position = np.copy(pursuer_position)

        if action == Actions.STAY.value:
            pass
        elif action == Actions.UP.value:
            new_pursuer_position[1] = max(pursuer_position[1] - 1, 0)
            self.pursuer_dir_idxs[pursuer_idx] = 6  # N
        elif action == Actions.DOWN.value:
            new_pursuer_position[1] = min(
                pursuer_position[1] + 1, self.map_dimensions[1] - 1
            )
            self.pursuer_dir_idxs[pursuer_idx] = 2  # S
        elif action == Actions.LEFT.value:
            new_pursuer_position[0] = max(pursuer_position[0] - 1, 0)
            self.pursuer_dir_idxs[pursuer_idx] = 4  # W
        elif action == Actions.RIGHT.value:
            new_pursuer_position[0] = min(
                pursuer_position[0] + 1, self.map_dimensions[0] - 1
            )
            self.pursuer_dir_idxs[pursuer_idx] = 0  # E
        elif action == Actions.UP_LEFT.value:
            new_pursuer_position[0] = max(pursuer_position[0] - 1, 0)
            new_pursuer_position[1] = max(pursuer_position[1] - 1, 0)
            self.pursuer_dir_idxs[pursuer_idx] = 5  # NW
        elif action == Actions.UP_RIGHT.value:
            new_pursuer_position[0] = min(
                pursuer_position[0] + 1, self.map_dimensions[0] - 1
            )
            new_pursuer_position[1] = max(pursuer_position[1] - 1, 0)
            self.pursuer_dir_idxs[pursuer_idx] = 7  # NE
        elif action == Actions.DOWN_LEFT.value:
            new_pursuer_position[0] = max(pursuer_position[0] - 1, 0)
            new_pursuer_position[1] = min(
                pursuer_position[1] + 1, self.map_dimensions[1] - 1
            )
            self.pursuer_dir_idxs[pursuer_idx] = 3  # SW
        elif action == Actions.DOWN_RIGHT.value:
            new_pursuer_position[0] = min(
                pursuer_position[0] + 1, self.map_dimensions[0] - 1
            )
            new_pursuer_position[1] = min(
                pursuer_position[1] + 1, self.map_dimensions[1] - 1
            )
            self.pursuer_dir_idxs[pursuer_idx] = 1  # SE
        elif action == Actions.ROTATE_CW.value:
            self.pursuer_dir_idxs[pursuer_idx] = (
                self.pursuer_dir_idxs[pursuer_idx] + 1
            ) % 8
        elif action == Actions.ROTATE_CCW.value:
            self.pursuer_dir_idxs[pursuer_idx] = (
                self.pursuer_dir_idxs[pursuer_idx] - 1
            ) % 8
        else:
            raise ValueError("Invalid action")

        self.pursuer_dirs[pursuer_idx] = self._directions[
            self.pursuer_dir_idxs[pursuer_idx]
        ]

        if not self.obstacle_map[new_pursuer_position[1], new_pursuer_position[0]]:
            self.pursuer_positions[pursuer_idx] = new_pursuer_position

    def _forward_evader_dynamics(self):
        self.evader_path_step = min(
            self.evader_path_step + 1, len(self.evader_path) - 1
        )
        evader_position = self.evader_position = self.evader_path[self.evader_path_step]
        return evader_position

    def _compute_evader_path(self):
        self.true_goal_location = self.goal_locations[
            np.random.randint(len(self.goal_locations))
        ]
        evader_path = find_shortest_path(
            self.starting_evader_position,
            self.true_goal_location,
            self.obstacle_map,
            allow_diagonal=self.allow_diagonal_evader_movement,
        )
        return evader_path

    def _evader_inview(self):
        for p_pos, p_dir in zip(self.pursuer_positions, self.pursuer_dirs):
            if check_fov(
                p_pos,
                self.evader_position,
                p_dir,
                self.fov_angle,
                self.fov_distance,
                self.obstacle_map,
            ):
                return True
        return False

    def _get_obs(self, is_in_view=False):
        """Get the current observation.

        Args:
            is_in_view (bool): Whether the evader is currently in view of any pursuer.
        """

        if not self.flatten_obs:
            return {
                "pursuer_position": np.array(
                    self.pursuer_positions, dtype=np.int64
                ),
                "evader_position": self.evader_position
                if is_in_view
                else np.array([-1, -1], dtype=np.int64),
                "goal_position": np.array(
                    self.goal_locations, dtype=np.int64
                ),
                "is_evader_in_view": int(self._evader_inview()),
            }
        else:  # flattened observation
            pursuer_flat = [self._to_flattened(p_pos) for p_pos in self.pursuer_positions]
            return {
                "pursuer_position": np.array(pursuer_flat, dtype=np.int64),
                "evader_position": self._to_flattened(self.evader_position)
                if is_in_view
                else -1,
                "goal_position": np.array(
                    [self._to_flattened(g) for g in self.goal_locations], dtype=np.int64
                ),
                "is_evader_in_view": int(self._evader_inview()),
            }

    def _to_flattened(self, position):
        return position[0] * self.map_dimensions[1] + position[1]

    def _from_flattened(self, index):
        x = index // self.map_dimensions[1]
        y = index % self.map_dimensions[1]
        return np.array([x, y])

    def close(self):
        return super().close()

    def _get_visible_cells(self, pursuer_pos, pursuer_dir):
        return get_visible_cells(
            pursuer_pos,
            pursuer_dir,
            self.fov_angle,
            self.fov_distance,
            self.obstacle_map,
        )


"""Hardcoded obstacle maps for testing purposes.
"""
OBSTACLE_MAP_20x20 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)

OBSTACLE_MAP_10x10 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=np.int64,
)

ROAD_MAP_10x10 = np.array(
    [
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    ]
)

ROAD_MAP_30x30 = np.array(
    [
        [1] * 30 if r < 10 or r >= 20 else [1] * 10 + [0] * 10 + [1] * 10
        for r in range(30)
    ]
)

# Set the vertical and horizontal bands to 0
for i in range(13, 17):
    ROAD_MAP_30x30[:, i] = 0
    ROAD_MAP_30x30[i, :] = 0

# Set the thick diagonal to 0
for i in range(0, 30):
    for j in range(0, 30):
        # Make the diagonal thicker by checking if the indices are close
        if abs(i - j) <= 1:
            ROAD_MAP_30x30[i, j] = 0


if __name__ == "__main__":
    import ns_gym 
    import gymnasium as gym
    import time

    goal_locations = [(29, 29), (15, 29), (29, 15), (0, 15), (15, 0)]

    env = gym.make(
        "ns_gym/VehicleTracking-v0",
        render_mode="human",
        num_pursuers=2,
        starting_pursuer_position=[
            (10, 10),
            (11, 11),
        ],  # Provide a list of start positions
        obstacle_map=ROAD_MAP_30x30,
        starting_evader_position=(0, 0),
        goal_locations=goal_locations,
        fov_distance=10,
        fov_angle=90 * np.pi / 180,
        game_mode="follow",
        is_evader_always_observable=True,
        allow_diagonal_evader_movement=True,
    )
    env.reset()

    # Action must be a list or array with length equal to num_pursuers
    done = False
    print(env.render())
    while not done:
        time.sleep(0.5)
        actions = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(actions)
        print(env.render())
        print(reward)
        print(obs)

    env.close()
