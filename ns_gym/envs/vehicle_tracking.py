import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
from gridworldenvutils.los import check_fov, get_visible_cells
from gridworldenvutils.path_finding import find_shortest_path
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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

    metadata = {"render_modes": ["ansi", "human", "rgb_array"], "render_fps": 4}

    def __init__(self, **kwargs) -> None:
        super().__init__()


        self.obstacle_map = kwargs.get("obstacle_map", np.zeros((5, 5)))
        self.evader_road_map = kwargs.get("evader_road_map", self.obstacle_map)
        self.map_dimensions = self.obstacle_map.shape

        self.num_pursuers = kwargs.get("num_pursuers", 1)
        # prob intended movement succeeds
        self.movement_stochasticity = kwargs.get("movement_stochasticity", 1.0)

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

        assert all(
            self.evader_road_map[goal[1], goal[0]] == 0 for goal in self.goal_locations
        ), f"All goal locations must be on free space in the evader road map "
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

        self._OVERSTEER = 11

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
        assert self.evader_road_map[self.evader_position[1], self.evader_position[0]] == 0, "Evader starting position must be on free space"
        self.evader_path_step = 0

        self.last_pursuer_actions = [Actions.STAY.value] * self.num_pursuers
        self._uncertainty_map = {Actions.UP.value: [Actions.UP_LEFT.value, Actions.UP_RIGHT.value], 
                                 Actions.DOWN.value: [Actions.DOWN_LEFT.value, Actions.DOWN_RIGHT.value],
                                 Actions.LEFT.value: [Actions.UP_LEFT.value, Actions.DOWN_LEFT.value], 
                                 Actions.RIGHT.value: [Actions.UP_RIGHT.value, Actions.DOWN_RIGHT.value],
                                 Actions.UP_LEFT.value: [Actions.UP.value, Actions.LEFT.value],
                                 Actions.UP_RIGHT.value: [Actions.UP.value, Actions.RIGHT.value],
                                 Actions.DOWN_LEFT.value: [Actions.DOWN.value, Actions.LEFT.value], 
                                 Actions.DOWN_RIGHT.value: [Actions.DOWN.value, Actions.RIGHT.value]}

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
        num_inview = self._evader_inview()
        is_in_view = num_inview > 0

        if is_captured and self.game_mode == "capture":
            reward = 1.0
        elif is_in_view:
            if self.game_mode == "follow":
                reward = num_inview
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
        self.last_pursuer_actions = [Actions.STAY.value] * self.num_pursuers

        self.evader_position = self.starting_evader_position
        self.evader_path = self._compute_evader_path()
        self.evader_path_step = 0
        is_in_view = self._evader_inview() > 0

        return self._get_obs(
            is_in_view=is_in_view or self.is_evader_always_observable
        ), {}
    
    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "rgb_array":
            return self._render_matplotlib_frame()
        elif self.render_mode == "human":
            # For human, we typically rely on the caller to show the rgb array or we print ansi
            # Returning rgb_array allows generic gym wrappers to handle visualization
            return self._render_matplotlib_frame()
        
    def _render_matplotlib_frame(self):
        """Renders the environment using Matplotlib to an RGB array.
        """


        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        
        # --- 1. Construct Base Map Layer ---
        # Value 0: Road (Light Grey)
        # Value 1: Off-Road/Background (White)
        # Value 2: Hard Obstacle (Black)
        display_grid = np.ones(self.map_dimensions) # Default to Off-Road (1)

        # If road map exists, mark Roads (0) as 0
        if hasattr(self, 'evader_road_map') and self.evader_road_map is not None:
            # Map says 0 is road, 1 is not. 
            # We want Road (0) to be value 0 in display_grid.
            display_grid[self.evader_road_map == 0] = 0.0
        else:
            # Fallback if no road map: Treat free space (0) in obstacle map as Road
            display_grid[self.obstacle_map == 0] = 0.0

        # Mark Hard Obstacles (2) - Overwrites everything
        display_grid[self.obstacle_map == 1] = 2.0

        # Define Colormap: [Road, Off-Road, Obstacle]
        cmap = mcolors.ListedColormap(['#D3D3D3', 'white', 'black']) # LightGrey, White, Black
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Render Map
        # origin='upper' ensures (0,0) is top-left.
        ax.imshow(display_grid, cmap=cmap, norm=norm, origin='upper', interpolation='nearest')

        # --- 2. Construct Visibility Layer ---
        vis_mask = np.zeros(self.map_dimensions, dtype=bool)
        
       
        for p_pos, p_dir in zip(self.pursuer_positions, self.pursuer_dirs):
            cells = self._get_visible_cells(p_pos, p_dir)
            if cells:
                # cells are (x, y) tuples
                xs, ys = zip(*cells)
                # NumPy expects [row, col] -> [y, x]
                vis_mask[ys, xs] = True

        # Create Yellow Transparent Overlay
        vis_overlay = np.zeros(self.map_dimensions + (4,)) 
        vis_overlay[vis_mask] = [0.678, 0.847, 0.902, 0.9] 

        # Overlay Visibility
        ax.imshow(vis_overlay, origin='upper', zorder=1, interpolation='nearest')

        # --- 3. Draw Grid & Agents ---
        # Setup ticks for perfect grid alignment
        ax.set_xticks(np.arange(-0.5, self.map_dimensions[0], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.map_dimensions[1], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Dynamic Agents
        if len(self.goal_locations) > 0:
            gx, gy = zip(*self.goal_locations)
            ax.scatter(gx, gy, c='green', marker='X', s=80, label='Goal', zorder=2, edgecolors='white')

        ax.scatter(self.evader_position[0], self.evader_position[1], c='red', marker='o', s=80, label='Evader', zorder=3, edgecolors='white')

        px, py = zip(*self.pursuer_positions)
        ax.scatter(px, py, c='blue', marker='^', s=80, label='Pursuer', zorder=4, edgecolors='white')

        # --- 4. Export ---
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[..., 1:] # ARGB -> RGB
        
        plt.close(fig)
        return img
        
    
    def _render_ansi(self):
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

        intended_action = action
        actual_action = intended_action
        p_success = self.movement_stochasticity

        if p_success < self.np_random.random():
            if intended_action== Actions.STAY.value:
                actual_action = self.last_pursuer_actions[pursuer_idx]
            elif intended_action in [Actions.ROTATE_CW.value, Actions.ROTATE_CCW.value]:
                actual_action = np.random.choice([Actions.STAY.value, self._OVERSTEER])
            else:
                actual_action = np.random.choice(
                    self._uncertainty_map.get(intended_action, [intended_action])
                )

        if actual_action == Actions.STAY.value:
            pass
        elif actual_action == Actions.UP.value:
            new_pursuer_position[1] = max(pursuer_position[1] - 1, 0)
            self.pursuer_dir_idxs[pursuer_idx] = 6  # N
        elif actual_action == Actions.DOWN.value:
            new_pursuer_position[1] = min(
                pursuer_position[1] + 1, self.map_dimensions[1] - 1
            )
            self.pursuer_dir_idxs[pursuer_idx] = 2  # S
        elif actual_action == Actions.LEFT.value:
            new_pursuer_position[0] = max(pursuer_position[0] - 1, 0)
            self.pursuer_dir_idxs[pursuer_idx] = 4  # W
        elif actual_action == Actions.RIGHT.value:
            new_pursuer_position[0] = min(
                pursuer_position[0] + 1, self.map_dimensions[0] - 1
            )
            self.pursuer_dir_idxs[pursuer_idx] = 0  # E
        elif actual_action == Actions.UP_LEFT.value:
            new_pursuer_position[0] = max(pursuer_position[0] - 1, 0)
            new_pursuer_position[1] = max(pursuer_position[1] - 1, 0)
            self.pursuer_dir_idxs[pursuer_idx] = 5  # NW
        elif actual_action == Actions.UP_RIGHT.value:
            new_pursuer_position[0] = min(
                pursuer_position[0] + 1, self.map_dimensions[0] - 1
            )
            new_pursuer_position[1] = max(pursuer_position[1] - 1, 0)
            self.pursuer_dir_idxs[pursuer_idx] = 7  # NE
        elif actual_action == Actions.DOWN_LEFT.value:
            new_pursuer_position[0] = max(pursuer_position[0] - 1, 0)
            new_pursuer_position[1] = min(
                pursuer_position[1] + 1, self.map_dimensions[1] - 1
            )
            self.pursuer_dir_idxs[pursuer_idx] = 3  # SW
        elif actual_action == Actions.DOWN_RIGHT.value:
            new_pursuer_position[0] = min(
                pursuer_position[0] + 1, self.map_dimensions[0] - 1
            )
            new_pursuer_position[1] = min(
                pursuer_position[1] + 1, self.map_dimensions[1] - 1
            )
            self.pursuer_dir_idxs[pursuer_idx] = 1  # SE
        elif actual_action == Actions.ROTATE_CW.value:
            self.pursuer_dir_idxs[pursuer_idx] = (
                self.pursuer_dir_idxs[pursuer_idx] + 1
            ) % 8
        elif actual_action == Actions.ROTATE_CCW.value:
            self.pursuer_dir_idxs[pursuer_idx] = (
                self.pursuer_dir_idxs[pursuer_idx] - 1
            ) % 8
        elif actual_action == self._OVERSTEER:
            if intended_action == Actions.ROTATE_CW.value:
                self.pursuer_dir_idxs[pursuer_idx] = (
                    self.pursuer_dir_idxs[pursuer_idx] + 2
                ) % 8
            elif intended_action == Actions.ROTATE_CCW.value:
                self.pursuer_dir_idxs[pursuer_idx] = (
                    self.pursuer_dir_idxs[pursuer_idx] - 2
                ) % 8
        else:
            raise ValueError("Invalid action")

        self.pursuer_dirs[pursuer_idx] = self._directions[
            self.pursuer_dir_idxs[pursuer_idx]
        ]

        if not self.obstacle_map[new_pursuer_position[1], new_pursuer_position[0]]:
            self.pursuer_positions[pursuer_idx] = new_pursuer_position

        if intended_action not in [Actions.ROTATE_CW.value, Actions.ROTATE_CCW.value]:
            self.last_pursuer_actions[pursuer_idx] = actual_action

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
            self.evader_road_map,
            allow_diagonal=self.allow_diagonal_evader_movement,
        )
        return evader_path

    def _evader_inview(self):
        num_inview = 0
        for p_pos, p_dir in zip(self.pursuer_positions, self.pursuer_dirs):
            if check_fov(
                p_pos,
                self.evader_position,
                p_dir,
                self.fov_angle,
                self.fov_distance,
                self.obstacle_map,
            ):
                num_inview += 1
        return num_inview

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
                "is_evader_in_view": int(self._evader_inview()>0),
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
                "is_evader_in_view": int(self._evader_inview()>0),
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
        render_mode="rgb_array",
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
