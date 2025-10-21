import numpy as np
import sys
from ..utils.distribution import *
from random import randint
from six import StringIO
from gym import Env, spaces, utils
import matplotlib.pyplot as plt

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}


class State:
    """
    State class
    """
    def __init__(self, index, time):
        self.index = index
        self.time = time


def random_map(map_size):
    nR, nC = map_size
    nH = int(0.2 * nR * nC) # Number of holes
    m = []
    for i in range(nR): # Generate ice floe
        m.append(nC * ["F"])
    m[0][0] = "S" # Generate start
    m[-1][-1] = "G" # Generate goal
    while nH > 0: # Generate holes
        i, j = (randint(0, nR-1), randint(0, nC-1))
        if m[i][j] == "F":
            m[i][j] = "H"
            nH -= 1
    for i in range(nR): # Formating
        m[i] = "".join(m[i])
    return m


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class NSFrozenLakeV0(Env):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.

    Non-Stationarity: when the transition function is stochastic , i.e. slippery ice,
    the probability of the resulting states from any action evolves randomly through
    time. The resulting transition function is L_p-Lipschitz.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", map_size=(5,5), is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            if map_name == "random":
                desc = random_map(map_size)
            else:
                desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape

        self.nS = nrow * ncol # n states
        self.nA = 4 # n actions
        self.nT = 10 # n timesteps
        self.action_space = spaces.Discrete(self.nA)
        self.is_slippery = is_slippery
        self.tau = 1 # timestep duration
        self.L_p = 2.0
        self.L_r = 0.0
        self.num_actions = 4
        #self.T = self.generate_transition_matrix()
        isd = np.array(self.desc == b'S').astype('float64').ravel() # Initial state distribution
        self.isd = isd / isd.sum()
        #self._seed()
        self.np_random = np.random.RandomState()
        self.t = 0
        self.reset()

    def _seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def convert_map_to_colors(self, map_):
        color_map = {
            'H': 'black',
            'F': 'white',
            'S': 'blue',
            'G': 'red'
        }
        return [[color_map[c] for c in row] for row in map_]

    def plot_map_with_coordinates(self, map_name, coordinates, state, action):
        map_ = MAPS[map_name]
        nrow, ncol = len(map_), len(map_[0])
        colors = self.convert_map_to_colors(map_)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot the map tiles
        for i in range(nrow):
            for j in range(ncol):
                ax.add_patch(plt.Rectangle((j, nrow - 1 - i), 1, 1, facecolor=colors[i][j]))

        # Overlay the normalized float coordinates
        for coord in coordinates:
            x, y = coord[1] * ncol, (1 - coord[0]) * nrow  # Use the first index for x
            ax.plot(x, y, 'bo')


        for s in range(self.nrow * self.ncol):
            coord = self.__encode_state(s)
            #print(f"State {s} raw coords: {coord}")

            x, y = coord[1] * ncol, (1 - coord[0]) * nrow
            #print(f"State {s} adjusted coords: {(x, y)}")

            ax.annotate(str(s), (x, y), ha='center', va='center', color='black', fontsize=10, weight='bold')
            ax.plot(x, y, 'gx')

        # Map the action to a string
        action_map = {0: "left", 1: "down", 2: "right", 3: "up"}
        a_string = action_map.get(action, "unknown")

        ax.set_xlim(0, ncol)
        ax.set_ylim(0, nrow)
        ax.set_xticks(range(ncol + 1))
        ax.set_yticks(range(nrow + 1))
        ax.grid(which='both')
        ax.set_title(f"State: {state}, Action: {a_string}")

        #plt.gca().invert_yaxis()  # So that (0,0) is at the top-left
        filename = f"State_{state}_Action_{a_string}.png"  # Reformat filename to avoid issues
        plt.savefig(filename)
        plt.close(fig)

    def reset(self, latent_code = 1, seed=0):
        """
        Reset the environment.
        IMPORTANT: Does not create a new environment.
        """
        #if latent_code == 1:
        #    self.time = 0
        #elif latent_code == 2:
        self.time = latent_code - 1
        self.t = 0
        self.np_random = np.random.RandomState(seed)
        self.state = State(categorical_sample(self.isd, self.np_random), self.time) # (index, time)
        self.lastaction = None # for rendering
        self.T = self.generate_transition_matrix()
        return self.state

    def display(self):
        print('Displaying NSFrozenLakeEnv-v0')
        print('map       :')
        print(self.desc)
        print('n states  :', self.nS)
        print('n actions :', self.nA)
        print('timeout   :', self.nT)

    def observe(self):
        """Return current state."""
        return self.__encode_state(self.state.index)

    def inc(self, row, col, a):
        """
        Given a position (row, col) and an action a, return the resulting position (row, col).
        """
        if a==0: # left
            col = max(col-1,0)
        elif a==1: # down
            row = min(row+1,self.nrow-1)
        elif a==2: # right
            col = min(col+1,self.ncol-1)
        elif a==3: # up
            row = max(row-1,0)
        return (row, col)

    def to_s(self, row, col):
        """
        From the state's position (row, col), retrieve the state index.
        """
        return row * self.ncol + col

    def to_m(self, s):
        """
        From the state index, retrieve the state's position (row, col).
        """
        row = int(s / self.ncol)
        col = s - row * self.ncol
        return row, col

    def distance(self, s1, s2):
        """
        Return the Manhattan distance between the positions of states s1 and s2
        """
        if (type(s1) == State) and (type(s2) == State):
            row1, col1 = self.to_m(s1.index)
            row2, col2 = self.to_m(s2.index)
        else:
            assert (type(s1) == int), 'Error: input state has wrong type: type={}'.format(type(s1))
            assert (type(s2) == int), 'Error: input state has wrong type: type={}'.format(type(s2))
            row1, col1 = self.to_m(s1)
            row2, col2 = self.to_m(s2)
        return abs(row1 - row2) + abs(col1 - col2)

    def equality_operator(self, s1, s2):
        """
        Return True if the input states have the same indexes.
        """
        return (s1.index == s2.index)

    def reachable_states(self, s, a):
        if (type(s) == State):
            row, col = self.to_m(s.index)
        else:
            assert (type(s) == int), 'Error: input state has wrong type: type={}'.format(type(s))
            row, col = self.to_m(s)
        rs = np.zeros(shape=self.nS, dtype=int)
        if self.is_slippery:
            for b in [(a-1)%4, a, (a+1)%4]:# Put back for 3 reachable states
            #for b in range(4):
                newrow, newcol = self.inc(row, col, b)
                rs[self.to_s(newrow, newcol)] = 1
        else:
            newrow, newcol = self.inc(row, col, a)
            rs[self.to_s(newrow, newcol)] = 1
        return rs

    def distances_matrix(self, states):
        """
        Return the distance matrix D corresponding to the states of the input array.
        D[i,j] = distance(si, sj)
        """
        n = len(states)
        D = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i+1, n):
                D[i,j] = self.distance(states[i], states[j])
                D[j,i] = self.distance(states[i], states[j])
        return D

    def generate_transition_matrix_a(self):
        T = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)
        print("im in")
        for s in range(self.nS):
            for a in range(self.nA):
                # Generate distribution for t=0
                rs = self.reachable_states(s, a)
                nrs = np.sum(rs)
                w = random_tabular(size=nrs)
                wcopy = list(w.copy())
                T[s,a,0,:] = np.asarray([0 if x == 0 else wcopy.pop() for x in rs], dtype=float)
                row, col = self.to_m(s)
                row_p, col_p = self.inc(row, col, a)
                s_p = self.to_s(row_p, col_p)
                T[s,a,0,s_p] += 1.0 # Increase weight on normally reached state
                T[s,a,0,:] /= sum(T[s,a,0,:])
                states = []
                for k in range(len(rs)):
                    if rs[k] == 1:
                        states.append(State(k,0))
                D = self.distances_matrix(states)
                # Build subsequent distributions st LC constraint is respected
                for t in range(1, self.nT): # t
                    w = random_constrained(w, D, self.L_p * self.tau)
                    wcopy = list(w.copy())
                    T[s,a,t,:] = np.asarray([0 if x == 0 else wcopy.pop() for x in rs], dtype=float)
                #if s == 4 and a==2:
                #    print("check inside",s,a,T[s,a,0,:])
        return T

    def generate_transition_matrix(self):
        T = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)

        intended_prob = 0.9

        for s in range(self.nS):
            for a in range(self.nA):
                # Generate distribution for t=0
                rs = self.reachable_states(s, a)
                nrs = np.sum(rs)

                # Assign intended probability to the intended direction
                row, col = self.to_m(s)
                row_p, col_p = self.inc(row, col, a)
                s_p = self.to_s(row_p, col_p)
                T[s, a, 0, s_p] = intended_prob

                # Distribute the slip probability among other reachable states
                slip_prob = (1.0 - intended_prob) / (nrs - 1)
                for s_next in range(self.nS):
                    if rs[s_next] == 1 and s_next != s_p:
                        T[s, a, 0, s_next] = slip_prob

                # Normalize the probabilities (this might be redundant but ensures correctness)
                T[s, a, 0, :] /= sum(T[s, a, 0, :])

                # The rest of the code remains unchanged
                states = []
                for k in range(len(rs)):
                    if rs[k] == 1:
                        states.append(State(k, 0))
                D = self.distances_matrix(states)
                w = T[s, a, 0, :][rs == 1]  # Initialize w using the probabilities for reachable states
                for t in range(1, self.nT):  # t
                    w = random_constrained(w, D, self.L_p * self.tau)
                    wcopy = list(w.copy())
                    T[s, a, t, :] = np.asarray([0 if x == 0 else wcopy.pop() for x in rs], dtype=float)

        return T

    def transition_probability_distribution(self, s, t, a):
        assert s.index < self.nS, 'Error: index bigger than nS: s.index={} nS={}'.format(s.index, self.nS)
        assert t < self.nT, 'Error: time bigger than nT: t={} nT={}'.format(t, self.nT)
        assert a < self.nA, 'Error: action bigger than nA: a={} nA={}'.format(a, self.nA)
        return self.T[s.index, a, t]

    def transition_probability(self, s_p, s, t, a):
        assert s_p.index < self.nS, 'Error: position bigger than nS: s_p.index={} nS={}'.format(s_p.index, self.nS)
        assert s.index < self.nS, 'Error: position bigger than nS: s.index={} nS={}'.format(s.index, self.nS)
        assert t < self.nT, 'Error: time bigger than nT: t={} nT={}'.format(t, self.nT)
        assert a < self.nA, 'Error: action bigger than nA: a={} nA={}'.format(a, self.nA)
        return self.T[s.index, a, t, s_p.index]

    def get_time(self):
        return self.state.time

    def dynamic_reachable_states(self, s, a):
        """
        Return a numpy array of the reachable states.
        Dynamic means that time increment is performed.
        """
        rs = self.reachable_states(s, a)
        srs = []
        for i in range(len(rs)):
            if rs[i] == 1:
                srs.append(State(i, s.time + self.tau))
        assert (len(srs) == sum(rs))
        return np.array(srs)

    def static_reachable_states(self, s, a):
        """
        Return a numpy array of the reachable states.
        Static means that no time increment is performed.
        """
        rs = self.reachable_states(s, a)
        srs = []
        for i in range(len(rs)):
            if rs[i] == 1:
                srs.append(State(i, s.time))
        assert (len(srs) == sum(rs))
        return np.array(srs)

    def transition(self, s, a, is_model_dynamic=True):
        """
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        The boolean is_model_dynamic indicates whether the temporal transition is applied
        to the state vector or not.
        """
        #d = self.transition_probability_distribution(s, s.time, a)
        #print("transition check left",s.index, a, s.time,self.transition_probability_distribution(s, s.time, 0))
        #print("transition check down", s.index, a, s.time, self.transition_probability_distribution(s, s.time, 1))
        ##print("transition check right", s.index, a, s.time, self.transition_probability_distribution(s, s.time, 2))
        #print("transition check up", s.index, a, s.time, self.transition_probability_distribution(s, s.time, 3))
        d = self.transition_probability_distribution(s, s.time, a)
        p_p = categorical_sample(d, self.np_random)
        if is_model_dynamic:
            s_p = State(p_p, s.time + self.tau)
        else:
            s_p = State(p_p, s.time)
        r = self.instant_reward(s, s.time, a, s_p)
        done = self.is_terminal(s_p)
        return s_p, r, done

    def instant_reward(self, s, t, a, s_p):
        """
        Return the instant reward for transition s, t, a, s_p
        """
        newrow, newcol = self.to_m(s_p.index)
        newletter = self.desc[newrow, newcol]
        if newletter == b'G':
            return +1.0
        elif newletter == b'H':
            return -1.0
        else:
            return 0.0

    def instant_reward_byindex(self, s_p):
        """
        Return the instant reward for transition s, t, a, s_p
        """
        newrow, newcol = self.to_m(s_p)
        newletter = self.desc[newrow, newcol]
        if newletter == b'G':
            return +1.0
        elif newletter == b'H':
            return -1.0
        else:
            return 0.0

    def instant_reward_bycoordinate(self, s_p):
        """
        Return the instant reward for transition s, t, a, s_p
        """
        #print(s_p)
        s_p = self.__decode_state(s_p, self.state.index, -1)
        #print(s_p)
        newrow, newcol = self.to_m(s_p)
        newletter = self.desc[newrow, newcol]
        if newletter == b'G':
            return +1.0
        elif newletter == b'H':
            return -1.0
        else:
            return 0.0

    def expected_reward(self, s, t, a):
        """
        Return the expected reward function at s, t, a
        """
        R = 0.0
        d = self.transition_probability_distribution(s, t, a)
        for i in range(len(d)):
            s_p = State(i, s.time + self.tau)
            r_i = self.instant_reward(s, t, a, s_p)
            R += r_i * d[i]
        return R

    def is_terminal(self, s):
        """
        Return True if the input state is terminal.
        """
        row, col = self.to_m(s.index)
        letter = self.desc[row, col]
        done = bytes(letter) in b'GH'
        if s.time + self.tau >= self.nT: # Timeout
            done = True
        return done

    #def step(self, a):
    #    s, r, done = self.transition(self.state, a, False)
    #    self.state = s
    #    self.lastaction = a
    #    return (s, r, done, {})

    def step(self, a):
        s, r, done = self.transition(self.state, a, False)
        self.state = s
        self.lastaction = a
        self.t += 1
        return (self.__encode_state(self.state.index), r, done, {})


    def __encode_state(self, state):
        """Converts the state index to normalized grid coordinates."""

        # Convert the single index to 2D grid coordinates
        row = state // self.ncol
        col = state % self.ncol

        # Normalize the coordinates
        norm_row = row / (self.nrow - 1)
        norm_col = col / (self.ncol - 1)

        return np.array([norm_row, norm_col])

    '''
    def __decode_state(self, coordinates):
        """Converts normalized grid coordinates back to the state index."""

        # Denormalize the coordinates
        row = round(coordinates[0] * (self.nrow - 1))
        #print(row)
        col = round(coordinates[1] * (self.ncol - 1))
        #print(col)
        # Convert 2D grid coordinates to a single index
        state = row * self.ncol + col

        return state
    '''

    def __decode_state(self, coordinates, current_state, action):
        """Converts normalized grid coordinates back to the state index based on action."""

        current_row = current_state // self.ncol
        current_col = current_state % self.ncol

        possible_states = []

        if action == -1:
            return current_state

        # Left action possibilities
        if action == 0:
            if current_col > 0:
                possible_states.append(current_state - 1)
            if current_row > 0:
                possible_states.append(current_state - self.ncol)
            if current_row < self.nrow - 1:
                possible_states.append(current_state + self.ncol)

        # Right action possibilities
        elif action == 2:
            if current_col < self.ncol - 1:
                possible_states.append(current_state + 1)
            if current_row > 0:
                possible_states.append(current_state - self.ncol)
            if current_row < self.nrow - 1:
                possible_states.append(current_state + self.ncol)

        # Down action possibilities
        elif action == 1:
            if current_row < self.nrow - 1:
                possible_states.append(current_state + self.ncol)
            if current_col > 0:
                possible_states.append(current_state - 1)
            if current_col < self.ncol - 1:
                possible_states.append(current_state + 1)

        # Up action possibilities
        elif action == 3:
            if current_row > 0:
                possible_states.append(current_state - self.ncol)
            if current_col > 0:
                possible_states.append(current_state - 1)
            if current_col < self.ncol - 1:
                possible_states.append(current_state + 1)

        # Find the closest state to the given coordinates
        min_distance = float('inf')
        best_state = current_state
        for state in possible_states:
            encoded_coordinates = self.__encode_state(state)
            distance = np.linalg.norm(coordinates - encoded_coordinates)
            if distance < min_distance:
                min_distance = distance
                best_state = state

        return best_state

    def one_hot(self, index, length=40):
        length = self.nrow * self.ncol
        array = np.zeros(length)
        array[index] = 1
        return array

    def set_state(self, index, latent_code = 1):
        self.time = latent_code - 1
        self.t = 0
        #self.np_random = np.random.RandomState()
        isd = self.one_hot(index)
        #print("cs",categorical_sample(self.isd, self.np_random))
        self.state = State(categorical_sample(isd, self.np_random), self.time) # (index, time)


    def is_done(self, episode_length=100, state=None):
        """Check if episode is over."""
        if state is None:
            s = self.state
        else:
            s = state
        if (self.t >= episode_length) or self.is_terminal(s):
            return True
        else:
            return False

    def render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.state.index // self.ncol, self.state.index % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
