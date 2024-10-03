import numpy as np
import sys
from utils.distribution import wass_dual
from six import StringIO
from gym import Env, spaces, utils
import math
import matplotlib.pyplot as plt

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

#MAPS = {
#    "bridge": [
#        "HHHHHHHHHHHH",
#        "FFFFFFFFFFFF",
#        "GFFFFSFFFFFG",
#        "FFFFFFFFFFFH",
#        "HHHHHHHHHHHH"
#    ]
#}

MAPS = {
    "bridge": [
        "HHHHHHHH",
        "FFFFFHHH",
        "GFFFSFFG",
        "FFFFFHHH",
        "HHHHHHHH"
    ]
}

#MAPS = {
#    "bridge": [
#        "SFFFFF",
#        "FHFFFH",
#        "FFFFFH",
#        "HFFFFG"
#    ]
#}


class State:
    """
    State class
    """
    def __init__(self, index, time):
        self.index = index
        self.time = time

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class NSBridgeV0(Env):
    """
    Non Stationary grid-world representing a bridge.
    As time goes by, it gets slippery.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, time=0, desc=None, map_name="bridge", is_slippery=True):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        print(self.nrow, self.ncol)
        self.num_actions = 4
        self.nS = nrow * ncol # n states
        self.nA = 4 # n actions
        self.nT = 10 # n timesteps
        self.action_space = spaces.Discrete(self.nA)
        self.is_slippery = is_slippery
        self.tau = 1 # timestep duration
        self.L_p = 1.0
        self.L_r = 0.0
        self.epsilon = 0.5 # 0: left part of the bridge is slippery; 1: right part of the bridge is slippery; in between: mixture of both
        self.T = self.generate_transition_matrix()
        isd = np.array(self.desc == b'S').astype('float64').ravel()
        self.isd = isd / isd.sum()
        print("isd",self.isd )
        #self._seed(0)

        #self.np_random = np.random.RandomState(1993)
        self.np_random = np.random.RandomState()
        self.time = time
        self.reset()
        self.t = 0
    def set_epsilon(self, epsilon):
        print('Epsilon set to:', epsilon)
        self.epsilon = epsilon
        self.T = self.generate_transition_matrix()

    def _seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    #def __encode_state(self, state):
    #    """One-hot encodes the integer action supplied."""
    #    a = np.array([0] * self.nS)
    #    a[state] = 1
    #    return a
    def one_hot(self, index, length=40):
        length = self.nrow * self.ncol
        array = np.zeros(length)
        array[index] = 1
        return array

    def __encode_state(self, state):
        """Converts the state index to normalized grid coordinates."""

        # Convert the single index to 2D grid coordinates
        row = state // self.ncol
        col = state % self.ncol

        # Normalize the coordinates
        norm_row = row / (self.nrow - 1)
        norm_col = col / (self.ncol - 1)

        return np.array([norm_row, norm_col])

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
            if current_row < self.nrow - 1:
                possible_states.append(current_state + self.ncol)
            if current_row > 0:
                possible_states.append(current_state - self.ncol)
            if current_col == 0:
                possible_states.append(current_state)
        # Right action possibilities
        elif action == 2:
            if current_col < self.ncol - 1:
                possible_states.append(current_state + 1)
            if current_col == self.ncol - 1:
                possible_states.append(current_col)
            if current_row < self.nrow - 1:
                possible_states.append(current_state + self.ncol)
            if current_row > 0:
                possible_states.append(current_state - self.ncol)

        # Down action
        elif action == 1:
            if current_row > 0:
                possible_states.append(current_state - self.ncol)
            if current_row < self.nrow - 1:
                possible_states.append(current_state + self.ncol)

        # Up action
        elif action == 3:
            if current_row < self.nrow - 1:
                possible_states.append(current_state + self.ncol)
            if current_row > 0:
                possible_states.append(current_state - self.ncol)

        min_distance = float('inf')
        best_state = current_state
        for state in possible_states:
            encoded_coordinates = self.__encode_state(state)
            distance = np.linalg.norm(coordinates - encoded_coordinates)
            if distance < min_distance:
                min_distance = distance
                best_state = state

        return best_state

    '''
    def __decode_state(self, coordinates):
        """Converts normalized grid coordinates back to the state index."""
        import math
        # Denormalize the coordinates
        row = round(coordinates[0] * (self.nrow - 1))
        #print(row)
        col = round(coordinates[1] * (self.ncol - 1))
        #print(col)
        # Convert 2D grid coordinates to a single index
        state = row * self.ncol + col

        return state
    '''

    # Convert map letters to colors
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



    def observe(self):
        """Return current state."""
        return self.__encode_state(self.state.index)
        #return self.state.index


    def reset(self, latent_code = 1):
        """
        Reset the environment.
        IMPORTANT: Does not create a new environment.
        """
        #if latent_code == 1:
        #    self.time = 0
        #elif latent_code == 2:
        self.time = latent_code - 1
        self.t = 0
        #self.np_random = np.random.RandomState()
        #print("cs",categorical_sample(self.isd, self.np_random))
        self.state = State(categorical_sample(self.isd, self.np_random), self.time) # (index, time)
        self.lastaction = None # for rendering
        return self.state

    def set_state(self, index, latent_code = 1):
        self.time = latent_code - 1
        self.t = 0
        #self.np_random = np.random.RandomState()
        isd = self.one_hot(index)
        #print("cs",categorical_sample(self.isd, self.np_random))
        self.state = State(categorical_sample(isd, self.np_random), self.time) # (index, time)


    def display(self):
        print('Displaying NSFrozenLakeEnv-v0')
        print('map       :')
        print(self.desc)
        print('n states  :', self.nS)
        print('n actions :', self.nA)
        print('timeout   :', self.nT)

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
            #for b in [(a-1)%4, a, (a+1)%4]:
            for b in range(4):
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

    def generate_transition_matrix(self):
        T = np.zeros(shape=(self.nS, self.nA, self.nT, self.nS), dtype=float)
        for s in range(self.nS):
            row, col = self.to_m(s)
            letter = self.desc[row, col]
            if letter != b'H': # s is not a Hole
                for a in range(self.nA):
                    T[s,a,0,:] = np.zeros(shape=self.nS)
                    #row, col = self.to_m(s)
                    row_p, col_p = self.inc(row, col, a)
                    s_p = self.to_s(row_p, col_p)
                    T[s,a,0,s_p] = 1.0
                    rs = self.reachable_states(s, a)
                    nrs = sum(rs)
                    if nrs == 1:
                        T[s,a,:,:] = T[s,a,0,:]
                    else:
                        w0 = np.array(T[s,a,0,:])
                        wsat = np.zeros(shape=w0.shape)
                        #print(wsat.shape)
                        if col < 4: #left part
                            wsat[s_p] = 0.1 * (1 - self.epsilon) + 0.9 * self.epsilon
                        else: #right part
                            wsat[s_p] = 0.9 * (1 - self.epsilon) + 0.1 * self.epsilon
                        wslip = (1 - wsat[s_p]) / 2
                        s_up = self.to_s(row-1, col)
                        s_dw = self.to_s(row+1, col)
                        wsat[s_up] += wslip
                        wsat[s_dw] += wslip #0.25
                        D = self.distances_matrix(range(self.nS))
                        l = self.tau * self.L_p / wass_dual(w0, wsat, D)

                        #print("l",l)
                        for t in range(1, self.nT):
                            if l * t < 1.0: #if fully bounded by LC: 1 <= l * t, this condition allows gradually move towards target function
                                T[s,a,t,:] = (1 - l * t) * w0 + t * l * wsat
                            else:
                                T[s,a,t,:] = wsat
                            '''
                            if t == 1 and s == 21 and a == 0:
                                print(s,a,T[s,a,t,:])
                            if t == 1 and s == 22 and a == 0:
                                print(s,a,T[s,a,t,:])
                            if t == 1 and s == 21 and a == 2:
                                print(s,a,T[s,a,t,:])
                            if t == 1 and s == 22 and a == 2:
                                print(s,a,T[s,a,t,:])
                            '''
                            #print(T[s,a,t,:])
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
        drs = []
        for i in range(len(rs)):
            if rs[i] == 1:
                drs.append(State(i, s.time))
        assert (len(drs) == sum(rs))
        return np.array(drs)

    def transition(self, s, a, is_model_dynamic=True):
        """
        Transition operator, return the resulting state, reward and a boolean indicating
        whether the termination criterion is reached or not.
        The boolean is_model_dynamic indicates whether the temporal transition is applied
        to the state vector or not.
        """
        print("time",s.time)
        #print(is_model_dynamic)
        d = self.transition_probability_distribution(s, s.time, a)
        p_p = categorical_sample(d, self.np_random)
        if is_model_dynamic:
            s_p = State(p_p, s.time + self.tau)
        else:
            s_p = State(p_p, s.time)
        r = self.instant_reward(s_p)
        done = self.is_terminal(s_p)
        return s_p, r, done

    def instant_reward(self, s_p):
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
            r_i = self.instant_reward(s_p)
            R += r_i * d[i]
        return R

    def is_terminal(self, s):
        """
        Return True if the input state is terminal.
        """
        row, col = self.to_m(s.index)
        letter = self.desc[row, col]
        done = bytes(letter) in b'GH'
        return done

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

    def step(self, a):
        s, r, done = self.transition(self.state, a, False)
        self.state = s
        self.lastaction = a
        self.t += 1
        return (self.__encode_state(self.state.index), r, done, {})

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


