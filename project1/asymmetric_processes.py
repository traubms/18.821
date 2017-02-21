"""
asymmetric_processes.py
Description: contains classes for easily performing simulations and
calculations about the behavior of asymmetric random processes for
18.821 Project #1. 
Includes:
    AsymmetricProcess
    CirculalrProcess
    BoundaryProcess
    State
    ncr
"""

import numpy as np 
from scipy.ndimage.interpolation import shift
import scipy.sparse as sparse
import operator as op
from abc import abstractmethod, ABCMeta

class AsymmetricProcess(object):
    """
    Absract class that provides functionality that both
    CircleProcess and BoundaryProcess use
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def random_state(self):
        """
        Returns a random state (State object)
        """
        pass

    @abstractmethod
    def step(self, state):
        """
        Iterates one step of the simulation
        Input: state: State object
        Output: new State object
        """
        pass

    @abstractmethod
    def transitions(self, state):
        """
        Given a state, returns dictionary of all transitions like
        {str(new_state) : Pr[new_state @ t+1 | state @ t], ...}
        """
        pass

    @abstractmethod
    def state_index(self, state_str):
        """
        Given a state string, e.g.  "0010011", maps it to an integer
        """
        pass

    @abstractmethod
    def enum_states(self):
        """
        Provides a list of all state strings
        """
        pass

    def run_sim(self, T, initial=None):
        """
        Runs a simulation of process with T time steps
        Inputs:
            T: number of time steps
            initial: if initial is None, then random state used. Otherwise, initial is beginning state (string, list or numpy array)
        Returns: T x N matrix of states
        """
        if initial is None:
            initial = self.random_state()
        else:
            initial = self.castToState(initial)

        result = np.zeros((T, self.N))
        result[0, :] = initial.values
        state = initial
        for t in range(1, T):
            state = self.step(state)
            result[t, :] = state.values
        return result

    def run_sims(self, iters, T, initial=None):
        """
        Runs iters number of simulations of process with T time steps
        Inputs:
            iters: number of simulations to run
            T: number of time steps
            initial: if initial is None, then random state used. Otherwise, initial is beginning state (string, list or numpy array)
        Returns: iters x T x N 3D-matrix of states
        """
        result = np.zeros((iters, T, self.N))
        for i in range(iters):
            result[i, :, :] = self.run_sim(T, initial)
        return result

    def matrices(self):
        """
        Let M be the number of possible states. Returns N x M State matrix
        and M x M transition matrix
        Outputs:
            S: N x M state matrix
            T: M x M transition matrix
            V: M x M transition matrix weighted by transition "speeds" (+1 move right, -1 move left)
        """
        states = self.enum_states()
        M = len(states)
        transitions_speeds = [(s, self.transitions(State(s)), self.speeds(State(s))) for s in states]
        S = np.zeros((self.N, M))
        T = sparse.csr_matrix((M, M))
        V = sparse.csr_matrix((M, M))
        
        k = 0
        for s, t, v in transitions_speeds:
            i = self.state_index(s)
            S[:, i] = State(s).values
            for r in t:
                j = self.state_index(r)
                T[j, i] = t[r]
                if r in v:
                    V[j, i] = v[r]
                k += 1
        V = T.multiply(V) # speed weighted by probabilities
        return S, T, V

    def castToState(self, state):
        """
        Double checks that it is state object of correct length
        """
        if not isinstance(state, State):
            s = State(state)
        else:
            s = state
        if len(state) != self.N:
            raise ValueError("Input state is not length %s: %s" % (self.N, state))
        return s


class CircleProcess(AsymmetricProcess):
    """
    Defines the asymmetric process with periodic boundary
    conditions and fixed number of particles
    """

    def __init__(self, N, k, q):
        """
        N: number of positions
        k: number of particles
        q: asymmetry parameter; 0 <= q <= 1
        """
        self.N = N
        self.k = k
        self.q = q
        self.memo = dict()

    def random_state(self):
        """
        Returns a random state (State object)
        """
        state = np.zeros(self.N)
        state[np.random.choice(self.N, self.k, replace=False)] = 1
        return State(state, cycle=True)

    def step(self, state):
        """
        Iterates one step of the simulation
        Input: state: State object
        Output: new State object
        """
        n_open = len(state.open_spots_to_right())
        r = np.random.random()
        if r * self.N < n_open:
            return state.move_right()
        elif r * self.N < n_open * (1 + self.q):
            return state.move_left()
        else:
            return state.copy()

    def transitions(self, state):
        """
        Given a state, returns dictionary of all transitions like
        {str(new_state) : Pr[new_state @ t+1 | state @ t], ...}
        """
        state = self.castToState(state)
        open_r = state.open_spots_to_right()
        open_l = state.open_spots_to_left()
        transitions = dict()
        for new in open_r:
            transitions[str(state.move_right(new))] = 1. / self.N
        if self.q > 0:
            for new in open_l:
                new_state = str(state.move_left(new))
                if new_state in transitions:
                    transitions[new_state] += 1. * self.q / self.N
                else:
                    transitions[new_state] = 1. * self.q / self.N
        transitions[str(state)] = 1. - (1. + self.q) / self.N * len(open_r)
        return transitions

    def speeds(self, state):
        """
        Given a state, returns dictionary of all transitions like
        {str(new_state) : Speed of going from state to new_state (either +1, 0, -1), ...}
        """
        state = self.castToState(state)
        open_r = state.open_spots_to_right()
        open_l = state.open_spots_to_left()
        speeds = dict()
        for new in open_r:
            speeds[str(state.move_right(new))] = 1.
        if self.q > 0:
            for new in open_l:
                new_state = str(state.move_left(new))
                speeds[new_state] = -1.
        speeds[str(state)] = 0
        return speeds


    def state_index(self, state_str):
        """
        Given a state string, e.g.  "0010011", maps it to an integer.
        Don't worry about the math.
        """
        if not isinstance(state_str, str): #just in case State object given
            state_str = str(self.castToState(state_str))
        N = len(state_str)
        k = 0
        for c in state_str: # Count 1s
            if c == "1":
                k += 1
        index = 0
        for i in range(len(state_str)): # Find index
            c = state_str[i]
            if c == "0" and k > 0:
                index += ncr(N-1, k-1)
            else:
                k -= 1
            N -= 1
        return index

    def enum_states(self, N=None, k=None):
        """
        Provides a list of all state strings
        """
        if N is None and k is None:
            N = self.N
            k = self.k
        if N == 0:
            return [""]
        elif (N, k) in self.memo:
            return self.memo[(N, k)]
        else:
            results = []
            if k > 0:
                results += ["1" + r for r in self.enum_states(N-1, k-1)]
            if k < N:
                results += ["0" + r for r in self.enum_states(N-1, k)]
            self.memo[(N, k)] = results
            return results

    # def speeds(self, sim):
    #     """
    #     Given a simulation, calculates if each at each step some particle
    #     moved left (-1), right (+1) or no change (0)
    #     Inputs: T x N simulation matrix (from run_sim)
    #     Outputs: (T-1) vector of speeds
    #     """
    #     N = sim.shape[1]
    #     com = np.dot(sim , np.array(range(N))) / N
    #     speeds = (((com - np.roll(com, 1))[1:] + .5) % 1 - .5)
    #     return speeds * N

class BoundaryProcess(AsymmetricProcess):
    """
    Defines asymmetric Boundary process where particles
    enter from left and leave on right
    """

    def __init__(self, N, q, a, b):
        """
        N: number of positions
        q: asymmetry parameter, 0 <= q <= 1
        a: enter-from-left parameter, 0 <= a <= 1
        b: exit-from-right parameter, 0 <= b <= 1
        """
        self.N = N
        self.q = q
        self.a = a
        self.b = b

    def random_state(self):
        """
        Turns each position on with probability 1/2
        """
        state = np.zeros(self.N)
        state[np.random.random(self.N) > .5] = 1
        return State(state, cycle=False)

    def step(self, state):
        """
        Iterates one step of the simulation
        Input: state: State object
        Output: new State object
        """
        n_open_l = state.num_open_to_left()
        n_open_r = state.num_open_to_right()
        r = np.random.random() * (self.N + 1)
        if r < n_open_r:
            return state.move_right()
        elif r < n_open_r + self.q * n_open_l:
            return state.move_left()
        elif state.empty_leftmost() and  r < n_open_r + self.q * n_open_l + self.a:
            return state.enter_from_left()
        elif not state.empty_rightmost() and  r < n_open_r + self.q * n_open_l + self.a + self.b:
            return state.exit_from_right()
        else:
            return state.copy()

    def transitions(self, state):
        """
        Given a state, returns dictionary of all transitions like
        {str(new_state) : Pr[new_state @ t+1 | state @ t], ...}
        """
        #raise ValueError("You need to implement this!!!")

        state = self.castToState(state)
        open_r = state.open_spots_to_right()
        open_l = state.open_spots_to_left()
        transitions = {}

        #rightmost particle leaves system
        if state.empty_rightmost() == False:
            transitions[str(state.exit_from_right())] = self.b / (self.N + 1)

        #particle enters system if there is space on the left
        if state.empty_leftmost() == True:
            transitions[str(state.enter_from_left())] = self.a / (self.N + 1)

        #transitions where a particle moves right that is not the rightmost particle
        for new in open_r:
            transitions[str(state.move_right(new))] = 1. / (self.N + 1)

        #transitions where a particle moves left that is not in the leftmost position
        if self.q > 0:
            for new in open_l:
                new_state = str(state.move_left(new))
                if new_state in transitions:
                    transitions[new_state] += 1. * self.q / (self.N + 1)
                else:
                    transitions[new_state] = 1. * self.q / (self.N + 1)

        transitions[str(state)] = 1. - sum(transitions.values())
        
        return transitions


    def state_index(self, state_str):
        """
        Given a state string, e.g.  "0010011", maps it to an integer
        Maps it to the binary -> decimal mapping
        """
        if not isinstance(state_str, str): #just in case State object given
            state_str = str(self.castToState(state_str))
        index = 0
        for i in range(len(state_str)):
            l = state_str[-(i+1)]
            if l == "1":
                index += 2**i
        return index

    def enum_states(self, N=None):
        """
        Provides a list of all state strings
        This will be length 2^N
        """
        if N is None:
            N = self.N
        results = []
        for i in range(2**N):
            s = bin(i)[2:]
            results += ["0" * (N - len(s)) + s]
        return results

    def speeds(self, state):
        """
        Given a state, returns dictionary of all transitions like
        {str(new_state) : Speed of going from state to new_state (either +1, 0, -1), ...}
        """
        state = self.castToState(state)
        open_r = state.open_spots_to_right()
        open_l = state.open_spots_to_left()
        speeds = dict()
        for new in open_r:
            speeds[str(state.move_right(new))] = 1.
        if self.q > 0:
            for new in open_l:
                new_state = str(state.move_left(new))
                speeds[new_state] = -1.
        if state.empty_leftmost() == True and self.a > 0:
            speeds[str(state.enter_from_left())] = 1.
        if not state.empty_rightmost() and self.b > 0:
            speeds[str(state.exit_from_right())] = 1.
        speeds[str(state)] = 0
        return speeds


        
class State:
    """
    Stores a state and provides helper functions for transitioning the state
    and extracting attributes, e.g. open spots to right, of state
    """

    def __init__(self, values, cycle=True):
        """
        values: list, numpy array, or string (e.g. "11000111") containing the values
        cycle: if True, then periodic boundary condition (default True)
        """
        if isinstance(values, str):
            values = list(values)
        self.values = np.array(values).astype(int)
        self.cycle = cycle
        self.N = len(values)
        self.eps = 1e-14

    def open_spots(self):
        """
        Returns indices of not-occupied spots
        """
        return np.where(self.values > self.eps)[0]

    def open_spots_to_right(self):
        """
        Returns indicies of not-occupied spots that are 
        to the right of occupied spots
        """
        if self.cycle:
            s = 1 - self.values
            r = s - np.roll(s, 1)
        else:
            z = self.values
            r = shift(z, 1, cval=0) - z
        return np.where(r > self.eps)[0]

    def open_spots_to_left(self):
        """
        Returns indicies of not-occupied spots that are 
        to the left of occupied spots
        """
        if self.cycle:
            s = 1 - self.values
            r = s - np.roll(s, -1)
        else:
            z = self.values
            r = shift(z, -1, cval=0) - z
        return np.where(r > self.eps)[0]

    def num_open_to_left(self):
        """
        Returns number of not-occupied spots that are 
        to the left of occupied spots
        """
        if self.cycle:
            s = 1 - self.values
            r = s - np.roll(s, -1)
        else:
            z = self.values
            r = shift(z, -1, cval=0) - z
        return np.sum(r[r > self.eps])

    def num_open_to_right(self):
        """
        Returns number of not-occupied spots that are 
        to the right of occupied spots
        """
        if self.cycle:
            s = 1 - self.values
            r = s - np.roll(s, 1)
        else:
            z = self.values
            r = shift(z, 1, cval=0) - z
        return np.sum(r[r > self.eps])

    def move_right(self, new=None):
        """
        Randomly elects one of the valid spots to move 
        right to and returns a new State with that transition
        Input: 
            new: the new location to fill from right;
                if spot not valid move, ValueError thrown;
                move_right(State("1010"), 1) --> State("0110")
        Output: new State object
        """
        spots = self.open_spots_to_right()
        if new is None:
            new = np.random.choice(spots, 1)
        else:
            if new not in spots:
                raise ValueError("Cannot move right to %s: %s" % (new, self))
        new_values = self.values.copy()
        new_values[new] = 1
        new_values[(new - 1) % self.N] = 0
        return State(new_values, self.cycle)

    def move_left(self, new=None):
        """
        Randomly elects one of the valid spots to move 
        left to and returns a new State with that transition
        Input: 
            new: the new location to fill from left;
                if spot not valid move, ValueError thrown;
                move_right(State("1010"), 1) --> State("1100")
        Output: new State object
        """
        spots = self.open_spots_to_left()
        if new is None:
            new = np.random.choice(spots, 1)
        else:
            if new not in spots:
                raise ValueError("Cannot move left to %s: %s" % (new, self))
        new_values = self.values.copy()
        new_values[new] = 1
        new_values[(new + 1) % self.N] = 0
        return State(new_values, self.cycle)

    def enter_from_left(self):
        """
        Returns new state with left-most position
        being filled with 1
        """
        new_values = self.values.copy()
        if new_values[0] > self.eps:
            raise ValueError("Cannot enter from left: %s" % new_values)
        new_values[0] = 1
        return State(new_values, self.cycle)

    def exit_from_right(self):
        """
        Returns new state with right-most position
        being filled with 0
        """
        new_values = self.values.copy()
        if new_values[-1] < 1:
            raise ValueError("Cannot exit from right: %s" % new_values)
        new_values[-1] = 0
        return State(new_values, self.cycle)

    def empty_leftmost(self):
        """
        Returns boolean if leftmost position is empty
        """
        return self.values[0] == 0

    def empty_rightmost(self):
        """ 
        Returns boolean if rightmost position is empty
        """
        return self.values[-1] == 0

    def copy(self):
        """
        Returns new instance of State object with identical attributes
        """
        return State(self.values.copy(), self.cycle)

    def __repr__(self):
        return "".join(self.values.astype(int).astype(str))

    def __str__(self):
        return repr(self)

    def __len__(self):
        return len(self.values)

def ncr(n, r):
    """
    Returns "n choose r"
    """
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom



