# -*- coding: utf-8 -*-
"""
staircase.py
Authors: mtraub, steele94
Description: contains classes for easily performing simulations and
calculations about the behavior of erosion processes for
18.821 Project #2. 
Includes:
    ErosionProcess
    State
    ncr
"""

import numpy as np 
from scipy.ndimage.interpolation import shift
import scipy.sparse as sparse
import operator as op
from abc import abstractmethod, ABCMeta
import sympy

class ErosionProcess(object):
    """
    Functions for simulating and analyzing erosion process
    """

    def __init__(self, height, width, rng=None, pmf=None):
        """
        rng is a random number generation function to produce number [0, 1)
            for where particles collide (default: uniform)
        pmf is a density function for the probability of a particle hitting
            at some height [0, height) (default: uniform)
        """
        self.height = height
        self.width = width
        if rng is None and pmf is None:
            pmf = lambda x: 1./ self.height
            rng = np.random.random
        elif rng is None or pmf is None:
            raise ValueError("rng and pmf must be provided if either is not None")
        self.rng = rng
        self.pmf = pmf

    def initial_state(self):
        return State(self.height, self.width)

    def step(self, state):
        """
        Iterates one step of the simulation
        Input: state: State object
        Output: new State object
        """
        particle = int(self.rng() * self.height) # draw a particle index
        return state.collision(particle)

    def transitions(self, state):
        """
        Given a state, returns dictionary of all transitions like
        {new_state : Pr[new_state @ t+1 | state @ t], ...}
        """
        colls = state.all_collisions()
        result = dict()
        for c in colls:
            result[state.collision(c)] = self.pmf(c)
        return result

    def time_until_decay(self, p = .5, max_T = 1000):
        """
        Returns probability distribution over T, where T is the 
        time where <= p fraction of initial structure is left
        """
        dist = {self.initial_state() : 1.}
        pT = np.zeros(max_T)
        absorbing = 0.
        for t in range(max_T):
            new_dist = dict()
            pT[t] += absorbing
            print t, len(dist)
            for s, prob in dist.items():
                p_newS_given_S = self.transitions(s)
                for newS, trans_prob in p_newS_given_S.items():
                    if newS.pct_area() <= p:
                        absorbing += prob * trans_prob
                    elif newS in new_dist:
                        new_dist[newS] += prob * trans_prob
                    else:
                        new_dist[newS] = prob * trans_prob
            dist = new_dist
        return pT

    def matrices(self):
        T = np.zeros((self.height * self.width + 1, self.height * self.width + 1))
        visited = set()
        queue = set([self.initial_state()])
        while len(queue) > 0 :
            s = queue.pop()
            C = s.num_blocks()
            p_newS_given_S = self.transitions(s)
            for newS, trans_prob in p_newS_given_S.items():
                if newS not in visited:
                    Cnew = s.num_blocks()
                    T[Cnew, C] += trans_prob
                    queue.add(newS)
                    visited.add(newS)
        return T

    def enum_states(self):
        """
        Provides a list of all state indices
        """
        return list(range(self.height * self.width))

    def run_sim(self, T, initial=None):
        """
        Runs a simulation of process with T time steps
        Inputs:
            T: number of time steps
            initial: if initial is None, then initial_state() state used
        Returns: length T list of states
        """
        if initial is None:
            initial = self.initial_state()
        elif not isinstance(initial, State):
            raise ValueError("initial must be a State object")

        path = [initial]
        for t in range(1, T):
            path += [self.step(path[t-1])]
        return path


class State(object):
    """
    Stores a state and provides helper functions for transitioning the state
    and extracting attributes, e.g. open spots to right, of state
    """

    def __init__(self, height, width, values=None):
        """
        values: list, numpy array, or string (e.g. "11000111") containing the values
        cycle: if True, then periodic boundary condition (default True)
        """
        self.height = height
        self.width = width
        self.is_wide = width > height 
        if values is None:
            if self.is_wide:
                values = np.ones(self.height) * self.width
            else:
                values = np.zeros(self.width + 1)
                values[-1] = self.height
        if self.is_wide:
            assert len(values) == self.height
        else:
            assert len(values) == self.width + 1
            assert sum(values) == self.height
        self.values = values

    def collision(self, i):
        updated = self.values.copy()
        if self.is_wide:
            level = self.values[i]
            while i < len(updated) and updated[i] == level:
                if updated[i] == 0:
                    break
                else:
                    updated[i] -= 1
                    i += 1
        else:
            j = -1
            while -j < len(updated) and updated[j] <= i:
                i -= updated[j]
                j -= 1
            if 0 < len(updated) + j:
                updated[j - 1] += updated[j] - i
                updated[j] = i
        return State(self.height, self.width, values=updated)

    def effective_width(self):
        if self.is_wide:
            nonempty = self.values[self.values > 0]
            if len(nonempty) > 0:
                return max(nonempty) - min(nonempty)
            else:
                return 0  
        else:
            nonempty = np.where(self.values[1:] > 0)[0]
            if len(nonempty) > 0:
                return max(nonempty) - min(nonempty)
            else:
                return 0 

    def effective_height(self):
        if self.is_wide:
            nonempty = np.where(self.values > 0)[0]
            if len(nonempty) > 0:
                return max(nonempty) - min(nonempty) + 1
            else:
                return 0
        else:
            return self.height - self.values[0]

    def num_blocks(self):
        if self.is_wide:
            return np.sum(self.values)
        else:
            return np.sum(self.values * range(len(self.values)))

    def pct_area(self):
        return 1. * self.num_blocks() / (self.height * self.width)

    def all_collisions(self):
        return list(range(self.height))

    def matrix(self):
        S = np.zeros((self.height, self.width))
        if self.is_wide:
            for h in range(len(self.values)):
                S[h, :self.values[h]] = 1
        else:
            h = 0
            for w in range(len(self.values)):
                S[h:(h+self.values[w]), :w] = 1
                h = h+self.values[w]
        return S

    def copy(self):
        """
        Returns new instance of State object with identical attributes
        """ 
        return State(self.height, self.width, self.values)

    def state_index(self):
        """
        Returns an integer repesenting a state
        """
        if self.is_wide:
            index = 0
            for i in range(self.height):
                index += self.values[i] * self.width**i
        else:
            i = 0
            for w in range(len(self.values)):
                for j in range(self.values[w]):
                    index += w * (self.width)**(i+j)
                i += self.values[w]
        return index

    def __equals__(self, other):
        return self.values == other.values

    def __repr__(self):
        return "#{" + str(self.values) + "}#"

    def __str__(self):
        return repr(self)

    def __len__(self):
        return len(self.values)

    def __eq__(self, other):
        return all(self.values == other.values)

    def __hash__(self):
        return str(self).__hash__()

def ncr(n, r):
    """
    Returns "n choose r"
    """
    if r < 0 or r > n:
        return 0
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom
