#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

# This file is part of project pyannote core
# https://github.com/pyannote/DEPRECATED-pyannote-algorithms/blob/develop/pyannote/algorithms/utils/viterbi.py
# As long as this part is no more maintained, this file provides
# updates for compatibility with numpy>=1.24

"""
Constrained Viterbi Decoding Module
------------------------------------
This module implements a constrained version of the Viterbi algorithm, which is used to decode the most probable 
state sequence from a sequence of observations given emission, transition, and (optionally) initial probabilities. 

Key features include:
- Handling of minimum consecutive state constraints:certain states may be required to persist for a minimum number 
  of consecutive frames. This is achieved by duplicating states and updating the transition, initial, emission, and 
  constraint matrices accordingly.
- Incorporation of state-level constraints: states can be forced (mandatory) or forbidden at specific time steps.
- Log-domain probability computations: All probabilities (emission, transition, and initial) are handled in the log 
  domain for numerical stability. A very low log-probability (LOG_ZERO) is used to represent near-zero probabilities.

The main function, viterbi_decoding, performs the forward pass to compute the maximum likelihood of state sequences 
and then uses backtracking to recover the optimal state path. It supports optional constraints on state durations and 
forbidden/mandatory states, making it versatile for applications such as speech recognition, speaker diarization, or any 
sequential decision-making tasks that require constrained decoding.

This file has been updated for compatibility with numpy>=1.24 and is part of the pyannote core project.
"""

from __future__ import unicode_literals

import numpy as np
import itertools

VITERBI_CONSTRAINT_NONE = 0
VITERBI_CONSTRAINT_FORBIDDEN = 1
VITERBI_CONSTRAINT_MANDATORY = 2


LOG_ZERO = np.log(1e-200)

# Handling 'consecutive' constraints is achieved by duplicating states.
# The following functions are here to help in this process.

def _update_transition(transition, consecutive):
    """
    Updates the transition matrix by duplicating states according to the consecutive constraints.
    State duplication is used to enforce a minimum duration for each state by expanding the state space,
    allowing the Viterbi algorithm to account for the requirement that a state must persist for a specified number
    of consecutive frames. The function maps the original transition probabilities to the appropriate entries in
    the duplicated transition matrix.
    """

    # initialize with LOG_ZERO everywhere
    # except on the +1 diagonal np.log(1)
    new_n_states = np.sum(consecutive)
    new_transition = LOG_ZERO * np.ones((new_n_states, new_n_states))
    for i in range(1, new_n_states):
        new_transition[i - 1, i] = np.log(1)

    n_states = len(consecutive)
    boundary = np.hstack(([0], np.cumsum(consecutive)))
    start = boundary[:-1]
    end = boundary[1:] - 1

    for i, j in itertools.product(range(n_states), repeat=2):
        new_transition[end[i], start[j]] = transition[i, j]

    return new_transition


def _update_initial(initial, consecutive):
    """
    Updates the initial probability vector to account for duplicated states based on consecutive constraints.
    This function assigns the original initial probabilities to the first instance of each duplicated state group,
    ensuring that the expanded state space reflects the required minimum duration for each state.
    """

    new_n_states = np.sum(consecutive)
    new_initial = LOG_ZERO * np.ones((new_n_states, ))

    n_states = len(consecutive)
    boundary = np.hstack(([0], np.cumsum(consecutive)))
    start = boundary[:-1]

    for i in range(n_states):
        new_initial[start[i]] = initial[i]

    return new_initial


def _update_emission(emission, consecutive):
    """
    Updates the emission probability matrix to account for duplicated states based on consecutive constraints.
    Each state's emission probabilities are duplicated 'c' times (where c is the required consecutive count) so that
    each duplicated state inherits the same emission characteristics as the original state.
    """
    
    return np.vstack(
        [np.tile(e, (c, 1))  # duplicate emission probabilities c times
        for e, c in zip(emission.T, consecutive)]).T


def _update_constraint(constraint, consecutive):
    """
    Updates the constraint matrix to account for duplicated states based on consecutive constraints.
    Although this function is implemented similarly to _update_emission (i.e., by duplicating each column 'c' times),
    it differs in its purpose: while _update_emission duplicates the emission probabilities to expand the state space,
    this function duplicates the constraint probabilities so that the enforced state constraints are correctly applied
    to each instance of the expanded (duplicated) states.
    """
    
    return np.vstack(
        [np.tile(e, (c, 1))  # duplicate constraint probabilities c times
        for e, c in zip(constraint.T, consecutive)]).T


def _update_states(states, consecutive):
    """
    Converts the expanded (duplicated) state indices from the Viterbi decoding back to the original state labels.
    This mapping is essential after decoding in the expanded state space, where states were duplicated to enforce
    minimum consecutive constraints, so that the final state sequence reflects the original state definitions.
    """

    boundary = np.hstack(([0], np.cumsum(consecutive)))
    start = boundary[:-1]
    end = boundary[1:]

    new_states = np.empty(states.shape)

    for i, (s, e) in enumerate(zip(start, end)):
        new_states[np.where((s <= states) & (states < e))] = i

    return new_states


def viterbi_decoding(emission, transition,
                     initial=None, consecutive=None, constraint=None):
      """
      Constrained Viterbi Decoding
  
      This function computes the most probable sequence of hidden states given a series of observations,
      using the Viterbi algorithm. It supports constraints to enforce minimum consecutive state durations 
      and to restrict certain states at specific times (e.g., forbidding or forcing states).
  
      Parameters:
          emission : numpy.ndarray, shape (n_samples, n_states)
              The log-probabilities for each state at each observation (E[t, i] is the emission log-probability
              of sample t for state i).
          transition : numpy.ndarray, shape (n_states, n_states)
              The log-probabilities of transitioning from one state to another (T[i, j] is the transition log-probability
              from state i to state j).
          initial : optional, numpy.ndarray, shape (n_states,)
              The initial log-probabilities for each state. If not provided, a uniform distribution is assumed.
          consecutive : optional, int or numpy.ndarray, shape (n_states,)
              The minimum number of consecutive observations that each state must persist.
              A value of 1 indicates no constraint. This is used to expand the state space to enforce duration constraints.
          constraint : optional, numpy.ndarray, shape (n_samples, n_states)
              A matrix specifying state constraints at each time step:
                  - 0: no constraint,
                  - 1: state is forbidden,
                  - 2: state is forced.
                  
      Returns:
          numpy.ndarray, shape (n_samples,)
              The most probable sequence of states after applying the constrained Viterbi decoding.
      
      The function adjusts the emission, transition, initial, and constraint matrices by duplicating states to satisfy the
      minimum consecutive constraints. It then performs a forward pass to compute the optimal path probabilities, followed by 
      backtracking to recover the best state sequence, and finally maps the duplicated state indices back to the original states.
      """

    # ~~ INITIALIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    T, k = emission.shape  # number of observations x number of states

    # no minimum-consecutive-states constraints
    if consecutive is None:
        consecutive = np.ones((k, ), dtype=int)

    # same value for all states
    elif isinstance(consecutive, int):
        consecutive = consecutive * np.ones((k, ), dtype=int)

    # (potentially) different values per state
    else:
        consecutive = np.array(consecutive, dtype=int).reshape((k, ))

    # at least one sample
    consecutive = np.maximum(1, consecutive)

    # balance initial probabilities when they are not provided
    if initial is None:
        initial = np.log(np.ones((k, )) / k)

    # no constraint?
    if constraint is None:
        constraint = VITERBI_CONSTRAINT_NONE * np.ones((T, k))

    # artificially create new states to account for 'consecutive' constraints
    emission = _update_emission(emission, consecutive)
    transition = _update_transition(transition, consecutive)
    initial = _update_initial(initial, consecutive)
    constraint = _update_constraint(constraint, consecutive)
    T, K = emission.shape  # number of observations x number of new states
    states = np.arange(K)  # states 0 to K-1

    # set emission probability to zero for forbidden states
    emission[
        np.where(constraint == VITERBI_CONSTRAINT_FORBIDDEN)] = LOG_ZERO

    # set emission probability to zero for all states but the mandatory one
    for t, k in zip(
        *np.where(constraint == VITERBI_CONSTRAINT_MANDATORY)
    ):
        emission[t, states != k] = LOG_ZERO

    # ~~ FORWARD PASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    V = np.empty((T, K))                # V[t, k] is the probability of the
    V[0, :] = emission[0, :] + initial  # most probable state sequence for the
                                        # first t observations that has k as
                                        # its final state.

    P = np.empty((T, K), dtype=int)  # P[t, k] remembers which state was used
    P[0, :] = states                 # to get from time t-1 to time t at
                                     # state k

    for t in range(1, T):

        # tmp[k, k'] is the probability of the most probable path
        # leading to state k at time t - 1, plus the probability of
        # transitioning from state k to state k' (at time t)
        tmp = (V[t - 1, :] + transition.T).T

        # optimal path to state k at t comes from state P[t, k] at t - 1
        # (find among all possible states at this time t)
        P[t, :] = np.argmax(tmp, axis=0)

        # update V for time t
        V[t, :] = emission[t, :] + tmp[P[t, :], states]

    # ~~ BACK-TRACKING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    X = np.empty((T,), dtype=int)
    X[-1] = np.argmax(V[-1, :])
    for t in range(1, T):
        X[-(t + 1)] = P[-t, X[-t]]

    # ~~ CONVERT BACK TO ORIGINAL STATES

    return _update_states(X, consecutive)
