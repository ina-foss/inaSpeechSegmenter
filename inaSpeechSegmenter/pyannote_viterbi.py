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

from __future__ import unicode_literals

import numpy as np
import itertools

VITERBI_CONSTRAINT_NONE = 0
VITERBI_CONSTRAINT_FORBIDDEN = 1
VITERBI_CONSTRAINT_MANDATORY = 2


LOG_ZERO = np.log(1e-200)

# handling 'consecutive' constraints is achieved by duplicating states
# the following functions are here to help in this process


# create new transition prob. matrix accounting for duplicated states.
def _update_transition(transition, consecutive):

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


# create new initial prob. matrix accounting for duplicated states.
def _update_initial(initial, consecutive):

    new_n_states = np.sum(consecutive)
    new_initial = LOG_ZERO * np.ones((new_n_states, ))

    n_states = len(consecutive)
    boundary = np.hstack(([0], np.cumsum(consecutive)))
    start = boundary[:-1]

    for i in range(n_states):
        new_initial[start[i]] = initial[i]

    return new_initial


# create new emission prob. matrix accounting for duplicated states.
def _update_emission(emission, consecutive):
    
    return np.vstack(
        [np.tile(e, (c, 1))  # duplicate emission probabilities c times
        for e, c in zip(emission.T, consecutive)]).T


# create new constraint matrix accounting for duplicated states
def _update_constraint(constraint, consecutive):
    
    return np.vstack(
        [np.tile(e, (c, 1))  # duplicate constraint probabilities c times
        for e, c in zip(constraint.T, consecutive)]).T


# convert sequence of duplicated states back to sequence of original states.
def _update_states(states, consecutive):

    boundary = np.hstack(([0], np.cumsum(consecutive)))
    start = boundary[:-1]
    end = boundary[1:]

    new_states = np.empty(states.shape)

    for i, (s, e) in enumerate(zip(start, end)):
        new_states[np.where((s <= states) & (states < e))] = i

    return new_states


def viterbi_decoding(emission, transition,
                     initial=None, consecutive=None, constraint=None):
    """(Constrained) Viterbi decoding

    Parameters
    ----------
    emission : array of shape (n_samples, n_states)
        E[t, i] is the emission log-probabilities of sample t at state i.
    transition : array of shape (n_states, n_states)
        T[i, j] is the transition log-probabilities from state i to state j.
    initial : optional, array of shape (n_states, )
        I[i] is the initial log-probabilities of state i.
        Defaults to equal log-probabilities.
    consecutive : optional, int or int array of shape (n_states, )
        C[i] is a the minimum-consecutive-states constraint for state i.
        C[i] = 1 is equivalent to no constraint (default).
    constraint : optional, array of shape (n_samples, n_states)
        K[t, i] = 1 forbids state i at time t.
        K[t, i] = 2 forces state i at time t.
        Use K[t, i] = 0 for no constraint (default).

    Returns
    -------
    states : array of shape (n_samples, )
        Most probable state sequence

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
