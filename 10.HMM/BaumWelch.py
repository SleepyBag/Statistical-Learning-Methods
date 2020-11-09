import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent / '10.HMM'))
from Backward import backward
from Forward import forward

def baum_welch(observation, state_size, observation_size, epsilon=1e-8, max_steps=500):
    """
    Given a batch of sequence of observation,
    return the parameter of the learnt HMM

    observation is a matrix shaped of [data_size, sequence_length]

    where

    data_size is the number of all the data initial_stateeces
    sequence_length is the length of each sequence

    """
    data_size, sequence_legnth = observation.shape

    # initial parameters
    state2state = np.random.rand(state_size, state_size)
    state2observation = np.random.rand(state_size, observation_size)
    initial_state = np.random.rand(state_size)
    state2state /= state2state.sum(axis=-1, keepdims=True)
    state2observation /= state2observation.sum(axis=-1, keepdims=True)
    initial_state /= initial_state.sum()

    for step in range(max_steps):
        pre_state2state, pre_state2observation, pre_initial_state = state2state, state2observation, initial_state

        # Expectation step, from parameters to probability of states
        state_prob_forward = forward(state2state, state2observation, initial_state, observation)[1]
        state_likelihood_backward = backward(state2state, state2observation, initial_state, observation)[1]
        state_likelihood = state_prob_forward * state_likelihood_backward + epsilon

        state_likelihood_wrt_observation = state2observation.T[observation]

        state_prob = state_likelihood / state_likelihood.sum(axis=-1, keepdims=True)
        state_trans_prob = state_prob_forward[:, :-1, :, None] * \
            state2state[None, None, :, :] * \
            state_likelihood_wrt_observation[:, 1:, None, :] * \
            state_likelihood_backward[:, 1:, None, :]
        state_trans_prob /= state_trans_prob.sum(axis=(-1, -2), keepdims=True)

        # Maximization step, from probability of states to parameters
        state2state = state_trans_prob.sum(axis=(0, 1)) / state_prob[:, :-1, :].sum(axis=(0, 1))[:, None]
        state2observation = ((observation[:, :, None] == np.arange(observation_size)[None, None, :])[:, :, None, :] *
                             state_prob[:, :, :, None]).sum(axis=(0, 1)) / state_prob.sum(axis=(0, 1))[:, None]
        initial_state = state_prob[:, 0].mean(axis=0)

        stride = np.mean([abs(pre_state2state - state2state).mean(),
                          abs(pre_state2observation - state2observation).mean(),
                          abs(pre_initial_state - initial_state).mean()])
        if stride < epsilon:
            break
    return state2state, state2observation, initial_state


if __name__ == '__main__':
    def demonstrate(observation, state_size, observation_size, desc):
        print(desc)
        state2state, state2observation, initial_state = baum_welch(observation, state_size, observation_size)
        print('state2state is:\n', np.round(state2state, 2))
        print('state2observation is:\n', np.round(state2observation, 2))
        print('initial_state is:\n', np.round(initial_state, 2))
        print('')

    # Example 1
    observation = np.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
    state_size = 2
    observation_size = 2
    demonstrate(observation, state_size, observation_size, "Example 1")

    # Example 2
    observation = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    state_size = 2
    observation_size = 2
    demonstrate(observation, state_size, observation_size, "Example 2")

    # Example 3
    observation = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    state_size = 2
    observation_size = 2
    demonstrate(observation, state_size, observation_size, "Example 3")

    # Example 3
    observation = np.array([[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]])
    state_size = 3
    observation_size = 3
    demonstrate(observation, state_size, observation_size, "Example 4")
