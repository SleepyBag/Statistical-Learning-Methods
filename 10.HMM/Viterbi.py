import numpy as np

def viterbi(state2state, state2observation, initial_state, observation):
    """
    Given a HMM with parameter (state2state, state2observation, initial_state)
    and the observation,
    return the most possible state sequence

    state2state is a matrix shaped of [state_size, state_size]
    state2observation is a matrix shaped of [state_size, observation_size]
    initial_state is a tensor shaped of [state_size], whose each dimension means the probability of each state
    observation is a tensor shaped of [sequence_length]
    observation_size is the number of all the possible observations
    """
    sequence_length, = observation.shape
    state_size, _ = state2state.shape

    state_prob = initial_state
    pre_state = np.zeros([sequence_length, state_size]).astype(int)
    for i, o in enumerate(observation):
        state_prob *= state2observation[:, o]
        if i != sequence_length - 1:
            trans_prob = state_prob[:, None] * state2state
            pre_state[i + 1] = trans_prob.argmax(axis=0)
            state_prob = trans_prob.max(axis=0)
    ans = np.zeros(sequence_length).astype(int)
    ans[-1] = state_prob.argmax()
    for i in range(sequence_length - 2, -1, -1):
        ans[i] = pre_state[i + 1, ans[i + 1]]
    return ans


if __name__ == '__main__':
    state2state = np.array([[.5, .2, .3],
                            [.3, .5, .2],
                            [.2, .3, .5]])
    state2observation = np.array([[.5, .5],
                                  [.4, .6],
                                  [.7, .3]])
    initial_state = np.array([.2, .4, .4])
    observation = np.array([0, 1, 0])
    print(viterbi(state2state, state2observation, initial_state, observation))
