import numpy as np

def sampling_from_prob_vec(prob_vec):
    cumsum_prob = np.cumsum(prob_vec)
    rand_sample = np.random.uniform(0.0, 1.0, 1)
    for ind_prob in range(len(cumsum_prob)):
        if ind_prob == 0:
            if 0 <= rand_sample < cumsum_prob[ind_prob]:
                return ind_prob
                break
        else:
            if cumsum_prob[ind_prob-1] <= rand_sample < cumsum_prob[ind_prob]:
                return ind_prob
                break
    if rand_sample >= cumsum_prob[-1]:
        print('-------warning_---------')
        assert ind_prob == len(cumsum_prob) - 1
        return ind_prob
    else:
        pass