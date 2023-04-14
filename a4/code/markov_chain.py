import numpy as np
from scipy import stats


class MarkovChain:
    def __init__(self, p1=None, pt=None):
        if p1 is not None and pt is not None:
            self.fit(p1, pt)

    def fit(self, p1, pt):
        self.p1 = p1
        self.pt = pt

    def sample(self, n_samples, d, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        samples = np.zeros((n_samples, d), dtype=int)

        states = len(self.p1)
        
        for t in range(d):
            for x_i in range(n_samples):
                if t == 0:
                    samples[x_i, t] = rng.choice(states, p = self.p1)
                else:
                    prev_state = samples[x_i, t-1]
                    transition_probs = self.pt[prev_state]
                    samples[x_i, t] = rng.choice(states, p = transition_probs)
                
        return samples
    
    def mc_conditionals(self, start_time, start_state, end_time, end_state):
        """
        Returns p(x_start_time = start_state | x_end_time = end_state) for specific values.
        """
        m_prev = np.zeros(self.p1.size)

        for time in range(end_time):

            if time == 0:
                p_x_j = self.p1
                m_prev = p_x_j

            elif (time == start_time - 1):
                ## This is the summation going into @start_time
                ## i.e \sum_x2 (p(x_3 = @start_state | x_2) * M_2)
                p_next_given_curr = self.pt.T[start_state]
                m_prev = p_next_given_curr @ m_prev.T

            elif (time == start_time):
                ## this is the simple multiplication coming out of @start_time
                ## i.e M_4 = p(x_4 | x_3 = @start_state) * M_3
                transitions = self.pt[start_state]
                m_prev = transitions * m_prev

            elif (time == end_time - 1):
                ## this is the summation going into @end_time
                ## i.e M_6 = p(x_6 = @end_state | x_5) * M_5
                transition_to_last_state = self.pt.T[end_state]
                m_prev = transition_to_last_state.T @ m_prev

            else:
                m_prev = self.pt @ m_prev

        return m_prev


    def marginals(self, d):
        # M = np.zeros((self.p1.size, d))
        M = np.zeros((d, self.p1.size))
        
        for time in range(d):
            if time == 0:
                margs = self.p1
            else:
                pi_j = M[time - 1]
                margs = self.pt.T @ pi_j.T 

            M[time] = margs

        return M

    def mode(self, d):
        """
        DP: M[j] = max{ p(x_j | x_{j-1}) * M[j-1]}
        """
        ## M[t, k] = highest probability achievable to get to time t and in state k
        M = np.zeros((d, self.p1.size))
        
        ## B[t, k] = the corresponding state which achieved the highest probability in M[t, k]
        B = np.zeros((d, self.p1.size))

        best_sequence = []
        
        for t in range(d):
            if (t == 0):
                m = self.p1
                M[t] = m
                B[t] = -1
            else:
                m_prev = M[t-1]
                probs = self.pt.T * m_prev
                max_probs = probs.max(axis = 1)
                max_probs_index = probs.argmax(axis = 1)

                M[t] = max_probs
                B[t] = max_probs_index

                index_with_highest_probability = max_probs.argmax()
                state_with_highest_probability = max_probs_index[index_with_highest_probability]

                best_sequence.append(state_with_highest_probability)

        return best_sequence
    
    
    

    
