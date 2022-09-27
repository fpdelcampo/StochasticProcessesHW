import numpy as np

# Problem 1

print("Problem 1")
transition_matrix_1 = np.matrix('0.99 0.01; 0.12 0.88')
transition_matrix_1_cubed = np.matmul(np.matmul(transition_matrix_1, transition_matrix_1), transition_matrix_1)
print(transition_matrix_1_cubed)

# Problem 2

print("\nProblem 2")
transition_matrix_2 = np.matrix('0 0.5 0.5; 0.75 0 0.25; 0.75 0.25 0')
transition_matrix_2_squared = transition_matrix_2**2
transition_matrix_2_cubed = transition_matrix_2**3
print(transition_matrix_2_squared)
print(transition_matrix_2_cubed)

# Problem 5

print("\nProblem 5")
transition_matrix_5 = np.matrix('0.4 0.6 0 0; 0.4 0 0.6 0; 0.4 0 0 0.6; 0 0 0 1')
transition_matrix_5_to_the_tenth = transition_matrix_5**10
print(transition_matrix_5_to_the_tenth)

def sim_flips():
    p_heads = 0.6
    p_tails = 0.4

    num_3_heads = 0
    for i in range(10000):
        num_heads = 0
        results = np.random.random_sample(10)
        for i in range(10):
            if results[i]<0.6:
                num_heads+=1
                if num_heads==3:
                    num_3_heads+=1
                    break
            else:
                num_heads=0
    return num_3_heads
n = sim_flips()            
print(n)

# Problem 6

print("\nProblem 6")
transition_matrix_6_i = np.matrix('0.5 0.4 0.1; 0.2 0.5 0.3; 0.1 0.3 0.6')
transition_matrix_6_i_transpose = transition_matrix_6_i.transpose()
evals_6_i, evecs_6_i = np.linalg.eig(transition_matrix_6_i_transpose)
stationary_probabilities_6_i = evecs_6_i[:,np.isclose(evals_6_i, 1)]
pi_i = np.array([-0.39615532, -0.68426829,  -0.61224005]).T
print(pi_i*transition_matrix_6_i-pi_i)
print(stationary_probabilities_6_i/stationary_probabilities_6_i.sum())
transition_matrix_6_ii = np.matrix('0.5 0.4 0.1; 0.3 0.4 0.3; 0.2 0.2 0.6')
transition_matrix_6_ii_transpose = transition_matrix_6_ii.transpose()
evals_6_ii, evecs_6_ii = np.linalg.eig(transition_matrix_6_ii_transpose)
stationary_probabilities_6_ii = evecs_6_ii[:,np.isclose(evals_6_ii, 1)]
print(stationary_probabilities_6_ii)
pi_ii = np.array([-0.57735027, -0.57735027, -0.57735027]).T
print(pi_ii*transition_matrix_6_ii_transpose-pi_ii)
print(stationary_probabilities_6_ii/stationary_probabilities_6_ii.sum())

# Problem 7

print("\nProblem 7")
transition_matrix_7 = np.matrix('0.6 0.4; 0.5 0.5')
transition_matrix_7_transpose = transition_matrix_7.transpose()
evals_7, evecs_7 = np.linalg.eig(transition_matrix_7_transpose)
stationary_probabilities_7 = evecs_7[:,np.isclose(evals_7, 1)]
print(stationary_probabilities_7/stationary_probabilities_7.sum())
print(transition_matrix_7**4)

# Problem 8

print("\n Problem 8")
transition_matrix_8 = np.matrix('0.9 0.1; 0.3 0.7')
transition_matrix_8_transpose = transition_matrix_8.transpose()
print(transition_matrix_8**3)
print(transition_matrix_8**4)
evals_8, evecs_8 = np.linalg.eig(transition_matrix_8_transpose)
stationary_probabilities_8 = evecs_8[:,np.isclose(evals_8, 1)]
print(stationary_probabilities_8/stationary_probabilities_8.sum())