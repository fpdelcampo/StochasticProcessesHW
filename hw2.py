import numpy as np

# Problem 1

print("Problem #1")
transition_matrix_1 = np.matrix('0.2 0.8 0; 0.33333333 0.22222222 0.44444444; 0.28571429 0.28571429 0.42857143').transpose()
evals_1, evecs_1 = np.linalg.eig(transition_matrix_1)
stationary_probabilities_1 = evecs_1[:,np.isclose(evals_1, 1)]
print(stationary_probabilities_1/stationary_probabilities_1.sum())

# Problem 2

print("\nProblem 2")
transition_matrix_2 = np.matrix('0.8 0.2; 0.75 0.25').transpose()
evals_2, evecs_2 = np.linalg.eig(transition_matrix_2)
stationary_probabilities_2 = evecs_2[:,np.isclose(evals_2, 1)]
print(stationary_probabilities_2/stationary_probabilities_2.sum())