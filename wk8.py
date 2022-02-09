import mdptoolbox
import numpy as np

P1 = np.array([[[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]],
               [[0.2, 0.8, 0,   0],
                [0,   0.2, 0.8, 0],
                [0,   0,   0.2, 0.8],
                [0,   0,   0,   1]]])
R1 = np.array([[-0.04, -0.04], [-0.04, -0.04], [-0.04, -0.04], [1, 1]])

print('check', mdptoolbox.util.check(P1, R1))
vi1 = mdptoolbox.mdp.ValueIteration(P1, R1, 0.9)
vi1.run()
# We can then display the values (utilities) computed, and look at the policy:
print('Values:\n', vi1.V)
print('Policy:\n', vi1.policy)