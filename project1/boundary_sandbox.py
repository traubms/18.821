
import numpy as np
import pandas as pd
#import untitled4
#from asymmetric_processes import CircleProcess, BoundaryProcess, State
import matplotlib.pyplot as plt
import seaborn as sns # comment out if you don't have installed

# New process object
N = 10
a = 1
b = .5
q = 0
k = 2
#p = CircleProcess(N, q, k)
p = BoundaryProcess(N, q, a, b)

# SIMULATE DISTRIBUTION
sim = p.run_sims(1000, 100, [1, 1, 0 ,0 ,0, 0, 0, 0, 0, 0])
dist = np.mean(sim, axis=0)

plt.figure(1)
sns.heatmap(dist)
plt.title("Simulated Distribution of Particles Over Time")
plt.xlabel("Expected number of Particles at Position")
plt.ylabel("Time")
plt.show(block=False)

# EXACTLY CALCULATE DISTRIBUTION
S, T = p.matrices()
M = T.shape[0]
x = np.zeros(M)
x[0] = 1 # this is picking first state, not setting first particle = 1

iters = 100
R = np.zeros((iters, N))
for i in range(iters):
    R[i] = S.dot(x)
    x = T.dot(x)

plt.figure(2)
sns.heatmap(R)
plt.title("Exact Distribution of Particles Over Time")
plt.xlabel("Expected number of Particles at Position")
plt.ylabel("Time")
plt.show(block=False)


plt.figure(3)
sns.heatmap(dist - R)
plt.title("Simulated Distribution - Exact Distribution")
plt.xlabel("Error in Expected number of Particles at Position")
plt.ylabel("Time")
plt.show()