import numpy as np
import pandas as pd
import asymmetric_processes
from asymmetric_processes import CircleProcess, BoundaryProcess, State
import matplotlib.pyplot as plt
import seaborn as sns # comment out if you don't have installed

# New process object
N = 10
k = 2
q = 0.
p = CircleProcess(N, k, q)
S, T = p.matrices()

values, vectors = np.linalg.eig(T.todense())

zzz = sorted(zip(range(len(values)), np.abs(values), np.angle(values, deg=True)), key=lambda x: -x[1])
print "EIGENVALUES"
print "IDX  MAG       ANGLE"
for i, m, a in zzz:
    print "%s" % i, "   %.4f    %+.3f°" % (m, a)




