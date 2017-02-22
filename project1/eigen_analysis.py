import numpy as np
import pandas as pd
import asymmetric_processes
from asymmetric_processes import CircleProcess, BoundaryProcess, State
import matplotlib.pyplot as plt
import seaborn as sns # comment out if you don't have installed

PERIODIC = False

if PERIODIC: # Periodic

    # New process object
    N = 10
    k = 2
    q = 0.
    p = CircleProcess(N, k, q)
    S, T, V = p.matrices()

    values, vectors = np.linalg.eig(T.todense())

    zzz = sorted(zip(range(len(values)), np.abs(values), np.angle(values, deg=True)), key=lambda x: -x[1])
    print "EIGENVALUES"
    print "IDX  MAG       ANGLE"
    for i, m, a in zzz:
        print "%s" % i, "   %.4f    %+.3f" % (m, a)

else: # Boundary

    # New Boundary object
    N = 8
    q = 0.
    a = .6
    b = .6
    p = BoundaryProcess(N, q, a, b)

    S, T, V = p.matrices()
    l, v = np.linalg.eig(T.todense())
    i_eq = np.argmax(np.abs(l))
    M = T.shape[0]

    proj_eig_v = pd.DataFrame(np.real(S.dot(v)))
    proj_eig_v = proj_eig_v / proj_eig_v.sum()

    plt.figure()
    proj_eig_v[i_eq].plot.bar()
    plt.xlabel("Position in Length %s Bitstring" % N)
    plt.title("Expected Number of Particles at Each Position in Steady State")
    plt.show()




