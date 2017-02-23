"""
speeds.py
Authors: mtraub
18.821 Project #1. 

Description: provides functions for analytically finding the
    speeds of states and the number of open spots to right for
    given N, k in the periodic case

Includes:
    count
    prob
    mid
    max_speed
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from asymmetric_processes import ncr
import seaborn as sns

def count(N, k, R):
    """
    Returns number of states with number of open spots to the left
    R given N, k on the periodic case according to Lisa's formula
    """
    return ncr(k-1, R) * ncr(N-k-1, R-1) + 2 * ncr(k-1, R-1)* ncr(N - k-1, R-1) + ncr(k-1, R-1) * ncr(N - k - 1, R)

@np.vectorize
def prob(N, k, R):
    """
    Returns probability of states with number of open spots to the left
    R given N, k on the periodic case according to Lisa's formula
    """
    return 1. * count(N, k, R) / ncr(N, k)

def mid(N, R):
    """
    Same as count for k = N/2
    """
    return 2 * ncr(N / 2 - 1, R-1) * ncr(N/2, R)

@np.vectorize
def max_speed(N, q=0):
    """
    Given N, q, returns the maximum expected speed over all k
    """
    if N % 2 == 0:
        return N**2 / (4. * (N - 1.)) * (1.-q) / N
    else:
        return (N + 1.) / 4. * (1.-q) / N

def main():
    plt.figure(1)
    NN = 100
    for KK in range(1, NN):
        pd.Series(dict([(1. * r/NN, 1. * prob(NN, KK, r)) for r in range(NN/2+2)])).plot(style="-")
    plt.xlabel("R / N")
    plt.title("Distribution of P(R | N, K) for Various K");
    plt.show(block=False)

    plt.figure(2)
    for N in [2, 6, 10, 14, 22, 30, 38, 50, 82, 150, 500, 1000]:
        pd.Series(dict([(1. * r/N, 1. *mid(N, r) / ncr(N, N/2)) for r in range(N/2+2)])).plot(label=N, style="-")
    plt.xlabel("R / N")
    plt.title("Distribution of P(R | N, K=N/2) for Various N")
    plt.legend()
    plt.show(block=False)

    rr = dict()
    for NN in range(2, 100):
        x, y = zip(*[(k, sum(prob(NN, k, range(k+1)) * range(k+1))) for k in range(0, NN+1)])
        rr[NN] = (np.max(y) / NN, max_speed(NN))
    rrr = pd.DataFrame(rr).transpose()
    
    plt.figure(3)
    rrr[0].plot()
    plt.title("Max Speed for N")
    plt.xlabel("N")
    plt.show(block=False)

    plt.figure(4)
    (rrr[0] - rrr[1]).plot()
    plt.title("Error between Computed and Analytical Max Speed")
    plt.xlabel("N")
    plt.show(block=False)

    plt.show()


if __name__ == "__main__":
    main()
