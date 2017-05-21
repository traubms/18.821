import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from geometry import Point, Polygon, RegularPolygon, CircleList, RandomPolygon, RandomFlatShape

def plot_poly(p, size=2, scatter=False, xlim=None, ylim=None, bisectors=True, legend = False):
    plt.figure(figsize=(5, 5))
    if scatter:
        x, y = p.scatter()
        plt.scatter(x, y)
    else:
        plt.plot(*p.lines())
        i = 0
        for point in p.points:
            plt.text(point.x, point.y, i, color="red")
            i += 1
    if bisectors:
        zz = list(range(p.N))
        for (i, j) in zip(zz, zz[2:] + zz[:2]):
            a, b = p.points[i], p.points[j]
            z =  -5 * (b - a).rotate_90() + .5 * (b + a)
            c = + .5 * (b + a) + 5 * (b - a).rotate_90()
            plt.plot([z.x, c.x ], (z.y, c.y), linestyle='--', label=(i, j))
        
    if ylim is None:
        ylim = (-size, size)
    if xlim is None:
        xlim = (-size, size)
    plt.ylim(ylim)
    plt.xlim(xlim)
    if legend:
        plt.legend()

def plot_circle(c, r, *args, **kwargs):
    cir = plt.Circle((c.x, c.y), r, *args, **kwargs)
    plt.gcf().gca().add_artist(cir)

def find_radii(p, iters= 5000, plot=True, win_size=2):
    if plot:
        plt.figure(figsize=(5, 5))
    Xs = [] # list of all x coord of all vertices
    Ys = [] # list of all y coord of all vertices
    for i in range(iters): 
        p = p.flip(i % p.N) # flip in sequence 
#         p = p.flip(int(np.random.random() * len(p))) #flip randomly
        x, y = p.scatter()
        Xs += x
        Ys += y
        if plot:
            plt.scatter(x, y)
            
    # find center my taking mean
    Xm, Ym = np.mean(Xs), np.mean(Ys) 
    
    # find all distance^2 to middle
    ds = sorted([(Xs[i] - Xm)**2 + (Ys[i] - Ym)**2 for i in range(len(Xs))])
    r_min = ds[0]**.5 
    r_max = ds[-1]**.5
    center = (Xm, Ym)
    
    print "CENTER:", center
    print "R_MIN: ", r_min
    print "R_MAX: ", r_max

    if plot:
        x, y = p.scatter()
        plt.scatter(x, y, color='red')
        plt.ylim((-win_size, win_size))
        plt.xlim((-win_size, win_size))
        plot_circle(Point(Xm, Ym), ds[-1]**.5, color='red', alpha=.1, linewidth=0)
        plot_circle(Point(Xm, Ym), ds[0]**.5, color='purple', alpha=.5)
        
    return center, r_min, r_max

def track_point(p, i, iters= 5000, plot=True, win_size=2):
    if plot:
        plt.figure(figsize=(5, 5))
    Xs = [] # list of all x coord of all vertices
    Ys = [] # list of all y coord of all vertices
    for j in range(iters): 
        p = p.flip(j % p.N) # flip in sequence 
#         p = p.flip(int(np.random.random() * len(p))) #flip randomly
        x, y = p.scatter()
        Xs += [x[i]]
        Ys += [y[i]]
        # if plot:
        #     plt.scatter(x, y)

    if plot:
        # x, y = p.scatter()
        plt.scatter(Xs, Ys, color='red')
        plt.ylim((-win_size, win_size))
        plt.xlim((-win_size, win_size))
        # plot_circle(Point(Xm, Ym), ds[-1]**.5, color='red', alpha=.1, linewidth=0)
        # plot_circle(Point(Xm, Ym), ds[0]**.5, color='purple', alpha=.5)
        
    # return center, r_min, r_max

def main():
    p = RandomPolygon(6) # random quadrilateral
    print "POINTS:"
    print p.points
    print
    # p = Polygon([(0, 0), (1, 0), (1, 3), (0, 1)])
    plot_poly(p, size=2)
    find_radii(p, iters=5000, plot=True)
    track_point(p, 0)
    plt.show()



if __name__ == "__main__":
    main()