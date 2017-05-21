import numpy as np
import math

class CircleList(list):

    def equals(self, other):
        x = self[:]
        i = 0
        if len(self) != len(other):
            return False
        while i < len(self) and not np.all(np.abs(np.array(x[i:] + x[:i]) - np.array(other)) < 1e-14):
            i += 1
        if i == len(self):
            return False
        else:
            return True

class Point(object):

    def __init__(self, x, y, before=None, after=None):
        self.x = 1. * x
        self.y = 1. * y
        self.before = None
        self.after = None

    def flip(self):
        # rl = self.before - self.after
        # rm = self.before - self
        # if abs(rl * rm) < 1e-14:
        #     dx = -1 * rm
        # else:
        #     dx = (rl * rm) / (rl * rl) * rl - rm
        # new = self - 2 * dx
        # return Point(new.x, new.y, self.before, self.after)
        a = self.before
        b = self.after
        c = self
        ab = a - b
        ac = a - c
        if abs(ab * ac) < 1e-14 or abs(ab*ab) < 1e-14:
            dx = -1 * ac
        else:
            dx = (ab * ac) / (ab * ab) * ab - ac
        m = .5 * ab + b 
        dy = c - m - dx
        new = c - 2 * dy
        return Point(new.x, new.y, self.before, self.after)

    def rotate_90(self):
        return Point(self.y, -self.x)

    def dot(self, C):
        return self.x * C.x + self.y * C.y

    def cross(self, C):
        return self.dot(C.rotate_90())

    def norm(self):
        return math.sqrt(self.norm_sq())

    def norm_sq(self):
        return self.x**2 + self.y**2

    def circumcenter(self):
        p1 = self.before
        p2 = self
        p3 = self.after

        r = abs(p1 - p2) * abs(p2 - p3) * abs(p3 - p1) / (2 * (p1 - p2).cross(p2 - p3))

        denom = 2 * (p1 - p2).cross(p2 - p3)**2
        alpha = (p2 - p3).norm_sq() * (p1 - p2).dot(p1 - p3) / denom
        beta = (p1 - p3).norm_sq() * (p2 - p1).dot(p2 - p3) / denom
        gamma = (p1 - p2).norm_sq() * (p3 - p1).dot(p3 - p2) / denom

        center = alpha * p1 + beta * p2 + gamma * p3

        return (center, r)

    def __abs__(self):
        return self.norm()

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return self + -1 * other

    def __mul__(self, C):
        if isinstance(C, Point):
            return self.dot(C)
        else:
            return Point(self.x * C, self.y * C)

    def __repr__(self):
        return "(%.2f, %.2f)" % (self.x, self.y)

    __str__ = __repr__
    __rmul__ = __mul__
    __radd__ = __add__
    __rsub__ = __sub__



class Polygon(object):



    def __init__(self, points):
        self.N = len(points)
        if isinstance(points[0], Point):
            self.points = [points[0]]
        else:
            self.points = [Point(*points[0])]
        for i in range(1, self.N):
            if isinstance(points[i], Point):
                p = points[i]
            else:
                x, y = points[i]
                p = Point(x, y)
            self.points += [p]
            self.points[i-1].after = self.points[i]
            self.points[i].before = self.points[i-1]
        self.points[0].before = self.points[-1]
        self.points[-1].after = self.points[0]

    def scatter(self):
        x, y = zip(*[(p.x, p.y) for p in self.points])
        return list(x), list(y)

    def lines(self):
        x, y = self.scatter()
        return x + [x[0]], y + [y[0]]

    def angles(self):
        edge_angles = []
        for i in range(self.N):
            edge = self.points[i] - self.points[(i-1) % self.N]
            if edge.x == 0:
                edge_angle = math.pi / 2.
            else:
                edge_angle = math.atan(edge.y / edge.x) 
            if edge.x < 0:
                edge_angle += math.pi
            edge_angles += [edge_angle]
        angles = np.array(edge_angles)
        next_angles = np.array(edge_angles[1:] + edge_angles[:1])
        return (next_angles - angles) % (2 * math.pi)

    def flip(self, i):
        new_points = self.points[:]
        new_points[i] = self.points[i].flip()
        return Polygon(new_points)

    def is_convex(self):
        angles = self.angles()
        return np.all(angles < np.pi)

    def area(self):
        xs, ys = zip(*[(point.x, point.y) for point in self.points])
        return .5 * ((np.array(xs) * np.array(ys[1:] + ys[:1])).sum() - (np.array(ys) * np.array(xs[1:] + xs[:1])).sum())

    def __len__(self):
        return len(self.points)

class RegularPolygon(Polygon):

    def __init__(self, N, r=1.):
        angles = np.linspace(0, np.pi * 2, N+1)[:-1]
        points = zip(np.cos(angles), np.sin(angles))
        super(RegularPolygon, self).__init__(points)

class RandomPolygon(Polygon):

    def __init__(self, N, r=1., convex=True, ordered=True):
        angles = np.random.random(N) * np.pi * 2
        if ordered:
            angles = sorted(angles)
        rs = np.sqrt(np.random.random(N)) * r
        points = [(rs[i] * math.cos(angles[i]), rs[i] * math.sin(angles[i])) for i in range(N)]
        super(RandomPolygon, self).__init__(points)

class RandomFlatShape(Polygon):

    def __init__(self, N, L=1., convex=True):
        rs = np.random.random(N) * L
        points = [(r, 0) for r in rs]
        super(RandomFlatShape, self).__init__(points)

class PolygonProcess(object):

    def step(poly, t):
        return poly.flip(t % len(poly))

    def run_sim(init, T):
        results = [init]
        p = init
        for t in range(T):
            p = step(p, t)
            results += [p]
        return results
