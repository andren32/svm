from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy as np
import pylab,random,math

EPSILON = 10e-5

'''
x1 and x2 are datapoints
'''
def linear_kernel(x1, x2):
    return np.dot(np.transpose(x1),x2) + 1

def quad_kernel(x1,x2):
    return (np.dot(x1,x2)+1)**2

def cubic_kernel(x1,x2):
    return (np.dot(x1,x2)+1)**3

sp = 4
def radial_kernel(x1,x2):
    diff = np.subtract(x1,x2)
    t = -(np.dot(diff,diff)/(2*np.power(sp,2)))
    return np.exp(t)

class SVM():
    def __init__(self):
        self.alpha = None
        self.kernel = None

    def train(self, datapoints, labels, kernfunc=linear_kernel):
        N = len(datapoints)
        P = [[0 for i in range(N)] for i in range(N)]
        self.kernel = kernfunc
        for i, datapoint_i in enumerate(datapoints):
            for j, datapoint_j in enumerate(datapoints):
                P[i][j] = float(labels[i])*float(labels[j])*float(kernfunc(datapoint_i, datapoint_j))

        q = [-1.0 for i in range(N)]
        h = [0.0 for i in range(N)]
        G = [[0.0 for i in range(N)] for i in range(N)]

        for i in range(N):
            G[i][i] = -1

        r = qp (matrix(P),matrix(q),matrix(G),matrix(h))
        alpha = list(r['x'])
        self.alpha = []
        self.t = []
        self.y = []

        for i, a in enumerate(alpha):
            if a > EPSILON:
                self.alpha.append(a)
                self.t.append(labels[i])
                self.y.append(datapoints[i])

    def classify(self, X):
        res = []
        for x in X:
            ind = 0
            for i, a in enumerate(self.alpha):
                ind += a*self.t[i]*self.kernel(x,self.y[i]);
            if ind > 0:
                res.append(1)
            elif ind < 0:
                res.append(-1)

        return res

def main():
    svm = SVM()

    random.seed(3242342)
    classA = [(random.normalvariate(1.5, 0.4), random.normalvariate(2.5, 1.8), 1.0)
                for i in range(5)] + [(random.normalvariate(1.5, 0.5),
                random.normalvariate(1.2, 0.8), 1.0) for i in range(5)]
    classB = [(random.normalvariate(1.5, 1.0), random.normalvariate(2.2, 1.4), -1.0)
                for i in range(10)]

    data = classA + classB
    random.shuffle(data)

    datapoints = []
    labels = []
    for d in data:
        datapoints.append([d[0],d[1]])
        labels.append(d[2])

    pylab.hold(True)
    pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
    pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')


    x_range=np.arange(-4,4,0.05)
    y_range=np.arange(-4,10,0.05)
    # svm.train(datapoints, labels, linear_kernel)
    # svm.train(datapoints, labels, quad_kernel)
    # svm.train(datapoints, labels, cubic_kernel)
    svm.train(datapoints, labels, radial_kernel)
    grid = matrix([[svm.classify([[x,y]])[0] for y in y_range] for x in x_range])
    pylab.contour(x_range, y_range, grid, (-1.0,0.0,1.0),colors=('red','black','blue'),linewidths=(1,3,1))

    pylab.show()






if __name__ == '__main__':
    main()
