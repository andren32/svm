from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy as np
import pylab,random,math


class SVM():
    """docstring for SVM."""
    def __init__(self):
        self.P = []
    '''
    x1 and x2 are datapoints
    '''
    def linear_kernel(self, x1, x2):
        #TODO: find out if this is correct, instructions asked to add 1 (to avoid 0?)
        return np.dot(x1,x2) + 1

    def build_p(self, datapoints, labels, kernfunc="linear"):
        N = len(datapoints)
        self.P = [[0 for i in range(N)] for i in range(N)]
        for i, datapoint_i in enumerate(datapoints):
            for j, datapoint_j in enumerate(datapoints):
                if kernfunc == "linear":
                    self.P[i][j] = labels[i]*labels[j]*self.linear_kernel(datapoint_i, datapoint_j)
                else:
                    # go linear as default
                    raise ValueError('Given kernel function does not exist. Received: ', kernfunc)




def main():
    svm = SVM()

    svm.build_p([[1,2,3],[2,3,4]], [-1, 1], "linear")
    print svm.P

main()
