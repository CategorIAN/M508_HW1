import numpy as np
import pandas as pd

class Grad_Descent:
    def __init__(self, data):
        self.data = data
        self.M = self.data.df.loc[:, self.data.features_ohe].to_numpy()
        self.y = self.data.df.loc[:, ["Target"]].to_numpy()

    def J(self, theta):
        v = self.M @ theta - self.y
        return 0.5 * np.dot(v.T, v)[0][0]

    def theta(self):
        return np.random.rand(len(self.data.features_ohe))

    def theta_update(self, theta, i):
        return theta + np.sum(self.y - self.M @ theta) * self.M[i].T

    def epoch(self, start_theta = None):
        def go(theta, i):
            print("======================================")
            print("Theta: {}".format(theta))
            print("index: {}".format(i))
            if i == self.M.shape[0]:
                return theta
            else:
                new_theta = self.theta_update(theta, i)
                return go(new_theta, i + 1)
        start_theta = self.theta() if start_theta is None else start_theta
        return go(start_theta, 0)
