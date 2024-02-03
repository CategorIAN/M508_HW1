import numpy as np
from tail_recursive import tail_recursive
import random


class Regression:
    def __init__(self, data):
        self.data = data
        self.M = self.data.df.loc[:, self.data.features_ohe].to_numpy()
        self.y = self.data.df.loc[:, ["Target"]].to_numpy()

    def J(self, theta):
        v = self.M @ theta - self.y
        return 0.5 * np.dot(v.T, v)[0][0]

    def theta(self):
        return np.random.rand(len(self.data.features_ohe))

    def theta_update(self, alpha):
        def f(theta, i):
            return theta + alpha * (self.y[i] - self.M[i] @ theta) * self.M[i].T
        return f

    def epoch(self, alpha, start_theta = None):
        @tail_recursive
        def go(theta, i):
            print("======================================")
            print("Theta: {}".format(theta))
            print("index: {}".format(i))
            if i == n:
                return theta
            else:
                new_theta = theta_update_func(theta, index[i])
                return go.tail_call(new_theta, i + 1)
        start_theta = self.theta() if start_theta is None else start_theta
        theta_update_func = self.theta_update(alpha)
        n = self.M.shape[0]
        index = random.sample(list(self.data.df.index), k=n)
        return go(start_theta, 0)

    def stochastic_gd(self, alpha, start_theta = None):
        def evaluate(theta = None, prev_error = None, prev_theta = None):
            if theta is None:
                print("=========================")
                print("Getting a Theta")
                result_theta = self.epoch(alpha, start_theta)
                return evaluate(theta = result_theta)
            else:
                print("=========================")
                print("Got Theta")
                if prev_error is None:
                    print("-----------------")
                    print("Getting an Error")
                    error = self.J(theta)
                    new_theta = self.epoch(alpha, theta)
                    return evaluate(theta = new_theta, prev_error = error, prev_theta = theta)
                else:
                    print("-----------------")
                    print("Got Error")
                    error = self.J(theta)
                    if error < prev_error:
                        print("~~~~~~~")
                        print("Decreased Error; Let Us Go Through Again")
                        new_theta = self.epoch(alpha, theta)
                        return evaluate(theta = new_theta, prev_error = error, prev_theta = theta)
                    else:
                        print("~~~~~~~")
                        print("It is not increasing in performance. We are done.")
                        return prev_theta
        return evaluate()




