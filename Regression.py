import numpy as np
from tail_recursive import tail_recursive
import random

class Regression:
    def __init__(self, data):
        self.data = data
        self.X_train_mat = self.data.X_train.to_numpy()
        self.X_test_mat = self.data.X_test.to_numpy()
        self.Y_train_vec = self.data.Y_train.to_numpy()
        self.Y_test_vec = self.data.Y_test.to_numpy()

    def J(self, X, Y, theta):
        '''
        :param X: data matrix of predictors
        :param Y: target vector
        :param theta: vector of regression weights
        :return: the error of the regression model
        '''
        v = theta[0] + X @ theta[1:] - Y
        return 0.5 * np.dot(v.T, v)[0][0]

    def theta_update(self, alpha):
        '''
        :param alpha: the learning rate
        :return: function that takes theta and index and returns updated theta
        '''
        def f(theta, i):
            '''
            :param theta: vector of regression weights
            :param i: index of dataset to use for updating
            :return: an updated vector of regression weights
            '''
            grad = ((theta[0] + self.X_train_mat[i] @ theta[1:] - self.Y_train_vec[i]) *
                    np.concatenate(([1], self.X_train_mat[i].T)))
            return theta - alpha * grad
        return f

    def epoch(self, alpha, start_theta = None):
        '''
        :param alpha: the learning rate
        :param start_theta: starting regression weights
        :return: an updated vector of regression weights after going through dataset
        '''
        @tail_recursive
        def go(theta, i):
            if i == n:
                return theta
            else:
                new_theta = theta_update_func(theta, index[i])
                return go.tail_call(new_theta, i + 1)
        start_theta = np.random.rand(1 + len(self.data.feats_enc)) if start_theta is None else start_theta
        theta_update_func = self.theta_update(alpha)
        n = self.X_train_mat.shape[0]
        index = random.sample(list(self.data.X_train.index), k=n)
        return go(start_theta, 0)

    def stochastic_gd(self, alpha, start_theta = None, singleRun = False):
        '''
        :param alpha: the learning rate
        :param start_theta: starting regression weights
        :param singleRun: finds regression weights after one epoch
        :return: trained vector of regression weights
        '''
        def evaluate(theta = None, prev_error = None, prev_theta = None):
            if theta is None:
                result_theta = self.epoch(alpha, start_theta)
                return evaluate(theta = result_theta)
            else:
                if prev_error is None:
                    error = self.J(self.X_train_mat, self.Y_train_vec, theta)
                    new_theta = self.epoch(alpha, theta)
                    return evaluate(theta = new_theta, prev_error = error, prev_theta = theta)
                else:
                    error = self.J(self.X_train_mat, self.Y_train_vec, theta)
                    if error < prev_error:
                        new_theta = self.epoch(alpha, theta)
                        return evaluate(theta = new_theta, prev_error = error, prev_theta = theta)
                    else:
                        return prev_theta
        return self.epoch(alpha, start_theta) if singleRun else evaluate()




