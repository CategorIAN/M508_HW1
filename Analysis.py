from Regression import Regression
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, data):
        self.data = data
        self.reg = Regression(data)

    def error(self, alpha, singleRun = False):
        '''
        :param alpha: the learning rate
        :param singleRun: finds regression weights after one epoch
        :return: the error of trained regression weights using learning rate
        '''
        print("======================")
        print("Alpha: {}".format(alpha))
        start_time = time.time()
        theta = self.reg.stochastic_gd(alpha, singleRun = singleRun)
        result = self.reg.J(theta)
        print("Error: {}".format(result))
        print("Time Elapsed: {} Seconds".format(time.time() - start_time))
        return result

    def getErrorDf(self, alphas, singleRun = False, figure = False):
        '''
        :param alphas: a list of learning rates to tune
        :param singleRun: finds regression weights after one epoch
        :return: a dataframe that relates learning rate to error from learning rate
        '''
        rows = pd.Series(alphas).map(lambda hyp: (hyp, self.error(hyp, singleRun)))
        col_titles = ["Alpha", "Error"]
        error_df = pd.DataFrame.from_dict(data=dict(rows), orient="index", columns=col_titles)
        error_df.to_csv("\\".join([os.getcwd(), str(self.data), "{}_Error.csv".format(str(self.data))]))
        if figure:
            plt.figure(1)
            plt.scatter(error_df["Alpha"], error_df["Error"], **{'color': 'blue', 'marker': 'o'})
            plt.xlabel("Learning Rate")
            plt.ylabel("Error")
            plt.title("Error vs. Learning Rate")
            plt.show()
        return error_df


