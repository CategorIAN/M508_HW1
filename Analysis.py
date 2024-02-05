from Regression import Regression
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, data):
        self.data = data
        self.reg = Regression(data)

    def error_theta(self, alpha, singleRun = False):
        '''
        :param alpha: the learning rate
        :param singleRun: finds regression weights after one epoch
        :return: the error of trained regression weights using learning rate
        '''
        print("======================")
        print("Alpha: {}".format(alpha))
        start_time = time.time()
        theta = self.reg.stochastic_gd(alpha, singleRun = singleRun)
        result = (self.reg.J(self.reg.X_test_mat, self.reg.Y_test_vec, theta),) + tuple(theta)
        print("Error: {}".format(result[0]))
        print("Time Elapsed: {} Seconds".format(time.time() - start_time))
        return result

    def getErrorDf(self, alphas, singleRun = False, figure = False):
        '''
        :param alphas: a list of learning rates to tune
        :param singleRun: finds regression weights after one epoch
        :param figure: displays a scatterplot of Error vs. Learning Rate
        :return: a dataframe that relates learning rate to error from learning rate
        '''
        rows = pd.Series(alphas).map(lambda hyp: (hyp,) + self.error_theta(hyp, singleRun))
        col_titles = ["Alpha", "Error", "Theta_Int"] + ["Theta_{}".format(f) for f in self.data.feats_enc]
        error_df = pd.DataFrame.from_dict(data=dict(rows), orient="index", columns=col_titles)
        error_df.to_csv("\\".join([os.getcwd(), str(self.data), "{}_Error.csv".format(str(self.data))]))
        if figure:
            plt.figure(1)
            plt.scatter(error_df["Alpha"], error_df["Error"], **{'color': 'blue', 'marker': 'o'})
            plt.xlabel("Learning Rate")
            plt.ylabel("J Error")
            plt.title("J Error vs. Learning Rate")
            plt.show()
            #---------------------------------------------------------------------------------------------
            plt.scatter(error_df["Alpha"][:40], error_df["Error"][:40], **{'color': 'green', 'marker': 'o'})
            plt.xlabel("Learning Rate")
            plt.ylabel("J Error")
            plt.title("J Error vs. Learning Rate")
            plt.show()
            # ---------------------------------------------------------------------------------------------
            plt.scatter(error_df["Alpha"][:30], error_df["Error"][:30], **{'color': 'red', 'marker': 'o'})
            plt.xlabel("Learning Rate")
            plt.ylabel("J Error")
            plt.title("J Error vs. Learning Rate")
            plt.show()
        return error_df

    def best_param(self, error_df):
        min_error = error_df["Error"].min()
        best_row = error_df.loc[lambda df: df["Error"] == min_error].iloc[0]
        best_row.to_csv("\\".join([os.getcwd(), str(self.data), "{}_Best.csv".format(str(self.data))]))
        return best_row["Alpha"]


