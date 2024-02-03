from Regression import Regression
import pandas as pd
import os

class Analysis:
    def __init__(self, data):
        self.data = data
        self.reg = Regression(data)

    def error(self, alpha):
        '''
        :param alpha: the learning rate
        :return: the error of trained regression weights using learning rate
        '''
        theta = self.reg.stochastic_gd(alpha)
        return self.reg.J(theta)

    def getErrorDf(self, alphas):
        '''
        :param alphas: a list of learning rates to tune
        :return: a dataframe that relates learning rate to error from learning rate
        '''
        rows = pd.Series(alphas).map(lambda hyp: (hyp, self.error(hyp)))
        col_titles = ["Alpha", "Error"]
        error_df = pd.DataFrame.from_dict(data=dict(rows), orient="index", columns=col_titles)
        error_df.to_csv("\\".join([os.getcwd(), str(self.data), "{}_Error.csv".format(str(self.data))]))
        return error_df

