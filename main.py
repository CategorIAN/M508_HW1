from DataDictionary import DataDictionary
from Regression import Regression
from Analysis import Analysis
import numpy as np

def f(i):
    if i == 1:
        D = DataDictionary().dataobject("StudentPerformance")
        GD = Regression(D)
        print(GD.stochastic_gd(alpha = 0.1))
    if i == 2:
        D = DataDictionary().dataobject("StudentPerformance")
        A = Analysis(D)
        A.getErrorDf(alphas = [0.02 * i for i in range(1, 15 + 1)], singleRun = True, figure = True)

if __name__ == '__main__':
    f(2)

