from DataDictionary import DataDictionary
from Regression import Regression
from Analysis import Analysis

def f(i):
    if i == 1:
        D = DataDictionary().dataobject("StudentPerformance")
        GD = Regression(D)
        print(GD.stochastic_gd(alpha = 0.1))
    if i == 2:
        D = DataDictionary().dataobject("StudentPerformance")
        A = Analysis(D)
        df = A.getErrorDf(alphas = [0.00001 * i for i in range(1, 50 + 1)], singleRun = True, figure = True)
        print(A.best_param(df))

if __name__ == '__main__':
    f(2)

