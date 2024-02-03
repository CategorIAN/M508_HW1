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
        A.getErrorDf([0.1, 0.2, 0.3])

if __name__ == '__main__':
    f(2)

