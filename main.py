from DataDictionary import DataDictionary
from Regression import Regression


def f(i):
    if i == 1:
        D = DataDictionary().dataobject("StudentPerformance")
        GD = Regression(D)
        theta = GD.theta()
        print(GD.stochastic_gd(alpha = 0.1))


if __name__ == '__main__':
    f(1)

