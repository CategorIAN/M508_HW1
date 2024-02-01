from DataDictionary import DataDictionary
from Grad_Descent import Grad_Descent


def f(i):
    if i == 1:
        D = DataDictionary().dataobject("ForestFires")
        GD = Grad_Descent(D)
        print(GD.J(None))


if __name__ == '__main__':
    f(1)

