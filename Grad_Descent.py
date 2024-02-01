import numpy as np
import pandas as pd

class Grad_Descent:
    def __init__(self, data):
        self.data = data

    def J(self, theta):
        M = self.data.df.loc[:, self.data.features_ohe].to_numpy()
        return M