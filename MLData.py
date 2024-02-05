import pandas as pd
import os
from functools import reduce
from sklearn.model_selection import train_test_split

class MLData:
    def __init__(self, name, data_loc, columns, target_name, classification, header):
        self.name = name
        self.data_loc = data_loc
        self.columns = columns
        self.target_name = target_name
        self.classification = classification
        if header:
            self.df = pd.read_csv(self.data_loc)
        else:
            self.df = pd.read_csv(self.data_loc, header=None)
            self.df.columns = self.columns
        target_column = self.df.pop(self.target_name)
        self.features = list(self.df.columns)
        self.df.insert(len(self.df.columns), "Target", target_column)
        self.indicatorEncoding()
        self.classes = pd.Index(list(set(self.df["Target"]))) if self.classification else None
        if not classification: self.df["Target"].apply(pd.to_numeric)
        self.df.to_csv("\\".join([os.getcwd(), self.name, "{}_Cleaned.csv".format(self.name)]))
        self.X_train, self.X_test, self.Y_train, self.Y_test = (
            train_test_split(self.df.loc[:, self.feats_enc], self.df.loc[:, ["Target"]], test_size=0.3, random_state=1))

    def __str__(self):
        return self.name

    def indicatorEncoding(self):
        def addFeature(feats, f):
            (feats_num, feats_binary, feats_cat) = feats
            try:
                self.df[f].apply(pd.to_numeric)
                return (feats_num + [f], feats_binary, feats_cat)
            except:
                cats = set(self.df[f])
                if len(cats) <= 2:
                    return (feats_num, feats_binary + [f], feats_cat)
                else:
                    return (feats_num, feats_binary, feats_cat + ["{}_{}".format(f, cat) for cat in cats[1:]])

        (feats_num, feats_binary, feats_cat) = reduce(addFeature, self.features, ([], [], []))
        self.feats_num = feats_num
        self.feats_binary = feats_binary
        self.feats_cat = feats_cat
        self.feats_enc = feats_num + feats_binary + feats_cat
        self.df = pd.DataFrame(dict([(f, self.df[f]) for f in feats_num] +
                                    [(f, self.df[f].map(lambda x: int(x == "Yes"))) for f in feats_binary] +
            [(f, self.df[f.split("_")[0]].map(lambda x: int(x == f.split("_")[-1]))) for f in feats_cat] +
                                    [("Target", self.df["Target"])]))
