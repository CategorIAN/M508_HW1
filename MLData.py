import pandas as pd
import os

class MLData:
    def __init__(self, name, data_loc, columns, target_name, replace, classification):
        self.name = name
        self.data_loc = data_loc
        self.df = pd.read_csv(self.data_loc, header=None)
        self.columns = columns
        self.df.columns = self.columns
        self.target_name = target_name
        target_column = self.df.pop(self.target_name)
        self.features = list(self.df.columns)
        self.df.insert(len(self.df.columns), "Target", target_column)
        self.replace = replace
        self.classification = classification
        self.one_hot()
        self.z_score_normalize()
        self.classes = pd.Index(list(set(self.df["Target"]))) if self.classification else None
        if not classification: self.df['Target'].apply(pd.to_numeric)

    def __str__(self):
        return self.name

    '''
        save: saves the dataframe to its folder with a given suffix
    '''

    def save(self, suffix=None):
        self.df.to_csv(os.getcwd() + '\\' + str(self) + '\\' + "{}_{}.csv".format(str(self), suffix))

    '''
    readcsv: will take the location of the file and convert it to pandas
    @return self.data - panda df of the csv for us to work with
    '''

    def readcsv(self):
        dn = self.data_loc
        self.df = pd.read_csv(dn)  # read the data location to pandas dataframe
        return self.df

    '''
    one_hot: transforms data by doing one hot encoding
    '''

    def one_hot(self):
        (features_numerical, features_categorical) = ([], [])
        features_categorical_ohe = []
        for f in self.features:
            try:
                self.df[f].apply(pd.to_numeric)  # sees if the column data can be considered numerica
                features_numerical.append(f)  # adds the column name as a numerical feature
            except:
                features_categorical.append(f)  # adds the column name as a categorical feature
                categories = set(self.df[f])  # creates set of all categories from this categorical feature
                for cat in categories:
                    features_categorical_ohe.append(
                        "{}_{}".format(f, cat))  # adds a one hot encoding categorical column
        self.features_numerical = features_numerical
        self.features_categorical = features_categorical
        one_hot_df = pd.get_dummies(self.df, columns=self.features_categorical)  # applies pandas one hot encoding
        self.features_ohe = features_numerical + features_categorical_ohe  # defines new feature set after ohe
        target_column = one_hot_df.pop('Target')  # remove target column
        one_hot_df.insert(len(one_hot_df.columns), 'Target', target_column)  # add target column to the end
        self.df = one_hot_df  # redefines dataframe by one hot encoding

    '''
    z_score_normalize: normalizes the data by applying the z score
    '''

    def z_score_normalize(self):
        for col in self.features_ohe:
            std = self.df[col].std()  # computes standard deviation
            if std != 0:
                self.df[col] = (self.df[col] - self.df[col].mean()) / std  # column is normalized by z score