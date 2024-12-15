from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import ast

class Preprocesser:

    df = None
    outliers = None
    columns = None
    columns_nominal = ['key', 'mode', 'time_signature']
    columns_numeric = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False)

    def __init__(self, df, outliers, columns):
        self.df = df
        self.outliers = outliers
        self.columns = columns
        if not self.df.empty:
            self.__build()
    
    def __build(self):
        self.__removeOutliers() # 39233 rows
        self.__groupbyID() # 33213 rows
        self.__removeSpecifiedColumns()

    def __groupbyID(self):
        """
        group by id and aggregate the genre into a list named genre_list
        args:
            --None
        return:
            --None
        """
        df = self.df.copy(deep=True)
        # grouped = self.df.groupby('id')['genre'].agg(list)
        # df['genre_list'] = self.df['id'].map(grouped)
        # df['genre_list'] = df['genre_list'].apply(lambda x: eval(str(x)) if isinstance(x, str) else x)
        # df['genre_list'].to_csv("test.csv", index=False)
        genre_list = df.groupby('id')['genre'].apply(list).reset_index()
        df = df.merge(genre_list, on='id', suffixes=('', '_list'))
        df['genre_list'] = df['genre_list'].apply(lambda x: list(set(x)) if isinstance(x, list) else eval(str(x)))
        df = df.drop_duplicates(subset=['id']).reset_index(drop=True)
        self.df = df.copy(deep=True)
        ########################################
        """
        test region, for groupby function
        """
        # df = self.df.groupby("id").agg({
        #     'danceability': 'mean',
        #     'energy': 'mean',
        #     'key': 'mode',
        #     'loudness': 'mean',
        #     'mode': 'mode',
        #     'speechiness': 'mean',
        #     'acousticness': 'mean',
        #     'instrumentalness': 'mean',
        #     'liveness': 'mean',
        #     'valence': 'mean',
        #     'tempo': 'mean',
        #     'duration_ms': 'mean',
        #     'time_signature': 'mode',
        #     'genre': 'mode',
        #     'song_name': 'mode',
        #     'Unnamed: 0': 'mode',
        #     'title': 'mode',
        #     'genre_list': 'first'
        # })
        ########################################
        
    def __removeOutliers(self):
        """
        remove the outliers in the dataframe
        args:
            --None
        return:
            --None
        """
        df = self.df[~self.outliers].reset_index(drop=True)
        self.df = df.copy(deep=True)

    def __removeSpecifiedColumns(self):
        """
        remove the specified columns in the dataframe
        args:
            --columns: list type, columns to be removed
        return:
            --None
        """
        self.df.drop(self.columns, axis=1, inplace=True)

    def __normalize(self):
        """
        normalize the dataframe
        args:
            --None
        return:
            --normalized_df: dataframe which has been normalized with numeric columns
        """
        normalized_df = self.df.copy(deep=True)
        normalized_df[self.columns_numeric] = self.scaler.fit_transform(self.df[self.columns_numeric])
        return normalized_df

    def __oneHotEncoding(self):
        """
        one-hot encoding the dataframe
        args:
            --None
        return:
            --encoded_df: dataframe which has been one-hot encoded with nominal columns
        """
        encoded_categories = self.encoder.fit_transform(self.df[self.columns_nominal])
        encoded_df = pd.DataFrame(encoded_categories, columns=self.encoder.get_feature_names_out(self.columns_nominal))
        return encoded_df

    def preprocess(self):
        """
        preprocess the dataframe including normalization and one-hot encoding
        args:
            --None
        return:
            --None
        """
        df_numeric = self.__normalize()
        df_nominal = self.__oneHotEncoding()
        df_combined = pd.concat([df_numeric, df_nominal], axis=1)
        print(f"preprocess done")
        return df_combined
    
    def labelEnconded(self):
        """
        get the label of the dataframe
        args:
            --None
        return:
            --output: list type, label of genre_list
        """
        labelList = self.df['genre_list'].tolist()
        output = [np.unique(sublist).tolist() for sublist in labelList]
        # count = 0
        # for sublist in output:
        #     if len(sublist) > 1:
        #         count += 1
        # print(count) # 2299, origin data is 4566
        return output