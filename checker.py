import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
from scipy.spatial.distance import mahalanobis

class Checker():

    df = None

    def __init__(self, df):
        self.df = df

    def __writeCSV(self, df, fileName):
        """
        Write the dataframe to a csv file
        args:
            --fileName: string
        return:
            --None
        """
        if os.path.exists(fileName):
            os.remove(fileName)
        df.to_csv(fileName, index=False)

    def __checkNoiseZScore(self, df, threshold):
        """
        Check the noise in the dataframe using Z-Score
        args:
            --df: dataframe
            --threshold: int
        return:
            --outliers: Series
        """
        zScores = df.apply(zscore)
        outliers = (np.abs(zScores) > threshold).any(axis=1)
        # print(f"Outliers in the dataframe with z-score method: \n{df[outliers]}")
        print(f"check noise with z-score done")
        return outliers
    
    def __checkNoiseMahalanobis(self, df, threshold):
        """
        Check the noise in the dataframe using Mahalanobis distance
        args:
            --df: dataframe
            --threshold: int
        return:
            --outliers: Series
        """
        mean_vector = np.mean(df, axis=0)
        cov_matrix = np.cov(df, rowvar=False)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        distances = df.apply(lambda row: mahalanobis(row, mean_vector, inv_cov_matrix), axis=1)
        outliers = distances > threshold
        # print(f"Outliers in the dataframe with Mahalanobis method: \n{df[outliers]}")
        print(f"check noise with Mahalanobis done")
        return outliers

    def checkMissing(self):
        """
        Check the missing values in the dataframe
        args:
            --None
        return:
            --None
        """
        missing = self.df.isnull().sum()
        # print(f"Missing values in the dataframe: \n{missing}")
        print(f"check missing values done")

    def describeStatistic(self):
        """
        Descriptive statistics of the dataframe
        args:
            --None
        return:
            --None
        """
        # for col in self.df.columns:
            # print(f"Descriptive statistics of the dataframe with column {col}:\n{self.df[col].describe()}")
        print(f"describe statistics done")
        pass
    
    def checkNoise(self):
        """
        Check the noise in the dataframe
        args:
            --None
        return:
            --outliers: Series
        """
        columns_toDrop = ["id", "duration_ms", "genre", "song_name", "Unnamed: 0", "title"]
        columns_nominal = ["key", "mode", "time_signature"]
        threshold = 3
        df_copied = self.df.copy(deep=True)
        df_copied.drop(columns_toDrop, axis=1, inplace=True)
        df_copied.drop(columns_nominal, axis=1, inplace=True)
        outliers_zscore = self.__checkNoiseZScore(df_copied, threshold)
        outliers_mahalanobis = self.__checkNoiseMahalanobis(df_copied, threshold)
        outliers = outliers_zscore & outliers_mahalanobis
        df_outliers = df_copied[outliers]
        self.__writeCSV(df_outliers, "outliers.csv")
        return outliers