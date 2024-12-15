import os
import pandas as pd

from checker import Checker
from preprocesser import Preprocesser
from cluster import Cluster

"""
explanation of features from spofify API documentation
https://developer.spotify.com/documentation/web-api/reference/get-audio-features
"""

def read_csv(csvFile):
    """
    Read the csv file and return the dataframe
    args:
        --.csv file: csv file
    return:
        --dataframe: dataframe
    """
    try:
        df = pd.read_csv(csvFile, dtype={"song_name": "str", 
                                         "key": "category",
                                         "mode": "category",
                                        "time_signature": "category"})
    except Exception as e:
        raise e
    return df

def main():

    """##############################################"""
    """
    initial setup, lncluding reading and check dataframe with differnet columns
    """
    dirPath = "./data/"
    fileName = "genres_v2.csv"
    csvFile = os.path.join(dirPath, fileName)
    df = read_csv(csvFile)
    # print(df.head())
    """##############################################"""

    """##############################################"""
    """
    remove the columns that are not needed
    """
    drop_columns = ["uri", "track_href", "analysis_url", "type"]
    df.drop(drop_columns, axis=1, inplace=True)
    """##############################################"""

    """##############################################"""
    """
    create an instance of the class checker to check the columns
    """
    checker = Checker(df)
    checker.describeStatistic()
    checker.checkMissing()
    outliers = checker.checkNoise()
    """##############################################"""

    """##############################################"""
    """
    data preprocessing, including data cleaning and normalization
    """
    sepcified_columns = ["Unnamed: 0", "title", "song_name"]
    preprocesser = Preprocesser(df, outliers, sepcified_columns)
    df_preprocessed = preprocesser.preprocess()
    df_label = preprocesser.labelEnconded()
    """##############################################"""

    """##############################################"""
    """
    clustering and evaluation
    """
    n_clusters = 15
    cluster = Cluster(df_preprocessed, df_label, n_clusters)
    # cluster.cluster_Kmeans() # K = 16
    # cluster.cluster_Agglomerative()
    cluster.cluster_DBSCAN()
    # cluster.cluster_GaussianMixture()
    # cluster.Kmeans()
    # cluster.Agglomerative()
    # cluster.DBSCAN()
    # cluster.GaussianMixture()
    """##############################################"""
if __name__ == "__main__":
    main()