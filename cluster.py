from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics.cluster import rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from visualizer import Visualizer

class Cluster():

    df = None
    df_preprocessed = None
    genreEncoded = None
    n_clusters = None
    genreList = None

    visualizer = None
    mlb = MultiLabelBinarizer()

    k_clusters = range(2, 20)
    eps = [0.75 + 0.0675 * i for i in range(15)]
    sse = []
    silhouette_scores = []
    rand_scores = []
    normalizedMI_scores = []
    adjustedMI_scores = []
    vMeasure_scores = []
    FowlkesMallows_scores = []

    confusionMatrix = None

    def __init__(self, df, genreList, n_clusters):
        self.df = df
        self.df_preprocessed = df.drop(['id', 'genre', 'genre_list'], axis=1)
        self.genreEncoded = self.mlb.fit_transform(genreList)
        self.genreList = genreList
        self.n_clusters = n_clusters

    def __get_max_class(self, classes): 
        counts = pd.Series(classes).value_counts()
        max_count = counts.max()
        max_classes = counts[counts == max_count].index.tolist() 
        return random.choice(max_classes)

    def __clusterResultTransform(self, df):
        cluster_class = df.groupby('cluster')['genre_list'].apply(lambda x: self.__get_max_class([genre for sublist in x for genre in sublist])).reset_index(name='cluster_class')
        df = df.merge(cluster_class, on='cluster')
        return df, cluster_class
    
    def __clean(self):
        self.sse = []
        self.silhouette_scores = []
        self.rand_scores = []
        self.normalizedMI_scores = []
        self.adjustedMI_scores = []
        self.vMeasure_scores = []
        self.FowlkesMallows_scores = []
        self.confusionMatrix = None
    
    def __evaluate(self, y_true, y_pred):
        randScore = rand_score(y_true, y_pred)
        self.rand_scores.append(randScore)
        nMIScore = normalized_mutual_info_score(y_true, y_pred)
        self.normalizedMI_scores.append(nMIScore)
        aMIScore = adjusted_mutual_info_score(y_true, y_pred)
        self.adjustedMI_scores.append(aMIScore)
        vMeasurScore = v_measure_score(y_true, y_pred)
        self.vMeasure_scores.append(vMeasurScore)
        FMScore= fowlkes_mallows_score(y_true, y_pred)
        self.FowlkesMallows_scores.append(FMScore)

    def __visualize(self, method, dirPath):
        self.visualizer = Visualizer(self.k_clusters, self.eps, method, dirPath)
        self.visualizer.visualization_Elbow(self.sse, method)
        self.visualizer.visualization_Silhouette(self.silhouette_scores, method)
        self.visualizer.visualization_Rand(self.rand_scores, method)
        self.visualizer.visualization_NMI(self.normalizedMI_scores, method)
        self.visualizer.visualization_AMI(self.adjustedMI_scores, method)
        self.visualizer.visualization_VMeasure(self.vMeasure_scores, method)
        self.visualizer.visualization_FowlkesMallows(self.FowlkesMallows_scores, method)
    
    def cluster_Kmeans(self):
        self.__clean()
        method = "Kmeans"
        dirPath = "./assets/Kmeans/"
        for k in self.k_clusters:
            df = self.df.copy(deep=True)
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.df_preprocessed)
            cluster_labels = kmeans.predict(self.df_preprocessed)
            df['cluster'] = cluster_labels
            self.sse.append(kmeans.inertia_)
            silhouetteScore = silhouette_score(self.df_preprocessed, cluster_labels)
            self.silhouette_scores.append(silhouetteScore)
            df, __ = self.__clusterResultTransform(df)
            predictions_binary = self.mlb.transform([[pred] for pred in df['cluster_class']])
            self.__evaluate(self.genreEncoded.ravel(), predictions_binary.ravel())
        self.__visualize(method, dirPath)

    def cluster_Agglomerative(self):
        self.__clean()
        method = "Hierarchical"
        dirPath = "./assets/Hierarchical/"
        for k in self.k_clusters:
            df = self.df.copy(deep=True)
            agg_clustering = AgglomerativeClustering(n_clusters=k)
            cluster_labels = agg_clustering.fit_predict(self.df_preprocessed)
            df['cluster'] = cluster_labels
            silhouetteScore = silhouette_score(self.df_preprocessed, cluster_labels)
            self.silhouette_scores.append(silhouetteScore)
            df, __ = self.__clusterResultTransform(df)
            predictions_binary = self.mlb.transform([[pred] for pred in df['cluster_class']])
            self.__evaluate(self.genreEncoded.ravel(), predictions_binary.ravel())
        self.__visualize(method, dirPath)

    def cluster_DBSCAN(self):
        self.__clean()
        method = "DBSCAN"
        dirPath = "./assets/DBSCAN/"
        for e in self.eps:
            df = self.df.copy(deep=True)
            dbscan = DBSCAN(eps=e, min_samples=32) # number of features = 31
            cluster_labels = dbscan.fit_predict(self.df_preprocessed)
            # print(len(set(cluster_labels)))
            df['cluster'] = cluster_labels
            silhouetteScore = silhouette_score(self.df_preprocessed, cluster_labels)
            self.silhouette_scores.append(silhouetteScore)
            df, __ = self.__clusterResultTransform(df)
            predictions_binary = self.mlb.transform([[pred] for pred in df['cluster_class']])
            self.__evaluate(self.genreEncoded.ravel(), predictions_binary.ravel())
        self.__visualize(method, dirPath)

    def cluster_GaussianMixture(self):
        self.__clean()
        method = "GMM"
        dirPath = "./assets/GMM/"
        for k in self.k_clusters:
            df = self.df.copy(deep=True)
            gmm = GaussianMixture(n_components=k, random_state=42)
            cluster_labels = gmm.fit_predict(self.df_preprocessed)
            df['cluster'] = cluster_labels
            silhouetteScore = silhouette_score(self.df_preprocessed, cluster_labels)
            self.silhouette_scores.append(silhouetteScore)
            df, __ = self.__clusterResultTransform(df)
            predictions_binary = self.mlb.transform([[pred] for pred in df['cluster_class']])
            self.__evaluate(self.genreEncoded.ravel(), predictions_binary.ravel())
        self.__visualize(method, dirPath)
    
    def __evaluateSingle(self, y_true, y_pred):
        randScore = rand_score(y_true, y_pred)
        print(f"Rand Index: {randScore}")
        nMIScore = normalized_mutual_info_score(y_true, y_pred)
        print(f"Normalized Mutual Information: {nMIScore}")
        aMIScore = adjusted_mutual_info_score(y_true, y_pred)
        print(f"Adjusted Mutual Information: {aMIScore}")
        vMeasurScore = v_measure_score(y_true, y_pred)
        print(f"V-measure: {vMeasurScore}")
        FMScore= fowlkes_mallows_score(y_true, y_pred)
        print(f"Fowlkes-Mallows Score: {FMScore}")

    def __evaluateTitle(self, method, n_clusters):
        print(f"Method: {method}")
        print(f"Number of Clusters: {n_clusters}")

    def __visualizeSingle(self, method, dirPath, confusionMatrixList):
        self.visualizer = Visualizer(self.k_clusters, self.eps, method, dirPath)
        self.visualizer.visualization_ConfusionMatrix(confusionMatrixList)
        if method == "Hierarchical":
            self.visualizer.visualization_Dendrogram(self.df_preprocessed)
    
    def __clusterCount(self, cluster_labels):
        df_counts = pd.Series(cluster_labels).value_counts().reset_index()
        df_counts.columns = ['cluster', 'cluster_count']
        return df_counts

    def Kmeans(self):
        method = "Kmeans"
        n_clusters = 16
        dirPath = "./assets/Kmeans/"
        self.__evaluateTitle(method, n_clusters)
        df = self.df.copy(deep=True)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(self.df_preprocessed)
        cluster_labels = kmeans.predict(self.df_preprocessed)
        df['cluster'] = cluster_labels
        df_counts = self.__clusterCount(cluster_labels)
        df, cluster_class = self.__clusterResultTransform(df)
        df_merged = pd.merge(cluster_class, df_counts, on='cluster')
        silhouetteScore = silhouette_score(self.df_preprocessed, cluster_labels)
        predictions_binary = self.mlb.transform([[pred] for pred in df['cluster_class']])
        confusionMatrixList = multilabel_confusion_matrix(self.genreEncoded, predictions_binary)
        print(df_merged)
        print(f"Silhouette Score: {silhouetteScore}")
        self.__evaluateSingle(self.genreEncoded.ravel(), predictions_binary.ravel())
        self.__visualizeSingle(method, dirPath, confusionMatrixList)
    
    def Agglomerative(self):
        method = "Hierarchical"
        n_clusters = 16
        dirPath = "./assets/Hierarchical/"
        self.__evaluateTitle(method, n_clusters)
        df = self.df.copy(deep=True)
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agg_clustering.fit_predict(self.df_preprocessed)
        df['cluster'] = cluster_labels
        df_counts = self.__clusterCount(cluster_labels)
        df, cluster_class = self.__clusterResultTransform(df)
        df_merged = pd.merge(cluster_class, df_counts, on='cluster')
        silhouetteScore = silhouette_score(self.df_preprocessed, cluster_labels)
        predictions_binary = self.mlb.transform([[pred] for pred in df['cluster_class']])
        confusionMatrixList = multilabel_confusion_matrix(self.genreEncoded, predictions_binary)
        print(df_merged)
        print(f"Silhouette Score: {silhouetteScore}")
        self.__evaluateSingle(self.genreEncoded.ravel(), predictions_binary.ravel())
        self.__visualizeSingle(method, dirPath, confusionMatrixList)
        
        
    def DBSCAN(self):
        method = "DBSCAN"
        eps = 1.5
        dirPath = "./assets/DBSCAN/"
        self.__evaluateTitle(method, eps)
        df = self.df.copy(deep=True)
        dbscan = DBSCAN(eps=eps, min_samples=32) # number of features = 31
        cluster_labels = dbscan.fit_predict(self.df_preprocessed)
        df['cluster'] = cluster_labels
        df_counts = self.__clusterCount(cluster_labels)
        df, cluster_class = self.__clusterResultTransform(df)
        df_merged = pd.merge(cluster_class, df_counts, on='cluster')
        silhouetteScore = silhouette_score(self.df_preprocessed, cluster_labels)
        predictions_binary = self.mlb.transform([[pred] for pred in df['cluster_class']])
        confusionMatrixList = multilabel_confusion_matrix(self.genreEncoded, predictions_binary)
        print(df_merged)
        print(f"Silhouette Score: {silhouetteScore}")
        self.__evaluateSingle(self.genreEncoded.ravel(), predictions_binary.ravel())
        self.__visualizeSingle(method, dirPath, confusionMatrixList)

    def GaussianMixture(self):
        method = "GMM"
        n_clusters = 12
        dirPath = "./assets/GMM/"
        self.__evaluateTitle(method, n_clusters)
        df = self.df.copy(deep=True)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        cluster_labels = gmm.fit_predict(self.df_preprocessed)
        df['cluster'] = cluster_labels
        df_counts = self.__clusterCount(cluster_labels)
        df, cluster_class = self.__clusterResultTransform(df)
        df_merged = pd.merge(cluster_class, df_counts, on='cluster')
        silhouetteScore = silhouette_score(self.df_preprocessed, cluster_labels)
        predictions_binary = self.mlb.transform([[pred] for pred in df['cluster_class']])
        confusionMatrixList = multilabel_confusion_matrix(self.genreEncoded, predictions_binary)
        print(df_merged)
        print(f"Silhouette Score: {silhouetteScore}")
        self.__evaluateSingle(self.genreEncoded.ravel(), predictions_binary.ravel())
        self.__visualizeSingle(method, dirPath, confusionMatrixList)