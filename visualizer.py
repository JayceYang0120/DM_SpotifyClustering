
import os

from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.metrics.cluster import rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, v_measure_score, fowlkes_mallows_score
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns

class Visualizer():

    k_clusters = None
    eps = None
    method = None
    dirPath = None

    def __init__(self, k_clusters, eps, method, dirPath):
        self.k_clusters = k_clusters
        self.eps = eps
        self.method = method
        self.dirPath = dirPath

    def __check(self, scores):
        return len(scores)

    def visualization_Elbow(self, sse, method="Other"):
        if not self.__check(sse):
            return
        if method != "DBSCAN":
            xValue = self.k_clusters
        else:
            xValue = self.eps
        plt.figure(figsize=(8, 5))
        plt.plot(xValue, sse, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('SSE (Inertia)')
        plt.grid()
        # plt.show()
        file_name = "elbow_" + self.method + ".png"
        file_path = os.path.join(self.dirPath, file_name)
        plt.savefig(file_path, format='png')
        plt.close()

    def visualization_Silhouette(self, silhouette_scores, method="Other"):
        if not self.__check(silhouette_scores):
            return
        if method != "DBSCAN":
            xValue = self.k_clusters
        else:
            xValue = self.eps
        plt.figure(figsize=(8, 5))
        plt.plot(xValue, silhouette_scores, marker='o')
        plt.title('Silhouette Score Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid()
        # plt.show()
        file_name = "silhouette_" + self.method + ".png"
        file_path = os.path.join(self.dirPath, file_name)
        plt.savefig(file_path, format='png')
        plt.close()

    def visualization_Rand(self, rand_scores, method="Other"):
        if not self.__check(rand_scores):
            return
        if method != "DBSCAN":
            xValue = self.k_clusters
        else:
            xValue = self.eps
        plt.figure(figsize=(8, 5))
        plt.plot(xValue, rand_scores, marker='o')
        plt.title('Random Index Evaluation')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Random Index')
        plt.grid()
        # plt.show()
        file_name = "rand_" + self.method + ".png"
        file_path = os.path.join(self.dirPath, file_name)
        plt.savefig(file_path, format='png')
        plt.close()

    def visualization_NMI(self, normalizedMI_scores, method="Other"):
        if not self.__check(normalizedMI_scores):
            return
        if method != "DBSCAN":
            xValue = self.k_clusters
        else:
            xValue = self.eps
        plt.figure(figsize=(8, 5))
        plt.plot(xValue, normalizedMI_scores, marker='o')
        plt.title('Normalized Mutual Information Evaluation')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Normalized Mutual Information')
        plt.grid()
        # plt.show()
        file_name = "NMI_" + self.method + ".png"
        file_path = os.path.join(self.dirPath, file_name)
        plt.savefig(file_path, format='png')
        plt.close()
    
    def visualization_AMI(self, adjustedMI_scores, method="Other"):
        if not self.__check(adjustedMI_scores):
            return
        if method != "DBSCAN":
            xValue = self.k_clusters
        else:
            xValue = self.eps
        plt.figure(figsize=(8, 5))
        plt.plot(xValue, adjustedMI_scores, marker='o')
        plt.title('Adjusted Mutual Information Evaluation')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Adjusted Mutual Information')
        plt.grid()
        # plt.show()
        file_name = "AMI_" + self.method + ".png"
        file_path = os.path.join(self.dirPath, file_name)
        plt.savefig(file_path, format='png')
        plt.close()
    
    def visualization_VMeasure(self, vMeasure_scores, method="Other"):
        if not self.__check(vMeasure_scores):
            return
        if method != "DBSCAN":
            xValue = self.k_clusters
        else:
            xValue = self.eps
        plt.figure(figsize=(8, 5))
        plt.plot(xValue, vMeasure_scores, marker='o')
        plt.title('V Measure Evaluation')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('V Measure')
        plt.grid()
        # plt.show()
        file_name = "VMeasure_" + self.method + ".png"
        file_path = os.path.join(self.dirPath, file_name)
        plt.savefig(file_path, format='png')
        plt.close()
    
    def visualization_FowlkesMallows(self, FowlkesMallows_scores, method="Other"):
        if not self.__check(FowlkesMallows_scores):
            return
        if method != "DBSCAN":
            xValue = self.k_clusters
        else:
            xValue = self.eps
        plt.figure(figsize=(8, 5))
        plt.plot(xValue, FowlkesMallows_scores, marker='o')
        plt.title('Fowlkes Mallows Evaluation')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Fowlkes Mallows')
        plt.grid()
        # plt.show()
        file_name = "FowlkesMallows_" + self.method + ".png"
        file_path = os.path.join(self.dirPath, file_name)
        plt.savefig(file_path, format='png')
        plt.close()

    def visualization_Dendrogram(self, X_scaled):
        Z = sch.linkage(X_scaled, method='ward')
        sch.dendrogram(Z) # 繪製dendrogram
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        # plt.show()
        file_name = "Dendrograms_" + self.method + ".png"
        file_path = os.path.join(self.dirPath, file_name)
        plt.savefig(file_path, format='png')
        plt.close()
    
    def visualization_ConfusionMatrix(self, confusionMatrix):
        for i, cm in enumerate(confusionMatrix): 
            plt.figure(figsize=(5, 4)) 
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
            plt.title(f'Confusion Matrix for Label {i}') 
            plt.xlabel('Predicted') 
            plt.ylabel('True') 
            # plt.show()
            file_name = "confusionMatrix_" + self.method + "_" + str(i) + ".png"
            file_path = os.path.join(self.dirPath, file_name)
            plt.savefig(file_path, format='png')
            plt.close()