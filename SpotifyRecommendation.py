import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

class SpotifyRecommendation:

    def __init__(self, csv):

        self.csv = csv

        self.dataset = None

        self.columns = None

    def set_columns(self):

        self.columns = self.dataset.columns
        pass
    
    def read_csv(self):

        data = pd.read_csv(self.csv)
        data = data.drop([data.columns[1], data.columns[2], data.columns[3]], axis=1)
        self.dataset = data
        self.set_columns()

        pass

    def clean_dataset(self):

        self.dataset = self.dataset.dropna()

        orc = OrdinalEncoder()
        self.dataset[self.columns[:-1]] = orc.fit_transform(self.dataset[self.columns[:-1]])

        pass

    '''def most_correlations(self, n_features):
        labels_to_drop = set()
        for i in range(len(self.columns)):
            for j in range(i + 1):
                labels_to_drop.add((self.columns[i], self.columns[j]))
    
        unstacked_corr = self.dataset.corr().unstack()
        dropped_duplicates = unstacked_corr.drop(labels_to_drop)
        ordered_spotify = dropped_duplicates.sort_values(ascending = False)
        return ordered_spotify[0:n_features + 1]'''
    
    def normalize(self):

        for column_name in self.columns:
            sc = StandardScaler()
            reshaped_column = self.dataset[column_name].values.reshape(-1,1)
            self.dataset[column_name] = pd.DataFrame(sc.fit_transform(reshaped_column))

        pass

    def best_model(self, features, target, n_clusters):

        X = self.dataset[features]
        y = self.dataset[target]

        train_X, val_X, train_y, val_y = train_test_split(X,y, train_size = 0.2, random_state = 42)

        models = {}
        scores = {}
        k_cluster_centers = {}

        for i in range(2, n_clusters + 1):

            kmeans = KMeans(n_clusters = i, random_state = 42, n_init = "auto").fit(train_X)
            models[i] = kmeans
            k_cluster_centers[i] = kmeans.cluster_centers_
            scores[i] = silhouette_score(train_X, kmeans.labels_, metric = "euclidean")
    
        best_k = min(scores, key = scores.get)

        best_cluster_centers = k_cluster_centers[best_k]

        return best_k, best_cluster_centers
    
    def recommendation(self, sample_track, features, target):

        k_value, clusters = self.best_model(features, target)

        X = self.dataset[features]
        y = self.dataset[target]

        train_X, val_X, train_y, val_y = train_test_split(X,y, train_size = 0.2, random_state = 42)

        model = KMeans(n_clusters= k_value, random_state= 42, n_init= "auto", 
                       cluster_centers_= clusters).fit(train_X)
        
        indices = model.predict(sample_track)

        return self.dataset[indices] 






    



        
    


    