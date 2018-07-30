import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


categorical_data = pd.read_csv('projects_with_categorical_data.csv')
categorical_data.drop('Unnamed: 0', axis=1, inplace=True)
categorical_data.dropna(inplace=True)
print('Data loaded. Working with {} rows.'.format(categorical_data.shape[0]))

small_data = categorical_data.sample(500000)

newdf = small_data.select_dtypes(include='number')

data = np.array(newdf)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print('Data Scaled')

n_clusters = np.arange(400, 600, 20)
sil_scores = []

for clusters in n_clusters:
    print("Working on {} clusters.".format(clusters))
    clusterer = KMeans(n_clusters=clusters, random_state=10, n_jobs=-2)
    cluster_labels = clusterer.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    print("For n_clusters =", clusters,
          "The average silhouette_score is :", silhouette_avg)
    sil_scores.append(silhouette_avg)
