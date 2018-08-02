import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from scipy.spatial import distance



#load data
categorical_data = pd.read_csv('data/projects_with_dummy_data.csv')

#clean data
categorical_data.drop('Unnamed: 0', axis=1, inplace=True)
categorical_data.dropna(inplace=True)
print('Data loaded. Working with {} samples.'.format(categorical_data.shape[0]))

#get numerical columns only
newdf = categorical_data.select_dtypes(include='number')
#turn data into a numpy array
data = np.array(newdf)

#scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print('Data Scaled')

### Find Optimal Clusters ###
n_clusters = np.arange(2, 11)
sil_scores = []
min_cluster_size = []
average_cluster_size = []

## K Means Clustering ###
for clusters in n_clusters:
    print("Working on {} clusters.".format(clusters))
    clusterer = KMeans(n_clusters=clusters, random_state=10)
    #here you use scaled data
    cluster_labels = clusterer.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    print("For n_clusters =", clusters,
          "The average silhouette_score is :", silhouette_avg)
    sil_scores.append(silhouette_avg)
    titles = categorical_data['Project Title'].values
    cluster_size = []
    for i in range(clusters):
        cluster = np.arange(0, scaled_data.shape[0])[cluster_labels == i]
        cluster_size.append(len(cluster))
    min_cluster_size.append(min(cluster_size))
    average_cluster_size.append(np.mean(cluster_size))

best_sil_score_idx = np.argmax(sil_scores)
optimum_k = n_clusters[best_sil_score_idx]

#Plot Average Silhouette Scores
fig = plt.figure()
plt.title("Average Silhouette Scores vs. Number of Clusters")
#plt.plot(n_clusters, min_cluster_size, label='Smallest Cluster Size')
plt.plot(n_clusters, sil_scores, label="Silhouette Scores")
#plt.plot(n_clusters, average_cluster_size, label = 'Average Cluster Size')
plt.axvline(x=optimum_k, color='black', linestyle='--', label='Best Number of Clusters: {}'.format(optimum_k))
plt.legend()
plt.savefig('images/average_silhouette_score_with_k_prototypes.png')
plt.show()


### Cluster Using Optimum Clusters
# clusterer = KMeans(n_clusters=optimum_k, random_state=10)
# clusters = clusterer.fit(scaled_data)
#
# centroids = clusters.cluster_centers_

## Print 10 titles in each cluster
# titles = categorical_data['Project Title'].values
# for i in range(optimum_k):
#     cluster = np.arange(0, data.shape[0])[labels == i]
#     sample_titles = np.random.choice(cluster, 5, replace=True)
#     print("cluster %d:" % i)
#     for title in sample_titles:
#         print("    %s" % titles[title])
