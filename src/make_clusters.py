import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


categorical_data = pd.read_csv('projects_with_categorical_data.csv')
categorical_data.drop('Unnamed: 0', axis=1, inplace=True)
categorical_data.dropna(inplace=True)
print('Data loaded. Working with {} rows.'.format(categorical_data.shape[0]))

#western_rural_projects = categorical_data[(categorical_data['rural']==1)  & (categorical_data['W']==1)]

newdf = categorical_data.select_dtypes(include='number')

data = np.array(newdf)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print('Data Scaled')

n_clusters = np.arange(200, 310, 10)
sil_scores = []

for clusters in n_clusters:
    print("Working on {} clusters.".format(clusters))
    clusterer = KMeans(n_clusters=clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    print("For n_clusters =", clusters,
          "The average silhouette_score is :", silhouette_avg)
    sil_scores.append(silhouette_avg)

best_sil_score_idx = np.argmax(sil_scores)
optimum_k = n_clusters[best_sil_score_idx]

fig = plt.figure()
plt.title("Average Silhouette Scores vs. Number of Clusters")
plt.plot(n_clusters, sil_scores)
plt.axvline(x=optimum_k, color='black', linestyle='--', label='Best Number of Clusters: {}'.format(optimum_k))
plt.legend()
plt.savefig('images/zoomed_silhouette_score')
plt.show()
