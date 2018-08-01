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
data = pd.read_csv('projects_with_categorial_data.csv')

#clean data
data.drop('Unnamed: 0', axis=1, inplace=True)
data.dropna(inplace=True)
print('Data loaded. Working with {} samples.'.format(data.shape[0]))

data_to_cluster = data[['Project Resource Category',
       'Project Subject Category Tree', 'Project Subject Subcategory Tree',
       'Project Type', 'School Metro Type', 'Region', 'Project Grade Level Category']]

### Find Optimal Clusters ###
n_clusters = np.arange(100, 1000, 100)
costs = []

for n in n_clusters:
    print("Working on {} clusters.".format(n))
    kproto = KModes(n_clusters=n, init='random', verbose=True)
    # here you use the unsclaed data and tell the model which columns are categorical
    # and which ones are numerical
    cluster_obj = kproto.fit(data)
    labels = cluster_obj.labels_
    cost = cluster_obj.cost_
    print('cost is: {}'.format(cost))
    costs.append(cost)


#Plot Average Silhouette Scores
fig, ax = plt.subplots()
plt.title("Cost vs. Number of Clusters - Random Centroid Initializations")
plt.plot(n_clusters, costs, linestyle='--', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
#plt.savefig('images/cost_with_k_modes_using_right_data_random_init.png')
plt.show()

## Cluster Using Optimum Clusters
# clusterer = KMeans(n_clusters=optimum_k, random_state=10)
# clusters = clusterer.fit(scaled_data)
#
# centroids = clusters.cluster_centers_
#
# # Print 10 titles in each cluster
# titles = categorical_data['Project Title'].values
# for i in range(optimum_k):
#     cluster = np.arange(0, data.shape[0])[labels == i]
#     sample_titles = np.random.choice(cluster, 5, replace=True)
#     print("cluster %d:" % i)
#     for title in sample_titles:
#         print("    %s" % titles[title])
