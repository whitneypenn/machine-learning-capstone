import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

#load data
data = pd.read_csv('projects_with_categorial_data.csv')
num_clusters = 300


#clean data
data.drop('Unnamed: 0', axis=1, inplace=True)
data.dropna(inplace=True)
print('Data loaded. Working with {} samples.'.format(data.shape[0]))

data_to_cluster = data[['Project Resource Category',
       'Project Subject Category Tree', 'Project Subject Subcategory Tree',
       'Project Type', 'School Metro Type', 'Region', 'Project Grade Level Category']]

## Cluster Using Optimum Clusters
kproto = KModes(n_clusters=num_clusters, init='random', verbose=True)
# here you use the unsclaed data and tell the model which columns are categorical
# and which ones are numerical
cluster_obj = kproto.fit(data_to_cluster)
labels = cluster_obj.labels_
centroids = cluster_obj.cluster_centroids_

centroid_df = pd.DataFrame(centroids, columns=data_to_cluster.columns)

for col in data_to_cluster:
    print(col)
    print('centroid', len(centroid_df[col].unique()))
    print('data', len(data_to_cluster[col].unique()))
    print('')
