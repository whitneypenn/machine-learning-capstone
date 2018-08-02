import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

#load data
data = pd.read_csv('data/categorical_data.csv')
test_data = pd.read_csv('data/test_categorical_data.csv')
num_clusters = 100


#clean data
data.drop('Unnamed: 0', axis=1, inplace=True)
data.dropna(inplace=True)
#clean test_data
test_data.drop('Unnamed: 0', axis=1, inplace=True)
test_data.dropna(inplace=True)
print('Data loaded. Working with {} samples.'.format(data.shape[0]))

data_to_cluster = data[['Project Resource Category',
       'Project Subject Category Tree', 'Project Subject Subcategory Tree',
       'Project Type', 'School Metro Type', 'Region', 'Project Grade Level Category']]
test_data_for_clustering = test_data[['Project Resource Category',
       'Project Subject Category Tree', 'Project Subject Subcategory Tree',
       'Project Type', 'School Metro Type', 'Region', 'Project Grade Level Category']]

## Cluster Using Optimum Clusters
kproto = KModes(n_clusters=num_clusters, init='random', verbose=True)
# here you use the unsclaed data and tell the model which columns are categorical
# and which ones are numerical
cluster_obj = kproto.fit(data_to_cluster)
labels = cluster_obj.labels_
centroids = cluster_obj.cluster_centroids_

#randomly sample from the test projects
for i in range(5):
    sample = test_data_for_clustering.sample()
    sample_title = test_data.iloc[sample.index.values]['Project Title'].values[0]

    #get a cluster prediction
    pred = cluster_obj.predict(sample)

    #find the projects with that label, select three of them:
    rec_idx = data_to_cluster[labels == pred].sample(3)
    rec_titles = data.iloc[rec_idx.index.values]['Project Title'].values

    print('Project {}'.format(i+1))
    print('Seed Project Title: ', sample_title)
    print("Recommendations: ")
    for rec in rec_titles:
        print('   ', rec)
    print(' ')
