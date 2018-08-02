import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt

#load data
data = pd.read_csv('data/projects_with_categorial_data.csv')

#clean data
data.drop('Unnamed: 0', axis=1, inplace=True)
data.dropna(inplace=True)
print('Data loaded. Working with {} samples.'.format(data.shape[0]))

data_to_cluster = data[['Project Resource Category',
       'Project Subject Category Tree', 'Project Subject Subcategory Tree',
       'Project Type', 'School Metro Type', 'Region', 'Project Grade Level Category']]

### Find Optimal Clusters ###
n_clusters = np.arange(2, 1003, 100)
costs = []

for n in n_clusters:
    print("Working on {} clusters.".format(n))
    kproto = KModes(n_clusters=n, init='random', verbose=False)
    # here you use the unsclaed data and tell the model which columns are categorical
    # and which ones are numerical
    cluster_obj = kproto.fit(data_to_cluster)
    labels = cluster_obj.labels_
    cost = cluster_obj.cost_
    costs.append(cost)

#Plot Average Silhouette Scores
optimum_k = 100
fig, ax = plt.subplots()
plt.title("Cost vs. Number of Clusters - Random Centroid Initializations")
plt.plot(n_clusters, costs, linestyle='--', marker='o')
plt.axvline(x=optimum_k, color='black', linestyle='--', label='Best Number of Clusters: {}'.format(optimum_k))
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.legend()
plt.savefig('images/cost_with_k_modes_using_right_data_random_init.png')
plt.show()
