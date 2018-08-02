# DonorsChoose.org Project Recommender  

## Background:
[DonorsChoose.org](https://www.donorschoose.org/)

## Question:

## Data


## Recommend Based on Clusters of Existing Categories
### K-means Clustering

![](images/average_silhouette_score.png)

Kmeans isn't suitable for high-dimensional categorical values, because you run into sparse data. In practice, this meant that a lot of my clusters at my best silhouette score were alone in their clusters.

I tried to see if I could find some point that traded off silhouette score with minimum cluster size or average cluster size.

![](images/silhouette_score_with_cluster_size.png)

No matter how many clusters I used, there were always clusters with only one project. Not useful for a recommender, since you need to be able to pull from the clusters to recommend other, similar projects.


### K-mode Clustering
K-mode works very similarly to KMeans in the mechanics: it finds centroids, calculates 'distance' from points to centroids, and iterates until the assignments don't move. Instead of distance, k-modes calculates 'dissimilarity'. Between each data point and the centroid, a value of one is given to features that are categorial mis-matches, and a value of 0 is given to the features that do match. The dissimilarity is a summation of those values, so the larger
the number of mismatches of categorical values between the data point and the centroid is, the more dissimilar the two objects. K-mode then assigns data points to centroids based on smallest values of dissimilarity. [More Information Here.](http://www.irma-international.org/viewtitle/10828/)

Cost function is defined as the sum distance of all points to their respective cluster centroids. (Total dissimilarity of all the points to their respective cluster centroids, so you're attempting to minimize the total dissimilarity among all the clusters.)
![](images/cost_with_k_modes_using_right_data_random_init.png)

#### Clusters vs DonorsChoose Categories

| Unique Values | Project Resource Category | Project Subject Category Tree | Project Subject Subcategory Tree | Project Type | School Metro Type | Region | Project Grade Level Category |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Number Of unique values| 17 | 46 | 320 | 3 | 5 | 4 | 4 |
| Number of unique values in 100 clusters | 9 | 23 | 41 | 3 | 5 | 4 | 4|
| Number of unique values in 200 clusters | 13 | 25 | 67 | 3 | 5 | 4 | 4|
| Number of unique values in 300 clusters | 16 | 28 | 82 | 3 | 5 | 4 | 4 |


## Recommend Based on Document Topic Distributions Using LDA


### Words to Vectors
I used SKLearn's CountVectorizer to calculate a term frequency matrix.

| Parameter | Value |
| --- | --- |
| ``max_df`` (corpus-specific stop words) | .9 |
| ``stop_words`` | 'english' + 'school' |
| ``max_features`` | 10000 |
| ``min_df`` | 1 |
| ``n_gram_range`` | (1, 1) |

### Hyperparameters

| Parameter | Value |
| --- | --- |
| ``doc_topic_prior`` (alpha) | None (1 / n_components) |
| ``topic_word_prior`` (Beta) | None (1 / n_components) |
| ``learning_method`` | 'online' |
| ``learning_decay`` | .7 |
| ``learning_offset`` | 50. |

* The ``doc_topic_prior`` and ``topic_word_prior``
* Online variational Bayes method. In each EM update, use mini-batch of training data to update the ``components_`` variable incrementally. Learning rate is controlled by ``learning_decay`` and the ``learning_offset`` parameters.

### Choosing Topics
#### 5 Topics
![](images/5_topic_lda.png)

#### 10 Topics
![](images/10_topic_lda.png)

#### 15 Topics
![](images/15_topic_lda.png)

#### 20 Topics
![](images/20_topic_lda.png)

#### Analysis
15 topics looks like the way to go - Each of the topics is specific and distinct. Here's what I named the topics.

| Topic Interpretation | Words |
| --- | --- |
| Art  | art, create, arts, creative, projects |
| Classroom Necessities | learning, classroom, learn, help, use |
| Books | reading, books, read, book, love |
| Student Basic Needs | supplies, need, materials, help, paper |
| Classroom Furniture | classroom, seating, learning, sit, chairs |
| Hands-on Science | science, world, hands, life, experience |
| Extracurricular Activities | friday, cheer, devote, flashlights, flashlight |
| Refugee Resources or International Projects | American, African, Hispanic, software, war |
| 9 | battle, Haiti, enticing, mouthpieces, ereaders |  
| Teacher Training | academy, training, conference, seat, sacks |
| Gardening and Food | food, healthy, play, garden, life |
| Technology Supplies | technology, math, use, ipad, ipads |
| STEM Labs | stem, problem, create, engineering, build |
| Music | music, instruments, play, band, program |
| Sports Equipment | team, equipment, sports, game, club |



## Tools Used
- Pandas
- Numpy
- MatPlotLib
- SKLearn
- [KModes](https://pypi.org/project/kmodes/)

## References
