import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def plot_perplexity(topics_list, train_perplexity_list, test_perplexity_list, best_count=None, file_name=None):
    fig, ax = plt.subplots()
    plt.title('Topics vs Perplexity')
    plt.plot(topics_list, train_perplexity_list, linestyle='--', marker='o', label='Train Perplexity')
    #plt.plot(topics_list, test_perplexity_list, linestyle='--', marker='o', label='Test Perplexity')
    if best_count:
        plt.axvline(x=best_count, color='black', linestyle='--', label='Best Number of Clusters: {}'.format(best_count))
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.legend()
    if file_name:
        plt.savefig('images/{}.png'.format(file_name))
    plt.show()


#input data
essays = pd.read_csv('project_essays.csv')

documents = (essays['Project Title'] + essays['Project Essay']).values

training_set_size = int(documents.shape[0]*.8)
test_set_size = int(documents.shape[0]*.2)

training_idx = np.random.randint(documents.shape[0], size=training_set_size)
test_idx = np.random.randint(documents.shape[0], size=test_set_size)
training, test = documents[training_idx], documents[test_idx]


# idx = (np.random.randint(documents.shape[0], size=training_set_size))
#
# #indices = numpy.random.permutation(x.shape[0])
# training_idx, test_idx = training_idx[:training_set_size], training_idx[training_set_size:]
# training, test = documents[training_idx,:], documents[test_idx,:]

count_vectorizer = CountVectorizer(analyzer='word',
                                    input='content',
                                    lowercase=True,
                                    max_df=.95,
                                    max_features=10000,
                                    min_df=1,
                                    ngram_range=(1, 1),
                                    preprocessor=None,
                                    stop_words='english',
                                    strip_accents=None,
                                    token_pattern='(?u)\\b\\w\\w+\\b',
                                    tokenizer=None,
                                    vocabulary=None)

train_tf = count_vectorizer.fit_transform(training)
test_tf = count_vectorizer.fit_transform(test)

topics = np.arange(2, 330, 30)
train_perplexities = []
test_perplexities = []


for n in topics:
    print('working on {} topics.'.format(n))

    lda = LatentDirichletAllocation(n_components=n, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    lda.fit(train_tf)

    train_gamma = lda.transform(train_tf)
    train_perplexity = lda.perplexity(train_tf, train_gamma)

    test_gamma = lda.transform(test_tf)
    test_perplexity = lda.perplexity(test_tf, test_gamma)

    train_perplexities.append(train_perplexity)
    test_perplexities.append(test_perplexity)


plot_perplexity(topics, train_perplexities, test_perplexities)
