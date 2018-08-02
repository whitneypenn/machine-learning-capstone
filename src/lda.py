import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text

def get_top_words(model, feature_names, n_top_words):
    topic_word_probs = []
    distr = (model.components_ / model.components_.sum(axis=1)[:, np.newaxis])
    #for topic_idx, topic in enumerate(model.components_):
    for topic_idx, topic in enumerate(distr):
        word_probs = []
        idx = 0
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            word = feature_names[i]
            prob = topic[i]
            word_probs.append((word, prob))
        topic_word_probs.append(word_probs)

    return topic_word_probs


def plot_perplexity(topics_list, perplexity_list, best_count=None, file_name=None):
    fig, ax = plt.subplots()
    plt.title('Topics vs Perplexity')
    plt.plot(topics_list, perplexity_list, linestyle='--', marker='o', label='Train Perplexity')
    if best_count:
        plt.axvline(x=best_count, color='black', linestyle='--', label='Best Number of Clusters: {}'.format(best_count))
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.legend()
    if file_name:
        plt.savefig('images/{}.png'.format(file_name))
    plt.show()

def test_perplexities(range_to_test, topic_frequency):
    perplexities = []
    for n in range_to_test:
        print('working on {} topics.'.format(n))

        lda = LatentDirichletAllocation(n_components=n, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)

        lda.fit(topic_frequency)

        perplexities.append(lda.perplexity(topic_frequency))

    plot_perplexity(range_to_test, perplexities)

def plot_top_words(num_rows, num_cols, top_words_list, image_name=None):
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10,num_rows*2))
    plt.subplots_adjust(left=.125, bottom=.1, right=.9, top=.9,
                    wspace=.8, hspace=.9)
    plt.suptitle('Distributions Over the Words for Each Topic', fontsize=16)
    for idx, (ax, topics) in enumerate(zip(np.ndarray.flatten(axs),top_words_list)):
        words = [i[0] for i in topics]
        probs = [i[1] for i in topics]

        y_pos = np.arange(len(words))
        ax.barh(y_pos, probs, align='center', color = 'navy')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('P(Words | Topic)')
        ax.set_title('Topic {}'.format(idx+1))
    if image_name:
        plt.savefig('images/{}.png'.format(image_name))
    plt.show()

if __name__=='__main__':
    essays = pd.read_csv('data/project_essays.csv')
    documents = (essays['Project Title'] + essays['Project Essay']).values


    # Set Things #
    my_additional_stop_words = ['school']
    my_stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)

    count_vectorizer = CountVectorizer(analyzer='word',
                                        input='content',
                                        lowercase=True,
                                        max_df=.9,
                                        max_features=10000,
                                        min_df=1,
                                        ngram_range=(1, 1),
                                        preprocessor=None,
                                        stop_words=my_stop_words,
                                        strip_accents=None,
                                        token_pattern='(?u)\\b\\w\\w+\\b',
                                        tokenizer=None,
                                        vocabulary=None)

    tf = count_vectorizer.fit_transform(documents)

    #Uncomment these lines to test your range of topics
    # topic_count = [3]
    #test_perplexities(topic_count, tf)

    n_top_words = 5
    ## if you want the plotting to be nice, make this a multiple of 5
    n_components = 15

    # Create an LDA model with a useful number of topics:
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)

    doc_topic_dist = lda.fit_transform(tf)

    # Inspect Results #

    tf_feature_names = count_vectorizer.get_feature_names()
    top_words = get_top_words(lda, tf_feature_names, n_top_words)
    plot_top_words(n_components//5, 5, top_words)


# LDA Recommender
