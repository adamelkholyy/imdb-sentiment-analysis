# IMDB Sentiment Analysis
<strong>Adam El Kholy</strong> \
<strong>University of Bath</strong> \
Last Updated: <strong>06/12/2023</strong>

Free to use under the Apache 2.0 license \
For use with the [IMDB reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) dataset available on Kaggle 

The following notebook allows you to train and evaluate the following models for the task of sentiment analysis using on the IMDB movie dataset
- Multinomial Naive Bayes (manual implementation)
- Gaussian Naive Bayes (manual implementation)
- Sklearn MNB
- Sklearn GNB
- Logistic Regression
- Support Vector Machines

See ```BERT.ipynb``` for the evaluation of BERT (cased and uncased) on the same task


```python
import os
import nltk
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
np.random.seed(42)
```

# Loading the Data


```python
"""
Directory structure: 
data/ 
    pos/
        1.txt
        2.txt
        ...
    neg/
        1.txt
        2.txt
        ...
where pos/ and neg/ contain positive and negative reviews respectively
"""
def read_corpus(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    corpus = []
    for file in files:
        with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
            document = f.read()
            corpus.append(document)
    return corpus
```


```python
# unpack data into corpus vars
positive_corpus = read_corpus("data/pos/")
negative_corpus = read_corpus("data/neg/")
corpus = positive_corpus + negative_corpus
positive_labels = len(positive_corpus)
negative_labels = len(negative_corpus)
corpus_length = len(corpus)

# sanity check
print(positive_corpus[0][:128])
print(negative_corpus[1][:128])
```

    Homelessness (or Houselessness as George Carlin stated) has been an issue for years but never a plan to help those on the street ...
    This film is about a male escort getting involved in a murder investigation that happened in the circle of powerful men's wives. ...
    

# Splitting the Data


```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

r_seed = 42

def get_test_train_dev_split(X):
    y = np.concatenate([np.ones(positive_labels), np.zeros(negative_labels)])

    # we first split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, 
                                                        shuffle=True, random_state=r_seed)
    
    # then split train into train and development set 
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, 
                                                      test_size=0.15, 
                                                      shuffle=True, random_state=r_seed)
   
    #68% train, 12% validation, 20% test
    return X_train, y_train, X_test, y_test, X_dev, y_dev

X_train, y_train, X_test, y_test, X_dev, y_dev = get_test_train_dev_split(stem_unigram_tf_idf)
print(X_train.shape)
print(X_test.shape)
print(X_dev.shape)
print(X_train.toarray()[0])

```

    (2720, 29379)
    (800, 29379)
    (480, 29379)
    [0.        0.0105175 0.        ... 0.        0.        0.       ]
    

# Feature Generation Using N-Grams


```python
from nltk import word_tokenize
from nltk import ngrams

def text_to_ngrams(sentence, n, remove_stopwords=True):
    # set of stopwords to remove
    stoplist = set(stopwords.words('english')) 
    if not remove_stopwords:
        stoplist = set()
    tokenised_words = [word for word in word_tokenize(sentence.lower()) 
                       if word not in stoplist and word not in string.punctuation and word != "br"] 
                       # a list of tokenised words with stop-words, punctuation, and <br>s removed
    # apply nltk n-grams algorithm
    zipped_grams = ngrams(tokenised_words, n) 
    return list(zipped_grams)
```


```python
sentence = "I am Ozymandias, king of kings, look upon my works ye mighty and despair"
grams = text_to_ngrams(sentence, 3)
print(grams)
for gram in grams:
    print(gram)
```

    [('ozymandias', 'king', 'kings'), ('king', 'kings', 'look'), ('kings', 'look', 'upon'), ('look', 'upon', 'works'), ('upon', 'works', 'ye'), ('works', 'ye', 'mighty'), ('ye', 'mighty', 'despair')]
    ('ozymandias', 'king', 'kings')
    ('king', 'kings', 'look')
    ('kings', 'look', 'upon')
    ('look', 'upon', 'works')
    ('upon', 'works', 'ye')
    ('works', 'ye', 'mighty')
    ('ye', 'mighty', 'despair')
    


```python
# convert the entire corpus to ngrams
def corpus_to_ngrams(corpus, n, remove_stopwords=True):
    new_corpus = []
    for text in corpus:
        new_corpus.append(text_to_ngrams(text, n, remove_stopwords))
    return new_corpus
```


```python
corpus_unigrams = corpus_to_ngrams(corpus, 1)
print(corpus_unigrams[0])
```

    [('homelessness',), ('houselessness',), ('george',), ('carlin',), ('stated',), ('issue',), ('years',), ('never',), ('plan',), ('help',), ('street',) ... ]
    


```python
corpus_bigrams = corpus_to_ngrams(corpus, 2)
print(corpus_bigrams[0])
```

    [('homelessness', 'houselessness'), ('houselessness', 'george'), ('george', 'carlin'), ('carlin', 'stated'), ('stated', 'issue'), ('issue', 'years'), ('years', 'never'), ('never', 'plan'), ('plan', 'help'), ('help', 'street') ... ]
    


```python
corpus_trigrams = corpus_to_ngrams(corpus, 3)
print(corpus_trigrams[0])
```

    [('homelessness', 'houselessness', 'george'), ('houselessness', 'george', 'carlin'), ('george', 'carlin', 'stated'), ('carlin', 'stated', 'issue'), ('stated', 'issue', 'years'), ('issue', 'years', 'never'), ('years', 'never', 'plan'), ('never', 'plan', 'help'), ('plan', 'help', 'street') ... 
    


```python
# with stopwords included
corps_unigrams_with_stopwords = corpus_to_ngrams(corpus, 1, remove_stopwords=False)
print(corps_unigrams_with_stopwords[0])
```

    [('homelessness',), ('or',), ('houselessness',), ('as',), ('george',), ('carlin',), ('stated',), ('has',), ('been',), ('an',), ('issue',), ('for',), ('years',), ('but',), ('never',), ('a',), ('plan',), ('to',), ('help',), ('those',), ('on',), ('the',), ('street',) ...]
    

# Feature Selection using Lemmatization and Stemming


```python
def apply_stemming(text):
    st = LancasterStemmer()
    word_list = [" ".join(st.stem(gram) for gram in ngram) for ngram in text]
                # stems the list of ngram tuples using nltk's LancasterStemmer
    return word_list
```


```python
stemmed_text = apply_stemming(grams)
for feature in stemmed_text:
    print(feature)
```

    ozymandia king king
    king king look
    king look upon
    look upon work
    upon work ye
    work ye mighty
    ye mighty despair
    


```python
def apply_lemmatization(text):
    lm = WordNetLemmatizer()
    word_list = [" ".join(lm.lemmatize(gram) for gram in ngram) for ngram in text]
                # lemmatizes the list of ngram tuples
    return word_list
```


```python
lemmatized_text = apply_lemmatization(grams)
for feature in lemmatized_text:
    print(feature)
```

    ozymandias king king
    king king look
    king look upon
    look upon work
    upon work ye
    work ye mighty
    ye mighty despair
    


```python
# apply a given stemming or lemmatization function to the corpus
def apply_to_corpus(func, corpus):
    new_corpus = []
    for text in corpus:
        new_corpus.append(func(text))
    return new_corpus
```


```python
lemmatized_unigrams = apply_to_corpus(apply_lemmatization, corpus_unigrams)
stemmed_unigrams = apply_to_corpus(apply_stemming, corpus_unigrams)
print(lemmatized_unigrams[0][:10])
print(stemmed_unigrams[0][:10])
```

    ['homelessness', 'houselessness', 'george', 'carlin', 'stated', 'issue', 'year', 'never', 'plan', 'help']
    ['homeless', 'houseless', 'georg', 'carlin', 'stat', 'issu', 'year', 'nev', 'plan', 'help']
    

# TF-IDF Feature Extraction

First we set about generating a shared vocabulary, containing the number of documents each unique word occurs in, so as to calculate TF and IDF values


```python
def generate_shared_vocabulary(corpus):
    words = {}
    for text in corpus:
        for word in set(text): 
            # set(text) removes duplicates, meaning the dictionary contains document frequency values 
            if word in words:
                words[word] += 1
            else:
                words[word] = 1
    return words
```


```python
shared_vocabulary = generate_shared_vocabulary(stemmed_unigrams)
```

Now we generate our TF-IDF matrix, where each row represents a document in the corpus. We utilise the scipy sparse matrix data structure in order to save memory


```python
from scipy import sparse

def generate_tf_idf_matrix(corpus, shared_vocabulary, one_hot=False):
    N = len(shared_vocabulary)
    shared_vocabulary_list = list(shared_vocabulary)
    matrix = sparse.lil_matrix(np.zeros([corpus_length, N]))
    # sparse list of lists to store our tf_idf values

    for i, text in enumerate(corpus):
        # calculate tf_idf for each feature in each document and insert in correct index
        for word in text: 
            index = shared_vocabulary_list.index(word)
            tf = text.count(word) / len(text)
            idf = np.log10(N / (shared_vocabulary[word] +1))
            # if using one_hot vectors for SVM and LogReg then simply insert 1
            matrix[i, index] = tf * idf if not one_hot else 1
    return matrix
```


```python
stem_unigram_tf_idf = generate_tf_idf_matrix(stemmed_unigrams, shared_vocabulary)
stem_unigram_tf_idf[0].toarray()
```




    array([[0.01005802, 0.00693199, 0.01359507, ..., 0.        , 0.        ,
            0.        ]])



# Multinomial Naïve Bayes


```python
"""
Calculates the likelihood that a feature x belongs to class C, p(x|C)
    labels: 1 for the class whose likelihood is being calculated, 0 for any others
    data: training data, i.e. our TF-IDF matrix of all documents
    alpha: value for laplace smoothing
"""
def calculate_likelihoods(data, labels, alpha=1.0):
    N = data.shape[1]
    likelihoods = np.zeros([N])
    for i in range(N):
        feature = data[:, i].toarray().flatten()
        likelihoods[i] = (np.sum(feature * labels) + alpha)  / (np.sum(labels) + alpha) 
        # likelihood calculation (p(X|C)) using laplace smoothing
    return likelihoods
        
```


```python
"""
Calculates the likelihoods for both classes given the training data as well as priors for both classes
"""
def train_multinomial_bayes(X_train, y_train):
    # inverted label array, denoting 1 for the negative class and 0 for the positive, used for calculating likelihood and priors
    inverted_y_train = np.array([not y for y in y_train]).astype(int)
    
    pos_likelihoods = calculate_likelihoods(X_train, y_train)
    neg_likelihoods = calculate_likelihoods(X_train, inverted_y_train)
    pos_log_likelihoods = np.log(pos_likelihoods)
    neg_log_likelihoods = np.log(neg_likelihoods)

    pos_prior = np.sum(y_train) / len(y_train)
    neg_prior = np.sum(inverted_y_train) / len(inverted_y_train)
    pos_log_prior = np.log(pos_prior)
    neg_log_prior = np.log(neg_prior)
    return pos_log_likelihoods, neg_log_likelihoods, pos_log_prior, neg_log_prior
    
```


```python
"""
Assigns a class label to a given document using likelihoods and priors
"""
def get_multinomial_class_label(data, document):
    pos_log_likelihoods, neg_log_likelihoods, pos_log_prior, neg_log_prior = data

    # unpack our sparse vector:
    features = np.nonzero(document)[0] 
    pos_total = 0
    neg_total = 0

    for index in features: 
        # sum log likelihoods for each feature, for both classes
        pos_total += pos_log_likelihoods[index]
        neg_total += neg_log_likelihoods[index]
        
    # add priors
    pos_total += pos_log_prior 
    neg_total += neg_log_prior
    class_label = 1 if pos_total > neg_total else 0 
    return class_label
```


```python
"""
Runs the entire pipeline for MNB and returns a predictions array
"""
def test_train_multinomial_bayes(train_data, train_labels, test_data):
    data = train_multinomial_bayes(train_data, train_labels)
    predictions = []
    for i, _ in enumerate(test_data):
        doc =  test_data[i].toarray().flatten()
        label = get_multinomial_class_label(data, doc)
        predictions.append(label)
    return predictions
    
```


```python
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# outputs full model evaluation
def evaluate_model(test_labels, predictions):
    print(f"accuracy: {accuracy_score(test_labels, predictions)}")
    print(f"precision: {precision_score(test_labels, predictions)}")
    print(f"recall: {recall_score(test_labels, predictions)}")
    print(f"f1 score: {f1_score(test_labels, predictions)}")
    print()
```

Evaluation for Multinomial Naive Bayes on the development set for stemmed unigrams


```python
predictions = test_train_multinomial_bayes(X_train, y_train, X_dev)
evaluate_model(y_dev, predictions)
```

    accuracy: 0.875
    precision: 0.9267015706806283
    recall: 0.7937219730941704
    f1 score: 0.8550724637681159
    

# Gaussian Naïve Bayes


```python
"""
Calculates the mean and standard distribution for a given feature using TF-IDF scores across all documents
    labels: 1 for the class whose likelihood is being calculated, 0 for any others
    data: training data, i.e. our TF-IDF matrix of all documents
    alpha: value for laplace smoothing
"""
def calculate_guassian_distributions(data, labels, alpha=1e-10):
    pos_distribution = []
    neg_distribution = []
    inverted_labels = np.array([not y for y in labels]).astype(int)

    # calculate means and standard deviations for each feature in order to compute distributions
    for i in range(data.shape[1]):
        feature = data[:, i].toarray().flatten() 

        # collects the instances of the feature being present in positive and negative class resp.
        pos_feature = feature * labels
        neg_feature = feature * inverted_labels
        pos_distribution.append((np.mean(pos_feature) + alpha, np.std(pos_feature) + alpha))
        neg_distribution.append((np.mean(neg_feature) + alpha, np.std(neg_feature) + alpha))
    return pos_distribution, neg_distribution


```


```python
"""
Calculates the likelihoods for both classes given the training data as well as priors for both classes
"""
def train_gaussian_bayes(X_train, y_train):
    # an inverted label array, denoting 1 for the negative class and 0 for the positive
    inverted_y_train = np.array([not y for y in y_train]).astype(int)

    pos_distribution, neg_distribution = calculate_guassian_distributions(X_train, y_train)
    pos_log_distribution = np.log(pos_distribution)
    neg_log_distribution = np.log(neg_distribution)

    pos_prior = np.sum(y_train) / len(y_train)
    neg_prior = np.sum(inverted_y_train) / len(inverted_y_train)
    pos_log_prior = np.log(pos_prior)
    neg_log_prior = np.log(neg_prior)

    return pos_log_distribution, neg_log_distribution, pos_log_prior, neg_log_prior
```


```python
# fits value (the TF-IDF score for a given feature in the input document) to the guassian distribution of said feature 
def gaussian(mean, sd, value):
    exponent = (- (value - mean)**2 ) / (2 * sd**2)
    value = (1 / np.sqrt(2 * np.pi * sd**2)) * np.exp(exponent)
    if np.isnan(value): # 0 if NaN
        return 0
    return value
```


```python
# produces a class label for a given document using our GNB
def get_gaussian_class_label(data, document):
    pos_distribution, neg_distribution, pos_log_prior, neg_log_prior = data
    features = np.nonzero(document)[0]
    pos_total = 0
    neg_total = 0
    for index in features: # calculates the likelihood using the mean, sd, and value for each feature
                           # usage: gaussian(mean, sd, x)
                           # pos_distribution[i] = (mean, sd), where i is feature index
        pos_total += gaussian(pos_distribution[index][0], pos_distribution[index][1], 
                            document[index] ) 
        neg_total += gaussian(neg_distribution[index][0], neg_distribution[index][1],
                            document[index] ) 
    pos_total += pos_log_prior
    neg_total += neg_log_prior
    label = 1 if pos_total > neg_total else 0
    return label
```


```python
# full GNB pipeline
def test_train_gaussian_bayes(train_data, train_labels, test_data):
    data = train_gaussian_bayes(train_data, train_labels)
    predictions = []
    for i, v in enumerate(test_data):
        doc =  test_data[i].toarray().flatten()
        label = get_gaussian_class_label(data, doc)
        predictions.append(label)
    return predictions
```

Evaluation on development set for Guassian Bayes using stemmed unigrams


```python
predictions = test_train_gaussian_bayes(X_train, y_train, X_dev)
evaluate_model(y_dev, predictions)
```

    accuracy: 0.8645833333333334
    precision: 0.8291666666666667
    recall: 0.8923766816143498
    f1 score: 0.8596112311015119
    
    

# Sklearn MNB and GNB Models


```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

# full pipeline for training and evaluation of sklearn models
def test_train_sklearn_models(X_train, y_train, X_test, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("Sklearn Multinomial Bayes")
    evaluate_model(y_test, predictions)

    clf2 = GaussianNB()
    clf2.fit(X_train.toarray(), y_train)
    predictions = clf2.predict(X_test.toarray())
    print("Sklearn Gaussian Bayes")
    evaluate_model(y_test, predictions)
test_train_sklearn_models(X_train, y_train, X_dev, y_dev)
```

    Sklearn Multinomial Bayes
    accuracy: 0.85625
    precision: 0.8155737704918032
    recall: 0.8923766816143498
    f1 score: 0.8522483940042827
    
    Sklearn Gaussian Bayes
    accuracy: 0.65
    precision: 0.611336032388664
    recall: 0.6771300448430493
    f1 score: 0.6425531914893617
    
    

For stemmed unigrams our own implementation of multinomial bayes achieves a similar accuracy and f1score, higher precision and a lower recall in comparison to the prebuilt sklearn model. Own implementation of Gaussian Bayes far outperforms sklearn's model on accuracy, precision, and f1 score

# MNB and GNB Evaluation


```python
# full training and evaluation for a given corpus using all models
def evaluate_on_corpus(corpus):
    shared_vocabulary = generate_shared_vocabulary(corpus)
    tf_idf_matrix = generate_tf_idf_matrix(corpus, shared_vocabulary)
    X_train, y_train, X_test, y_test, X_dev, y_dev = get_test_train_dev_split(tf_idf_matrix)

    multinomial_predictions = test_train_multinomial_bayes(X_train, y_train, X_dev)
    print("Multinomial Bayes")
    evaluate_model(y_dev, multinomial_predictions)
    gaussian_predictions = test_train_gaussian_bayes(X_train, y_train, X_dev)
    print("Gaussian Bayes")
    evaluate_model(y_dev, gaussian_predictions)
    test_train_sklearn_models(X_train, y_train, X_dev, y_dev)
    return tf_idf_matrix
```

In testing our best feature set was unigrams with stemming. Let's now evaluate on the test set


```python
X_train, y_train, X_test, y_test, X_dev, y_dev = get_test_train_dev_split(stem_unigram_tf_idf)
multinomial_predictions = test_train_multinomial_bayes(X_train, y_train, X_test)
print("Multinomial Bayes")
evaluate_model(y_test, multinomial_predictions)
gaussian_predictions = test_train_gaussian_bayes(X_train, y_train, X_test)
print("Gaussian Bayes")
evaluate_model(y_test, gaussian_predictions)
test_train_sklearn_models(X_train, y_train, X_test, y_test)
```

    Multinomial Bayes
    accuracy: 0.80875
    precision: 0.8768328445747801
    recall: 0.7292682926829268
    f1 score: 0.796271637816245
    
    Gaussian Bayes
    accuracy: 0.8275
    precision: 0.8105022831050228
    recall: 0.8658536585365854
    f1 score: 0.8372641509433961
    
    Sklearn Multinomial Bayes
    accuracy: 0.82
    precision: 0.8093023255813954
    recall: 0.848780487804878
    f1 score: 0.8285714285714286
    
    Sklearn Gaussian Bayes
    accuracy: 0.6525
    precision: 0.6578947368421053
    recall: 0.6707317073170732
    f1 score: 0.6642512077294686
    
    

# Logistic Regression

Let's first compare the use of one hot matrices to our usual TF-IDF vectors


```python
one_hot_matrix = generate_tf_idf_matrix(stemmed_unigrams, shared_vocabulary, one_hot=True)
print(one_hot_matrix[0].toarray())
```

    [[1. 1. 1. ... 0. 0. 0.]]
    


```python
from sklearn.linear_model import LogisticRegression

# test base logistic regression model's accuracy using tf-idf features
X_train, y_train, X_test, y_test, X_dev, y_dev = get_test_train_dev_split(stem_unigram_tf_idf)
clf = LogisticRegression(random_state=r_seed, solver="sag")
clf.fit(X_train, y_train)
clf.score(X_dev, y_dev)
```




    0.8104166666666667




```python
# test base logistic regression model's accuracy using one-hot vectors
X_train, y_train, X_test, y_test, X_dev, y_dev = get_test_train_dev_split(one_hot_matrix)
clf = LogisticRegression(random_state=r_seed, solver="sag")
clf.fit(X_train, y_train)
clf.score(X_dev, y_dev)
```

    0.8333333333333334
    

Clearly one hot matrices lead to higher performance. Let's try evaluating for stemmed unigrams


```python
predictions = clf.predict(X_dev)
evaluate_model(y_dev, predictions)
```

    accuracy: 0.8333333333333334
    precision: 0.8122270742358079
    recall: 0.8340807174887892
    f1 score: 0.8230088495575221
    
    

Now let's evaluate lemmatized unigrams


```python
lem_uni_one_hot = generate_tf_idf_matrix(lemmatized_unigrams, 
                                         generate_shared_vocabulary(lemmatized_unigrams), one_hot=True)
X_train, y_train, X_test, y_test, X_dev, y_dev = get_test_train_dev_split(lem_uni_one_hot)
clf = LogisticRegression(random_state=r_seed, solver="sag").fit(X_train, y_train)
predictions = clf.predict(X_dev)
evaluate_model(y_dev, predictions)
```

    accuracy: 0.8583333333333333
    precision: 0.8414096916299559
    recall: 0.8565022421524664
    f1 score: 0.8488888888888888
    

Finally let's evaluate stemmed unigrams without stopword removal


```python
stemmed_unigrams_stopwords = apply_to_corpus(apply_stemming, corps_unigrams_with_stopwords)
stem_uni_stop_one_hot = generate_tf_idf_matrix(stemmed_unigrams_stopwords, 
                                               generate_shared_vocabulary(stemmed_unigrams_stopwords), one_hot=True)
X_train, y_train, X_test, y_test, X_dev, y_dev = get_test_train_dev_split(stem_uni_stop_one_hot)
clf = LogisticRegression(random_state=r_seed, solver="sag")
clf.fit(X_train, y_train)
predictions = clf.predict(X_dev)
evaluate_model(y_dev, predictions)
```

    accuracy: 0.8375
    precision: 0.8111587982832618
    recall: 0.8475336322869955
    f1 score: 0.8289473684210525
    

The best performing feature set was thus lemmatization with stop-word removal

# Support Vector Machines

Let's define our SVM classifier and evaluate stemmed unigrams with stopword removal


```python
from sklearn import svm

# svm classifer train and evaluation pipeline
def svm_classifier(feature_matrix):
    X_train, y_train, X_test, y_test, X_dev, y_dev = get_test_train_dev_split(feature_matrix)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_dev)
    evaluate_model(y_dev, predictions)
    
svm_classifier(one_hot_matrix)
```

    accuracy: 0.8520833333333333
    precision: 0.8247863247863247
    recall: 0.8654708520179372
    f1 score: 0.8446389496717724
    
    

SVM with lemmatized unigrams


```python
svm_classifier(lem_uni_one_hot)
```

    accuracy: 0.85
    precision: 0.8212765957446808
    recall: 0.8654708520179372
    f1 score: 0.8427947598253275
    
    

SVM with stemmed unigrams and no stopword removal


```python
svm_classifier(stem_uni_stop_one_hot)
```

    accuracy: 0.8520833333333333
    precision: 0.8247863247863247
    recall: 0.8654708520179372
    f1 score: 0.8446389496717724
    
    

Our highest performing set for the SVM classifier was lemmatization with stopword removal

<h1>Logistic Regression Hyperparameter Optimisation

For our LogReg classifier we found that parallelizing does not increase performance. The tuned hyperparameters of our fine tuned model were as follows
+ C = 1.5
+ solver = "lbfgs"
+ penalty = "l2"
+ n_jobs = None (i.e. not parallelized)


```python
clf = LogisticRegression(C=1.5, 
                         solver="lbfgs", 
                         random_state=r_seed, 
                         penalty="l2", 
                         n_jobs=None)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
evaluate_model(y_test, predictions)
```

    accuracy: 0.83625
    precision: 0.8329355608591885
    recall: 0.8512195121951219
    f1 score: 0.8419782870928829
    
    

# SVM Hyperparameter Optimisation </h1>

The final tuned hyperparameters of our SVM model were as follows
+ C = 0.9
+ kernel = "rbf" 
+ gamma = "scale"


```python
clf = svm.SVC(C=0.9, 
              kernel="rbf", 
              gamma="scale")
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
evaluate_model(y_test, predictions)
```

    accuracy: 0.82625
    precision: 0.8086560364464692
    recall: 0.8658536585365854
    f1 score: 0.8362779740871613
    
    
