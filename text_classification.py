
# coding: utf-8

# In[89]:

import nltk
from collections import Counter
import pandas as pd
import string
import numpy as np
import sklearn


# # Text Classification [30pts]
# In this problem (again!), you will be analyzing the Twitter data we extracted using [this](https://dev.twitter.com/overview/api) api. This time, we extracted the tweets posted by the following six Twitter accounts: `realDonaldTrump, mike_pence, GOP, HillaryClinton, timkaine, TheDemocrats`
# 
# For every tweet, we collected two pieces of information:
# - `screen_name`: the Twitter handle of the user tweeting and
# - `text`: the content of the tweet.
# 
# We divided the tweets into three parts - train, test and hidden test - the first two of which are available to you in CSV files. For train, both the `screen_name` and `text` attributes were provided but for test, `screen_name` was hidden.
# 
# The overarching goal of the problem is to "predict" the political inclination (Republican/Democratic) of the Twitter user from one of his/her tweets. The ground truth (i.e., true class labels) is determined from the `screen_name` of the tweet as follows
# - `realDonaldTrump, mike_pence, GOP` are Republicans
# - `HillaryClinton, timkaine, TheDemocrats` are Democrats
# 
# Thus, this is a binary classification problem. 
# 
# The problem proceeds in three stages:
# 1. **Text processing (8pts)**: We will clean up the raw tweet text using the various functions offered by the [nltk](http://www.nltk.org/genindex.html) package.
# 2. **Feature construction (10pts)**: In this part, we will construct bag-of-words feature vectors and training labels from the processed text of tweets and the `screen_name` columns respectively.
# 3. **Classification (12pts)**: Using the features derived, we will use [sklearn](http://scikit-learn.org/stable/modules/classes.html) package to learn a model which classifies the tweets as desired. 
# 
# As mentioned earlier, you will use two new python packages in this problem: `nltk` and `sklearn`, both of which should be available with anaconda. However, NLTK comes with many corpora, toy grammars, trained models, etc, which have to be downloaded manually. This assignment requires NLTK's stopwords list and WordNetLemmatizer. Install them using:
# 
#   ```python
#   >>>nltk.download('stopwords')
#   >>>nltk.download('wordnet')
#   ```
# 
# Verify that the following commands work for you, before moving on.
# 
#   ```python
#   >>>lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()
#   >>>stopwords=nltk.corpus.stopwords.words('english')
#   ```
# 
# Let's begin!

# ## 1. Text Processing [6pts + 2pts]
# 
# You first task to fill in the following function which processes and tokenizes raw text. The generated list of tokens should meet the following specifications:
# 1. The tokens must all be in lower case.
# 2. The tokens should appear in the same order as in the raw text.
# 3. The tokens must be in their lemmatized form. If a word cannot be lemmatized (i.e, you get an exception), simply catch it and ignore it. These words will not appear in the token list.
# 4. The tokens must not contain any punctuations. Punctuations should be handled as follows: (a) Apostrophe of the form `'s` must be ignored. e.g., `She's` becomes `she`. (b) Other apostrophes should be omitted. e.g, `don't` becomes `dont`. (c) Words must be broken at the hyphen and other punctuations. 
# 
# Part of your work is to figure out a logical order to carry out the above operations. You may find `string.punctuation` useful, to get hold of all punctuation symbols. Your tokens must be of type `str`. Use `nltk.word_tokenize()` for tokenization once you have handled punctuation in the manner specified above.

# In[159]:

def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    result = []
    _text = text.lower()
    for char in string.punctuation:
        if char != "'":
            _text = _text.replace(char, " ")
    tokens = nltk.word_tokenize(_text.replace("'s", "").replace("'", ""))
    for token in tokens:
        try:
            result.append(str(lemmatizer.lemmatize(token)))
        except:
            continue
    return result


# You can test the above function as follows. Try to make your test strings as exhaustive as possible. Some checks are:
#     
#    ```python
#    >>> process("I'm doing well! How about you?")
#    ['im', 'doing', 'well', 'how', 'about', 'you']
#    ```
# 
#    ```python
#    >>> process("Education is the ability to listen to almost anything without losing your temper or your self-confidence.")
#    ['education', 'is', 'the', 'ability', 'to', 'listen', 'to', 'almost', 'anything', 'without', 'losing', 'your', 'temper', 'or', 'your', 'self', 'confidence']
#    ```

# In[160]:

# # AUTOLAB_IGNORE_START
# print process("This is a sample test inputs for  processing.")
# print process("Let''s'|don't|take|''practical?>data,\@sciences\':""I''d say }that's #worthless and ''!")
# print process("I'm doing well! How about you?")
# print process("Educations are the ability to listen to almost anything without losing your temper or your self-confidence.")
# print process("this")
# # AUTOLAB_IGNORE_STOP


# # In[139]:

# # AUTOLAB_IGNORE_START
# tweets = pd.read_csv("tweets_train.csv", na_filter=False)
# print tweets.head()
# # AUTOLAB_IGNORE_STOP


# You will now use the `process()` function we implemented to convert the pandas dataframe we just loaded from tweets_train.csv file. Your function should be able to handle any data frame which contains a column called `text`. The data frame you return should replace every string in `text` with the result of `process()` and retain all other columns as such. Do not change the order of rows/columns.

# In[162]:

def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ process all text in the dataframe using process_text() function.
    Inputs
        df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs
        pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                        the output from process_text() function. Other columns are unaffected.
    """
    processed_text = df['text'].apply(lambda text: process(text, lemmatizer))
    return df.assign(text=processed_text)

# # AUTOLAB_IGNORE_START
# processed_tweets = process_all(tweets)
# print processed_tweets.head()
# # AUTOLAB_IGNORE_STOP


# # In[163]:

# # AUTOLAB_IGNORE_START
# print tweets['text'][0]
# print processed_tweets['text'][0]
# AUTOLAB_IGNORE_STOP


# The output should be:
# 
#    ```python
#     >>> print processed_tweets.head()
#           screen_name                                               text
#     0             GOP  [rt, gopconvention, oregon, vote, today, that,...
#     1    TheDemocrats  [rt, dwstweets, the, choice, for, 2016, is, cl...
#     2  HillaryClinton  [trump, calling, for, trillion, dollar, tax, c...
#     3  HillaryClinton  [timkaine, guiding, principle, the, belief, th...
#     4        timkaine  [glad, the, senate, could, pas, a, thud, milco...
#     ```

# ## 2. Feature Construction [4pts + 4pts + 2pts]
# The next step is to derive feature vectors from the tokenized tweets. In this section, you will be constructing a bag-of-words TF-IDF feature vector.
# 
# But before that, as you may have guessed, the number of possible words is prohibitively large and not all of them may be useful for our classification task. Our first sub-task is to determine which words to retain, and which to omit. The common heuristic is to construct a frequency distribution of words in the corpus and prune out the head and tail of the distribution. The intuition of the above operation is as follows. Very common words (i.e. stopwords) add almost no information regarding similarity of two pieces of text. Conversely, very rare words tend to be typos. 
# 
# As NLTK has a list of in-built stop words which is a good substitute for head of the distribution, we will now implement a function which identifies rare words (tail). We will consider a word rare if it occurs not more than once in whole of tweets_train.csv.
# 
# Using `collections.Counter` will make your life easier.
#    ```python
#    >>> Counter(['sample', 'test', 'input', 'processing', 'sample'])
#     Counter({'input': 1, 'processing': 1, 'sample': 2, 'test': 1})
#    ```
# For details on other operations you can perform with Counter, see [this](https://docs.python.org/2/library/collections.html#collections.Counter) page.

# In[183]:

def get_rare_words(processed_tweets):
    """ use the word count information across all tweets in training data to come up with a feature list
    Inputs:
        processed_tweets: pd.DataFrame: the output of process_all() function
    Outputs:
        list(str): list of rare words, sorted alphabetically.
    """
    counter = Counter()
    for _, row in processed_tweets.iterrows():
        counter.update(row['text'])
    return sorted([word for word, count in counter.items() if count == 1])

# # AUTOLAB_IGNORE_START
# rare_words = get_rare_words(processed_tweets)
# print len(rare_words) # should give 19623
# # AUTOLAB_IGNORE_STOP


# Construct a sparse matrix of features for each tweet with the help of `sklearn.feature_extraction.text.TfidfVectorizer`. Remember to ignore the rare words obtained above and NLTK's stop words during the feature creation step. You must leave other optional parameters (e.g., `vocab`, `norm`, etc) at their default values. Is the number of features returned by `TfidfVectorizer` lesser than expected? Any idea why?

# In[198]:

def create_features(processed_tweets, rare_words):
    """ creates the feature matrix using the processed tweet text
    Inputs:
        tweets: pd.DataFrame: tweets read from train/test csv file, containing the column 'text'
        rare_words: list(str): one of the outputs of get_feature_and_rare_words() function
    Outputs:
        sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
                                                we need this to tranform test tweets in the same way as train tweets
        scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
    """
    stopwords=nltk.corpus.stopwords.words('english')
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stopwords + rare_words)
    count = sklearn.feature_extraction.text.CountVectorizer(stop_words=stopwords + rare_words)
    text = []
    for _, row in processed_tweets.iterrows():
        text.append(' '.join(row['text']))
        
    return tfidf, tfidf.fit_transform(text), count, count.fit_transform(text)

# # AUTOLAB_IGNORE_START
# (tfidf, X) = create_features(processed_tweets, rare_words)
# # AUTOLAB_IGNORE_STOP


# # In[188]:

# # AUTOLAB_IGNORE_START
# print len(processed_tweets)
# print X.shape
# # AUTOLAB_IGNORE_STOP


# Also for each tweet, assign a class label (0 or 1) using its `screen_name`. Use 0 for realDonaldTrump, mike_pence, GOP and 1 for the rest.

# In[179]:

def create_labels(processed_tweets):
    """ creates the class labels from screen_name
    Inputs:
        tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
    Outputs:
        numpy.ndarray(int): dense binary numpy array of class labels
    """
    labels = []
    zeros = set(['realDonaldTrump', 'mike_pence', 'GOP'])
    for _, row in processed_tweets.iterrows():
        if row['screen_name'] in zeros:
            labels.append(0)
        else:
            labels.append(1)
            
    return np.array(labels)

# # AUTOLAB_IGNORE_START
# y = create_labels(processed_tweets)
# print y
# # AUTOLAB_IGNORE_STOP


# ## 3. Classification [4pts + 4pts + 4pts]
# And finally, we are ready to put things together and learn a model for the classification of tweets. The classifier you will be using is [`sklearn.svm.SVC`](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC) (Support Vector Machine). [Here](http://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html) is a nice introduction to SVMs for the curious minded. (But we will cover it in the class soon!)
# 
# At the heart of SVMs is the concept of kernel functions, which determines how the similarity/distance between two data points in computed. `sklearn`'s SVM provides four kernel functions: `linear`, `poly`, `rbf`, `sigmoid` (details [here](http://scikit-learn.org/stable/modules/svm.html#svm-kernels)) but you can also implement your own distance function and pass it as an argument to the classifier.
# 
# Through the various functions you implement in this part, you will be able to learn a classifier, score a classifier based on how well it performs and use it for prediction tasks.
# 
# Specifically, you will carry out the following tasks in order:
# 1. Implement the `learn_classifier()` function assuming `kernel` is always one of {`linear`, `poly`, `rbf`, `sigmoid`}. Stick to default values for any other optional parameters.
# 2. Implement the `evaluate_classifier()` function which scores a classifier based on accuracy.
# 3. Call `learn_classifier()` and `evaluate_classifier()` for each of the four kernel modes and determine what performs the best (this code has been written for you already).
# 4. Go back to `learn_classifier()` and fill in the best mode.

# In[195]:

def learn_classifier(X_train, y_train, kernel='best'):
    """ learns a classifier from the input features and labels using the kernel function supplied
    Inputs:
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features_and_labels()
        y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_features_and_labels()
        kernel: str: kernel function to be used with classifier. [best|linear|poly|rbf|sigmoid]
                    if 'best' is supplied, reset the kernel parameter to the value you have determined to be the best
    Outputs:
        sklearn.svm.classes.SVC: classifier learnt from data
    """
    if kernel == 'best':
        kernel = 'linear' # fill the best mode after you finish the evaluate_classifier() function
    
    classifer = sklearn.svm.SVC(kernel=kernel)
    classifer.fit(X_train, y_train)
    return classifer

# # AUTOLAB_IGNORE_START
# classifier = learn_classifier(X, y, 'linear')
# # AUTOLAB_IGNORE_STOP


# Now that we know how to learn a classifier, the next step is to evaluate it, ie., characterize how good its classification performance is. This step is necessary to select the best model among a given set of models, or even tune hyperparameters for a given model.
# 
# There are two questions that should now come to your mind:
# 1. **What data to use?** The data used to evaluate a classifier is called **validation data**, and it is usually different from the data used for training (for reasons we will learn about in the later lectures). However, in this problem, you will use training data as the validation data as well.
# 
# 2. **And what metric?** There are several evaluation measures available in the literature (e.g., accuracy, precision, recall, F-1,etc) and different fields have different preferences for specific metrics due to different goals. We will go with accuracy. According to wiki, **accuracy** of a classifier measures the fraction of all data points that are correctly classified by it; it is the ratio of the number of correct classifications to the total number of (correct or incorrect) classifications.
# 
# Now, implement the following function.

# In[192]:

def evaluate_classifier(classifier, X_validation, y_validation):
    """ evaluates a classifier based on a supplied validation data
    Inputs:
        classifier: sklearn.svm.classes.SVC: classifer to evaluate
        X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features
        y_train: numpy.ndarray(int): dense binary vector of class labels
    Outputs:
        double: accuracy of classifier on the validation data
    """
    y_p = classifier.predict(X_validation)
    return 1.0 - np.mean(y_p ^ y_validation)

# # AUTOLAB_IGNORE_START
# accuracy = evaluate_classifier(classifier, X, y)
# print accuracy # should give 0.954850271708
# # AUTOLAB_IGNORE_STOP


# Use the following code to determine which classifier is the best.

# In[193]:

# # AUTOLAB_IGNORE_START
# for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
#     classifier = learn_classifier(X, y, kernel)
#     accuracy = evaluate_classifier(classifier, X, y)
#     print kernel,':',accuracy
# # AUTOLAB_IGNORE_STOP


# We're almost there! It's time to write a nice little wrapper function that will use our model to classify unlabeled tweets from tweets_test.csv file.

# In[199]:

def classify_tweets(tfidf, classifier, unlabeled_tweets):
    """ predicts class labels for raw tweet text
    Inputs:
        tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
        classifier: sklearn.svm.classes.SVC: classifier learnt
        unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
    Outputs:
        numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
    """
    processed_tweets = process_all(unlabeled_tweets)
    text = []
    for _, row in processed_tweets.iterrows():
        text.append(' '.join(row['text']))
    return classifier.predict(tfidf.transform(text))
    
# # AUTOLAB_IGNORE_START
# classifier = learn_classifier(X, y, 'best')
# unlabeled_tweets = pd.read_csv("tweets_test.csv", na_filter=False)
# y_pred = classify_tweets(tfidf, classifier, unlabeled_tweets)
# # AUTOLAB_IGNORE_STOP

