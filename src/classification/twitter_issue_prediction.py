#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:49:43 2020

@author: jiangli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from collections import OrderedDict
from gensim import corpora
import gensim.downloader as api
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import LabelPowerset
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from skmultilearn.ensemble import RakelO
import text_cleaning as tc  # textcleaning.py
import keywords  # keywords.py


class descriptive_stats:
    """A series of operations for descriptive stats."""

    def __init__(self, df, colname):
        self.df = df
        self.colname = colname
        self.cat_dict = {}

    def count_issues(self):
        """Calculate number of unique issues."""
        for tweet in self.df[self.colname]:
            single_issues = str(tweet).split(', ')
            for single_issue in single_issues:
                single_issue = single_issue.strip()
                try:
                    self.cat_dict[single_issue] += 1
                except KeyError:
                    self.cat_dict[single_issue] = 1
        print('There are {} unique issues.\n'.format(len(self.cat_dict)))

    def bar_plot(self):
        """Plot a bar graph orgnized by issue counts."""
        sorted_cat_dict = sorted(self.cat_dict.items(),
                                 key=lambda x: x[1], reverse=True)
        sorted_cat_dict = OrderedDict(sorted_cat_dict)
        plt.figure(figsize=(12, 12))
        sns.barplot(list(sorted_cat_dict.values()),
                    list(sorted_cat_dict.keys()),
                    alpha=0.8, orient='h')
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Issues', fontsize=12)
        plot_dir = create_dir('plots')
        plt.savefig(plot_dir +'/issue_counts.png')
        print('Descriptive plot saved in this directory:\n{}\n'.format(plot_dir))


class MyCorpus(object):
    """Creates an interator that yields sentences (lists of str)."""

    def __init__(self, corpus):
        self.corpus = corpus
        self.dictionary = corpora.Dictionary(corpus)
        self.id = self.dictionary.token2id
        self.tokens_only = True

    def __iter__(self):
        for idx, document in enumerate(self.corpus):
            if not self.tokens_only:
                yield TaggedDocument(document, [idx])
            else:
                yield document

    def __call__(self, tokens_only):
        self.tokens_only = tokens_only
        return self


def infer_pretrained_vector(sentence, wv):
    vectors = []
    for token in sentence:
        try:
            vectors.append(wv[token])
        except:
            continue
    vectors = np.array(vectors)
    vectors = np.mean(vectors, axis=0)
    return vectors


def encode_y(df, colname, return_binarizer=False):
    """Binary encoding for y."""
    target = df[colname].apply(lambda x: x.split(', '))
    mlb = MultiLabelBinarizer()
    y_vect = mlb.fit_transform(target)
    if return_binarizer:
        return mlb
    else:
        return y_vect


def aloc_accuracy_score(y_pred, y_true, binarizer):
    """
    A helper function to calculate the accuracy,
    where at least one of predicted categories is correct
    """
    tally = {}
    pred_labels = binarizer.inverse_transform(y_pred)
    true_labels = binarizer.inverse_transform(y_true)

    for row_num, row in enumerate(pred_labels):
        for cat in row:
            if cat in true_labels[row_num]:
                tally[row_num] = 1
                break
            else:
                tally[row_num] = 0
    score = np.mean(list(tally.values()))
    return score


def create_custom_scorer():
    scoring = {'acc': 'accuracy',
               'aloc': make_scorer(aloc_accuracy_score,
                                   greater_is_better=True,
                                   binarizer=encode_y.binarizer),
               }
    return scoring


def pipeline(file_path, target_col):
    # read file
    df = pd.read_excel(file_path)
    # generate descriptive stats
    desc = descriptive_stats(df, target_col)
    desc.count_issues()
    desc.bar_plot()
    # encode target variable
    y = encode_y(df, target_col)
    binarizer = encode_y(df, target_col, True)
    return df, y, binarizer


def build_corpus(df, text_col, train_corpus):
    """
    Create genarator objects for streaming large body of texts
    so that not everything is read in-memory
    """

    # preprocess and clean text
    texts = tc.preprocess_text(df, [text_col]).squeeze()

    # create a generator object of mappings (document ID and tokens)
    if train_corpus:
        train_sentences = MyCorpus(texts)(tokens_only=False)
        return train_sentences

    # create a generator object of tokens
    else:
        test_sentences = list(MyCorpus(texts)(tokens_only=True))
        return test_sentences


def create_dir(name):
    parent_dir = str(Path(__file__).resolve().parents[2])
    if os.path.isdir(os.path.join(parent_dir, name)):
        new_dir = os.path.join(parent_dir, name)
    else:
        new_dir = os.path.join(parent_dir, name)
        os.mkdir(new_dir)
    return new_dir


def embed_corpus(train_sentences, test_sentences, method):
    '''
    Word embeddings for the corpus using either custom Doc2vec or pretrained vectors
    '''

    if method == 'doc2vec':
        # Using cutom Doc2vec
        model = Doc2Vec(vector_size=64, min_count=3, alpha=0.02, min_alpha=0.001,
                    window=5, epochs=40)
        model.build_vocab(train_sentences)
        print('\nCreating word embeddings...')
        model.train(train_sentences, total_examples=model.corpus_count, epochs=model.epochs)

        # save model
        model_dir = os.path.join(create_dir('model'), 'doc2vec')
        model.save(model_dir)
        print('\nThe model is saved in the following directory:\n{}\n'.format(model_dir))
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

        # get the feature vector for the corpus (word2vec)
        corpus_vect = list(map(lambda x: model.infer_vector(x), test_sentences))
        corpus_vect = np.array(corpus_vect)
        return corpus_vect

    # Using Pretrained Word Vectors
    if method == 'pretrained':
        # pretrained_wv = api.load('word2vec-google-news-300')
        pretrained_wv = api.load("glove-twitter-100")
        print('Downloading pretrained vectors ...')
        corpus_vect = list(map(lambda x: infer_pretrained_vector(x, pretrained_wv), test_sentences))

        for idx, value in enumerate(corpus_vect):
            if np.isnan(value).all():
                corpus_vect[idx] = np.zeros((pretrained_wv['king'].shape[0],), dtype=float)

        corpus_vect = np.array(corpus_vect)
        return corpus_vect


def extract_bow_features(data_path, df, col_name):
    """
    Extract features based on the self-defined keywords
    using the bag-of-words model
    """

    kwd = keywords.read_issues(data_path)
    keyword_vectors = df[col_name].apply(lambda x: keywords.create_vectors(x, kwd))
    keyword_vectors = np.array(list(keyword_vectors))
    return keyword_vectors


def build_clf(X, y, top_estimator, base_estimator):
    """

    Parameters
    ----------
    X : NumPy Array
        The array is generated by concatenating corpus and keywords vectors.
    y : NumPy Array
        y.shape[0] should be equal to X.shape[0].
    top_estimator : str
        One of ['ClassifierChain', 'LabelPowerset', 'Rake10'].
    base_estimator : str
        One of ['ClassifierChain', 'LabelPowerset', 'Rake10'].


    In Scikit-learn, only the following support multilabel classification:

    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.ExtraTreeClassifier
    sklearn.ensemble.ExtraTreesClassifier
    sklearn.neighbors.KNeighborsClassifier
    sklearn.neural_network.MLPClassifier
    sklearn.neighbors.RadiusNeighborsClassifier
    sklearn.ensemble.RandomForestClassifier
    sklearn.linear_model.RidgeClassifierCV

    We have to use the `multiclass` or `multioutput` modules
    if the base classifier is not among the above listed.

    # In the multilabel learning literature,
    # OvR is also known as the binary relevance method.
    # An indicator matrix - a matrix of shape (n_samples, n_classes)
    # turns on multilabel classification.
    
    Returns
    -------
    A fitted classifier object.

    """
    if base_estimator in ['KNeighbors', 'MLP', 'RandomForest']:
        if base_estimator == 'KNeighbors':
            """
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
            """
            from sklearn.neighbors import KNeighborsClassifier
            n_neighbors = int(input('Enter the number of neighbors\n'))
            weights = input('Enter the weight function used in prediction\n')
            leaf_size = int(input('Enter the leaf size\n'))
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, leaf_size=leaf_size)
            classifier.fit(X, y)
            print('Training...\n')
            return classifier
        
        elif base_estimator == 'MLP':
            """
            https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
            """
            from sklearn.neural_network import MLPClassifier
            hidden_layer_sizes = tuple(eval(input('Enter the size of the hidden layers.\n e.g: (64,64,64)'\
                'indicates a simple three-layer feedforward neural network with 64 neurons in each layer\n')))
            classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver='lbfgs') # use lbgfs because it's a small dataset
            classifier.fit(X, y)
            print('Training...\n')
            return classifier
        
        elif base_estimator == 'RandomForest':
            """
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
            """
            from sklearn.ensemble import RandomForestClassifier
            n_estimators = int(input('Enter the number of trees in the forest.\n'))
            max_depth = eval(input('Enter the maximum depth of the tree (int or None)\n'))
            min_samples_split = eval(input('The minimum number of samples required to split an internal node\n'))
            min_samples_leaf = eval(input('The minimum number of samples required to be at a leaf node\n'))
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            classifier.fit(X, y)
            print('Training...\n')
            return classifier
    
    else:
        if base_estimator == 'LogisticRegression':
            """
            http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
            """
            from sklearn.linear_model import LogisticRegression
            penalty = input('Enter the norm used in the penalization. One of {l1, l2, elasticnet, none}\n')
            tol = float(input('Enter the tolerance for stopping criteria; default=1e-4. Bigger number means faster convergence\n'))
            C = float(input('Enter the regularization strength. Smaller values specify stronger regularization.'\
                'Must be a positive number.\n'))
            solver = input('Enter the optimizer. One of {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}\n')
            max_iter = int(input('Enter the maximum number of iterations.\n'))
            base_classifier = LogisticRegression(penalty=penalty, tol=tol, C=C, solver=solver, max_iter=max_iter)
        
        elif base_estimator == 'SGD':
            """
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
            """
            from sklearn.linear_model import SGDClassifier
            loss = input('Enter the loss function. One of {‘hinge’, ‘log’, ‘modified_huber’, ‘squared_hinge’}\n')
            penalty = input('Enter the penalty. One of {‘l2’, ‘l1’, ‘elasticnet’}\n')
            tol = float(input('Enter the tolerance for stopping criteria; default=1e-4. Bigger number means faster convergence\n'))
            learning_rate = input('Enter the learning rate: Either ‘optimal’ or ‘adaptive’\n')
            eta0 = eval(input('Enter the initial learning rate\n'))
            base_classifier = SGDClassifier(loss=loss, penalty=penalty, tol=tol, learning_rate=learning_rate,
                                            eta0=eta0, alpha=0.0008)

        if top_estimator == 'Rakelo':
            classifier = RakelO(
                base_classifier=base_classifier,
                base_classifier_require_dense=[True, True],
                labelset_size=y.shape[1] // 4,
                model_count=6
            )
            classifier.fit(X, y)
            print('Training...\n')
            return classifier
        
        else:
            classifier = globals()[top_estimator](base_classifier)
            classifier.fit(X, y)
            print('Training...\n')
            return classifier


def tune_lr_hyperparam(clf, X, y):
    lr_param = {'base_estimator__C': np.logspace(-3, 3, 5),
                'base_estimator__solver': ['newton-cg', 'saga', 'sag']
            }

    grid_rfr = GridSearchCV(clf, param_grid=lr_param,
                            cv=KFold(n_splits=5, shuffle=True),
                            scoring=create_custom_scorer(),
                            refit='acc')
    grid_rfr.fit(X, y)
    print('Best parameters: {}'.format(grid_rfr.best_params_))
    return grid_rfr.best_estimator_

