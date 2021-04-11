#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:42:56 2020

@author: jiangli
"""

from sys import argv
import time
import numpy as np
import pandas as pd
import pickle
import twitter_issue_prediction as tip
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train_file_path", type=str, help="The training file path")
parser.add_argument("train_target_col", type=str, help="The training file's label column name")
parser.add_argument("train_text_col", type=str, help="The training file's text column name")
parser.add_argument("keywords_path", type=str, help="The path to the keyword JSON file")
parser.add_argument("embedding_file_path", type=str, help="The embedding file's path")
parser.add_argument("embedding_text_col", type=str, help="The embedding file's text column name")
parser.add_argument("--top_estimator", type=str, default='Rakelo', help="The main classifier", 
                    choices=['ClassifierChain', 'Rakelo', 'LabelPowerset', 
                             'MultiOutputClassifier', 'OneVsRestClassifier'])
parser.add_argument("--base_estimator", type=str, default='LogisticRegression', help="The base classifier",
                    choices=['KNeighbors', 'MLP', 'RandomForest', 'LogisticRegression', 'SGD'])
parser.add_argument("--embedding_method", type=str, default='doc2vec', help="The word embedding method",
                    choices=['doc2vec', 'pretrained'])
args = parser.parse_args()


def main():
    df_train, y_train, binarizer = tip.pipeline(args.train_file_path, args.train_target_col)
    # store binarizer object in a local file
    binarizer_dir = tip.create_dir('model')
    with open(binarizer_dir + '/binarizer.obj', 'wb') as handle:
        pickle.dump(binarizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    embedding_df = pd.read_excel(args.embedding_file_path, nrows=8000)

    # preprocess and clean text
    vocab_sentences = tip.build_corpus(embedding_df, args.embedding_text_col, True)
    infer_sentences = tip.build_corpus(df_train, args.train_text_col, False)

    corpus_vect = tip.embed_corpus(vocab_sentences, infer_sentences, args.embedding_method)
    keyword_vect = tip.extract_bow_features(args.keywords_path, df_train, args.train_text_col)

    X_train = np.concatenate((corpus_vect, keyword_vect), axis=1)

    classifier = tip.build_clf(X_train, y_train, args.top_estimator, args.base_estimator)

    # store classifier object in a local file
    clf_dir = tip.create_dir('model')
    with open(clf_dir + '/classifier_param.obj', 'wb') as handle:
        pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    print('Training started.\n')
    start_time = time.time()
    main()
    print('Training finished.')
    print('Runtime: {:.2f} seconds\n'.format(time.time() - start_time))
