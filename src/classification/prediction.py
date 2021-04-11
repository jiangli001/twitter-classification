#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:45:21 2020

@author: jiangli
"""

import pickle
import pandas as pd
from sys import argv
import time
import numpy as np
import scipy
from gensim.models.doc2vec import Doc2Vec
import twitter_issue_prediction as tip

PRED_FILE_PATH = argv[1]
PRED_TEXT_COL = argv[2]
EMBEDDING_DIR = argv[3]
CLF_DIR = argv[4]
BINARIZER_DIR = argv[5]
KEYWORDS_PATH = argv[6]
THRESHOLD = argv[7]


def main():
    # load classifer
    with open(CLF_DIR, 'rb') as handle:
        classifier = pickle.load(handle)

    # load binarizer
    with open(BINARIZER_DIR, 'rb') as handle:
        binarizer = pickle.load(handle)

    # load Doc2vec model
    doc2vec_model = Doc2Vec.load(EMBEDDING_DIR)

    # load prediction files
    df_pred = pd.read_excel(PRED_FILE_PATH)

    # word embeddings
    print('Creating word embeddings ...\n')
    infer_sentences = tip.build_corpus(df_pred, PRED_TEXT_COL, False)
    corpus_vect = np.array(list(map(lambda x: doc2vec_model.infer_vector(x), infer_sentences)))
    keyword_vect = tip.extract_bow_features(KEYWORDS_PATH, df_pred, PRED_TEXT_COL)
    X_pred = np.concatenate((corpus_vect, keyword_vect), axis=1)

    # prediction
    """currently not supporting Rakelo and Random Forest"""
    
    print('Generating predicted labels...\n')
    
    try:
        proba_predictions = classifier.predict_proba(X_pred)
        if isinstance(proba_predictions, scipy.sparse.lil.lil_matrix):
            proba_predictions = proba_predictions.toarray()
        if proba_predictions.shape[0] == X_pred.shape[0]:
            predictions = (proba_predictions > eval(THRESHOLD)).astype(int)
    except:
        predictions = classifier.predict(X_pred)
    
    pred_labels = binarizer.inverse_transform(predictions)
    df_pred['predicted_labels'] = pred_labels

    # write to excel and save file locally
    excel_dir = tip.create_dir('result')
    df_pred.to_excel(excel_dir + '/prediction_result.xlsx')
    print('Result saved in this directory:\n{}\n'.format(excel_dir))


if __name__ == "__main__":
    print('Prediction started.\n')
    start_time = time.time()
    main()
    print('Prediction finished.')
    print('Runtime: {:.2f} seconds'.format(time.time() - start_time))
