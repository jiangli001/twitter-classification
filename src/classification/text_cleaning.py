#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:30:19 2020

@author: jiangli
"""

import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from collections import defaultdict


# remove contractions
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = text.strip()
    return text


# clean the word of any punctuation or special characters
def clean_punc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'', sentence)
    cleaned = re.sub(r'&amp;',r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ', cleaned)
    cleaned = re.sub(r'[/(){}\[\]\|@,;]', r' ', cleaned)
    cleaned = re.sub(r'[^0-9a-z #+_]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


# Lemmatize each word based on its part of speech
def lemmatize_sentence(sentence, tokenize=False):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []

    # tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]

    for word, tag in nltk.pos_tag(filtered_sentence):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos))

    if tokenize == False:
        lemmatized_sentence = ' '.join(lemmatized_tokens) # back to string
        return lemmatized_sentence
    else:
        return lemmatized_tokens


def preprocess_text(data, col_name: list):
    new_data = data.copy()
    print('Preprocessing text ...\n')
    for c in col_name:
        new_data[c] = new_data[c].apply(clean_text).apply(clean_punc).apply(lemmatize_sentence, tokenize=True)
    return new_data[col_name]


def remove_infrequent_words(documents):
    texts = [document for document in documents]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    return texts

