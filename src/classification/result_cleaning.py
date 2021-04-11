#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:19:25 2020

@author: jiangli
"""

path = '/Users/jiangli/Desktop/tweet-classification/result/prediction_result.xlsx'

import pandas as pd
import re
df = pd.read_excel(path)

def strip_punc(text):
    cleaned_text = re.sub(r"[(|)|'|\"]", r"", text)
    cleaned_text = cleaned_text.strip(',').split(', ')
    return cleaned_text

df['corrected issue(s)'] = df['predicted_labels'].apply(lambda x: strip_punc(x))


df = df.explode('corrected issue(s)')
df.drop(columns=['predicted_labels'], inplace=True)


labeled_df = pd.read_excel('/Users/jiangli/Desktop/tweet-classification/data/labeled_tweets_v2.xlsx')

labeled_df['corrected issue(s)'] = labeled_df['corrected issue(s)'].apply(lambda x: strip_punc(x))
labeled_df = labeled_df.explode('corrected issue(s)')


concat_df = labeled_df.append(df)
concat_df['corrected issue(s)'].replace('', 'Others', inplace=True)

concat_df.to_excel('/Users/jiangli/Desktop/tweet-classification/result/cleaned_prediction_labeled_combined.xlsx')

