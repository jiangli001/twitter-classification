#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 18:35:49 2020

@author: jiangli
"""

import json
import itertools
import numpy as np
import re

def read_issues(path):

    """Read the json as a dict"""

    with open(path) as json_data:
        kw_dict = json.load(json_data)

    keywords = list(itertools.chain.from_iterable(kw_dict.values()))
    keywords = list(dict.fromkeys(keywords)) # remove duplicates

    return keywords


def create_vectors(text, keywords):
    """
    Create a vector based on the number of appearances of keywords in text.
    The length of the vector is the total number of unique keywords.
    """
    freq = {}
    for keyword in keywords:
        # binary alternative:
        # if keyword.lower() in text.lower():
        #     freq[keyword] = 1
        # elif keyword not in freq.keys():
        #     freq[keyword] = 0
        freq[keyword] = len(re.findall(keyword, str(text)))
    return np.fromiter(freq.values(), int)

