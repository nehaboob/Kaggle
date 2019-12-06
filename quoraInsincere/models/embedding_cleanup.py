#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:12:58 2019

@author: neha
"""

import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
import os
import collections
from tqdm import tqdm

TRAIN_PATH="../input/train.csv"
TEST_PATH="../input/test.csv"

def get_embeddings():
    embeddings = {}
    f = open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
    for i, line in tqdm(enumerate(f)):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word.lower()] = coefs
    f.close()
    
    return embeddings

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    for p in "/-":
        text = text.replace(p, ' ')
    for p in "'`‘":
        text = text.replace(p, '')
        
    punct = set('?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~°√' + '“”’')
    for p in punct:
        text = text.replace(p, f' {p} ') 
    
    text = text.replace('\d+', ' # ')
    text = text.strip().lower()
    return text

def build_word_dict():
    train_df = pd.read_csv(TRAIN_PATH)
    contents = train_df["question_text"]

    words = list()
    for content in tqdm(contents):
        for word in word_tokenize(clean_str(content)):
            words.append(word)

    word_counter = collections.Counter(words).most_common()
    word_dict = dict()
    word_dict["<pad>"] = 0
    word_dict["<unk>"] = 1
    word_dict["<eos>"] = 2
    for word, _ in word_counter:
        word_dict[word] = len(word_dict)

    return word_dict, word_counter

def word_coverage(word_counter, embeddings):
    missed_counts = 0
    total_words = 0
    missed = {}
    for word in word_counter:
        total_words = total_words+word_counter[word]
        if word not in embeddings:
            missed_counts = missed_counts+word_counter[word]
            missed[word] = word_counter[word]
    #print(collections.Counter(missed).most_common(20))
    print('missed text coverage ', missed_counts/total_words)
    print('missed word dict coverage ', len(missed)/len(word_dict))
    print(missed_counts, total_words)
    return missed
            

embeddings = get_embeddings()
word_dict, word_counter = build_word_dict()
word_counter = dict(word_counter)
missed = word_coverage(word_counter, embeddings)


sorted(missed.items(), key=lambda x: x[1], reverse=True)[:20]



