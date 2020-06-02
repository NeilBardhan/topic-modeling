import os
import time
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
#from gensim.parsing.preprocessing import STOPWORDS

os.chdir('..')

PATH = os.getcwd()
DATA_PATH = PATH + '//headlines-data//'

def load_data(data_file):
    df = pd.read_csv(data_file, header = 'infer')
    return df

def remove_stopwords(row):
#    return ['a', 'b', 'c']
    return [token for token in row['headline_text_tokens'] if token not in stopwords.words('english')]

def preprocessing(text_data):
    ## tolower all strings (optional)
    ## strip leading and trailing whitespaces
    ## remove empty rows
    ## Tokenize the strings
    ## remove stopwords
    ## summary statistics - headline length
    ## date statistics - which date/month/year has most headlines
    ## day of the week statistics
    ## initial sentiment analysis
    text_data['headline_text'] = text_data['headline_text'].str.lower()
    text_data['headline_text'] = text_data['headline_text'].str.strip()
    text_data['headline_text'].replace('', np.nan, inplace=True)
    text_data.dropna(subset=['headline_text'], inplace=True)
    mean_headline_len = text_data['headline_text'].str.len().mean()
    print("Mean headline length ->", round(mean_headline_len), "characters.")
    text_data['headline_text_tokens'] = text_data['headline_text'].str.split()
    text_data['keywords'] = text_data.apply(remove_stopwords, axis=1)
    text_data.to_csv(DATA_PATH + 'million-headlines-keywords.csv', index=False, header=True)
    return text_data

fname = 'million-headlines.csv'

start = time.time()
text_data = load_data(DATA_PATH + fname)
print("Read time ->", round(time.time() - start, 2), "seconds.")

start = time.time()
text_data = preprocessing(text_data)
print("Preprocessing time ->", round(time.time() - start, 2), "seconds.")
#print(text_data.head())
