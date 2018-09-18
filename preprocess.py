from __future__ import print_function
import string
import re
from bs4 import BeautifulSoup
import argparse
import sys
import os
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict

def parse_args():
    """
    Creates and returns the Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess your data for HAN model")
    parser.add_argument("--datadir", dest="datadir",
                    help="Path to the data file.",
                    default="./data/yelp.csv", type=str)
    parser.add_argument("--outputdir", dest="outputdir",
                    help="Path to the output that contains the preprocessed data.",
                    default="./data/", type=str)
    parser.add_argument("--col_text", dest="col_text",
                    help="text column of text in the dataset",
                    default="text", type=str)
    parser.add_argument("--col_label", dest="col_label",
                    help="label column in the dataset",
                    default="label", type=str)
    parser.add_argument("--max_sent_len", dest="max_sent_len",
                    help="maximum sentence length for the preprocessed vector",
                    default="20", type=int)
    parser.add_argument("--no_of_samples", dest="no_of_samples",
                    help="Number of samples in the dataset to preprocess",
                    default="10", type=int)
    parser.add_argument("--min_word_freq", dest="min_word_freq",
                    help="minimum word frequency required in document to keep it in vocabulary",
                    default="1", type=int)
    args = parser.parse_args()
    return args

def clean_str(string, max_sent_len):
    """
    adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = BeautifulSoup(string, "lxml").text
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    s =string.strip().lower().split(" ")
    if len(s) > max_sent_len:
        return s[0:max_sent_len]
    return s

def preprocess_text(df, col_text, max_sent_len):
    for docs in df[col_text]:
        idx = 0
        if docs[:2] == "b\"":
            docs = docs[2:-1]
        x.append(clean_str(docs,max_sent_len))
    return x

def preprocess(df, col_text, max_sent_len, min_word_freq):
    x = preprocess_text(df, col_text, max_sent_len)

    stoplist = stopwords.words("english") + list(string.punctuation)
    stemmer = SnowballStemmer("english")
    x = [[stemmer.stem(word.lower()) for word in sent  if word not in stoplist] for sent in x]

    frequency = defaultdict(int)
    for sent in x:
        for word in sent:
                frequency[word] += 1
    x = [[token for token in sent if frequency[token] >= min_word_freq ]
             for sent in x]

    return x

if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.datadir):
        sys.exit("datadir doesnot exist")
    if not os.path.exists(args.outputdir):
        sys.exit("outputdir doesnot exist")

    df=pd.read_csv(args.datadir)[:args.no_of_samples]

    x = []
    x = preprocess(df, args.col_text, args.max_sent_len, args.min_word_freq)
    out = dict()
    out["label"] = list(df[args.col_label])
    out["text"] = x

    output = pd.DataFrame(out)
    args.outputdir = os.path.join(args.outputdir, '') # to add a trailing /
    file_name = ".".join(args.datadir.split("/")[-1].split(".")[:-1]) + "_preprocessed.csv"
    output.to_csv(args.outputdir+file_name)
