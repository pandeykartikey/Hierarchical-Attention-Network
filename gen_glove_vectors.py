import os
import re
import sys
import itertools
import argparse
import pandas as pd
import more_itertools
from glove import Corpus, Glove
from multiprocessing import Pool

def parse_args():
    """
    Creates and returns the Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess your data for HAN model")
    parser.add_argument("--datadir", dest="datadir",
                    help="Path to the data file.",
                    default="./data/yelp_preprocessed.csv", type=str)
    parser.add_argument("--outputdir", dest="outputdir",
                    help="Path to the output that contains the glove dictionary.",
                    default="./weights/", type=str)
    parser.add_argument("--col_text", dest="col_text",
                    help="text column of text in the dataset",
                    default="text", type=str)
    parser.add_argument("--window", dest="window",
                    help="window size of glove model",
                    default="5", type=int)
    parser.add_argument("--vector_size", dest="vector_size",
                    help="Size of vector required",
                    default="200", type=int)
    parser.add_argument("--lr", dest="lr",
                    help="Learning rate",
                    default="0.01", type=float)
    parser.add_argument("--epoch", dest="epoch",
                    help="number of epochs",
                    default="30", type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.datadir):
        sys.exit("datadir doesnot exist")
    if not os.path.exists(args.outputdir):
        sys.exit("outputdir doesnot exist")

    df =  pd.read_csv(args.datadir)

    for idx in range(df[args.col_text].size):
        df[args.col_text][idx] = [ re.match(".\'(\w+)\'*",x).group(1) for x in df[args.col_text][idx].split(",") if re.match(".\'(\w+)\'*",x) is not None]

    texts = list(more_itertools.collapse(df[args.col_text], levels=0))
    corpus = Corpus()
    corpus.fit(texts, window=args.window)
    glove = Glove(no_components=args.vector_size, learning_rate=args.lr)
    glove.fit(matrix=corpus.matrix, epochs=args.epoch, no_threads = Pool()._processes, verbose = True)
    glove.add_dictionary(corpus.dictionary)

    args.outputdir = os.path.join(args.outputdir, '') # to add a trailing /
    file_name = ".".join(args.datadir.split("/")[-1].split(".")[:-1]) + ".glove.model"
    glove.save(args.outputdir + file_name)
