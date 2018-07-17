"""
@author: Michael Guarino
"""

import os
import subprocess
import platform
import urllib
import tarfile

class prjPaths:
    def __init__(self, getDataset=True):
        self.SRC_DIR = os.path.abspath(os.path.curdir)
        self.ROOT_MOD_DIR = "/".join(self.SRC_DIR.split("/")[:-1])
        self.ROOT_DATA_DIR = os.path.join(self.ROOT_MOD_DIR, "data")
        self.LIB_DIR = os.path.join(self.ROOT_MOD_DIR, "lib")
        self.CHECKPOINT_DIR = os.path.join(self.LIB_DIR, "chkpts")
        self.CHECKPOINTS_HAN = os.path.join(self.CHECKPOINT_DIR, "han_chkpts/")
        self.CHECKPOINTS_HAN_GRU = os.path.join(self.CHECKPOINT_DIR, "han_gru_chkpts/")
        self.CHECKPOINTS_CNN = os.path.join(self.CHECKPOINT_DIR, "cnn_chkpts/")
        self.LOGS_DIR = os.path.join(self.LIB_DIR, "logs")

        if getDataset:
            osType = platform.system()
            if osType == "Windows":
                print("manually download data set from 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'"
                      " and set getDataset=False when prjPaths is called in *_master.py script")
                exit(0)
            elif osType is not "Linux":
                osType = "OSX"

            if not os.path.exists(self.ROOT_DATA_DIR):
                os.mkdir(path=self.ROOT_DATA_DIR)
            filename="{}/aclImdb_v1.tar.gz".format(self.ROOT_DATA_DIR)
            ACLIMDB_DIR = "{}/aclImdb".format(self.ROOT_DATA_DIR)
            urllib.request.urlretrieve("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", filename)
            if (filename.endswith("tar.gz")):
                tar = tarfile.open(filename, "r:gz")
                tar.extractall(ACLIMDB_DIR)
                tar.close()
            elif (filename.endswith("tar")):
                tar = tarfile.open(filename, "r:")
                tar.extractall(ACLIMDB_DIR)
                tar.close()
            #subprocess.Popen("sh getIMDB.sh {}".format(osType), shell=True, stdout=subprocess.PIPE).wait()
    # end
# end

import tensorflow as tf
import os
import csv
import re
import itertools
import more_itertools
import pickle
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from keras.preprocessing import sequence

class IMDB:
    @staticmethod
    def csvExist():
        paths = prjPaths()
        csvExists = "imdb.csv" in os.listdir(paths.ROOT_DATA_DIR)
        return csvExists
    # end

    def __init__(self, action=None):
        self.paths = prjPaths()
        self.ROOT_DATA_DIR = self.paths.ROOT_DATA_DIR
        assert(action in ["create", None]), "invalid action"

        if action == "create":
            # directory structure
            train_dir = "{}/{}".format(self.ROOT_DATA_DIR, "train")
            test_dir = "{}/{}".format(self.ROOT_DATA_DIR, "test")

            trainPos_dir = "{}/{}".format(train_dir, "pos")
            trainNeg_dir = "{}/{}".format(train_dir, "neg")

            testPos_dir = "{}/{}".format(test_dir, "pos")
            testNeg_dir = "{}/{}".format(test_dir, "neg")

            self.data = {"trainPos": self._getDirContents(trainPos_dir),
                         "trainNeg": self._getDirContents(trainNeg_dir),
                         "testPos": self._getDirContents(testPos_dir),
                         "testNeg": self._getDirContents(testNeg_dir)}
    # end

    def _getDirContents(self, path):

        dirFiles = os.listdir(path)
        dirFiles = [os.path.join(path, file) for file in dirFiles]

        return dirFiles
    # end

    def _getID_label(self, file, binary):
        splitFile = file.split("/")
        testOtrain = splitFile[5]
        filename = os.path.splitext(splitFile[-1])[0]
        id, label = filename.split("_")
        if binary:
            if int(label) < 5:
                label = 0
            else:
                label = 1

        return [id, label, testOtrain]
    # end

    def _loadTxtFiles(self, dirFiles):
        TxtContents = list()
        for file in dirFiles:
            with open(file, encoding="utf8") as txtFile:
                content = txtFile.read()
                id, label, testOtrain = self._getID_label(file, binary=True)
                TxtContents.append({"id": id,
                                    "content": content,
                                    "label": label,
                                    "testOtrain": testOtrain})
        return TxtContents
    # end

    def _writeTxtFiles(self, TxtContents):
        csvFileName = "{}/imdb.csv".format(self.ROOT_DATA_DIR)
        with open(csvFileName, "a") as csvFile:
            fieldNames = ["id", "content", "label", "testOtrain"]
            writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
            writer.writeheader()

            for seq in TxtContents:
                writer.writerow({"id": seq["id"],
                                 "content": seq["content"],
                                 "label": seq["label"],
                                 "testOtrain": seq["testOtrain"]})
    # end

    def _clean_str(self, string):
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
        return string.strip().lower().split(" ")
    # end

    def _oneHot(self, y):
        y = list(map(int, y))
        lookuplabels = {v: k for k, v in enumerate(sorted(list(set(y))))}
        recoded_y = [lookuplabels[i] for i in y]
        labels = tf.constant(recoded_y)
        max_label = tf.reduce_max(labels)
        labels_OHE = tf.one_hot(labels, max_label+1)

        with tf.Session() as sess:
            l = sess.run(labels)
            y_ohe = sess.run(labels_OHE)
        sess.close()
        return y_ohe
    # end

    def _index(self, data, type):
        def _apply_index(data, type):
            if type == "han":
                indexed = [[[unqVoc_LookUp[char] for char in seq] for seq in doc] for doc in data]
            else:
                indexed = [[unqVoc_LookUp[char] for char in seq] for seq in data]
            return indexed
        # end

        x_train, x_test = data

        unqVoc = set(list(more_itertools.collapse(x_train[:] + x_test[:])))
        unqVoc_LookUp = {k: v+1 for v, k in enumerate(unqVoc)}
        self.vocab_size = len(list(unqVoc_LookUp))

        # save lookup table
        pickle._dump(unqVoc_LookUp, open("{}/{}_unqVoc_Lookup.p".format(self.paths.LIB_DIR,
                                                                        type), "wb"))

        x_train = _apply_index(data=x_train, type=type)
        x_test = _apply_index(data=x_test, type=type)

        return [x_train, x_test]
    # end

    def _saver(self, type, data, vals):
        x_train, y_train, x_test, y_test = data
        if type == "han":
            persist = {}
    # end

    def get_batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
         adapted from Denny Britz https://github.com/dennybritz/cnn-text-classification-tf.git
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
    # end

    def hanformater(self, inputs):
        batch_size = len(inputs)

        document_sizes = np.array([len(doc) for doc in inputs], dtype=np.int32)
        document_size = document_sizes.max()

        sentence_sizes_ = [[len(sent) for sent in doc] for doc in inputs]
        sentence_size = max(map(max, sentence_sizes_))

        b = np.zeros(shape=[batch_size, document_size, sentence_size], dtype=np.int32)  # == PAD

        sentence_sizes = np.zeros(shape=[batch_size, document_size], dtype=np.int32)
        for i, document in enumerate(inputs):
            for j, sentence in enumerate(document):
                sentence_sizes[i, j] = sentence_sizes_[i][j]
                for k, word in enumerate(sentence):
                    b[i, j, k] = word
        return b, document_sizes, sentence_sizes
    # end

    def createManager(self):
        for key in self.data.keys():
            self.data[key] = self._loadTxtFiles(self.data[key])
            self._writeTxtFiles(self.data[key])
    # end

    def partitionManager(self, type=None):
        print("partitionManager function")
        csvfile = "{}/{}".format(self.ROOT_DATA_DIR, "imdb.csv")
        df = pd.read_csv(csvfile)

        # partition data
        train = df.loc[df["testOtrain"] == "train"]
        test = df.loc[df["testOtrain"] == "test"]

        if type == "han":
            create3DList = lambda df: [[self._clean_str(seq) for seq in "|||".join(re.split("[.?!]", docs)).split("|||")]
                                    for docs in df["content"].as_matrix()]

            x_train = create3DList(df=train)
            x_test = create3DList(df=test)
            print("x_train: {}".format(len(x_train)))
            print("x_test: {}".format(len(x_test)))

            x_train, x_test = self._index(data=[x_train[:], x_test[:]], type=type)
            self.max_seq_len = max([len(seq) for seq in itertools.chain.from_iterable(x_train + x_test)])
            self.max_sent_len = max([len(sent) for sent in (x_train+x_test)])
        else:
            x_train = [self._clean_str(seq) for seq in train["content"].as_matrix()]
            x_test = [self._clean_str(seq) for seq in test["content"].as_matrix()]

            print("x_train: {}".format(len(x_train)))
            print("x_test: {}".format(len(x_test)))

            x_train, x_test = self._index(data=[x_train[:], x_test[:]], type=type)

            # pad sequences
            self.max_seq_len = max([len(seq) for seq in (x_train[:] + x_test[:])])
            x_train = sequence.pad_sequences(x_train, maxlen=self.max_seq_len)
            x_test = sequence.pad_sequences(x_test, maxlen=self.max_seq_len)

        y_train = train["label"].tolist()
        y_test = test["label"].tolist()

        #OHE classes
        y_train = self._oneHot(y_train)
        y_test = self._oneHot(y_test)

        return[x_train, y_train, x_test, y_test]
    # end

# end
