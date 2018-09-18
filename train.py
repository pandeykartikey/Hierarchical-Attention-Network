import os
import re
import sys
import time
import torch
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from glove import Corpus, Glove
# from model import SentenceRNN as Sentence
from model import SentenceCNN as Sentence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    """
    Creates and returns the Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess your data for HAN model")
    parser.add_argument("--datadir", dest="datadir",
                    help="Path to the data file.",
                    default="./data/yelp_preprocessed.csv", type=str)
    parser.add_argument("--glovedir", dest="glovedir",
                    help="Path to the glove model file.",
                    default="./weights/yelp_preprocessed.glove.model", type=str)
    parser.add_argument("--outputdir", dest="outputdir",
                    help="Path to the output that contains the training.",
                    default="./runs/", type=str)
    parser.add_argument("--weightdir", dest="weightdir",
                    help="Path to the directory that contains various runs weights training.",
                    default="./weights/", type=str)
    parser.add_argument("--col_text", dest="col_text",
                    help="text column of text in the dataset",
                    default="text", type=str)
    parser.add_argument("--col_label", dest="col_label",
                    help="label column in the dataset",
                    default="label", type=str)
    parser.add_argument("--max_sent_len", dest="max_sent_len",
                    help="maximum sentence length for the preprocessed vector",
                    default="20", type=int)
    parser.add_argument("--batch_size", dest="batch_size",
                    help="Size of a batch",
                    default="4", type=int)
    parser.add_argument("--train_percent", dest="train_percent",
                    help="train_percent",
                    default="0.8", type=float)
    parser.add_argument("--val_percent", dest="val_percent",
                    help="val_percent",
                    default="0.2", type=float)
    parser.add_argument("--hid_size", dest="hid_size",
                    help="Hidden Layer size",
                    default="100", type=int)
    parser.add_argument("--embedsize", dest="embedsize",
                    help="Embedding Layer size",
                    default="200", type=int)
    parser.add_argument("--window", dest="window",
                    help="window size of CNN model",
                    default="3", type=int)
    parser.add_argument("--lr", dest="lr",
                    help="Learning rate",
                    default="0.01", type=float)
    parser.add_argument("--momentum", dest="momentum",
                    help="Momentum",
                    default="0.9", type=float)
    parser.add_argument("--epoch", dest="epoch",
                    help="number of epochs",
                    default="30", type=int)
    parser.add_argument("--id", dest="id",
                    help="Uniques ID of each run",
                    default=str(time.time()), type=str)
    args = parser.parse_args()
    return args

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def gen_batch(x,y,batch_size):
#     k = random.sample(range(len(x)-1),batch_size)
    j = random.randint(0,len(x)-1)
    k = [0]*len(cls_arr)
    x_batch=[]
    y_batch=[]

    while sum(k) < batch_size:
        if k[int(y[j])] <= int((batch_size/len(cls_arr))+1):
          k[int(y[j])] = k[int(y[j])] +1
          x_batch.append(x[j])
          y_batch.append(y[j])
        j = j+1
        if(j==len(x)):
          j=0
    return [x_batch,y_batch]

def validation_accuracy(batch_size, x_val,y_val,sent_attn_model):
    acc = []
    val_length = len(x_val)

    for j in range(int(val_length/batch_size)):
        x,y = gen_batch(x_val,y_val,batch_size)
        # state_word = sent_attn_model.init_cell()
        # state_sent = sent_attn_model.init_hidden_sent()
        y_pred = sent_attn_model(x)
        # y_pred, _ = sent_attn_model(x, state_sent, state_word)
        max_index = y_pred.max(dim = 1)[1]
        correct = (max_index == torch.LongTensor(y)).sum()
        acc.append(float(correct)/batch_size)
    return np.mean(acc)

def train_data(batch_size, review, targets, sent_attn_model, sent_optimizer, criterion):

    # state_word = sent_attn_model.init_cell()
    # state_sent = sent_attn_model.init_hidden_sent()
    sent_optimizer.zero_grad()

    y_pred = sent_attn_model(review)
    # y_pred, _ = sent_attn_model(review, state_sent, state_word)

    loss = criterion(y_pred, torch.LongTensor(targets))

    max_index = y_pred.max(dim = 1)[1]
    correct = (max_index == torch.LongTensor(targets)).sum()
    acc = float(correct)/batch_size

    loss.backward()

    sent_optimizer.step()

    return float(loss), acc

def train_early_stopping(batch_size, x_train, y_train, x_val, y_val, sent_attn_model,
                         sent_attn_optimiser, loss_criterion, num_epoch,df_name, weightdir,
                         print_loss_every = 50, code_test=True):
    start = time.time()
    loss_full = []
    loss_epoch = []
    acc_epoch = []
    acc_full = []
    val_acc = []
    epoch_counter = 0
    train_length = len(x_train)
    for i in range(1, num_epoch + 1):
        loss_epoch = []
        acc_epoch = []
        for j in range(int(train_length/batch_size)):
            x,y = gen_batch(x_train,y_train,batch_size)
            loss,acc = train_data(batch_size, x, y, sent_attn_model, sent_attn_optimiser, loss_criterion)
            loss_epoch.append(loss)
            acc_epoch.append(acc)
            if (code_test and j % int(print_loss_every/batch_size) == 0) :
                print ("Loss at %d paragraphs, %d epoch,(%s) is %f" %(j*batch_size, i, timeSince(start), np.mean(loss_epoch)))
                print ("Accuracy at %d paragraphs, %d epoch,(%s) is %f" %(j*batch_size, i, timeSince(start), np.mean(acc_epoch)))

        loss_full.append(np.mean(loss_epoch))
        acc_full.append(np.mean(acc_epoch))
        torch.save(sent_attn_model.state_dict(), weightdir+"sent_attn_model_"+df_name+".pth")
        print ("Loss after %d epoch,(%s) is %f" %(i, timeSince(start), np.mean(loss_epoch)))
        print ("Train Accuracy after %d epoch,(%s) is %f" %(i, timeSince(start), np.mean(acc_epoch)))

        val_acc.append(validation_accuracy(batch_size, x_val, y_val, sent_attn_model))
        print ("Validation Accuracy after %d epoch,(%s) is %f" %(i, timeSince(start), val_acc[-1]))
    return loss_full,acc_full,val_acc

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.datadir):
        sys.exit("datadir doesnot exist")
    if not os.path.exists(args.outputdir):
        sys.exit("outputdir doesnot exist")
    if not os.path.exists(args.glovedir):
        sys.exit("glovedir doesnot exist")

    glove  = Glove.load(args.glovedir)
    df =  pd.read_csv(args.datadir)

    for idx in range(df[args.col_text].size):
        df[args.col_text][idx] = [ re.match(".\'(\w+)\'*",x).group(1) for x in df[args.col_text][idx].split(",") if re.match(".\'(\w+)\'*",x) is not None]

    x = [[glove.dictionary[token] for token in text]
         for text in list(df[args.col_text])]

    weights = torch.FloatTensor(glove.word_vectors)
    vocab_size = len(glove.word_vectors)

    y = df[args.col_label].tolist()
    X_pad = [sub_list + [0] * (args.max_sent_len - len(sub_list)) for sub_list in x]
    cls_arr = np.sort(df[args.col_label].unique()).tolist()
    classes = len(cls_arr)
    y =  [torch.FloatTensor([cls_arr.index(label)]) for label in y]

    length = df.shape[0]
    train_len = int(args.train_percent*length)
    val_len = int(args.val_percent*train_len)
    x_train = X_pad[:train_len]
    x_test = X_pad[train_len:]
    x_val = x_train[:val_len]
    x_train = x_train[val_len:]

    y_train = y[:train_len]
    y_test = y[train_len:]
    y_val = y_train[:val_len]
    y_train = y_train[val_len:]

    # sent_attn
    sent_attn = Sentence(vocab_size, args.embedsize, args.batch_size, args.max_sent_len, args.hid_size, args.window,classes)
    # sent_attn = Sentence(vocab_size, args.embedsize, args.batch_size, args.hid_size, classes)

    sent_attn.embed.from_pretrained(weights)
    torch.backends.cudnn.benchmark=True

    sent_optimizer = torch.optim.SGD(sent_attn.parameters(), lr=args.lr, momentum=args.momentum)

    criterion = nn.NLLLoss()

    args.weightdir = os.path.join(args.weightdir, "") # to add a trailing /
    args.outputdir = os.path.join(args.outputdir, "") # to add a trailing /
    if not os.path.exists(args.outputdir+args.id):
        os.makedirs(args.outputdir+args.id)
    outputdir = os.path.join(args.outputdir, args.id)
    outputdir = os.path.join(outputdir, "")

    df_name = ".".join(args.datadir.split("/")[-1].split(".")[:-1]) + args.id

    loss_full, acc_full, val_acc = train_early_stopping(args.batch_size, x_train, y_train, x_val,
                                y_val, sent_attn, sent_optimizer, criterion, args.epoch, df_name, args.weightdir, 0, False)

    plt.plot(loss_full)
    plt.ylabel("Training Loss")
    plt.xlabel("Epoch")
    plt.savefig(outputdir+"loss_"+df_name+".png")
    plt.clf()

    plt.plot(acc_full)
    plt.ylabel("Training Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(outputdir+"train_acc_"+df_name+".png")
    plt.clf()

    plt.plot(val_acc)
    plt.ylabel("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(outputdir+"val_acc_"+df_name+".png")
