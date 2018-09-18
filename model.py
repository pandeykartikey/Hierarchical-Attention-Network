import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)

class SentenceRNN(nn.Module):
    def __init__(self,vocab_size,embedsize, batch_size, hid_size,c):
        super(SentenceRNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        self.cls = c
        self.embed = nn.Embedding(vocab_size, embedsize)
        ## Sentence Encoder
        self.sentRNN = nn.GRU(embedsize, hid_size, bidirectional=True)
        ## Sentence Attention
        self.sentattn = nn.Linear(2*hid_size, 2*hid_size)
        self.attn_combine = nn.Linear(2*hid_size, 2*hid_size,bias=False)
        self.doc_linear = nn.Linear(2*hid_size, c)

    def forward(self, inp, hid_state_sent, cell_state):
        emb_out  = self.embed(torch.LongTensor(inp).view(-1, self.batch_size))

        out_state, hid_state = self.sentRNN(emb_out, hid_state_sent)
        sent_annotation = F.tanh(self.sentattn(hid_state.view(1, self.batch_size, -1)))
        attn = F.softmax(self.attn_combine(sent_annotation),dim=1)

        doc = attention_mul(hid_state.view(1, self.batch_size, -1), attn)
        d = self.doc_linear(doc)
        cls = F.log_softmax(d.view(-1,self.cls),dim=1)
        return cls, hid_state

    def init_hidden_sent(self):
            return Variable(torch.zeros(2, self.batch_size, self.hid_size))

    def init_cell(self):
            return Variable(torch.zeros(2, self.batch_size, self.hid_size))

class SentenceCNN(nn.Module):
    def __init__(self, vocab_size, embedsize,  batch_size,  max_sent_len, hid_size, window, c):
        super(SentenceCNN, self).__init__()
        self.batch_size = batch_size
        self.embedsize = embedsize
        self.hid_size = hid_size
        self.cls = c
        self.max_sent_len = max_sent_len
        self.window = window
        self.embed = nn.Embedding(vocab_size, embedsize)
        ## Sentence Encoder
        self.sentCNN = nn.Conv2d(1, hid_size, (window, embedsize))
        self.maxpool = nn.MaxPool1d(max_sent_len-window)
        self.tan1 = nn.Linear(hid_size, hid_size)
        ## Sentence Attention
        self.sentattn = nn.Linear(hid_size, hid_size)
        self.attn_combine = nn.Linear(hid_size, hid_size,bias=False)
        self.doc_linear = nn.Linear(hid_size, c)

    def forward(self, inp):
        emb_out  = self.embed(torch.LongTensor(inp).view(-1, self.batch_size))
        emb_out = emb_out.view(self.batch_size, 1, -1, self.embedsize)

        cnn_out = self.sentCNN(emb_out)
        m = nn.Tanh()
        tan_out = m(cnn_out.view(-1, self.batch_size, self.hid_size))
        max_out = self.maxpool(tan_out.view(self.batch_size, self.hid_size, -1))

        sent_annotation = m(self.sentattn(max_out.view(1, self.batch_size, -1)))
        attn = F.softmax(self.attn_combine(sent_annotation),dim=1)

        doc = attention_mul(max_out.view(1, self.batch_size, -1), attn)
        d = self.doc_linear(doc)
        cls = F.log_softmax(d.view(-1,self.cls),dim=1)
        return cls
