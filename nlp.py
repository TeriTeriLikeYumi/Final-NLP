#librabry
import requests
import math
import matplotlib.pyplot as plt
import shutil
from getpass import getpass
from PIL import Image, UnidentifiedImageError
from requests.exceptions import HTTPError
from io import BytesIO
from pathlib import Path

import os
from os import listdir
from os.path import isfile, join
from xml.dom import minidom

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from collections import defaultdict
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup, AutoModel, BertTokenizerFast, BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import matplotlib.pyplot as plt

import urllib.request
import requests
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.decomposition import PCA

# Specifying GPU
device = torch.device ('cuda')

#Training dataset
path = 'D:/GitHubFile/Final-NLP/data/train'
with open(path+'/en.txt') as f:
    lines = f.readlines()

###
id2gender_train = {}
for item in lines:
    n = 0
    l = len (item)
    if item[-7] == 'f':
        v = 'F'
        n = -10
    else:
        v = 'M'
        n = -8
    k = item[:n]
    id2gender_train[k] = v

###
id2texts_train = {}
path1 = path + '/text/'
for k in id2gender_train.keys():
  file = minidom.parse (path1 + k + '.xml')
  docs = file.getElementsByTagName ('document')
  l = len (docs)
  aux = [docs[i].childNodes[0].data for i in range (0, l)]
  id2texts_train[k] = aux

ids = [k for k in list(id2texts_train.keys()) for i in range (0, len(id2texts_train[k]))]
texts = []
for k in list(id2texts_train.keys()):
  for t in id2texts_train[k]:
    # Cleaning
    t = re.sub ('\S+@\S+', ' ', t)
    t = re.sub ('@\S+', ' ', t)
    t = re.sub ('http\S+', ' ', t)
    t = t.replace ('....', '.')
    t = t.replace ('...', '.')
    t = t.replace ('..', '.')
    t = t.replace ('   ', ' ')
    t = t.replace ('  ', ' ')
    t = t.replace (',,,,', ',')
    t = t.replace (',,,', ',')
    t = t.replace (',,', ',')
    if t[0] == ' ':
      t = t[1:]
    s = len (t)
    if s != 0 and t[s-1] == ' ':
      t = t[:s-1]
    texts.append (t)
df = pd.DataFrame (data={'ids':ids, 'texts':texts})
df.to_csv ('D:/GitHubFile/Final-NLP/train_texts.csv', index=False)

#Testing dataset
path = 'D:/GitHubFile/Final-NLP/data/test'
with open(path+'/en.txt') as f:
    lines = f.readlines()
###    
id2gender_test = {}
for item in lines:
  n = 0
  l = len (item)
  if item[-7] == 'f':
    v = 'F'
    n = -10
  else:
    v = 'M'
    n = -8
  k = item[:n]
  id2gender_test[k] = v
  
id2texts_test = {}
path1 = path + '/text/'
for k in id2gender_test.keys():
  file = minidom.parse (path1 + k + '.xml')
  docs = file.getElementsByTagName ('document')
  l = len (docs)
  aux = [docs[i].childNodes[0].data for i in range (0, l)]
  id2texts_test[k] = aux

ids = [k for k in list(id2texts_test.keys()) for i in range (0, len(id2texts_test[k]))]
texts = []
for k in list(id2texts_test.keys()):
  for t in id2texts_test[k]:
    # Cleaning
    t = re.sub ('\S+@\S+', ' ', t)
    t = re.sub ('@\S+', ' ', t)
    t = re.sub ('http\S+', ' ', t)
    t = t.replace ('....', '.')
    t = t.replace ('...', '.')
    t = t.replace ('..', '.')
    t = t.replace ('   ', ' ')
    t = t.replace ('  ', ' ')
    t = t.replace (',,,,', ',')
    t = t.replace (',,,', ',')
    t = t.replace (',,', ',')
    if t[0] == ' ':
      t = t[1:]
    s = len (t)
    if s != 0 and t[s-1] == ' ':
      t = t[:s-1]
    texts.append (t)
df = pd.DataFrame (data={'ids':ids, 'texts':texts})
df.to_csv ('D:/GitHubFile/Final-NLP/test_texts.csv', index=False)


train_texts = pd.read_csv ('D:/GitHubFile/Final-NLP/train_texts.csv')
test_texts = pd.read_csv ('D:/GitHubFile/Final-NLP/test_texts.csv')

bert = BertModel.from_pretrained ('bert-base-cased')
tokenizer = BertTokenizer.from_pretrained ('bert-base-cased')

ids = pd.unique(train_texts['ids']).tolist()
id2texts_train = {}
for i in ids:
  id2texts_train[i] = []
  df1 = train_texts[train_texts['ids'] == i]
  for j in range (0, 10):
    id2texts_train[i].append (' '.join (list(df1['texts'].astype(str))[10*j:10*(j+1)]))
    
ids = pd.unique(test_texts['ids']).tolist()
id2texts_test = {}
for i in ids:
  id2texts_test[i] = []
  df1 = test_texts[test_texts['ids'] == i]
  for j in range (0, 10):
    id2texts_test[i].append (' '.join (list(df1['texts'].astype(str))[10*j:10*(j+1)]))
    
seq_len = [len (x.split()) for ls in list(id2texts_train.values()) for x in ls]
max_len = max(seq_len)
pd.Series (seq_len).hist(bins = 30)
max_len

seq_len = [len (x.split()) for ls in list(id2texts_test.values()) for x in ls]
max_len = max(seq_len)
pd.Series (seq_len).hist(bins = 30)
max_len

max_len = 256

id2tokens_train = {}
for k in list(id2texts_train.keys()):
  tokens = tokenizer.batch_encode_plus (id2texts_train[k], max_length=max_len, pad_to_max_length=True, add_special_tokens=True, truncation=True)
  id2tokens_train[k] = tokens

id2tokens_test = {}
for k in list(id2texts_test.keys()):
  tokens = tokenizer.batch_encode_plus (id2texts_test[k], max_length=max_len, pad_to_max_length=True, add_special_tokens=True, truncation=True)
  id2tokens_test[k] = tokens
  
# Convert lists to tensors
ids = [x for x in range(0, len(id2tokens_train)) for i in range(0, 10)]
ids = torch.tensor (ids)
train_seq = [x for k in list(id2tokens_train.keys()) for x in id2tokens_train[k]['input_ids']]
train_seq = torch.tensor (train_seq)
train_mask = [x for k in list(id2tokens_train.keys()) for x in id2tokens_train[k]['attention_mask']]
train_mask = torch.tensor (train_mask)
train_y = [0 if id2gender_train[k] == 'F' else 1 for k in list(id2tokens_train.keys()) for i in range(0, 10)]
train_y = torch.tensor (train_y)

ids2 = [x for x in range(0, len(id2tokens_test)) for i in range(0, 10)]
ids2 = torch.tensor (ids2)
test_seq = [x for k in list(id2tokens_test.keys()) for x in id2tokens_test[k]['input_ids']]
test_seq = torch.tensor (test_seq)
test_mask = [x for k in list(id2tokens_test.keys()) for x in id2tokens_test[k]['attention_mask']]
test_mask = torch.tensor (test_mask)
test_y = [0 if id2gender_test[k] == 'F' else 1 for k in list(id2tokens_test.keys()) for i in range(0, 10)]
test_y = torch.tensor (test_y)

batch_size = 32

train_data = TensorDataset (train_seq, train_mask, train_y, ids)
train_sampler = RandomSampler (train_data)
train_dataloader = DataLoader (train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset (test_seq, test_mask, test_y, ids2)
test_sampler = SequentialSampler (test_data)
test_dataloader = DataLoader (test_data, sampler=test_sampler, batch_size=batch_size)

classes = 2

class BertGenderClassifier (nn.Module):
  def __init__ (self, bert, classes):
    super (BertGenderClassifier, self).__init__()

    self.bert = bert
    self.dropout = nn.Dropout (p=0.1)
    self.out = nn.Linear (self.bert.config.hidden_size, classes)

  def forward (self, input_ids, attention_mask):
    _, bert_output = self.bert (input_ids=input_ids, 
                                attention_mask=attention_mask, return_dict=False)
    #print (bert_output)
    output = self.dropout (bert_output)
    
    return self.out (output)
  
# Create an instance of our model and push it to GPU
model = BertGenderClassifier (bert, classes)
model = model.to (device)

plt.bar (['Female', 'Male'], [train_y.tolist().count(0), train_y.tolist().count(1)])

plt.bar (['Female', 'Male'], [test_y.tolist().count(0), test_y.tolist().count(1)])

epochs = 8 #20

optimizer = AdamW (model.parameters(), lr=2e-5)

total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup (optimizer, num_warmup_steps=0, num_training_steps=total_steps)

ce_loss = nn.CrossEntropyLoss().to (device)

# A function for training 
def train_epochs (model, dataloader, ce_loss, optimizer, device, scheduler, entry_size):

  model = model.train ()

  losses = []
  correct_predictions_count = 0

  F_correct = 0
  F_incorrect = 0
  M_correct = 0
  M_incorrect = 0

  for data in dataloader:
    input_ids = data[0].to (device)
    attention_mask = data[1].to (device)
    targets = data[2].to (device)

    outputs = model (input_ids=input_ids, attention_mask=attention_mask)

    _, preds = torch.max (outputs, dim=1)
    loss = ce_loss (outputs, targets)

    correct_predictions_count += torch.sum (preds == targets)
    losses.append (loss.item())

    loss.backward ()

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()

    optimizer.zero_grad()

    F_correct += torch.sum ((preds == 0) & (preds == targets))
    F_incorrect += torch.sum ((preds == 0) & (preds != targets))
    M_correct += torch.sum ((preds == 1) & (preds == targets))
    M_incorrect += torch.sum ((preds == 1) & (preds != targets))

  return correct_predictions_count.double() / entry_size, np.mean(losses), F_correct / (F_correct + F_incorrect), M_correct / (M_correct + M_incorrect), F_correct / (F_correct + M_incorrect), M_correct / (M_correct + F_incorrect) 

def eval_model (model, dataloader, ce_loss, device, entry_size):
  model = model.eval()
  
  losses = []
  correct_predictions_count = 0

  F_correct = 0
  F_incorrect = 0
  M_correct = 0
  M_incorrect = 0

  with torch.no_grad():
    for data in dataloader:
      input_ids = data[0].to (device)
      attention_mask = data[1].to (device)
      targets = data[2].to (device)

      outputs = model (input_ids=input_ids, attention_mask=attention_mask)

      _, preds = torch.max (outputs, dim=1)

      loss = ce_loss (outputs, targets)

      l = len (targets[:])

      correct_predictions_count += torch.sum (preds == targets)
      losses.append (loss.item())

      F_correct += torch.sum ((preds == 0) & (preds == targets))
      F_incorrect += torch.sum ((preds == 0) & (preds != targets))
      M_correct += torch.sum ((preds == 1) & (preds == targets))
      M_incorrect += torch.sum ((preds == 1) & (preds != targets))

  return correct_predictions_count.double() / entry_size, np.mean(losses), F_correct / (F_correct + F_incorrect), M_correct / (M_correct + M_incorrect), F_correct / (F_correct + M_incorrect), M_correct / (M_correct + F_incorrect)

import torch
torch.cuda.empty_cache()

# For saving the history
history = defaultdict(list)
best_accuracy = 0

for epoch in range (epochs):
  print(f'Epoch {epoch + 1}/{epochs}')
  print('-' * 10)

  train_acc, train_loss, train_F_Percision, train_M_Percision, train_F_Recall, train_M_Recall = train_epochs (model, train_dataloader, ce_loss, optimizer, device, scheduler, len(train_data) )

  print(f'Train loss {train_loss} accuracy {train_acc} Female Percision {train_F_Percision} Male Percision {train_M_Percision} Female Recall {train_F_Recall} Male Recall {train_M_Recall}')

  val_acc, val_loss, val_F_Percision, val_M_Percision, val_F_Recall, val_M_Recall = eval_model(model, test_dataloader, ce_loss, device, len(test_data) )

  print(f'Val loss {val_loss} accuracy {val_acc} Female Percision {val_F_Percision} Male Percision {val_M_Percision} Female Recall {val_F_Recall} Male Recall {val_M_Recall}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'BERTModel')
    best_accuracy = val_acc