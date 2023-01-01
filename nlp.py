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

import torch
from huggingface_hub import HfApi, Repository
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTForImageClassification

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

path = 'D:\GitHubFile\Final NLP\data\train'
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
    k = item[-n]
    id2gender_train[k] = v
    
###
id2texts_train = {}
path1 = path + '/text/'
for k in id2texts_train.keys():
    file = minidom.parse(path1 + k + '.xml')
    docs = file.getElementsByTagName('document')
    l = len(docs)
    aux = [docs[i].childNodes[0].data for i in range (0,1)]
    id2texts_train[k] = aux

ids = [k for k in list(id2texts_train.keys()) for i in range(0, len(id2texts_train[k]))]
texts = []
for k in list(id2texts_train.keys()):
    for t in id2gender_train[k]:
        #Cleaning
        t = re.sub ('\S+@\S+', '', t)
        t = re.sub ('@\S+', '', t)
        t = re.sub ('http\S+', '', t)
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
        s = len(t)
        if s != 0 and t[s-1] == ' ':
            t = t[:s-1]
        texts.append(t)
df = pd.DataFrame (data={'ids':ids, 'texts':texts})
df.to_csv ('D:\GitHubFile\Final NLP\train_texts.csv', index=False)



#Test
path = 'D:\GitHubFile\Final NLP\data\test'

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

ds = [k for k in list(id2texts_test.keys()) for i in range (0, len(id2texts_test[k]))]
texts = []
for k in list(id2texts_test.keys()):
  for t in id2texts_test[k]:
    # Cleaning
    t = re.sub ('\S+@\S+', '', t)
    t = re.sub ('@\S+', '', t)
    t = re.sub ('http\S+', '', t)
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
df.to_csv ('D:\GitHubFile\Final NLP\test_texts.csv', index=False)