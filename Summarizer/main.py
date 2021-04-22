from sentence_handler import*
from bertModel import *
from cluster_handler import *
import time
import numpy as np
import sys
import torch
import os
import re
# This is the main file for The summarizer. Here we have an objevt for the bert model, 
# Clustering and formatting the sentence


#k is the numer of sentences the user has sent. 
#The filename is reading Scan.txt this contains the text the user wishes to summarize
k = int(sys.argv[1])
type(k)
filename ="Scan.txt"
f = open("./Summarizer/Scan.txt", "r")
text = f.read()

#Formatting the sourcetext. This was a last minuite bug fix
# Reason for this is Json has illegal charachters
text = text.replace('\n', ' ').replace('\r', ' ').replace('\"', ' ').replace('\\', ' ').replace('\/', ' ').replace('\b', ' ').replace('\f', ' ').replace('\t', ' ').replace('\r', ' ')
re.sub("\text\text+" , " ", text)

# Creating Objects
model = BertModel()
cluster_model = Clusterer()
sentence_handler = SentenceHandler()

#Tokenize the text into single sentences
sentences = sentence_handler.tokenize(text)


#If the user selects more sentences than there is in the source text they will recieve the source text
if k > len(sentences):
    k = len(sentences)
    


#Getting Feature vectors of each sentence
sentence_embeddings = model.get_embeddings(sentences)

#Clustering Feature Vector with user inputted value of k
values = cluster_model.cluster(sentence_embeddings, k)


sorted_values = sorted(values)
result = []


#Adding sentences together
for i, sentence in enumerate(sorted_values):
    s = sentences[sentence]
    result.append(s)
    result.append(" ")

s ='   '.join(result)
s = ''.join([i if ord(i) < 128 else ' ' for i in s])
s = s.replace('\n', ' ').replace('\r', '')

#Formatting the text into a JSON format
re.sub("\text\text+" , " ", s)
res = "{\"summary\": \""+ s
res += "\"}"



print(res)