#coding=utf-8
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer 
from keras.layers import Input,Embedding,LSTM,Dense,Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import codecs
import re

filename = 'data/thrones1.txt'
letters = ' abcdefghijklmnopqrstuvwxyz'#<>is used for some spicific words e.g. <s> used for ending a story

def txtpreprocessing(filename, letters):
    raw_text = codecs.open(filename,'r','utf-8').read().lower()
    #tear the word and the signs apart
    
    new_text = ''.join([' '+x if (x not in letters)&(x!='\n') else x for x in raw_text])
    new_text = ''.join([x+' ' if (x not in letters)&(x!='\r') else x for x in new_text])
    # divide the sequence to words
    segment_space = [s for s in range(len(new_text)) if new_text[s]==' ']
    if segment_space[0]!=0:# to confirm we have the first word
        segment_space = [0]+segment_space
    raw_words = [re.sub(r' ','',new_text[segment_space[w]:segment_space[w+1]]) for w in range(len(segment_space)-1)]
    #delete the null word which is empty
    return [nw for nw in raw_words if nw !=u'']

new_words = txtpreprocessing(filename, letters)
vocablist = []
for v in new_words:
    if v not in vocablist:
        vocablist.append(v)
word_index =  dict((c, i) for i, c in enumerate(vocablist))
print len(vocablist),len(new_words)
token = Tokenizer(num_words=None,filters = '')
token.fit_on_texts(new_words)
words_num = len(new_words)
word_index = token.word_index
word_index[u'<unk>']=0#for unknown words
vocab_num = len(word_index)
output = open('vocab_list.pkl','wb')
pickle.dump(word_index,output)
output.close()
