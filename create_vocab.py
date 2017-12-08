#coding=utf-8
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer 
from keras.layers import Input,Embedding,LSTM,Dense,Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import os
import codecs
import re
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
def dickeys2str(dictionary, ignores):
    keys = ''
    for i in range(len(dictionary.keys())):
        keys_in = dictionary.keys()[i]
        if keys_in not in  ignores:
            keys  += keys_in
    return keys
def dic_merge(dic1,dic2):
    for i in range(signum):
        if siglist.items()[i][0] not in worlist.keys():
            worlist[siglist.items()[i][0]] = siglist.items()[i][1]+wornum

filename = 'data/Aesop\'s Fables.txt'
letters = ' abcdefghijklmnopqrstuvwxyz'#<>is used for some spicific words e.g. <s> used for ending a story

new_words = txtpreprocessing(filename, letters)
#token = Tokenizer(num_words=None,filters = letters)
#token.fit_on_texts(new_words)
#sign_index = token.word_index
#sign_index['<unk>'] = 0
#sign_num=len( sign_index)

#print sign_index
#output = open('lists/sign_list.pkl','wb')
#pickle.dump(sign_index,output)
#output.close()
##########################################################
sign_f = open('lists/sign_list.pkl','rb')
siglist = pickle.load(sign_f)
signum = len(siglist)
sign_f.close()
token = Tokenizer(num_words=None,filters = dickeys2str(siglist,'<unk>'))

token.fit_on_texts(new_words)
word_index = token.word_index
word_num=len( word_index)
word_index = dict((w, word_index[w]-1) for w in word_index.keys())
print '\r\n' in word_index.keys()
print word_index
output = open('lists/word_list.pkl','wb')
pickle.dump(word_index,output)
output.close()
################################################################


#word_f = open('lists/word_list.pkl','rb')
#worlist = pickle.load(word_f)
#wornum = len(worlist)
#word_f.close()

#sign_f = open('lists/sign_list.pkl','rb')
#siglist = pickle.load(sign_f)
#signum = len(siglist)
#sign_f.close()
#print len(worlist)
#for i in range(signum):
    #if siglist.items()[i][0] not in worlist.keys():
        #worlist[siglist.items()[i][0]] = siglist.items()[i][1]+wornum

#output = open('lists/total_list.pkl','wb')
#print len(worlist)
#pickle.dump(worlist,output)
#output.close()
