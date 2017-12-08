#coding=utf-8
import sys
import numpy as np
import pprint,pickle
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence as txt2word
import codecs
import os 
import re
import time
def txtpreprocessing(raw_text, letters):
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

def predict_next(pattern,sequence_len,model):    
    x = np.reshape(pattern, (1, sequence_len, 1))
    prediction = model.predict(x, batch_size = 1,verbose=0)
    predict_index = np.argmax(prediction)
    return predict_index
    


word_f = open('lists/word_list.pkl','rb')
wordlist = pickle.load(word_f)
vocanum = len(wordlist)
word_f.close()

rever_wordlist = dict(zip(wordlist.values(),wordlist.keys()))

sign_f = open('lists/sign_list.pkl','rb')
signlist = pickle.load(sign_f)
signnum = len(signlist)
sign_f.close()

rever_signlist = dict(zip(signlist.values(),signlist.keys()))

total_f = open('lists/total_list.pkl','rb')
totallist = pickle.load(total_f)
totalnum = len(totallist)
total_f.close()

rever_totallist = dict(zip(totallist.values(),totallist.keys()))


modelfile1 = "output/Aesop_7word_256X2/weights-improvement-719-0.3024.hdf5"
model1 = load_model(modelfile1)
sequence_len1 = 7
modelfile2 = "output/Aesop_17sign_256X2/weights-improvement-289-2.6502.hdf5"
model2 = load_model(modelfile2)
sequence_len2 = 17

letters = ' abcdefghijklmnopqrstuvwxyz<>'#<>is used for some spicific words e.g. <s> used for ending a story



filename = "op2.txt"
raw_text = codecs.open(filename,'r','utf-8').read().lower()
new_words = txtpreprocessing(raw_text, letters)
n_words = len(new_words)

input_seqs1 = [0]*(sequence_len1-1)+[wordlist[w] for w in new_words if w in wordlist ]
n_seqs1 = len(input_seqs1)

input_seqs2 = [0]*(sequence_len2-1)+[totallist[w] for w in new_words  if w in totallist ]

result1=' '
i=0


while result1!=u'<s>':
    pattern1 =input_seqs1[i:i+sequence_len1]
    predict_index1 = predict_next(pattern1,sequence_len1,model1)
    result1 = rever_wordlist[predict_index1]    
    if i+sequence_len1>=n_seqs1: 
        input_seqs1.append(predict_index1)
        input_seqs2.append(predict_index1)
    i +=1
         
txt_out1 = ''.join([rever_wordlist[i]+' ' for i in input_seqs1])
print txt_out1     
#print raw_text

result2 = ' '
n = 0

while rever_totallist[input_seqs2[n+sequence_len2]]!=u'<s>':
    pattern2 =input_seqs2[n:n+sequence_len2]
    predict_index2 = predict_next(pattern2,sequence_len2,model2)
    result2 = rever_signlist[predict_index2] 
    print result2
    if (n+sequence_len1>=n_words)&(predict_index2!=0): 
        input_seqs2.insert(n+sequence_len2+1,totallist[result2])
    n +=1


txt_out2 = ''.join([rever_totallist[i]+' ' for i in input_seqs2])
print txt_out2
print "\nDone."
