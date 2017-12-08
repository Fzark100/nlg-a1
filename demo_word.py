#coding=utf-8
import sys
import numpy as np
import pprint,pickle
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence as txt2word
import codecs
import os 
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

word_f = open('lists/word_list.pkl','rb')
wordlist = pickle.load(word_f)
vocanum = len(wordlist)
word_f.close()

rever_wordlist = dict(zip(wordlist.values(),wordlist.keys()))

filename = "outputs2/Aesop's Fables_5words_256_256_dropout0.2/weights-improvement-09-0.0000.hdf5"


model = load_model(filename)
filename2 = "op4.txt"

letters = ' abcdefghijklmnopqrstuvwxyz<>'#<>is used for some spicific words e.g. <s> used for ending a story

new_words = txtpreprocessing(filename2, letters)
new_words = [w for w in new_words if w in wordlist]
n_words = len(new_words)
print n_words
input_index = [wordlist.get(value) for value in new_words]
sequence_len = 5
generate_len = 100
output = ' '
for i in range(n_words+generate_len-sequence_len):
	patterns=input_index[i:i+sequence_len]
	x = np.reshape(patterns, (1, sequence_len, 1))
	#print test_pattern[i:i+sequence_len]

	prediction = model.predict(x, batch_size = 1,verbose=0)
	predict_index = np.argmax(prediction)
	
	result = rever_wordlist[predict_index]	
	if i+sequence_len>=n_words: 
         output = output+result+' '
         #if result==u'<s>':
              #break 
         input_index.append(predict_index)
print output

#test_f = "test/test1.txt"
#test_word = txtpreprocessing(test_f, letters)
#test_word = [w for w in test_word if w in wordlist]
#ref_word = test_word[n_words:]
#ref_text = [w+' '  for w in ref_word]
#hyp_text = [rever_wordlist[i]+' ' for i in input_index[n_words:len(test_word)]]

#fo = open("test/test1_hyp.txt", "w")
#fo.writelines( hyp_text )
#fo.close()
#fo = open("test/test1_ref.txt", "w")
#fo.writelines( ref_text )
#fo.close()
#print "\nDone."
