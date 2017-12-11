 #some NLP algorithms in Python (for language models)
from keras.preprocessing.text import text_to_word_sequence as txt2word
from keras.preprocessing.text import Tokenizer 
import math
import sys
import numpy as np
import pprint,pickle
from keras.models import load_model
import codecs
import os 
import re
from nlgeval import compute_metrics
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
#change file as required
filename = 'data/Aesop\'s Fables.txt'
letters = ' abcdefghijklmnopqrstuvwxyz'

raw_word = txtpreprocessing(filename, letters)
word_f = open('lists/word_list.pkl','rb')
wordlist = pickle.load(word_f)
vocanum = len(wordlist)
word_f.close()

rever_wordlist = dict(zip(wordlist.values(),wordlist.keys()))
words_num = len(raw_word)
vocab_num = len(wordlist)
seq_len = 3

storage_tri = dict()
storage_bi = dict()

#create an n-gram list
for i in range(words_num-seq_len+1):
  gram = tuple(raw_word[i:i+seq_len])
  if gram in storage_tri:
      storage_tri[gram] += 1
  else:
      storage_tri[gram] = 1

  if gram[:seq_len-1] in storage_bi:
      storage_bi[gram[:seq_len-1]] += 1
  else:
      storage_bi[gram[:seq_len-1]] = 1
#add the last two words into bi-gram
gram = tuple(raw_word[words_num-2:words_num])
if gram in storage_bi:
    storage_bi[gram] += 1
else:
    storage_bi[gram] = 1
#all poss trigrams
vocabs = wordlist.keys()

storage_uni = dict((k,raw_word.count(k)) for k in vocabs)
def possibility(pattern,storage_tri,storage_bi,storage_uni):
    lambd1 = 0.01
    lambd2 = 0.1
    lambd3 = 0.89
    vocab_num = len(storage_uni)
    if pattern in storage_tri:
        return lambd3*storage_tri[pattern]/storage_bi[pattern[:seq_len-1]]+lambd2*storage_bi[pattern[1:]]/storage_uni[pattern[1]]+lambd1*storage_uni[pattern[2]]/vocab_num
    else:
        if pattern in storage_bi:
            return lambd2*storage_bi[pattern[1:]]/storage_uni[pattern[1]]+lambd1*storage_uni[pattern[2]]/vocab_num
        else:
            return lambd1*storage_uni[pattern[2]]/vocab_num
                
#test data
def perplexity(test_text,storage_tri,storage_bi,storage_uni):
    test_words = txt2word(str(test_text))
    eval = dict()
    text_len = len(test_words)
    p = 0.0
    seq_len = 3
    #perplexity
    for i in range(text_len-seq_len+1):
        gram = tuple(test_words[i:i+seq_len])
        poss = possibility(gram,storage_tri,storage_bi,storage_uni)
        p += math.log(poss)
    p=math.exp(p*(-1.0/text_len))
    return p



modelname = "outputs2/Aesop's Fables_5words_256_256_dropout0.5/weights-improvement-999-1.1113.hdf5"
model = load_model(modelname)

test_f = "data/Grimm's Fairy Tales.txt"
new_words = txtpreprocessing(test_f, letters)
new_words = [w for w in new_words if w in wordlist]
n_words = len(new_words)
input_index = [wordlist.get(value) for value in new_words]   

sequence_len = 5
generate_len = 33

def txt_generation(test_index,rever_wordlist,sequence_len,generate_len): 
    output = ''
    n_words = len(test_index)
    for i in range(n_words+generate_len-sequence_len):
        patterns=test_index[i:i+sequence_len]
        x = np.reshape(patterns, (1, sequence_len, 1))
        #print test_pattern[i:i+sequence_len]
        prediction = model.predict(x, batch_size = 1,verbose=0)
        predict_index = np.argmax(prediction)
        result = rever_wordlist[predict_index]	
        if i+sequence_len>=n_words: 
            output = output+result+' '
            #if result==u'<s>':
            #break 
            test_index.append(predict_index)
    return output
sample_num = 30
generated_txt = []
reference_txt = []
perplexitys = 0
for i in range(sample_num):
    start_test = np.random.randint(0,n_words-sequence_len)
    test_txt = new_words[start_test:start_test+sequence_len]
    ref_txt = new_words[start_test+sequence_len:start_test+sequence_len+generate_len]
    ref_txt = ''.join([w+' '  for w in ref_txt])
    test_index = [wordlist.get(value) for value in test_txt]  
    txtgen = txt_generation(test_index,rever_wordlist,sequence_len,generate_len)
    perplexitys +=perplexity(txtgen,storage_tri,storage_bi,storage_uni)
    generated_txt.append(txtgen+'\r\n')
    reference_txt.append(ref_txt+'\r\n')
print perplexitys/sample_num
fo1 = open("hyp.txt", "w")
fo1.writelines( generated_txt )
fo1.close()
fo2 = open("ref.txt", "w")
fo2.writelines( reference_txt )
fo2.close()
metrics_dict = compute_metrics(hypothesis="hyp.txt",references=["ref.txt"])


#print metrics_dict
