#some NLP algorithms in Python (for language models)
from keras.preprocessing.text import text_to_word_sequence as txt2word
from keras.preprocessing.text import Tokenizer 
import pickle
import math
#change file as required
filename = 'data/Aesop\'s Fables.txt'
raw_text = open(filename).read()
#preprocessing the texts
#firstly detect all the characters which is not needed
raw_word = txt2word(str(raw_text),filters = 'abcdefghijklmnopqrstuvwxyz1234567890<>')
token = Tokenizer(num_words=None)
token.fit_on_texts(raw_word)

temp = token.word_index.keys()
temp2 = ''
for i in range(len(temp)):
    temp2 = temp2+temp[i]
#remove the unused chars, and get word lists
raw_word = txt2word(str(raw_text),filters = str(temp2)+'!"#$%&()*+,-./:;=?@[]^_`{|}~\t\n')
token = Tokenizer(num_words=None,filters = str(temp2)+'!"#$%&()*+,-./:;=?@[]^_`{|}~\t\n')
token.fit_on_texts(raw_word)
print raw_word[0:200]
words_num = len(raw_word)
vocab_num = len(token.word_index)
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
vocabs = token.word_index.keys()

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
print 'test start '
test_text = open('test_example.txt').read()

test_words = txt2word(str(test_text),filters = str(temp2)+'!"#$%&()*+,-./:;=?@[]^_`{|}~\t\n')
eval = dict()
text_len = len(test_words)
print text_len
p = 0.0
#perplexity
for i in range(text_len-seq_len+1):
    gram = tuple(test_words[i:i+seq_len])
    poss = possibility(gram,storage_tri,storage_bi,storage_uni)
    p += math.log(poss)
p=math.exp(p*(-1.0/text_len))
print('Perplexity: ')
print(p)

