# Load LSTM network and generate text
import sys
import numpy as np

from keras.models import load_model
import codecs
import os 
import pprint,pickle


sequence_len = 100;
generate_len = 100 ;
chars = [u'\t', u'\n', u'\r', u' ', u'!', u'&', u'(', u')', u',', u'-', u'.', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u':', u';', u'?', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'\u2018', u'\u2019', u'\u201c', u'\u201d', u'\u2026', u'\uff0e']
n_voca = len(chars)
char_to_int = dict((c, i) for i, c in enumerate(chars))

int_to_char = dict((i, c) for i, c in enumerate(chars))

sign_f = open('lists/sign_list.pkl','rb')
signlist = pickle.load(sign_f)
signnum = len(signlist)
sign_f.close()

rever_signlist = dict(zip(signlist.values(),signlist.keys()))

filename = "output/Aesop_100char_lower_256X2/weights-improvement-197-1.1269.hdf5"
model = load_model(filename)

filename2 = "test/test_example2.txt"
raw_text = codecs.open(filename2,'r','utf-8').read()
test_text = ' '*sequence_len+raw_text
n_chars = len(test_text)

#print char_to_int
test_pattern = [char_to_int.get(value,char_to_int[' ']) for value in test_text]
print "Seed:"
print "\"", raw_text, "\""
# generate characters

#before n_chars, the system is reading the given text
i=0
#model.predict(np.reshape(patterns, (n_chars-sequence_len, sequence_len, 1)), verbose=0)
while i+sequence_len <n_chars:
    patterns=test_pattern[i:i+sequence_len]
    x = np.reshape(patterns, (1, sequence_len, 1))
    #print test_pattern[i:i+sequence_len]
    x = x / float(n_voca)
    prediction = model.predict(x, batch_size = 1,verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    if result in signlist:
        test_pattern.insert(i+sequence_len+1,index)
        i +=1
        n_chars +=1
    i +=1 
txt_out2 = ''.join([int_to_char[i] for i in test_pattern])
print txt_out2
print "\nDone."
