# Load LSTM network and generate text
import sys
import numpy as np

from keras.models import load_model
import codecs
import os 


sequence_len = 100;
generate_len = 1000;
chars = [u'\t', u'\n', u'\r', u' ', u'!', u'&', u'(', u')', u',', u'-', u'.', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u':', u';', u'?', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'\u2018', u'\u2019', u'\u201c', u'\u201d', u'\u2026', u'\uff0e']
n_voca = len(chars)
char_to_int = dict((c, i) for i, c in enumerate(chars))

int_to_char = dict((i, c) for i, c in enumerate(chars))

filename = "output/Aesop_100char_256X2/weights-improvement-197-1.3270.hdf5"
model = load_model(filename)

filename2 = "op2.txt"
raw_text = codecs.open(filename2,'r','utf-8').read()
test_text = ' '*sequence_len+raw_text
n_chars = len(test_text)

#print char_to_int
test_pattern = [char_to_int.get(value,char_to_int[' ']) for value in test_text]
print "Seed:"
print "\"", raw_text, "\""
# generate characters

patterns = []
patterns.extend(test_pattern[i:i+sequence_len] for i in range(n_chars-sequence_len))
#before n_chars, the system is reading the given text

#model.predict(np.reshape(patterns, (n_chars-sequence_len, sequence_len, 1)), verbose=0)
for i in range(n_chars+generate_len):
	patterns=test_pattern[i:i+sequence_len]
	x = np.reshape(patterns, (1, sequence_len, 1))
	#print test_pattern[i:i+sequence_len]
	
	x = x / float(n_voca)
	prediction = model.predict(x, batch_size = 1,verbose=0)
	index = np.argmax(prediction)
	result = int_to_char[index]
	
	if i+sequence_len>=n_chars:
         sys.stdout.write(result)
         test_pattern.append(index)

print "\nDone."
