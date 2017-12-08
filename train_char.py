 
# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import os 
import codecs
from optparse import OptionParser
import re

parser = OptionParser()
parser.add_option("-f","--file",dest = "filename", help = "input the training set file",default = "data/test1.txt", metavar ="FILE")
parser.add_option("-l","--length",dest = "seq_length", help = "input the length of sequences",default = 100,type=int)
parser.add_option("-n","--network",dest = "net_structure", help = "decide the network structure, needs more than one hidden layers",action = "append",default = [],type=int)
parser.add_option("-e","--epoch",dest = "epoch", help = "input the number of training epoch",default = 20,type=int)
parser.add_option("-b","--batch",dest = "batch_size", help = "decide the training batch size",default = 256,type=int)
parser.add_option("-d","--mkdir",dest = "mkdir", help = "decide the output destination",default = "output")
parser.add_option("-v","--verbose",action = "store_true",dest = "verbose", help = "wether see the output in the terminal")

parser.add_option("-w","--weights",dest = "load_weights",help="load prepared weights",default = [])
parser.add_option("-s","--stop",dest = "stop", help = "set the EarlyStopping",default = -1,type=int)

(options, args) = parser.parse_args()
# load ascii text and covert to lowercase
filename =options.filename
raw_text = codecs.open(filename,'r','utf-8').read()
#raw_text = raw_text.lower()
# create mapping of unique chars to integers
#chars = sorted(list(set(raw_text)))
chars = [u'\t', u'\n', u'\r', u' ', u'!', u'&', u'(', u')', u',', u'-', u'.', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u':', u';', u'?', u'A', u'B', u'C', u'D', u'E', u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R', u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y', u'z', u'\u2018', u'\u2019', u'\u201c', u'\u201d', u'\u2026', u'\uff0e']
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data

# prepare the dataset of input to output pairs encoded as integers
seq_length = options.seq_length
dataX = []
dataY = []
raw_text = ' '*seq_length+raw_text
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
seq_in = raw_text[0:seq_length]
for i in range(seq_length, n_chars-1, 1):
    seq_out = raw_text[i+1]
    if seq_out not in chars:
        seq_out = ' ' 
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    seq_in = seq_in[1:]+seq_out
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
net_structure = options.net_structure
hidden_layers = len(net_structure)
if hidden_layers==0:
    hidden_layers=2
    net_structure=[256,256]#for default condition
model = Sequential()
for i in range(0,hidden_layers):
    if i==0:
        model.add(LSTM(net_structure[0], input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    elif i==hidden_layers-1:
        model.add(LSTM(net_structure[i]))
    else:
        model.add(LSTM(net_structure[i], return_sequences=True))   
    model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
#load weights if exists
if len(options.load_weights)!=0:
    model.load_weights(options.load_weights)
    searchobj = re.search('weights-improvement-(.*)-(.*).hdf5',options.load_weights,re.M|re.I)
    ini_epoch = int(searchobj.group(1))
else:
    ini_epoch = 0

model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
save_path = options.mkdir
if not os.path.exists(save_path):
    os.makedirs(save_path)
if save_path[len(save_path)-1]!="/":
    save_path=save_path+"/"
filepath=save_path+"weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=options.verbose, save_best_only=True, mode='min')
if options.stop<0:
    callbacks_list = [checkpoint]
else:
    stopingpoint = EarlyStopping(monitor='loss',patience=options.stop, verbose=1, mode='auto')
    callbacks_list = [checkpoint,EarlyStopping]

# fit the model
model.fit(X, y, epochs=options.epoch, initial_epoch = ini_epoch, batch_size=options.batch_size, callbacks=callbacks_list) 


