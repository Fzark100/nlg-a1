 
# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np
from keras.layers import Input,Embedding,LSTM,Dense, Concatenate
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import os 
import codecs
from optparse import OptionParser
import re
import word2vec
def cosine_proximity_R(y_true, y_pred):
    def lz_normalize(x,axis):
        norm = K.sqrt(K.sum(K.square(x),axis = axis,keepdim=True))
        return K.maximum(x,K.epsilon())/K.maximum(norm,K.epsilon())
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.sum(1. - y_true * y_pred, axis=-1)

parser = OptionParser()
parser.add_option("-f","--file",dest = "filename", help = "input the training set file",default = "data/test1.txt", metavar ="FILE")
parser.add_option("-l","--length",dest = "seq_length", help = "input the length of sequences",default = 100,type=int)
#parser.add_option("-n","--network",dest = "net_structure", help = "decide the network structure, needs more than one hidden layers",action = "append",default = [],type=int)
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
raw_text = raw_text[:500000]
vectorfile = 'lists/fanren_vec.bin'
word_vectors = word2vec.load(vectorfile)
vector_dim = word_vectors.vectors.shape[1]
seq_length = options.seq_length



raw_text = ' '*seq_length+raw_text

#raw_text = raw_text.lower()
# create mapping of unique chars to integers
#chars = sorted(list(set(raw_text)))
char_to_vec = dict((c, word_vectors[c]) for c in word_vectors.vocab)
#char_to_vec[' ']=[0]*vector_dim
#char_to_vec['\r']=[1]*vector_dim
#char_to_vec['\n']=[0.99]*vector_dim
#char_to_vec['\t']=[0.5]*vector_dim

# summarize the loaded data

# prepare the dataset of input to output pairs encoded as integers


seqs = [char_to_vec[w] for w in raw_text if w in char_to_vec]
n_letters = len(seqs)
n_chars = len(word_vectors.vectors)
print("Total Characters: ", n_letters)
print("Total Vocab: ", n_chars)

datain = []
for i in range(n_letters-seq_length):
    datain.append(seqs[i:i+seq_length])
    
print len(datain),len(datain[0]),len(datain[0][0])
# reshape X to be [samples, time steps, features]
#datain = np.reshape(datain, (-1, vector_dim, seq_length))
#print len(datain),len(datain[0]),len(datain[0][0])

# normalize
#datain = datain / float(n_chars)
# one hot encode the output variable
y = seqs[seq_length:]
y = np.reshape(y, (-1, vector_dim, ))
print seqs[seq_length]
print y[0]
# define the LSTM model
#net_structure = options.net_structure
#hidden_layers = len(net_structure)
#if hidden_layers==0:
    #hidden_layers=2
    #net_structure=[256,256]#for default condition
inputs = Input(shape=(vector_dim,seq_length,))
lstm1 = LSTM(512,dropout = 0.2,return_sequences=True)(inputs)
lstm2 = LSTM(512,dropout = 0.2,return_sequences=False)(lstm1)
lstm1_R = LSTM(512,dropout = 0.2,return_sequences=True,go_backwards = True)(inputs)
lstm2_R = LSTM(512,dropout = 0.2,return_sequences=False,go_backwards = True)(lstm1_R)
merge1 = Concatenate()([lstm2,lstm2_R])
predictions = Dense(vector_dim,activation='softmax')(merge1)

model = Model(inputs = inputs, outputs=predictions)

#for i in range(0,hidden_layers):
    #if i==0:
        #model.add(LSTM(net_structure[0], input_shape=(datain.shape[1], datain.shape[2]), return_sequences=True))
    #elif i==hidden_layers-1:
        #model.add(LSTM(net_structure[i]))
    #else:
        #model.add(LSTM(net_structure[i], return_sequences=True))   
    #model.add(Dropout(0.2))
#model.add(Dense(y.shape[1], activation='softmax'))
#load weights if exists
if len(options.load_weights)!=0:
    model.load_weights(options.load_weights)
    searchobj = re.search('weights-improvement-(.*)-(.*).hdf5',options.load_weights,re.M|re.I)
    ini_epoch = int(searchobj.group(1))
else:
    ini_epoch = 0

model.compile(loss='cosine_proximity', optimizer='adam')
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
    callbacks_list = [checkpoint,stopingpoint]

# fit the model
model.fit(datain, y, epochs=options.epoch, initial_epoch = ini_epoch, batch_size=options.batch_size, callbacks=callbacks_list) 


