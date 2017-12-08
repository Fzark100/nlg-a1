 
# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np
from keras.layers import Input,Embedding,LSTM,Dense, Concatenate
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import os 
import codecs
from keras.utils import np_utils

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
parser.add_option("-d","--mkdir",dest = "mkdir", help = "decide the output destination",default = "output")
parser.add_option("-v","--verbose",action = "store_true",dest = "verbose", help = "wether see the output in the terminal")

parser.add_option("-w","--weights",dest = "load_weights",help="load prepared weights",default = [])
parser.add_option("-s","--stop",dest = "stop", help = "set the EarlyStopping",default = -1,type=int)

(options, args) = parser.parse_args()
# load ascii text and covert to lowercase
filename =options.filename
raw_text = codecs.open(filename,'r','utf-8').read()
#raw_text = raw_text[:5000]
seq_length = options.seq_length

raw_text = ' '*seq_length+raw_text

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))


# summarize the loaded data

# prepare the dataset of input to output pairs encoded as integers


seqs = [char_to_int[w] for w in raw_text if w in char_to_int]
n_letters = len(seqs)
n_chars = len(chars)
print("Total Characters: ", n_letters)
print("Total Vocab: ", n_chars)

def generate_training_dataset(seqs,seq_length,n_letters,n_chars,batchsize):
    i = 0
    while 1:
        if (1+i)*batchsize+seq_length-1 <= n_letters:
            datain = [seqs[i*batchsize+n:i*batchsize+n+seq_length] for n in range(batchsize)]
            datain = np.reshape(datain, (-1,seq_length,1))
            dataout = [np_utils.to_categorical(seqs[i*batchsize+m+seq_length],n_chars) for m in range(batchsize)]
            dataout = np.reshape(dataout, (-1,n_chars))
            i += 1
        else :
            datain = [seqs[i*batchsize+n:i*batchsize+n+seq_length] for n in range(n_letters-batchsize*i-seq_length)]
            datain = np.reshape(datain, (-1,seq_length,1))
            dataout = [np_utils.to_categorical(seqs[i*batchsize+m+seq_length],n_chars) for m in range(n_letters-batchsize*i-seq_length)]
            dataout = np.reshape(dataout, (-1,n_chars))
            i=0
        yield({'input_1':datain},{'dense_1':dataout})
        

# reshape X to be [samples, time steps, features]
#print len(datain),len(datain[0]),len(datain[0][0])

# normalize
#datain = datain / float(n_chars)
# one hot encode the output variable
# define the LSTM model
#net_structure = options.net_structure
#hidden_layers = len(net_structure)
#if hidden_layers==0:
    #hidden_layers=2
    #net_structure=[256,256]#for default condition
inputs = Input(shape=(seq_length,1))
lstm1 = LSTM(256,dropout = 0.2,return_sequences=True)(inputs)
lstm2 = LSTM(256,dropout = 0.2,return_sequences=True)(lstm1)
lstm3 = LSTM(256,dropout = 0.2,return_sequences=True)(lstm2)
lstm4 = LSTM(256,dropout = 0.2,return_sequences=False)(lstm3)

lstm1_R = LSTM(256,dropout = 0.2,return_sequences=True,go_backwards = True)(inputs)
lstm2_R = LSTM(256,dropout = 0.2,return_sequences=True)(lstm1_R)
lstm3_R = LSTM(256,dropout = 0.2,return_sequences=True)(lstm2_R)

lstm4_R = LSTM(256,dropout = 0.2,return_sequences=False,go_backwards = True)(lstm3_R)
merge1 = Concatenate()([lstm4,lstm4_R])
predictions = Dense(n_chars,activation='softmax')(merge1)

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
    callbacks_list = [checkpoint,stopingpoint]
batchsize = 256
# fit the model
model.fit_generator(generate_training_dataset(seqs,seq_length,n_letters,n_chars,batchsize), steps_per_epoch = (n_letters+1-seq_length)/batchsize , epochs=options.epoch, initial_epoch = ini_epoch, callbacks=callbacks_list) 


