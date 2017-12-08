#coding=utf-8
import numpy as np
import pickle
from keras.layers import Input,Embedding,LSTM,Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard

from keras.utils import np_utils

import os
import codecs
import re
from optparse import OptionParser
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
            datain = [seqs[i*batchsize+n:i*batchsize+n+seq_length]  for n in range(batchsize) if n <(n_letters-batchsize*i-seq_length)]
            datain = np.reshape(datain, (-1,seq_length,1))
            dataout = [np_utils.to_categorical(seqs[i*batchsize+m+seq_length],n_chars) for m in range(batchsize) if m <(n_letters-batchsize*i-seq_length)]
            dataout = np.reshape(dataout, (-1,n_chars))
            i=0
        yield({'input_1':datain},{'dense_1':dataout})
parser = OptionParser()
parser.add_option("-f","--file",dest = "filename", help = "input the training set file",default = "data/Aesop\'s Fables_pro_end.txt", metavar ="FILE")
parser.add_option("-l","--length",dest = "seq_length", help = "input the length of sequences",default = 5,type=int)
parser.add_option("-n","--network",dest = "net_structure", help = "decide the network structure, needs more than one hidden layers",action = "append",default = [],type=int)

parser.add_option("--drop",dest = "dropout", help = "input the value of dropout rate",default = 0.2,type=int)
parser.add_option("-e","--epoch",dest = "epoch", help = "input the number of training epoch",default = 1000,type=int)
parser.add_option("-b","--batch",dest = "batch_size", help = "decide the training batch size",default = 256,type=int)
parser.add_option("-d","--mkdir",dest = "mkdir", help = "decide the output destination",default = "output/temp")
parser.add_option("-v","--verbose",action = "store_true",dest = "verbose", help = "wether see the output in the terminal")

parser.add_option("-w","--weights",dest = "load_weights",help="load prepared weights",default = [])
parser.add_option("-s","--stop",dest = "stop", help = "set the EarlyStopping",default = -1,type=int)
parser.add_option("-p","--period",dest = "period", help = "set the period of weights saving",default = 10,type=int)

(options, args) = parser.parse_args()
filename =options.filename
letters = ' abcdefghijklmnopqrstuvwxyz<>'#<>is used for some spicific words e.g. <s> used for ending a story

new_words = txtpreprocessing(filename, letters)

word_f = open('lists/word_list.pkl','rb')
wordlist = pickle.load(word_f)
vocanum = len(wordlist)
word_f.close()


seq_len = options.seq_length

seqs = [wordlist[w] for w in new_words if w in wordlist ]
seqs =  [0]*(seq_len-1)+seqs
words_num = len(seqs)


print("vocabulary length: ", vocanum)
print("words count: ", words_num)
#datain = []
#for i in range(words_num-seq_len):
    #datain.extend(seqs[i:i+seq_len])
#datain = np.reshape(datain,(-1,seq_len,1))
#matrixs = np_utils.to_categorical(seqs)

net_structure = options.net_structure
hidden_layers = len(net_structure)
if hidden_layers==0:
    hidden_layers=2
    net_structure=[256,256]#for default condition
batchsize = options.batch_size
inputs = Input(shape=(seq_len,1))
if hidden_layers==1:
    lstm = LSTM(net_structure[hidden_layers-1],dropout = options.dropout,activation = 'relu',return_sequences=False)(inputs)
else:
    for i in range(0,hidden_layers):
        if i==0:
            x = LSTM(net_structure[i],dropout = options.dropout,activation = 'relu',return_sequences=True)(inputs)
        elif i == hidden_layers-1:
            lstm = LSTM(net_structure[i],dropout = options.dropout,activation = 'relu',return_sequences=False)(x)
        else:
            x = LSTM(net_structure[i],dropout = options.dropout,activation = 'relu',return_sequences=True)(x)   


#lstm = LSTM(net_structure[hidden_layers-1],dropout = options.dropout,activation = 'relu',return_sequences=(hidden_layers!=1))(inputs)
predictions = Dense(vocanum,activation='softmax')(lstm)

model = Model(inputs = inputs, outputs=predictions)
if len(options.load_weights)!=0:
    model.load_weights(options.load_weights)
    searchobj = re.search('weights-improvement-(.*)-(.*).hdf5',options.load_weights)
    ini_epoch = int(searchobj.group(1))
else:
    ini_epoch = 0

model.compile('adam','categorical_crossentropy')
save_path = options.mkdir
if save_path =='auto':
    read_txt_name = filename.split('/')[len(filename.split('/'))-1]
    read_txt_name = read_txt_name[:len(read_txt_name)-4]
    net_structure_str = ''.join([str(num)+'_' for num in net_structure])
    save_path = 'outputs2/'+read_txt_name+'_'+str(seq_len)+'words_'+net_structure_str+'dropout'+str(options.dropout)+'/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if save_path[len(save_path)-1]!="/":
    save_path=save_path+"/"
filepath=save_path+"weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=options.verbose, save_best_only=True, mode='min',period =options.period)
modelVisual = TensorBoard(log_dir="keras_logs/")
if options.stop<0:
    callbacks_list = [checkpoint,modelVisual]
else:
    stopingpoint = EarlyStopping(monitor='loss',patience=options.stop, verbose=options.verbose, mode='min')
    callbacks_list = [checkpoint,modelVisual,stopingpoint]

# fit the model
model.fit_generator(generate_training_dataset(seqs,seq_len,words_num,vocanum,batchsize), steps_per_epoch = (words_num+1-seq_len)/batchsize ,verbose = 1, epochs=options.epoch, initial_epoch = ini_epoch, callbacks=callbacks_list) 

# fit the model
#model.fit(datain, matrixs[seq_len:words_num], epochs=options.epoch, initial_epoch = ini_epoch,  batch_size=options.batch_size,verbose = 0, callbacks=callbacks_list) 
