#coding=utf-8

import codecs
import word2vec
filename ='data/fanren.txt'
raw_text = codecs.open(filename,'r','utf-8').read()
#put space in each characters
#f = codecs.open('data/fanren_pro.txt','w','utf-8')
#for i in raw_text:
    #f.write(i+' ',)
#f.close()

chars = sorted(list(set(raw_text)))
#print len(raw_text)
#char_to_int = dict((c, i) for i, c in enumerate(chars))
#print char_to_int[' ']
print len(chars),len(raw_text)

filename ='data/fanren_pro.txt'
outputfile = 'lists/fanren_vec.bin'
#word2vec.word2vec(filename,outputfile,size=20,min_count = 0,window = 10,verbose = True)
model = word2vec.load(outputfile)
chars2 =  model.vocab
print [c for c in chars if c not in chars2]
#print model[u'(']
char_to_vec = dict((c, model[c]) for c in model.vocab)
print model.vectors.shape[1]
char_to_vec[' ']=[0]*model.vectors.shape[1]
print char_to_vec[' ']
