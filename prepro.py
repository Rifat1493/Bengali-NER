import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences

def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename,encoding="utf-8")
    sentences = []
    sentence = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n" or line[1]=="\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
			
        splits = line.split('\t')
        if (splits[-1]=='\n'):
            continue
        sentence.append([splits[0],splits[-1]])

    if len(sentence) >0:
        sentences.append(sentence)
        sentence = []
    return sentences


def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches,batch_len



def createMatrices(sentences, word2Idx, label2Idx,char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']    
        
    dataset = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:
        wordIndices = []    
        
        charIndices = []
        labelIndices = []
        
        for word,char,label in sentence:  
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                
                try:
                   charIdx.append(char2Idx[x])
                except:
                   charIdx.append(char2Idx["UNKNOWN"])  
                
                
            #Get the label and map to int            
            wordIndices.append(wordIdx)
            
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])
           
        dataset.append([wordIndices, charIndices, labelIndices]) 
        
    return dataset

def iterate_minibatches(dataset,batch_len): 
    start = 0
    for i in batch_len:
        tokens = []
     
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t,ch,l = dt
            l = np.expand_dims(l,-1)
            tokens.append(t)
            
            char.append(ch)
            labels.append(l)
        yield np.asarray(labels),np.asarray(tokens),np.asarray(char)

def addCharInformatioin(Sentences):
    for i,sentence in enumerate(Sentences):
        for j,data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0],chars,data[1]]
    return Sentences

def padding(Sentences):
    maxlen = 101
    for sentence in Sentences:
        char = sentence[1]
        for x in char:
            maxlen = max(maxlen,len(str(x)))
    for i,sentence in enumerate(Sentences):
        Sentences[i][1] = pad_sequences(Sentences[i][1],101,padding='post')
    return Sentences
