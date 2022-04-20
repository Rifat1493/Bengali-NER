import numpy as np 
from validation import compute_f1
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate,GRU
from prepro import readfile,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
from keras.utils import plot_model
np.random.seed(2)
from keras_contrib.layers import CRF


epochs = 35

def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens,char, labels = data
        tokens = np.asarray([tokens])     
        
        char = np.asarray([char])
        pred = model.predict([tokens,char], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels


trainSentences = readfile("train_data.txt")
testSentences = readfile("test_data.txt")

#testSentences.pop(0)
#trainSentences.pop(0)
#trainSentences[0].pop(0)
#testSentences[0].pop(0)
trainSentences = addCharInformatioin(trainSentences)

testSentences = addCharInformatioin(testSentences)

labelSet = set()
words = {}


"""for sentence in trainSentences:
    for token,char,label in sentence:
        labelSet.add(label)
        words[token.lower()] =True"""



for dataset in [trainSentences,testSentences]:
    for sentence in dataset:
        
        for token,char,label in sentence:
            labelSet.add(label)
            words[token.lower()] = True
       
# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

#del label2Idx['\n']
word2Idx = {}
wordEmbeddings = []

fEmbeddings = open("./model/bn_w2v_model.text", encoding="utf-8")

#check=fEmbeddings.readlines()
for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(300) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25,300)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
        

   
        
wordEmbeddings = np.array(wordEmbeddings)

char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " ০১২৩৪৫৬৭৮৯ ্ । ো অআ া ও য় ৌ এ ে ই ী ি ঈ ঐ ৈ ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ত থ দ ধ ণ ট ঠ ড ঢ ন স হ ড় ঢ় য় ৎ ং ঃ ঁ প ফ ব ভ ম য র ল শ ষ উ ু ূঊ।,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)

train_set = padding(createMatrices(trainSentences,word2Idx,  label2Idx, char2Idx))
test_set = padding(createMatrices(testSentences, word2Idx, label2Idx,char2Idx))


idx2Label = {v: k for k, v in label2Idx.items()}

"""from sklearn.model_selection import train_test_split
train_set, test_set= train_test_split(train_set, test_size=0.1)"""


train_batch,train_batch_len = createBatches(train_set)


test_batch,test_batch_len = createBatches(test_set)



words_input = Input(shape=(None,),dtype='int32',name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)
character_input=Input(shape=(None,101,),name='char_input')
embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
#dropout= Dropout(0.25)(embed_char_out)
conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(embed_char_out)
maxpool_out=TimeDistributed(MaxPooling1D(101))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
#char = Dropout(0.25)(char)
output = concatenate([words,char])
# output = Bidirectional(LSTM(200, return_sequences=True,dropout=0.25, recurrent_dropout=0.25))(output)
output = Bidirectional(GRU(200, return_sequences=True))(output)


# output = TimeDistributed(Dense(50, activation="relu"))(output)

# crf = CRF(len(label2Idx),sparse_target=True)  # CRF layer
# output = crf(output)  # outpu
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)


model = Model(inputs=[words_input,character_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
# model.compile(optimizer='nadam',loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()
# plot_model(model, to_file='model.png')
# plot_model(model, to_file='model1.png')
epochs=35
for epoch in range(epochs):    
    print("Epoch %d/%d"%(epoch,epochs))
    a = Progbar(len(train_batch_len))
    for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
        labels, tokens,char = batch       
        model.train_on_batch([tokens,char], labels)
        a.update(i)
    print(' ')

    
    



pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))









###saving model in keras

from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
 
# evaluate loaded model on test data
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
score = loaded_model.evaluate(X, Y, verbose=0)







idx2word = {v: k for k, v in word2Idx.items()}
print(idx2word[1480])




#   Performance on test dataset       
predLabels, correctLabels = tag_dataset(test_set)    
#####creating the test result
f2 = open("predicted_label_t10.txt","a",encoding="utf-8")

for i,j,g in zip(testSentences,predLabels,correctLabels):
   
   f2.write("\n")
        
   for b,k,m in zip(i,j,g):
       
       #print(b[0],k)
       f2.write(b[0]+"\t"+idx2Label[k].strip()+"\t"+idx2Label[m])


     
f2.close()



f3= open("predicted_label_t10.txt","r",encoding="utf-8")   
b=f3.readlines()

y_pred=[]
y_true=[]

for i in b:
    if(i=="\n"):
        continue
    
    sent=i.strip().split("\t")
    if(len(sent)<3):
        continue
    y_pred.append(sent[1])
    y_true.append(sent[2])
    
f3.close()
    
    




import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.metrics import confusion_matrix

label=["B-PER","I-PER","B-LOC","I-LOC","B-ORG","I-ORG","TIM","O"]

cm=confusion_matrix(y_true, y_pred, labels=label)


sns.set(style='whitegrid', palette='bright', font_scale=1)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
plt.figure(figsize=(6, 6))
sns_plot=sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=True, fmt="d");
plt.title("Bangla NER Classification")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()

cm.sum()

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred,average="macro")




### without character embeddings ###

trainSentences = readfile("train_data.txt")
testSentences = readfile("test_data.txt")
labelSet = set()
words = {}

epochs=20

"""for sentence in trainSentences:
    for token,char,label in sentence:
        labelSet.add(label)
        words[token.lower()] = """



for dataset in [trainSentences,testSentences]:
    for sentence in dataset:
        
        for token,label in sentence:
            labelSet.add(label)
            words[token.lower()] = True
       
# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

#del label2Idx['\n']
word2Idx = {}
wordEmbeddings = []

fEmbeddings = open("./model/bn_w2v_model.text", encoding="utf-8")

for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(300) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25,300)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
               
wordEmbeddings = np.array(wordEmbeddings)

def createMatrices(sentences, word2Idx, label2Idx,char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
  
        
    dataset = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:
        wordIndices = []    
        
    
        labelIndices = []
        
        for word,label in sentence:  
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]                 
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            
                
                
            #Get the label and map to int            
            wordIndices.append(wordIdx)
            
            
            labelIndices.append(label2Idx[label])
           
        dataset.append([wordIndices, labelIndices]) 
        
    return dataset

char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " ০১২৩৪৫৬৭৮৯ ্ । ো অআ া ও য় ৌ এ ে ই ী ি ঈ ঐ ৈ ক খ গ ঘ ঙ চ ছ জ ঝ ঞ ত থ দ ধ ণ ট ঠ ড ঢ ন স হ ড় ঢ় য় ৎ ং ঃ ঁ প ফ ব ভ ম য র ল শ ষ উ ু ূঊ।,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)

train_set = createMatrices(trainSentences,word2Idx,  label2Idx, char2Idx)
test_set =createMatrices(testSentences, word2Idx, label2Idx,char2Idx)


idx2Label = {v: k for k, v in label2Idx.items()}

"""from sklearn.model_selection import train_test_split
train_set, test_set= train_test_split(train_set, test_size=0.1)"""


train_batch,train_batch_len = createBatches(train_set)


test_batch,test_batch_len = createBatches(test_set)

def iterate_minibatches(dataset,batch_len): 
    start = 0
    for i in batch_len:
        tokens = []
     
      
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t,l = dt
            l = np.expand_dims(l,-1)
            tokens.append(t)
            
           
            labels.append(l)
        yield np.asarray(labels),np.asarray(tokens)


words_input = Input(shape=(None,),dtype='int32',name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)
#character_input=Input(shape=(None,101,),name='char_input')
#embed_char_out=TimeDistributed(Embedding(len(char2Idx),100,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
#dropout= Dropout(0.25)(embed_char_out)
#conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=100, padding='same',activation='tanh', strides=1))(embed_char_out)
#maxpool_out=TimeDistributed(MaxPooling1D(101))(conv1d_out)
#char = TimeDistributed(Flatten())(maxpool_out)
#char = Dropout(0.25)(char)
#output = concatenate([words,char])
output = Bidirectional(LSTM(200, return_sequences=True))(words)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
model = Model(inputs=[words_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()
#plot_model(model, to_file='model.png')
#plot_model(model, to_file='model1.png')
epochs=35
for epoch in range(epochs):    
    print("Epoch %d/%d"%(epoch,epochs))
    a = Progbar(len(train_batch_len))
    for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
        labels, tokens = batch       
        model.train_on_batch([tokens], labels)
        a.update(i)
    print(' ')


def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens, labels = data
        tokens = np.asarray([tokens])     
        
       
        pred = model.predict([tokens], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels

predLabels, correctLabels = tag_dataset(test_set)    

#####creating the test result
f2 = open("predicted_label_t7.txt","a",encoding="utf-8")

for i,j,g in zip(testSentences,predLabels,correctLabels):
   
   f2.write("\n")
        
   for b,k,m in zip(i,j,g):
       
       #print(b[0],k)
       f2.write(b[0]+"\t"+idx2Label[k].strip()+"\t"+idx2Label[m])


     
f2.close()



f3= open("predicted_label_t7.txt","r",encoding="utf-8")   
b=f3.readlines()

y_pred=[]
y_true=[]

for i in b:
    if(i=="\n"):
        continue
    
    sent=i.strip().split("\t")
    if(len(sent)<3):
        continue
    y_pred.append(sent[1])
    y_true.append(sent[2])
    
f3.close()
    
    



import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.metrics import confusion_matrix

label=["B-PER","I-PER","B-LOC","I-LOC","B-ORG","I-ORG","TIM","O"]

cm=confusion_matrix(y_true, y_pred, labels=label)


sns.set(style='whitegrid', palette='bright', font_scale=1)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
plt.figure(figsize=(6, 6))
sns_plot=sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=True, fmt="d");
plt.title("Bangla NER Classification")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred,average="macro")      


