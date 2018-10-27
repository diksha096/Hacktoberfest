#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 00:55:19 2018

@author: dikshasharma
"""


import tensorflow as tf
import re
import time
import numpy as np

lines=open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')

#create a dictionary to map lines with their ids
idline={}
for line in lines:
    line=line.split(' +++$+++ ')
    if len(line)==5:
        idline[line[0]]=line[4]
        
#create a list of all the conversation ids
conversation_ids=[]
for conversation in conversations[:-1]:
    conversation=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversation_ids.append(conversation.split(','))
    
    
#create a list for questions and answers
questions=[]
answers=[]
for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(idline[conversation[i]])
        answers.append(idline[conversation[i+1]])
        
def clean_text(text):
    text=text.lower()
    text=re.sub(r"i'm"," i am",text)
    text=re.sub(r"he's"," he is",text)
    text=re.sub(r"she's"," she is",text)
    text=re.sub(r"that's"," that is",text)
    text=re.sub(r"what's"," what is",text)
    text=re.sub(r"where's"," where is",text)
    text=re.sub(r"\'ll"," will",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text=re.sub(r"\'d"," would",text)
    text=re.sub(r"[- #@;() / <> + = {} ~ | ? .]"," ",text)
    return text

#clean questions
clean_questions=[]
for question in questions:
    clean_questions.append(clean_text(question))

#clean answers
clean_answers=[]
for answer in answers:
    clean_answers.append(clean_text(answer))
    
wordcount={}
for question in clean_questions:
    for word in question.split():
        if word not in wordcount:
            wordcount[word]=1
        else:
            wordcount[word]+=1
for answer in clean_answers:
    for word in answer.split():
        if word not in wordcount:
            wordcount[word]=1
        else:
            wordcount[word]+=1
threshold=20
questionword2int={}
word_number=0
for word,count in wordcount.items():
    if count>=threshold:
        questionword2int[word]=word_number
        word_number+=1

answerword2int={}
word_number=0
for word,count in wordcount.items():
    if count>=threshold:
        answerword2int[word]=word_number
        word_number+=1
        
tokens=['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionword2int[token]=len(questionword2int)+1

for token in tokens:
    answerword2int[token]=len(answerword2int)+1
    
answerint2word={w_i:w for w,w_i in answerword2int.items()}

#Adding end of string token at the end of every answer
for i in range(0,len(clean_answers)):
    clean_answers[i]+=' <EOS>'
    
questions_to_int=[]
for question in clean_questions:
    ints=[]
    for word in question.split():
        if word not in questionword2int:
            ints.append(questionword2int['<OUT>'])
        else:
            ints.append(questionword2int[word])
    questions_to_int.append(ints)

answers_to_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in answerword2int:
            ints.append(answerword2int['<OUT>'])
        else:
            ints.append(answerword2int[word])
    answers_to_int.append(ints)

sorted_clean_questions=[]
sorted_clean_answers=[]

for length in range(1,25+1):
    for i in enumerate(questions_to_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])
            
#STEP2 SEQ2SEQ MODEL
def model_inputs():
    inputs=tf.placeholder(tf.int32,['None','None'],name='input')
    targets=tf.placeholder(tf.int32,['None','None'],name='target')
    lr=tf.placeholder(tf.float32,name='learning rate')
    keep_prob=tf.placeholder(tf.float32,name='keep prob')
    
    return inputs,targets,lr,keep_prob

def prepprocess_targets(targets,word2int,batch_size):
    left_sie=tf.fill([batch_size,1],word2int['<SOS>'])
    right_side=tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1])
    prepocessed_targtes=tf.concat([left_sie,right_side],1,)
    return prepocessed_targtes
    
def encoder_rnn_layer(rnn_inputs,rnn_size,num_layers,keep_prob,sequence_length):
    lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
    _,encoder_state=tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                    cell_bw=encoder_cell,sequence_length=sequence_length,inputs=rnn_inputs,dtype=int32)
    return encoder_state

#decoding the training set
def decode_training_set(encoder_state,decoder_cell,):
    
    
    
    
    