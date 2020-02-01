# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:28:35 2020

@author: NFran
"""

### mount goodle drive
from google.colab import drive
drive.mount('/content/gdrive')

### install missing packages
%%capture
pip install ktrain


### Setup
# import packages
import numpy as np
import math
import os
import random
import seaborn as sn
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

import ktrain
from ktrain import text

test_size = 0.30
seed = 8071992

random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)


def process_network_data(speech, topicCode):
  
  '''
  This function prepares the two main variables of interest, text (speech, bill, etc.) 
  and corresponding code to be fed into the network.
  '''

  # sort based on topic
  topicCode, speech = zip(*sorted(zip(topicCode, speech)))
  speech = list(speech)

  # get topicCode in desired fromat (and recode some things)
  topicCode = np.array(topicCode)
  topicCode= topicCode.astype(int)
  topicCode = np.where(topicCode == 99, 0, topicCode) # replace 99 with 0 for easier presentation
  topicCode = np.where(topicCode == 100, 0, topicCode) # replace 100 with 0 for easier presentation (it is either 99 or 100 in any given dataset)
  topicCode= topicCode.astype(str) # make them strings

  return speech, topicCode



def get_categories(topicCode):

    '''
    This function gives a list with the unique categories that the network needs
    to be trained to categorize.
    '''
    categories = np.unique(topicCode)
    categories= categories.astype(int)
    categories= categories.astype(str)
    categories = list(categories)
    
    
    
    
    
def partition_training_data(speech, topicCode, test_size = 0.2):

  '''
  This function splits the training data to training and testing data.
  Can be used again to split the test data to test and validation data.
  '''

  data = list(zip(speech, topicCode)) # link the lists
  random.shuffle(data)    # shuffle

  data_train, data_test = train_test_split(data,test_size=test_size) # partition


  ## check that test data include every category
  unique1 = []
  for i in range(len(data_train)):
    if data_train[i][1] not in unique1:
      unique1.append(data_train[i][1])

  unique2 = []
  for i in range(len(data_test)):
    if data_test[i][1] not in unique2:
      unique2.append(data_test[i][1])
  
  if len(unique1) != len(unique2):
    raise ValueError('Missing categories from the test sample:', len(unique1), 'vs', len(unique2))


  return data_train, data_test
    
    
    
    

def nn_inputs(data_train, data_test = [], data_validate = []):

  '''
  This function creates the variables that will be fed into the neural network 
  in their final form. It receives as input the training/testing/validation data.
  '''

  labels_train = []
  texts_train = []

  texts_train[:], labels_train[:] = zip(*data_train) # unlink the lists


  if data_test != []:
    labels_test = []
    texts_test = []

    texts_test[:], labels_test[:] = zip(*data_test)

  if data_validate != []:
    labels_validate = []
    texts_validate = []

    texts_validate[:], labels_validate[:] = zip(*data_validate)


  if data_test == [] and data_validate == []:
    return texts_train, labels_train

  elif data_test != [] and data_validate == []:
    return texts_train, labels_train, texts_test, labels_test

  elif data_test != [] and data_validate != []:
    return texts_train, labels_train, texts_test, labels_test, texts_validate, labels_validate







def evaluation_metrics(predicted_categories, true_categories, accuracy = True, confusionMatrix = True, Precision = True, Recall = True, f1 = True, cohen = True):

  '''
  This function creates evaluation metrics for the test data coded by the trained
  neural network
  '''
  
  y_pred = [None] * len(predicted_categories)
  y_prob = [None] * len(predicted_categories)
  for i in range(len(predicted_categories)):
      y_prob[i] = np.max(predicted_categories[i]) # get highest probability
      y_pred[i] = np.argmax(predicted_categories[i]) # get class with highest probability
      y_pred[i] = np.str(y_pred[i]) # turn them to strings


  # Accuracy
  accuracy = accuracy_score(y_pred, true_categories)
  print('accuracy:', accuracy)

  # Confusion matrix
  confusion = confusion_matrix(y_pred, true_categories)

  # Precision 
  precision = precision_score(y_pred, true_categories, average = 'weighted')
  print('precision:', precision)
  # Recall
  recall = recall_score(y_pred, true_categories, average = 'weighted')
  print('recall:', recall)
  # F1 score
  f1 = f1_score(y_pred, true_categories, average = 'weighted')
  print('f1:', f1)
  # Cohen's kappa
  cohen = cohen_kappa_score(y_pred, true_categories)
  print('Cohen kappa:', cohen)

  metrics = [accuracy, precision, recall, f1, cohen]

  return metrics, confusion








### Prepare data (Nigeria)
# Give the location of the file 
loc1 = ("/content/gdrive/My Drive/Denmark/Nigeria.xlsx")
# load data set
Nigeria_bills = pd.read_excel(loc1, encoding = "utf-8")






# drop if topic is not coded
Nigeria_bills_coded = Nigeria_bills.dropna(subset=['Eda Codes'])

# Create major topic code (here we have coded only the sub-topic, whose first 1 or 2 digits are the same as its major topic)
codes = list(Nigeria_bills_coded['Eda Codes'])

for i in range(len(codes)):
  length = len(str(codes[i]))
  if length == 6:
    n = 2
  else:
    n = 1

  codes[i] = codes[i] // 10 ** (int(math.log(codes[i], 10)) - n + 1)




## use custom functions

# prepare main variables 
speech, topicCode =  process_network_data(Nigeria_bills_coded.Bill, codes)
# get categories
categories = get_categories(topicCode)
# partition data
data_train, data_test = partition_training_data(speech, topicCode, test_size = test_size)
# create train and test inputs
texts_train, labels_train, texts_test, labels_test = nn_inputs(data_train, data_test)






### Preprocess the Dataset using ktrain
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=texts_train, y_train=labels_train,
                                                                       x_test=texts_test, y_test=labels_test,
                                                                       maxlen=75, 
                                                                       preprocess_mode='bert',
                                                                       class_names = categories)



### Load pretrained model and wrap it in a ktrain.Learner object
model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)

learner = ktrain.get_learner(model,train_data=(x_train, y_train), batch_size=6)




### Train and Fine-Tune the Model on the training dataset
learner.fit_onecycle(0.0001, 7)   # set training rate and number of epochs


    
    
### Evaluations

predictor = ktrain.get_predictor(learner.model, preproc)
# feed test data to the trained model
y_probs = predictor.predict(texts_test, return_proba=True) # probabilities for each class
# evaluate results
metrics, confusion = evaluation_metrics(y_probs, labels_test)

# plot confusion matrix
plt.figure(figsize=(10, 8))
plt.title('Heatmap for Nigerian test data')
sn.heatmap(confusion, annot=True, cmap="BuPu")
    
    
    
    
# predictor.save('/content/gdrive/My Drive/CMSE_BERT/Nigeria_code_net')




### Code uncoded data from Nigeria

# create data frame with uncoded observations
Nigeria_bills_uncoded = Nigeria_bills[pd.isnull(Nigeria_bills['Eda Codes'])]

# prepare main variables 
speech, topicCode =  process_network_data(Nigeria_bills_uncoded.Bill, codes) # topic code here will be null for all cases

# predict new observations
y_probs = predictor.predict(speech, return_proba=True) # probabilities for each class


# get highest probability
y_pred = [None] * len(y_probs)
y_prob = [None] * len(y_probs)
for j in range(len(y_probs)):
  y_prob[j] = np.max(y_probs[j]) # get highest probability
  y_pred[j] = np.argmax(y_probs[j]) # get class with highest probability

























    
    
    
    






