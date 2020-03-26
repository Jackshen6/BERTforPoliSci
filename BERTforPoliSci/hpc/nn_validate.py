# -*- coding: utf-8 -*-
"""NN_validate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-agxBJAhDhrlgE2--tRbFMCUU1B83pPg

Notebook description:
Train network based on US bills.
"""


### Setup
# import packages
import numpy as np
import random
import seaborn as sn
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import ktrain
from ktrain import text

### import custom functions
import sys
import bert_cap as bc


test_size = 0.30
seed = 100000

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Load US bills
loc = ("./US-Legislative-congressional_bills_19.3_3_2.csv")
# load data set 1
bills = pd.read_csv(loc, encoding = "utf-8")

## prepare dataset

# drop if before 1990
bills = bills[bills.year >= 1990]

## use custom functions

# prepare main variables 
speech, topicCode = bc.recode_data(bills.description, bills.majortopic)
# get categories
categories = bc.get_categories(topicCode)
# partition data
data_train, data_test = bc.partition_training_data(speech, topicCode, test_size = test_size)
# create train and test inputs
texts_train, labels_train, texts_test, labels_test = bc.nn_inputs(data_train, data_test)

### Preprocess the Data and Create a Transformer Model
MODEL_NAME = 'bert-base-uncased'
t = text.Transformer(MODEL_NAME, maxlen=40, class_names=categories)
trn = t.preprocess_train(texts_train, labels_train)
val = t.preprocess_test(texts_test, labels_test)
model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=20)

### Train and Fine-Tune the Model on the training dataset
learner.fit_onecycle(0.0001, 2)   # set training rate and number of epochs

predictor = ktrain.get_predictor(learner.model, preproc = t)
# feed test data to the trained model
y_probs = predictor.predict(texts_test, return_proba=True) # probabilities for each class
# evaluate results
metrics, confusion = bc.evaluation_metrics(y_probs, labels_test)

# plot confusion matrix
plt.figure(figsize=(10, 8))
plt.title('Heatmap for US test data')
sn.heatmap(confusion, annot=True, cmap="BuPu")
