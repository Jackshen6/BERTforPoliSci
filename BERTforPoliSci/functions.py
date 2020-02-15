# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:39:57 2020.

@author: NFran
"""

# Custom Functions to run BERT

### Working Functions for small BERT via ktrain library
# all of these functions can be used to run big BERT. It will be necessary to write a few extra functions in order to bypass ktrain.


def process_network_data(speech, topicCode):
  """
  This function prepares the two main variables of interest, text (speech, bill, etc.) and corresponding code to be fed into the network. Output variables are in string format.
  """
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
    """
    This function gives a list with the unique categories that the network needs to be trained to categorize.
    """
    categories = np.unique(topicCode)
    categories= categories.astype(int)
    categories= categories.astype(str)
    categories = list(categories)
    
    
    
    
    
def partition_training_data(speech, topicCode, test_size = 0.2):
  """
  This function splits the training data to training and testing data. 
  Can be used again to split the test data to test and validation data.
  """

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

  """
  This function prepares the variables that will be fed into the neural network so that they can be preprocessed. 
  It receives as input the training/testing/validation data.
  """

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

  """
  This function creates evaluation metrics for the test data coded by the trained
  neural network. Possible metrics are accuracy, confusion matrix, precision, recall, f1 score, cohen's k.
  """
  
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


###############################################################################
  
# Additional functions in stubbed form
  
def preprocess(x_train, y_train, x_test, y_test):
    
    """
    This function will do the necessary preprocessing steps to feed the data into BERT.
    """
    
    return x_train, y_train, x_test, y_test




def load_model(location, model_name):
    
    """
    This function will load the pretrained BERT model of our choice from the disk.
    """
    
    return model




def create_learner(parameter_a, parameter_b, etc):
    
    """
    This function will create the learner object and set the parameters. 
    """
    
    return learner


def train_net(learning_rate, num_epochs):
    
    """
    This function begins the training. It sets the training rate and number of epochs.
    It returnes the trained network.
    """
    
    print(training_process_statistcs)
    return trained_net




def classify_new(trained_net, unclassified_text):
    
    """
    This function will classify new text using the trained network.
   """
    
    return(classified_text)




















