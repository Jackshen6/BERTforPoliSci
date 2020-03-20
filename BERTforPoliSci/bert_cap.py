# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:51:48 2020

@author: NFran
"""

# Custom functions necessary to use BERT to code data based on the
# Comparative Agendas Project typology.


# imports
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score


def recode_data(speech, topicCode, sub_to_major=0):
    '''
    This function recodes hand coded data to make their processing more straightforward:
    - It recodes the "other" category to be 0
    - It can extract the major topic from the sub-topic
    '''

    # sort based on topic
    topicCode, speech = zip(*sorted(zip(topicCode, speech)))
    speech = list(speech)

    topicCode = np.array(topicCode)
    topicCode = topicCode.astype(int)

    # recode "other" category to 0
    topicCode = np.where(topicCode == 99, 0, topicCode)
    topicCode = np.where(topicCode == 100, 0, topicCode)

    # if sub_to_major == 1 replace sub-code with major code
    if sub_to_major == 1:
        topicCode = topicCode // 100

    topicCode = np.where(topicCode < 0, 0, topicCode)
    topicCode = topicCode.astype(str)

    return speech, topicCode


def get_categories(topicCode):
    '''
    This function gives a list with the unique categories that the network needs
    to be trained to categorize.
    '''
    categories = np.unique(topicCode)
    categories = categories.astype(int)
    categories = categories.astype(str)
    categories = list(categories)

    return categories


def partition_training_data(speech, topicCode, test_size=0.2, retry=1):
    '''
    This function splits the training data to training and testing data.
    Can be used again to split the test data to test and validation data.
    '''

    data = list(zip(speech, topicCode))  # link the lists
    random.shuffle(data)    # shuffle

    unbalanced = 1

    while unbalanced == 1:    # repeat until training and testing samples have all categories

        data_train, data_test = train_test_split(
            data, test_size=test_size)  # partition

        # check that test data include every category
        unique1 = np.unique([item[1] for item in data_train])
        unique2 = np.unique([item[1] for item in data_test])

        if len(unique1) != len(unique2):
            # if retry is set to 0, raise error and exit
            if retry == 0:
                unbalanced = 0
                raise ValueError(
                    'Missing categories from the test sample:',
                    len(unique1),
                    'vs',
                    len(unique2))
            else:
                # else run again
                print('Missing categories from the test sample, trying again.')
        else:
            # when samples are balanced, exit
            print('All categories included in test sample.')
            unbalanced = 0

    return data_train, data_test


def nn_inputs(data_train, data_test=[], data_validate=[]):
    '''
    This function creates the variables that will be fed into the neural network
    in their final form. It receives as input the training/testing/validation data.
    '''

    labels_train = []
    texts_train = []

    texts_train[:], labels_train[:] = zip(*data_train)  # unlink the lists

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


def evaluation_metrics(
        y_pred,
        true_categories,
        accuracy=True,
        confusionMatrix=True,
        Precision=True,
        Recall=True,
        f1=True,
        cohen=True):
    '''
    This function creates evaluation metrics for the test data coded by the trained
    neural network
    '''

    # Accuracy
    accuracy = accuracy_score(y_pred, true_categories)
    print('accuracy:', accuracy)

    # Confusion matrix
    confusion = confusion_matrix(y_pred, true_categories)

    # Precision
    precision = precision_score(y_pred, true_categories, average='weighted')
    print('precision:', precision)
    # Recall
    recall = recall_score(y_pred, true_categories, average='weighted')
    print('recall:', recall)
    # F1 score
    f1 = f1_score(y_pred, true_categories, average='weighted')
    print('f1:', f1)
    # Cohen's kappa
    cohen = cohen_kappa_score(y_pred, true_categories)
    print('Cohen kappa:', cohen)

    metrics = [accuracy, precision, recall, f1, cohen]

    return metrics, confusion
