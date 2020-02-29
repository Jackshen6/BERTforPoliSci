
# Topic: Unit test the BERT classification pipeline
# Author: Nikolas Frantzeskakis
# Date: 2/28/2019



### note to self: to conduct tests run the following in the command line
# set Pythonpath=C:\Users\NFran\Dropbox\Courses MSU\4. Spring 2020\CMSE 802, Computational Modeling\BERTforPoliSci\BERTforPoliSci;%Pythonpath%




import bert_pipeline as bp
import pytest


def test_process_network_data():

    ex_speech = list()
    ex_speech.append("It is the economy stupid!")
    ex_speech.append("Hello World")
    ex_code = [1, 99]
    
    speech, topicCode = bp.process_network_data(ex_speech,ex_code)
    
    # test if function returns output in the correct format
    assert isinstance(topicCode[0], str)
    # test if function transforms topic codes as required
    assert topicCode[1] == "0"




def test_get_categories():
    
    ex_topics = ['0','0','1','1','1','1','2','3','3','4','4','4','5','6']

    cats = bp.get_categories(ex_topics)

    # test if function returns the right number of categories
    assert len(cats) == 7
    # test if function returns output in the correct format
    assert type(cats) == list
    assert isinstance(ex_topics[0], str)
    
    


def test_partition_training_data():
    
    ex_speech = list()
    ex_speech.append("It is the economy stupid!")
    ex_speech.append("It is the economy stupid!")
    ex_speech.append("It is the economy stupid!")
    ex_speech.append("It is the economy stupid!")
    ex_speech.append("It is the economy stupid!")
    ex_speech.append("Hello World")
    ex_speech.append("Hello World")
    ex_speech.append("Hello World")
    ex_speech.append("Hello World")    
    ex_speech.append("Hello World")
    
    ex_code = ['1', '1', '1', '1', '1', '0', '0', '0', '0', '0'] 
    
    
    # Test if the function shuffled both variables in the same way
    data_train, data_test = bp.partition_training_data(ex_speech, ex_code, test_size = 0.5)
    
    for i in range(len(data_train)):
        if data_train[i][0] == "It is the economy stupid!":
            assert data_train[i][1] == '1'
        else:
            assert data_train[i][1] == '0'
    
    # Test if the function raises the error when training and test samples do not have the same categories
    with pytest.raises(ValueError) as excinfo:
            data_train, data_test = bp.partition_training_data(ex_speech, ex_code, test_size = 0.1)

    assert "Missing categories from the test sample:" in str(excinfo.value)



    
