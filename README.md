# BERTforPoliSci

# <center>  What are they talking about? Text Classification using a Neural Network Approach </center>

<center>By Nikolaos Frantzeskakis</center>


# Program Description

At this point I have code developed a number of functions which allow a user to easily make use of BERT, which is made available by Google (Devlin 2018). BERT is to the best of my knowledge the most advanced publicly available neural netowork trained for NLP tasks. It became available about a year ago, at which point it comfortably broke a large number of records on tasks like text categorization and next sentence prediction. Nonetheless, I am not aware of any papers (published or in progress) which are making use of BERT for research in the discipline of political science.

In order to deploy BERT, I made use of ktrain (Maiya 2018). This is a python library which allows users to easily modify the original model in order to tackle the problem at hand. Through ktrain I can use the "small" version of BERT, which is 32 layers deep, as opposed to the "large" version which has 64 layers.

As a next step, I want to prfessionalize my existing code by adding more thorough documentation, unit testing the code, and adding extra robustness checks.

A more ambitious goal is to bypass ktrain. While this is a great library for someone to start working with BERT, it makes a number of important choices for you. First and foremost, it is not possible to use the large version of BERT through ktrain. So, I would like to create a program that will be able to easily make use of the large version of BERT through a few functions. 

Finally, "large" BERT is very heavy for normal machines to run. To overcome this issue, I plan to develop this program using HPCC, MSU's high performance computing cluster. 
