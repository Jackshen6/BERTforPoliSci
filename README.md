# BERTforPoliSci

# <center>  What are they talking about? Text Classification using a Neural Network Approach </center>

<center>By Nikolaos Frantzeskakis</center>

# Overview

My broader research agenda has been developed around the use of text analysis to study African parliaments. African parliaments have long been understudied. One major reason is the lack of systematic quantitative data on legislatures and members of parliament. Nonetheless, the proliferation of text data available online in combination with the development of text analysis create the opportunity to create datasets that will enable us to study legislatures in Africa in more depth than ever before.

One area of research regards parliamentary bills. These bills, which are introduced and discussed in parliament, are crucial forms of policy output. By studying them we can infer a lot regarding a country's policy priorities. A vast literature has developed around the study of parliamentary bills in OECD countries.

A large branch of this research has been conducted based on the typology developed by the Comparative Agendas Project (Baumgartner et al. 2006). At it's heart, the Comparative Agendas Project is a typology of 22 major categories and about 200 sub-categories in which we can categorize political speech or text based on its content. As a result, in case we are interested in studying parliamentary bills, we can produce quantitative datasets describing the type of output of the legislature, on which we can then run statistical models.

While such data are available for most OECD countries, they are not available for any African countries. I am involved in a number of projects which aim to generate data that will allow us to further study African parliaments. Up to this point I have written code that is capable of categorizing bills, as well as parliamentary transcripts with a very high accuracy rate. My goal for this project is to professionalize my existing code and extend its capabilities.



# Program Description

At this point I have code developed a number of functions which allow a user to easily make use of BERT, which is made available by Google (Devlin 2018). BERT is to the best of my knowledge the most advanced publicly available neural netowork trained for NLP tasks. It became available about a year ago, at which point it comfortably broke a large number of records on tasks like text categorization and next sentence prediction. Nonetheless, I am not aware of any papers (published or in progress) which are making use of BERT for research in the discipline of political science.

In order to deploy BERT, I made use of ktrain (Maiya 2018). This is a python library which allows users to easily modify the original model in order to tackle the problem at hand. Through ktrain I can use the "small" version of BERT, which is 32 layers deep, as opposed to the "large" version which has 64 layers.

As a next step, I want to prfessionalize my existing code by adding more thorough documentation, unit testing the code, and adding extra robustness checks.

A more ambitious goal is to bypass ktrain. While this is a great library for someone to start working with BERT, it makes a number of important choices for you. First and foremost, it is not possible to use the large version of BERT through ktrain. So, I would like to create a program that will be able to easily make use of the large version of BERT through a few functions. 

Finally, "large" BERT is very heavy for normal machines to run. To overcome this issue, I plan to develop this program using HPCC, MSU's high performance computing cluster. 



# Project Goals and Timeline

While this project is quite ambitious it is not difficult to set a number of milestones. First, I will work on creating a git repository for the project. The second step will be to complete the existing code by further developing the documentation for the existing functions and making it more robust by additing more validity checks and unit testing. These things should not take more than three weeks.

The next step will be to create a simple pipeline that will be making use of large BERT to code some existing data. At this stage it will be acceptable for the code to be ran on Google colaboratoy. This step will be completed by 2/14/2020. The following step will be to modularize the code, unit test the program and make it run on HPCC. A first complete draft of the program will be available by 3/20/2019. This should allow for ample time to finish the project until 4/20/2019.



# Anticipating Challenges  

While the first steps of the project are very simple, completing all the steps laid out in the earlier sections is going to be very challenging. The first has to do with the number of packages I will need to work with. Keras and Tensorflow, which are basic packages I will be using, were recently updated. On the other hand, libraries making it possible to implement BERT had not been updated until very recently. So, the first hurdle will be to design an environment where it will be possible to use all the necessary libraries in their optimal versions.

The second difficulty will be developing the first verion of the pipeline for large Bert. Ktrain makes it very easy to implement BERT's simpler version, but creating the pipeline without such a safety net will be considerably more difficult.



# References
Baumgartner, Frank R., Christoffer Green-Pedersen, and Bryan D. Jones. "Comparative studies of policy agendas." Journal of European Public Policy 13.7 (2006): 959-974.

Devlin, Jacob, et al. "Bert: Pre-training of deep bidirectional transformers for language understanding." *arXiv preprint arXiv:1810.04805* (2018).

Maiya, Arun S. "ktrain: A Lightweight Wrapper for Keras to Help Train Neural Networks." DOI: https://github.com/amaiya/ktrain (2018).

