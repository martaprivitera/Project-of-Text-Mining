# Project-of-Text-Mining

## 01_Data_Preparation: preparation of the dataset. 

Following actions have been performed:

-	 handling missing data
-	 reducing the corpus dimension
-	 handling default class ‘0’
-	 filtering out non english reviews
-	 mapping classes to label (binary classification)

## 02_Data_PreProcessing: general pre-processing

-	remotion of numbers
-	normalization
-	output file: cleaning dataset for the task

## 03_1_TF_IDF_Word2Vec:

-	training and evaluation of classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost) using TF_IDF 
-	training and evaluation of classifiers (Logistic Regression, Decision Tree, Random Forest, KNN) using Word2Vec

## 03_2_GloVe_Baseline_Model:

-	Baseline Model (1)
-	Embedding layer
-	LSTM layer with 100 units and 10% dropout 
-	Dense layer with sigmoid as activation function
-	Baseline Model (2)
-	Embedding layer
-	LSTM layer with 128 units and 10% dropout 
-	Dense layer with sigmoid as activation function

## 03_3_GloVe_LSTM_Conv:

LSTM Neural Network (1):

-	Embedding layer
-	Three LSTM layers, each with 256 units and 50% dropout
-	Dropout layers that introduced a 20% dropout rate 
-	Dense layer with sigmoid as activation function

LSTM Neural Network (2):

-	Augmenting complexity by introducing two additional dense layers one with 128 nodes and another with 64 nodes.

LSTM Neural Network (3):

-	Retaining only a single LSTM layer with 256 units
-	Eliminating the two additional dense layers introduced
-	Dense layer with sigmoid as activation function
  
CONV Neural Network (1):

-	Embedding layer
-	Three 1D convolutional layers 
-	Dropout layers with a dropout rate of 0.5 were added after the second and third layers
-	Global Max Pooling layer 
-	Two dense layers 256 units each, dropout 50% and ReLU activation
-	Dense layer with sigmoid as activation function

CONV Neural Network (2):
  
-	Reduced number of nodes in the layers and incorporated kernel regularization into the final two dense layers
  
CONV Neural Network (3):

-	Removed callback dynamic learning rate and kernel regularizers from the dense layers
  
CONV Neural Network (4):

-	Adding two kernel regularizers to the first two layers of the neural network 

## 03_4_GloVe_Hybrid_Networks:

Model (1)
-	Embedding layer
-	LSTM layer with 128 units, a dropout rate 20% and recurrent dropout of 20% 
-	1D convolutional layer with 128 filters and a kernel size of 5 
-	Global max pooling 1D layer
-	Two fully connected dense layers (first layer with 64 units and ReLU activation, second with 1 unit and sigmoid activation

Model (1.bis)
-	Reducing epochs 


# 04_Topic_modeling:

Second task, training and evaluation of different models. 
For each of the 4 classes, models have been trained on subsets of docs (20k, 15k, 10k, 5k) and on different numbers of topics (from 2 to 15).
Best model selection in each class according coherence value.

Model trained:
-	LSA models 
-	LDA models (from sklearn, Gensim and Multicore)
-	BERT
-	Top2Vec models

In order to run this file, please, follow these steps:

1.	run the first and the second chunks in the “Install” section. Then select “restart session” under the second chunk and put a # before all the content in the second chunk.
2.	rerun all the sections: “Install”, “Import and Drive Mount”, “Dataset Preparation”, “LSA models”, “LDA models with sklearn”, “LDA models with Gensim”, “LDA models with Multicore”, and “BERTopic models”.
3.	in order to run the “Top2Vec” section it is necessary to restart the session, then run the “Import and Drive Mount”, “Data Preparation” sections and then the “Top2Vec” section

