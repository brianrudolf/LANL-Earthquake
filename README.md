# LANL-Earthquake [Kaggle Competition](https://www.kaggle.com/c/LANL-Earthquake-Prediction/)
A machine learning pipeline for Kaggle competition predicting the time remaining before a laboratory earthquake occurs. 

## Data available 
The competition makes available one large sample of data with two variables; measured acoustic data from the experiment and the time remaining before a laboratory earthquake occurs. There are over 600 million pairs of data in the data provided for training. The test data contains over 2600 sequences of acoustic data (150,000 points long), and the goal is to predict the time remaining before the next earthquake occurs for each of the test sequences. 

One particular note on the training data is that a very large acoustic emission occurs prior to each earthquake. (notebook to be added to show this)

## Defining divisions of the data
The provided test data is noted by the competition as being continuous data samples within each test sequence, but overall the data does not represent a continuous sequence of data. Upon analyzing the training data, very large acoustic emmisions are observed similar to those occuring in the training data. (notebook to be added to show this) It is therefore decided to include these acoustic samples in the training data rather than dismiss them as abnormal outliers. The working theory is that the test samples are taken from within time intervals prior to an earthquake, but do not contain earthquakes themselves. Thus, similar data samples for training and validating a regression model will need to be created. 

## Creating data samples
To create data samples from data between but not including earthquakes, the time (index within the data) when earthquakes occur need to be identified. The script 'identify_quakes.py' reads through all of the labelled data searching for earthquakes and records where that quake occurs. The earthquake is nominally denoted by 'time_to_faulure' reaching zero, however in the data this value is not definitely reached, so the steps between the temporal data is observed and the program searches for a (relatively) large positive change to detect when a new earthquake interval period begins.

The purpose here is to create a record of these earthquakes for further data manipulation without having to run this scan again.
Having a record of the earthquakes allows for downstream data manipulation without repeating this work over again.

The goal now is to create data samples that are 150,000 points long (to match training and test data formats). The script 'create_raw_samples.py' uses a rolling window / aperture to create more data samples than simply separating the available data into non-overlapping sequences. By using this method, segments of acoustic data will be duplicated across different data samples and it is imperative that all of the data from each earthquake sequence be used exclusively in either training data or cross validation data. If data samples from the same earthquake interval are shuffled and then split into train and validate, it could lead to identical data existing within the train and validation set due to the overlapping window used in data sample creation.

The purpose of separating out this sample creation step into its own script is to allow for different feature extraction methods on these data samples to be tested downstream without having to re-create these samples (which would be redundant and a waste of time).

## Feature generation
To build a regression model to predict the time remaining before an earthquake occurs, each of the data samples need to be characterized by a set of features. As it is not immediately clear how best to characterize the time sequences, many statistical features are gathered. A future improvement to this pipeline will see the features reduced by eliminating features that are shared / similar across data samples (scikit learn's [feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) could be used for example). 

The script 'create_features.py' creates four sets of train and validation data based on a 4-fold cross validation scheme. With 16 earthquake intervals, each set of data uses 12 intervals for the training data and the remaining 4 intervals for the cross validation data. The features for the samples within each interval are generated once, and a subsequent loop created the 4 different sets of train/val data, shuffles the data points, and saves them to disk for later use. Feature generation is defined within the script 'quake_features.py', where the Numba library is used to speed up computation and run the for loop in parallel across multiple CPU cores (future improvement would like to see the feature extraction vectorized). 

## Model training
The 'train_model.py' script efficiently imports the training and validation data, fits a scikit learn scaler to the training data, transforms the train and validation data with it, trains a SVR model, and finally scores said model with the mean absolute error of both the training and validation data as well as providing a the R^2 score from the scikit learn library.

## Prediction on test data samples
To complete the exercise for this competition, time to failure values are predicted for each of the test data samples provided. The script 'predict_on_test.py' loads in the saved model and data scaler (the best model can be selected from the 'train_model' script based on the model's performance on the validation set), generates features for the test data, scales said features in the same manner as the training data, and makes a prediction for the time remaining. A csv file is then created for submission to Kaggle. 
