# Activity Recognition

![image](https://user-images.githubusercontent.com/45201620/92718617-a9008080-f37f-11ea-9d41-27026db7912c.png)

## Requirements
* MATLAB
* Dataset from the UCI Machine Learning Repository

## Dataset
Dataset was obtained from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/index.php)

The dataset has 561 predictor variables and the response variable as Activity that can classify into 6 different classes or categories Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying. The sensor values were obtained from a Samsung Smartphone's accelerometer and gyroscope at the rate of 50 Hz. The dataset was then divided into test and training dataset in the ration 7:3 respectively.
The sensor data was pre-processed using noise filters and then sampled.

Using Machine Learning to predict the classes would require to train the model over the training data with the input and output values. The model can then be tested for the test data to check the performance on previously unseen data. This would give information on the model if the model overfits or underfits on the test data.

To import the dataset:
```
$ readtable('')
```
