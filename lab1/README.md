# Lab 1 README

In this lab assignment, we built a scalable serverless machine learning system with a feature, training and batch interference pipeline as we, as well as a user interface. This task was completed using three frameworks/platforms:
* Hopsworks as feature store
* Modal for compute
* Huggingface as UI

## Machine Learning for the Titanic dataset

Our first step in this assignment (apart from building and running the Iris Flower Dataset as a serverless system) was to locally build a machine learning model for the Titanic dataset. This process was done in 4 steps and can be found under the folder `titanic_local_modelling`:
* Data Exploration (data_exploration.ipynb)
* Preprocessing (preprocessing.ipynb)
* Feature Engineering (feature_engineering.ipynb)
* Training a model (modelling.ipynb)

### Data Exploration
In the data exploration, we investigated the data - what types is the data, what range is it and how many values are missing. We also plotted the features against the label Survived to get an understanding of how the values of the features might be linked to the outcome (survived or not).

### Preprocessing
Our next step was processing the data. This meant dropping features which had no correlation to the outcome (survived or not) like PassangerID and Ticket. These two features are pretty much unique values for each passanger and had no correlation to the outcome.  

We also dealt with missing values for the Cabin (687/891 values missing), Age (177/891 values missing) and Embarked (2/891 values missing) features. For the first of these features, Cabin, we stripped the first letter of the cabin and mapped it to an integer (A:1, B:2, C:3 etc). We then filled all the missing values with an unique integer 0. For the Age feature, we extracted the mean and standard deviation of the values and generated random values using this information and filled the missing values with these. For the Embarked feature, we filled the missing values with the most common embarked value.  

We then transformed/converted features in to easier to handle datatypes. Such as Fare into integer, Sex into numeric and Embarked into numeric.

### Feature Engineering
The next step was to create new features. This included extracting the titles of the names and the dropping the name feature. The name was previously more or less unique for each passanger and had little to no correlation to the outcome. The title however, could be usefull. We did some feature crossing/combinations such as Age multiplied with Pclass and extracting if a passanger was traveling alone or not.

### Modelling
The final step was training and testing different models. The notebook for this (modelling.ipynb) has been cleaned up and only the final model, RandomForest, is shown there.

## Combining our local model with Hopsworks, Modal and Huggingface

The second step of the assignment was to upload the local preprocessing, feature engineering and modelling steps to the scalable serverless machine learning frameworks. This step was completed by first going through, understanding and running the provided Iris Flower Dataset code. Secondly we replaced the given code with our preprocessing, feature engineering and modelling. The result of this can be seen the following files:
* titanic-feature-pipeline.py
* titanic-training-pipeline.py
* titanic-feature-pipeline-daily.py
* titanic-batch-interference-pipeline.py

And the final UI on Huggingface can be seen at:
* https://huggingface.co/spaces/Workinn/titanic
* https://huggingface.co/spaces/Workinn/titanic-monitor