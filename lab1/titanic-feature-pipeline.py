import os
import modal
from hsfs.feature import Feature

LOCAL = True
DATASET_URL = "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv"

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().pip_install(["hopsworks", "pandas", "numpy"])


    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()


def g():
    import hopsworks
    import pandas as pd
    import re
    import numpy as np

    project = hopsworks.login()
    fs = project.get_feature_store()
    data_df = pd.read_csv(DATASET_URL, index_col=0)

    data_df = data_df.drop(['Ticket'], axis=1)

    # Cabin Feature
    deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}

    data_df['Cabin'] = data_df['Cabin'].fillna("U0")
    data_df['Deck'] = data_df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    data_df['Deck'] = data_df['Deck'].map(deck)
    data_df['Deck'] = data_df['Deck'].fillna(0)
    data_df['Deck'] = data_df['Deck'].astype(int)

    # we can now drop the cabin feature
    data_df = data_df.drop(['Cabin'], axis=1)

    # Age Feature
    mean = data_df["Age"].mean()
    std = data_df["Age"].std()

    is_null = data_df["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    # fill NaN values in Age column with random values generated
    age_slice = data_df["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data_df["Age"] = age_slice
    data_df["Age"] = data_df["Age"].astype(int)

    # Embarked Feature
    most_common_val = data_df['Embarked'].describe()[2]
    data_df['Embarked'] = data_df['Embarked'].fillna(most_common_val)

    # Transforming / Converting Features
    data_df['Fare'] = data_df['Fare'].astype(int)
    genders = {"male": 0, "female": 1}
    data_df['Sex'] = data_df['Sex'].map(genders)
    ports = {"S": 0, "C": 1, "Q": 2}
    data_df['Embarked'] = data_df['Embarked'].map(ports)

    # Extract `Titles` from `Name` Feature
    titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    # extract titles
    data_df['Title'] = data_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    data_df['Title'] = data_df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data_df['Title'] = data_df['Title'].replace('Mlle', 'Miss')
    data_df['Title'] = data_df['Title'].replace('Ms', 'Miss')
    data_df['Title'] = data_df['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    data_df['Title'] = data_df['Title'].map(titles)
    # filling NaN with 0, to get safe
    data_df['Title'] = data_df['Title'].fillna(0)
    data_df = data_df.drop(['Name'], axis=1)

    data_df['Age_Class'] = data_df['Age'] * data_df['Pclass']

    data_df['Relatives'] = data_df['SibSp'] + data_df['Parch']
    data_df.loc[data_df['Relatives'] > 0, 'Not_alone'] = 0
    data_df.loc[data_df['Relatives'] == 0, 'Not_alone'] = 1
    data_df['Not_alone'] = data_df['Not_alone'].astype(int)

    data_df['Fare_Per_Person'] = data_df['Fare'] / (data_df['Relatives'] + 1)
    data_df['Fare_Per_Person'] = data_df['Fare_Per_Person'].astype(int)

    # Add id for the feature store
    data_df['Id'] = data_df.index

    data_df['Age'] = data_df['Age'].astype("int64")
    data_df['Age_Class'] = data_df['Age_Class'].astype("int64")
    data_df['Deck'] = data_df['Deck'].astype("int64")
    data_df['Embarked'] = data_df['Embarked'].astype("int64")
    data_df['Fare'] = data_df['Fare'].astype("int64")
    data_df['Fare_Per_Person'] = data_df['Fare_Per_Person'].astype("int64")
    data_df['Id'] = data_df['Id'].astype("int64")
    data_df['Not_alone'] = data_df['Not_alone'].astype("int64")
    data_df['Parch'] = data_df['Parch'].astype("int64")
    data_df['Pclass'] = data_df['Pclass'].astype("int64")
    data_df['Relatives'] = data_df['Relatives'].astype("int64")
    data_df['Sex'] = data_df['Sex'].astype("int64")
    data_df['SibSp'] = data_df['SibSp'].astype("int64")
    data_df['Survived'] = data_df['Survived'].astype("int64")
    data_df['Title'] = data_df['Title'].astype("int64")

    titanic_fg = fs.get_or_create_feature_group(
        name="titanic",
        version=1,
        primary_key=['Id'],
        description="Processed titanic survivors dataset")
    titanic_fg.insert(data_df, write_options={"wait_for_job": False})


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()
