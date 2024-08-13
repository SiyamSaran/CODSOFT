
#import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#importing data
titanic_data=pd.read_csv(r"E:\\Titanic-Dataset.csv")
print(titanic_data.head(15))
titanic_data.shape
titanic_data.info()

#Preprocessing the data
titanic_data.isnull().sum()
titanic_data=titanic_data.drop(columns='Cabin',axis=1)
titanic_data.fillna({'Age': titanic_data['Age'].mean()}, inplace=True)
titanic_data.isnull().sum()
print(titanic_data['Embarked'].mode())

print(titanic_data['Embarked'].mode()[0])
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data.isnull().sum()


titanic_data.describe()

#Visualizing the data
sns.set()
sns.countplot(titanic_data['Sex'])
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)


titanic_data['Sex'].value_counts()
titanic_data['Embarked'].value_counts()
pd.set_option('future.no_silent_downcasting', True)
titanic_data.replace({'Sex': {'male': 1, 'female': 2}, 'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, inplace=True)

#Identifying  the features
X = titanic_data.drop(columns =['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y = titanic_data['Survived']

#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()


#Fitting the model with the training data
model.fit(X_train, y_train)
X_train_prediction = model.predict(X_train)


accuracy = accuracy_score(y_train,  X_train_prediction)
print("Accuracy of the training data:", accuracy)


X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(y_test, X_test_prediction)
print("Accuracy of testing data:", test_accuracy)

#Evaluating the model
def get_passenger_details():
    Pclass = int(input("Enter Passenger Class (1, 2, or 3): "))
    Sex = input("Enter Sex (male or female): ").lower()
    Age = float(input("Enter Age: "))
    SibSp = int(input("Enter number of siblings/spouses aboard: "))
    Parch = int(input("Enter number of parents/children aboard: "))
    Fare = float(input("Enter Fare: "))
    Embarked = input("Enter Port of Embarkation (S, C, Q): ").upper()
    Sex = 0 if Sex == 'male' else 1
    Embarked = {'S': 0, 'C': 1, 'Q': 2}[Embarked]
    passenger_data = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [Sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [Embarked]
    })
    prediction = model.predict(passenger_data)
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    print(f"The passenger would have: {result}")

get_passenger_details()