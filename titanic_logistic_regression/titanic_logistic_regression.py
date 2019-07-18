#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: samuel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



df_train = pd.read_csv("titanic/train.csv")
df_test = pd.read_csv("titanic/test.csv")
y_test = pd.read_csv('titanic/gender_submission.csv')



print('Create heat map of Null data in training and test sets \n')

print('\n'*5 + 'TRAINING DATASET')

sns.heatmap(df_train.isnull(), cbar=False, yticklabels=False)
plt.show()
plt.clf()
print('\n'*5 + 'TESTNG DATASET')

sns.heatmap(df_test.isnull(), yticklabels=False, cbar = False)
plt.show()
plt.clf()
print("Cabin data has to many null values so it will be ignored in analysis...")

print('\n'*5 + 'Different Graphs to explore Data')
sns.set_style("darkgrid")


sns.countplot(x='Survived', data=df_train, hue  = 'Sex')
plt.show()
plt.clf()

print('\n'*5)
sns.countplot(x='Survived', data=df_train, hue  = 'Pclass')
plt.show()

#Cleaning Data


def impute_age_calc(c):
    if pd.isnull (c[0]) :
        return df_train[df_train['Pclass'] == c[1]]['Age'].mean()  #returning mean of class that nan is in
    else:
        return c[0]





y_test = y_test.drop('PassengerId', axis = 1)

df_test = pd.concat([df_test, y_test], axis = 1)
   
#Setting age based of passenger class (wealther people tend to older)
df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age_calc, axis = 1)
df_test['Age'] = df_train[['Age','Pclass']].apply(impute_age_calc, axis = 1)



sex = pd.get_dummies(df_train['Sex'], drop_first = True)
embark = pd.get_dummies(df_train['Embarked'], drop_first = True)
df_train = pd.concat([df_train,sex,embark],axis = 1)

sex = pd.get_dummies(df_test['Sex'], drop_first = True)
embark = pd.get_dummies(df_test['Embarked'], drop_first = True)
df_test= pd.concat([df_test,sex,embark],axis = 1)


df_train.drop(['Cabin', 'Name', 'Ticket','Sex','Embarked', 'PassengerId'], axis = 1, inplace = True)
df_test.drop(['Cabin', 'Name', 'Ticket','Sex','Embarked', 'PassengerId'], axis = 1, inplace = True)
df_train.dropna(inplace = True)
df_test.dropna(inplace = True)


#Running Logistic Regression
X_train = df_train.drop('Survived',axis = 1)
y_train = df_train['Survived']

X_test = df_test.drop('Survived',axis = 1)

y_test = df_test['Survived']


logReModel = LogisticRegression()

logReModel.fit(X_train,y_train)

predictions = logReModel.predict(X_test)


print("Final info")
print(classification_report(y_test,predictions))

